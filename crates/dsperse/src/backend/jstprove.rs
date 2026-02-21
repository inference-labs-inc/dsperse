use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::error::{DsperseError, Result};

#[derive(Debug)]
pub struct JstproveBackend {
    binary: PathBuf,
    compress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub succeeded: usize,
    pub failed: usize,
    pub errors: Vec<(usize, String)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipeWitnessJob {
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub witness: String,
}

#[derive(Serialize)]
struct BatchManifest<T> {
    jobs: Vec<T>,
}

impl JstproveBackend {
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            compress: true,
        }
    }

    pub fn with_compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    pub fn from_env() -> Result<Self> {
        let path = std::env::var("JSTPROVE_BINARY")
            .map_err(|_| DsperseError::Backend("JSTPROVE_BINARY not set".into()))?;
        Ok(Self::new(path))
    }

    pub fn compile(
        &self,
        circuit_path: &Path,
        metadata_path: &Path,
        architecture_path: &Path,
        wandb_path: Option<&Path>,
    ) -> Result<()> {
        let mut cmd = self.base_command();
        cmd.arg("run_compile_circuit")
            .arg("-c")
            .arg(circuit_path)
            .arg("-m")
            .arg(metadata_path)
            .arg("-a")
            .arg(architecture_path);

        if let Some(wandb) = wandb_path {
            cmd.arg("-b").arg(wandb);
        }
        if !self.compress {
            cmd.arg("--no-compress");
        }

        run_checked(cmd)
    }

    pub fn witness(
        &self,
        circuit_path: &Path,
        input_path: &Path,
        output_path: &Path,
        witness_path: &Path,
        metadata_path: &Path,
        wandb_path: Option<&Path>,
    ) -> Result<()> {
        let mut cmd = self.base_command();
        cmd.arg("run_gen_witness")
            .arg("-c")
            .arg(circuit_path)
            .arg("-i")
            .arg(input_path)
            .arg("-o")
            .arg(output_path)
            .arg("-w")
            .arg(witness_path)
            .arg("-m")
            .arg(metadata_path);

        if let Some(wandb) = wandb_path {
            cmd.arg("-b").arg(wandb);
        }
        if !self.compress {
            cmd.arg("--no-compress");
        }

        run_checked(cmd)
    }

    pub fn witness_piped(
        &self,
        circuit_path: &Path,
        metadata_path: &Path,
        jobs: &[PipeWitnessJob],
        wandb_path: Option<&Path>,
    ) -> Result<BatchResult> {
        let mut cmd = self.base_command();
        cmd.arg("run_pipe_witness")
            .arg("-c")
            .arg(circuit_path)
            .arg("-m")
            .arg(metadata_path);

        if let Some(wandb) = wandb_path {
            cmd.arg("-b").arg(wandb);
        }
        if !self.compress {
            cmd.arg("--no-compress");
        }

        let manifest = BatchManifest { jobs: jobs.to_vec() };
        let payload = serde_json::to_vec(&manifest)
            .map_err(|e| DsperseError::Backend(format!("serialize witness jobs: {e}")))?;

        run_piped(cmd, &payload)
    }

    pub fn prove(
        &self,
        circuit_path: &Path,
        witness_path: &Path,
        proof_path: &Path,
        metadata_path: &Path,
    ) -> Result<()> {
        let mut cmd = self.base_command();
        cmd.arg("run_prove_witness")
            .arg("-c")
            .arg(circuit_path)
            .arg("-w")
            .arg(witness_path)
            .arg("-p")
            .arg(proof_path)
            .arg("-m")
            .arg(metadata_path);

        if !self.compress {
            cmd.arg("--no-compress");
        }

        run_checked(cmd)
    }

    pub fn verify(
        &self,
        circuit_path: &Path,
        input_path: &Path,
        output_path: &Path,
        witness_path: &Path,
        proof_path: &Path,
        metadata_path: &Path,
    ) -> Result<()> {
        let mut cmd = self.base_command();
        cmd.arg("run_gen_verify")
            .arg("-c")
            .arg(circuit_path)
            .arg("-i")
            .arg(input_path)
            .arg("-o")
            .arg(output_path)
            .arg("-w")
            .arg(witness_path)
            .arg("-p")
            .arg(proof_path)
            .arg("-m")
            .arg(metadata_path);

        run_checked(cmd)
    }

    fn base_command(&self) -> Command {
        let mut cmd = Command::new(&self.binary);
        cmd.env("RUST_BACKTRACE", "1");
        cmd
    }
}

fn run_checked(mut cmd: Command) -> Result<()> {
    tracing::debug!(cmd = ?cmd, "jstprove");

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| DsperseError::Backend(format!("spawn jstprove: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        return Err(DsperseError::Backend(format!(
            "jstprove exited {code}: {stderr}"
        )));
    }

    Ok(())
}

fn run_piped(mut cmd: Command, stdin_payload: &[u8]) -> Result<BatchResult> {
    tracing::debug!(cmd = ?cmd, payload_len = stdin_payload.len(), "jstprove piped");

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| DsperseError::Backend(format!("spawn jstprove: {e}")))?;

    let payload = stdin_payload.to_vec();
    let stdin_handle = child.stdin.take();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        if let Some(mut stdin) = stdin_handle {
            use std::io::Write;
            stdin.write_all(&payload)?;
        }
        Ok(())
    });

    let output = child
        .wait_with_output()
        .map_err(|e| DsperseError::Backend(format!("wait jstprove: {e}")))?;

    match writer.join() {
        Ok(Err(e)) => {
            return Err(DsperseError::Backend(format!("stdin write: {e}")));
        }
        Err(_) => {
            return Err(DsperseError::Backend("stdin writer thread panicked".into()));
        }
        Ok(Ok(())) => {}
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        return Err(DsperseError::Backend(format!(
            "jstprove exited {code}: {stderr}"
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_last_json_object(&stdout)
}

fn parse_last_json_object(output: &str) -> Result<BatchResult> {
    for line in output.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with('{') {
            return serde_json::from_str(trimmed)
                .map_err(|e| DsperseError::Backend(format!("parse batch result: {e}")));
        }
    }
    Err(DsperseError::Backend(
        "no JSON found in jstprove output".into(),
    ))
}
