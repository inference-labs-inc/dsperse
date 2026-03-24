use std::collections::HashMap;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{DsperseError, Result};

pub struct PublishConfig {
    pub api_url: String,
    pub auth_token: String,
    pub circuit_id: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
    pub circuit_type: String,
    pub proof_system: String,
    pub timeout: u64,
    pub activate: bool,
}

pub struct PublishResult {
    pub circuit_id: String,
    pub files_uploaded: usize,
}

fn sha256_file(path: &Path) -> Result<String> {
    let data = fs::read(path).map_err(|e| DsperseError::io(e, path))?;
    let hash = Sha256::digest(&data);
    Ok(format!("{hash:x}"))
}

fn collect_files(dir: &Path) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| DsperseError::io(e, dir))? {
        let entry = entry.map_err(|e| DsperseError::io(e, dir))?;
        let path = entry.path();
        if path.is_file() {
            let name = entry
                .file_name()
                .to_str()
                .ok_or_else(|| DsperseError::Other("non-UTF-8 filename".into()))?
                .to_string();
            files.push((name, path));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

pub fn publish(dir: &Path, config: &PublishConfig) -> Result<PublishResult> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DsperseError::Other(format!("tokio runtime: {e}")))?;

    rt.block_on(publish_async(dir, config))
}

async fn publish_async(dir: &Path, config: &PublishConfig) -> Result<PublishResult> {
    if !dir.is_dir() {
        return Err(DsperseError::Other(format!(
            "directory not found: {}",
            dir.display()
        )));
    }

    let entries = collect_files(dir)?;
    if entries.is_empty() {
        return Err(DsperseError::Other("no files to publish".into()));
    }

    tracing::info!(count = entries.len(), "hashing files");
    let mut file_map: HashMap<String, String> = HashMap::new();
    let mut file_paths: HashMap<String, std::path::PathBuf> = HashMap::new();
    for (name, path) in &entries {
        let hash = sha256_file(path)?;
        tracing::info!(file = %name, hash = %hash, "hashed");
        file_map.insert(name.clone(), hash);
        file_paths.insert(name.clone(), path.clone());
    }

    let client = reqwest::Client::new();
    let api_url = config.api_url.trim_end_matches('/');

    let input_schema: serde_json::Value = serde_json::json!({});

    let body = serde_json::json!({
        "id": config.circuit_id,
        "metadata": {
            "name": config.name,
            "description": config.description,
            "author": config.author,
            "version": config.version,
            "type": config.circuit_type,
            "proof_system": config.proof_system,
            "netuid": null,
            "weights_version": null,
            "timeout": config.timeout,
            "input_schema": input_schema,
        },
        "files": file_map,
    });

    tracing::info!(url = %api_url, id = %config.circuit_id, "registering circuit");
    let resp = client
        .post(format!("{api_url}/admin/circuits"))
        .header("Authorization", format!("Bearer {}", config.auth_token))
        .json(&body)
        .send()
        .await
        .map_err(|e| DsperseError::Other(format!("register request: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp
            .text()
            .await
            .unwrap_or_else(|_| "<no body>".to_string());
        return Err(DsperseError::Other(format!(
            "register failed ({status}): {text}"
        )));
    }

    let register_resp: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| DsperseError::Other(format!("parse register response: {e}")))?;

    let upload_urls = register_resp["upload_urls"]
        .as_object()
        .ok_or_else(|| DsperseError::Other("missing upload_urls in response".into()))?;

    let mut uploaded = 0usize;
    for (filename, url_val) in upload_urls {
        let url = url_val
            .as_str()
            .ok_or_else(|| DsperseError::Other(format!("non-string URL for {filename}")))?;

        let path = file_paths
            .get(filename)
            .ok_or_else(|| DsperseError::Other(format!("no local file for {filename}")))?;

        let data = fs::read(path).map_err(|e| DsperseError::io(e, path))?;
        let size = data.len();

        tracing::info!(file = %filename, size, "uploading");
        let put_resp = client
            .put(url)
            .body(data)
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("upload {filename}: {e}")))?;

        if !put_resp.status().is_success() {
            let status = put_resp.status();
            return Err(DsperseError::Other(format!(
                "upload {filename} failed ({status})"
            )));
        }

        uploaded += 1;
    }

    if config.activate {
        tracing::info!("activating circuit");
        let activate_resp = client
            .patch(format!(
                "{api_url}/admin/circuits/{}",
                config.circuit_id
            ))
            .header("Authorization", format!("Bearer {}", config.auth_token))
            .json(&serde_json::json!({ "is_active": true }))
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("activate request: {e}")))?;

        if !activate_resp.status().is_success() {
            let status = activate_resp.status();
            let text = activate_resp
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(DsperseError::Other(format!(
                "activate failed ({status}): {text}"
            )));
        }
    }

    Ok(PublishResult {
        circuit_id: config.circuit_id.clone(),
        files_uploaded: uploaded,
    })
}
