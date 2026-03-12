fn main() {
    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("Failed to compile ONNX proto");

    if let Some(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let rev = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("cargo:rustc-env=DSPERSE_GIT_REV={rev}");
    }
    if let Some(output) = std::process::Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let head_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("cargo:rerun-if-changed={head_path}");
    }

    if let Some(output) = std::process::Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let head_ref = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if let Some(output) = std::process::Command::new("git")
            .args(["rev-parse", "--git-path", &head_ref])
            .output()
            .ok()
            .filter(|o| o.status.success())
        {
            let ref_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rerun-if-changed={ref_path}");
        }
    }
}
