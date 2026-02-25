fn main() {
    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("Failed to compile ONNX proto");

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let rev = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rustc-env=DSPERSE_GIT_REV={rev}");
        }
    }
    println!("cargo:rerun-if-changed=.git/HEAD");
}
