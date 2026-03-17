fn main() {
    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("Failed to compile ONNX proto");

    let git_rev = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    if let Some(ref rev) = git_rev {
        println!("cargo:rustc-env=DSPERSE_GIT_REV={rev}");
    }

    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let display_version = match (pkg_version.as_str(), &git_rev) {
        ("0.0.0", Some(rev)) => format!("dev-{rev}"),
        ("0.0.0", None) => "dev".to_string(),
        (v, Some(rev)) => format!("{v}+{rev}"),
        (v, None) => v.to_string(),
    };
    println!("cargo:rustc-env=DSPERSE_DISPLAY_VERSION={display_version}");
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
