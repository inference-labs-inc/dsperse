use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DsperseVersion {
    pub dsperse_version: String,
    pub dsperse_rev: Option<String>,
    pub jstprove_version: String,
    pub jstprove_rev: Option<String>,
}

pub fn dsperse_artifact_version() -> DsperseVersion {
    let jst_ver = jstprove_circuits::api::jstprove_artifact_version();
    DsperseVersion {
        dsperse_version: env!("CARGO_PKG_VERSION").to_string(),
        dsperse_rev: option_env!("DSPERSE_GIT_REV").map(String::from),
        jstprove_version: jst_ver.crate_version,
        jstprove_rev: Some(jst_ver.git_rev),
    }
}
