use std::fs;
use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::error::{DsperseError, Result};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(300);

pub struct PublishConfig {
    pub api_url: String,
    pub auth_token: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
    pub proof_system: String,
    pub timeout: u64,
    pub activate: bool,
}

pub struct PublishResult {
    pub model_id: String,
    pub components_uploaded: usize,
    pub components_skipped: usize,
    pub weights_uploaded: usize,
    pub weights_skipped: usize,
}

fn auth_header(token: &str) -> String {
    format!("Bearer {token}")
}

pub fn publish(dir: &Path, config: &PublishConfig) -> Result<PublishResult> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DsperseError::Other(format!("tokio runtime: {e}")))?;

    rt.block_on(publish_async(dir, config))
}

async fn publish_async(dir: &Path, config: &PublishConfig) -> Result<PublishResult> {
    let manifest_path = dir.join("manifest.msgpack");
    if !manifest_path.is_file() {
        return Err(DsperseError::Other(format!(
            "manifest.msgpack not found in {}",
            dir.display()
        )));
    }

    let manifest_bytes =
        fs::read(&manifest_path).map_err(|e| DsperseError::io(e, &manifest_path))?;
    let manifest: serde_json::Value = rmp_serde::from_slice(&manifest_bytes)
        .map_err(|e| DsperseError::Other(format!("failed to parse manifest: {e}")))?;

    let components = manifest["components"]
        .as_array()
        .ok_or_else(|| DsperseError::Other("manifest missing components array".into()))?;

    let dag = manifest["dag"]
        .as_array()
        .ok_or_else(|| DsperseError::Other("manifest missing dag array".into()))?;

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| DsperseError::Other(format!("http client: {e}")))?;
    let api = config.api_url.trim_end_matches('/');
    let auth = auth_header(&config.auth_token);

    let mut components_uploaded = 0usize;
    let mut components_skipped = 0usize;
    let mut weights_uploaded = 0usize;
    let mut weights_skipped = 0usize;

    for comp in components {
        let sha = comp["sha256"]
            .as_str()
            .ok_or_else(|| DsperseError::Other("component missing sha256".into()))?;
        let files: Vec<String> = comp["files"]
            .as_array()
            .ok_or_else(|| DsperseError::Other("component missing files".into()))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let check = client
            .get(format!("{api}/components/{sha}"))
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("check component {sha}: {e}")))?;

        if check.status().is_success() {
            tracing::info!(sha = %sha, "component exists, skipping");
            components_skipped += 1;
            continue;
        }
        if check.status().as_u16() != 404 {
            let status = check.status();
            let text = check.text().await.unwrap_or_default();
            return Err(DsperseError::Other(format!(
                "probe component {sha} returned unexpected status ({status}): {text}"
            )));
        }

        let proof_system = comp["proof_system"]
            .as_str()
            .unwrap_or(&config.proof_system);
        let comp_name = comp["name"].as_str().unwrap_or(sha);

        tracing::info!(sha = %sha, files = files.len(), "registering component");
        let register_resp = client
            .post(format!("{api}/admin/components"))
            .header("Authorization", &auth)
            .json(&serde_json::json!({
                "sha256": sha,
                "name": comp_name,
                "description": "",
                "proof_system": proof_system,
                "files": files,
            }))
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("register component {sha}: {e}")))?;

        let reg_status = register_resp.status();
        if reg_status.as_u16() == 409 {
            tracing::info!(sha = %sha, "component already registered (conflict)");
            components_skipped += 1;
            continue;
        }
        if !reg_status.is_success() {
            let text = register_resp.text().await.unwrap_or_default();
            if text.contains("already exists") {
                tracing::info!(sha = %sha, "component already registered");
                components_skipped += 1;
                continue;
            }
            return Err(DsperseError::Other(format!(
                "register component {sha} failed ({reg_status}): {text}"
            )));
        }

        let resp_body: serde_json::Value = register_resp
            .json()
            .await
            .map_err(|e| DsperseError::Other(format!("parse component response: {e}")))?;

        let upload_urls = resp_body["upload_urls"]
            .as_object()
            .ok_or_else(|| DsperseError::Other("missing upload_urls for component".into()))?;

        let comp_dir = dir.join("components").join(sha);
        for (filename, url_val) in upload_urls {
            let url = url_val
                .as_str()
                .ok_or_else(|| DsperseError::Other(format!("non-string URL for {filename}")))?;
            let file_path = comp_dir.join(filename);
            let data = fs::read(&file_path).map_err(|e| DsperseError::io(e, &file_path))?;

            tracing::info!(file = %filename, size = data.len(), "uploading component file");
            let put = client
                .put(url)
                .timeout(UPLOAD_TIMEOUT)
                .header("Content-Type", "application/octet-stream")
                .body(data)
                .send()
                .await
                .map_err(|e| DsperseError::Other(format!("upload {filename}: {e}")))?;

            if !put.status().is_success() {
                return Err(DsperseError::Other(format!(
                    "upload component file {filename} failed ({})",
                    put.status()
                )));
            }
        }

        components_uploaded += 1;
    }

    let mut all_weight_refs: Vec<&serde_json::Value> = Vec::new();
    for comp in components {
        if let Some(weights) = comp["weights"].as_array() {
            all_weight_refs.extend(weights);
        }
    }

    let mut uploaded_wbs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for wref in &all_weight_refs {
        let sha = wref["sha256"]
            .as_str()
            .ok_or_else(|| DsperseError::Other("weight ref missing sha256".into()))?;

        if uploaded_wbs.contains(sha) {
            continue;
        }

        let size = wref["size_bytes"].as_u64().unwrap_or(0);

        let check = client
            .get(format!("{api}/models/wb/{sha}"))
            .header("Range", "bytes=0-0")
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("check wb {sha}: {e}")))?;

        if check.status().is_success() || check.status().as_u16() == 206 {
            tracing::info!(sha = %sha, "weight blob exists, skipping");
            weights_skipped += 1;
            uploaded_wbs.insert(sha.to_string());
            continue;
        }
        if check.status().as_u16() != 404 {
            let status = check.status();
            let text = check.text().await.unwrap_or_default();
            return Err(DsperseError::Other(format!(
                "probe wb {sha} returned unexpected status ({status}): {text}"
            )));
        }

        let name = wref["role"].as_str().unwrap_or("");
        tracing::info!(sha = %sha, size, "registering weight blob");
        let wb_resp = client
            .post(format!("{api}/admin/models/wb"))
            .header("Authorization", &auth)
            .json(&serde_json::json!({
                "sha256": sha,
                "name": name,
                "size_bytes": size,
            }))
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("register wb {sha}: {e}")))?;

        let wb_status = wb_resp.status();
        if wb_status.as_u16() == 409 {
            tracing::info!(sha = %sha, "weight blob already registered (conflict)");
            weights_skipped += 1;
            uploaded_wbs.insert(sha.to_string());
            continue;
        }
        if !wb_status.is_success() {
            let text = wb_resp.text().await.unwrap_or_default();
            if text.contains("already exists") {
                tracing::info!(sha = %sha, "weight blob already registered");
                weights_skipped += 1;
                uploaded_wbs.insert(sha.to_string());
                continue;
            }
            return Err(DsperseError::Other(format!(
                "register wb {sha} failed ({wb_status}): {text}"
            )));
        }

        let wb_body: serde_json::Value = wb_resp
            .json()
            .await
            .map_err(|e| DsperseError::Other(format!("parse wb response: {e}")))?;

        match wb_body["upload_url"].as_str() {
            Some(upload_url) => {
                let wb_path = dir.join("wb").join(sha);
                let data = fs::read(&wb_path).map_err(|e| DsperseError::io(e, &wb_path))?;

                tracing::info!(sha = %sha, size = data.len(), "uploading weight blob");
                let put = client
                    .put(upload_url)
                    .timeout(UPLOAD_TIMEOUT)
                    .header("Content-Type", "application/octet-stream")
                    .body(data)
                    .send()
                    .await
                    .map_err(|e| DsperseError::Other(format!("upload wb {sha}: {e}")))?;

                if !put.status().is_success() {
                    return Err(DsperseError::Other(format!(
                        "upload wb {sha} failed ({})",
                        put.status()
                    )));
                }
                weights_uploaded += 1;
            }
            None => {
                return Err(DsperseError::Other(format!(
                    "registry returned no upload URL for weight blob {sha}"
                )));
            }
        }

        uploaded_wbs.insert(sha.to_string());
    }

    let model_info = &manifest["model"];
    let model_name = model_info["name"].as_str().unwrap_or(config.name.as_str());
    let model_author = model_info["author"]
        .as_str()
        .unwrap_or(config.author.as_str());
    let model_version = model_info["version"]
        .as_str()
        .unwrap_or(config.version.as_str());
    let model_timeout = model_info["timeout"].as_u64().unwrap_or(config.timeout);
    let input_schema = &model_info["input_schema"];
    let dsperse_version = model_info["dsperse_version"].as_str();
    let jstprove_version = model_info["jstprove_version"].as_str();

    let composition = serde_json::json!({
        "version": 1,
        "components": components,
        "dag": dag,
    });

    let mut model_hasher = Sha256::new();
    model_hasher.update(model_name.as_bytes());
    model_hasher.update(b"\x00");
    model_hasher.update(model_author.as_bytes());
    model_hasher.update(b"\x00");
    model_hasher.update(model_version.as_bytes());
    model_hasher.update(b"\x00");
    model_hasher.update(model_timeout.to_le_bytes());
    model_hasher.update(b"\x00");
    let comp_json = serde_json::to_string(&composition)
        .map_err(|e| DsperseError::Other(format!("serialize composition: {e}")))?;
    model_hasher.update(comp_json.as_bytes());
    let model_id = format!("{:x}", model_hasher.finalize());

    tracing::info!(id = %model_id, "creating model");
    let model_resp = client
        .post(format!("{api}/admin/models"))
        .header("Authorization", &auth)
        .json(&serde_json::json!({
            "id": model_id,
            "metadata": {
                "name": model_name,
                "description": config.description,
                "author": model_author,
                "version": model_version,
                "netuid": null,
                "weights_version": null,
                "timeout": model_timeout,
                "input_schema": input_schema,
                "dsperse_version": dsperse_version,
                "jstprove_version": jstprove_version,
            },
            "composition": composition,
        }))
        .send()
        .await
        .map_err(|e| DsperseError::Other(format!("create model: {e}")))?;

    if !model_resp.status().is_success() {
        let status = model_resp.status();
        let text = model_resp.text().await.unwrap_or_default();
        if !text.contains("already exists") {
            return Err(DsperseError::Other(format!(
                "create model failed ({status}): {text}"
            )));
        }
        tracing::info!(id = %model_id, "model already exists");
    }

    if config.activate {
        tracing::info!(id = %model_id, "activating model");
        let activate_resp = client
            .patch(format!("{api}/admin/models/{model_id}"))
            .header("Authorization", &auth)
            .json(&serde_json::json!({ "is_active": true }))
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("activate: {e}")))?;

        if !activate_resp.status().is_success() {
            let status = activate_resp.status();
            let text = activate_resp.text().await.unwrap_or_default();
            return Err(DsperseError::Other(format!(
                "activate failed ({status}): {text}"
            )));
        }
    }

    Ok(PublishResult {
        model_id,
        components_uploaded,
        components_skipped,
        weights_uploaded,
        weights_skipped,
    })
}
