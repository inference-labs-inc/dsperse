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

        // Verify the component by probing each file the manifest
        // expects to live in blob storage, not by asking for a
        // metadata row.  The registry registers the component row
        // as soon as POST /admin/components returns, but the
        // per-file PUTs against the pre-signed upload URLs happen
        // afterwards -- any failure there (timeout, network blip,
        // interrupted publish process) leaves the row present with
        // no backing files.  A plain GET /components/{sha} sees the
        // row and reports "exists, skipping", so every subsequent
        // publish run re-skips the broken component and the model
        // stays permanently half-uploaded from downstream
        // consumers' perspective.
        //
        // Mirror the byte-level presence check weight-blob uploads
        // below already use: HEAD each expected file by issuing a
        // single-byte ranged GET to the blob path.  If every file
        // is present, the component is genuinely done and we skip.
        // If every file is missing, proceed to the normal
        // register + upload path.  If the set is partially present
        // (registered but mid-upload), surface an actionable error
        // instead of silently continuing, because re-registering
        // via POST /admin/components will 409 and the current flow
        // has no way to request fresh upload URLs for that sha.
        // A manifest entry with zero files is malformed -- the
        // empty-list case would otherwise make both
        // `missing.is_empty()` and `present == files.len()` true
        // below, silently classifying the component as present
        // without any actual bytes verified.  Fail loud.
        if files.is_empty() {
            return Err(DsperseError::Other(format!(
                "component {sha} has no files listed in the manifest; refusing to treat as present"
            )));
        }

        let mut present = 0usize;
        let mut missing: Vec<String> = Vec::new();
        for filename in &files {
            let file_url = format!("{api}/components/{sha}/files/{filename}");
            let probe = client
                .get(&file_url)
                .header("Range", "bytes=0-0")
                .send()
                .await
                .map_err(|e| DsperseError::Other(format!("probe {sha}/{filename}: {e}")))?;
            // A Range: bytes=0-0 GET against a blob path has two
            // legitimate success replies: 206 (partial content,
            // what the blob store returns when it honours the
            // range) and 200 (full content, what it returns when
            // it ignores the range for an empty body or tiny file).
            // Any other 2xx (201 Created, 202 Accepted, 204 No
            // Content) is ambiguous for a GET on a CAS path and
            // should not be interpreted as "file present".
            let status = probe.status();
            match status.as_u16() {
                200 | 206 => present += 1,
                404 => missing.push(filename.clone()),
                _ => {
                    let text = probe.text().await.unwrap_or_default();
                    return Err(DsperseError::Other(format!(
                        "probe component {sha}/{filename} returned unexpected status ({status}): {text}"
                    )));
                }
            }
        }

        if missing.is_empty() && present == files.len() {
            tracing::info!(sha = %sha, "component files present, skipping");
            components_skipped += 1;
            continue;
        }

        // Ambiguity guard: a partial upload (some files present,
        // some missing) is unrecoverable without more surgery than
        // we can do from here -- we can't ask the registry to
        // re-issue pre-signed URLs for the specific missing files
        // without dropping and re-registering the whole component,
        // and that would also orphan any valid uploads still in
        // place.  Surface it as an explicit error so the operator
        // can decide.
        if present > 0 {
            return Err(DsperseError::Other(format!(
                "component {sha} is partially uploaded: {present}/{} files present, \
                 missing: {:?}.  A previous publish registered the component row \
                 and successfully PUT some files but not others.  Manual recovery: \
                 `curl -X DELETE -H 'Authorization: Bearer $REGISTRY_AUTH_TOKEN' \
                 {api}/admin/components/{sha}` to drop the stale row, then re-run \
                 publish.",
                files.len(),
                missing
            )));
        }

        let proof_system = comp["proof_system"]
            .as_str()
            .unwrap_or(&config.proof_system)
            .to_uppercase();
        let comp_name = comp["name"].as_str().unwrap_or(sha);
        let register_body = serde_json::json!({
            "sha256": sha,
            "name": comp_name,
            "description": "",
            "proof_system": proof_system,
            "files": files,
        });

        tracing::info!(sha = %sha, files = files.len(), "registering component");
        let mut register_resp = client
            .post(format!("{api}/admin/components"))
            .header("Authorization", &auth)
            .json(&register_body)
            .send()
            .await
            .map_err(|e| DsperseError::Other(format!("register component {sha}: {e}")))?;
        let mut reg_status = register_resp.status();

        // Self-heal the "metadata row exists, every file is 404"
        // state the presence check above just proved we are in:
        // the row was created by a prior publish whose per-file
        // PUTs never completed, and the server now 409s any fresh
        // register attempt for this sha.  The only way to request
        // fresh pre-signed upload URLs is to drop the stale row
        // and POST again.  Safe because the presence check already
        // confirmed zero files are live on the blob store, so
        // nothing is orphaned by the DELETE.
        if reg_status.as_u16() == 409 {
            tracing::warn!(
                sha = %sha,
                "component row exists but zero files are live; dropping stale row and \
                 re-registering so fresh upload URLs can issue"
            );
            let delete_resp = client
                .delete(format!("{api}/admin/components/{sha}"))
                .header("Authorization", &auth)
                .send()
                .await
                .map_err(|e| DsperseError::Other(format!("delete stale component {sha}: {e}")))?;
            let del_status = delete_resp.status();
            if !del_status.is_success() && del_status.as_u16() != 404 {
                let text = delete_resp.text().await.unwrap_or_default();
                return Err(DsperseError::Other(format!(
                    "delete stale component {sha} failed ({del_status}): {text}"
                )));
            }
            register_resp = client
                .post(format!("{api}/admin/components"))
                .header("Authorization", &auth)
                .json(&register_body)
                .send()
                .await
                .map_err(|e| DsperseError::Other(format!("re-register component {sha}: {e}")))?;
            reg_status = register_resp.status();
        }

        if !reg_status.is_success() {
            let text = register_resp.text().await.unwrap_or_default();
            if text.contains("already exists") || reg_status.as_u16() == 409 {
                // Persistent conflict after auto-recovery: surface
                // details so the operator can investigate rather
                // than silently skipping.
                return Err(DsperseError::Other(format!(
                    "component {sha} registration persistently conflicts after auto-DELETE \
                     retry ({reg_status}): {text}"
                )));
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
    if let Some(artifacts) = manifest["artifacts"].as_array() {
        all_weight_refs.extend(artifacts);
    }
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

    let artifacts = manifest["artifacts"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let composition = serde_json::json!({
        "version": 1,
        "artifacts": artifacts,
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
