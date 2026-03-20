use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::{DsperseError, Result};

pub struct SliceAssets {
    pub circuit_bundle: HashMap<String, Vec<u8>>,
    pub onnx_bytes: Option<Vec<u8>>,
}

impl SliceAssets {
    pub fn load_from_dslice(slices_dir: &Path, slice_id: &str) -> Result<Self> {
        let archive_path = slices_dir.join(format!("{slice_id}.dslice"));
        if !archive_path.exists() {
            return Ok(Self {
                circuit_bundle: HashMap::new(),
                onnx_bytes: None,
            });
        }

        let file =
            std::fs::File::open(&archive_path).map_err(|e| DsperseError::io(e, &archive_path))?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| {
            DsperseError::Slicer(format!(
                "reading dslice archive {}: {e}",
                archive_path.display()
            ))
        })?;

        let mut bundle_files: HashMap<String, Vec<u8>> = HashMap::new();
        let mut onnx_bytes = None;

        for i in 0..zip.len() {
            let mut entry = zip
                .by_index(i)
                .map_err(|e| DsperseError::Slicer(format!("reading zip entry {i}: {e}")))?;
            let name = entry.name().to_string();

            if let Some(bundle_name) = name.strip_prefix("jstprove/circuit.bundle/") {
                if !bundle_name.is_empty() && !entry.is_dir() {
                    let mut buf = Vec::with_capacity(entry.size() as usize);
                    entry.read_to_end(&mut buf).map_err(|e| {
                        DsperseError::Slicer(format!("reading {name} from dslice: {e}"))
                    })?;
                    bundle_files.insert(bundle_name.to_string(), buf);
                }
            } else if name.ends_with(".onnx") && name.starts_with("payload/") {
                let mut buf = Vec::with_capacity(entry.size() as usize);
                entry
                    .read_to_end(&mut buf)
                    .map_err(|e| DsperseError::Slicer(format!("reading onnx from dslice: {e}")))?;
                onnx_bytes = Some(buf);
            }
        }

        Ok(Self {
            circuit_bundle: bundle_files,
            onnx_bytes,
        })
    }

    pub fn write_bundle_to_dir(&self, dir: &Path) -> std::io::Result<()> {
        if self.circuit_bundle.is_empty() {
            return Ok(());
        }
        std::fs::create_dir_all(dir)?;
        for (name, bytes) in &self.circuit_bundle {
            std::fs::write(dir.join(name), bytes)?;
        }
        Ok(())
    }
}
