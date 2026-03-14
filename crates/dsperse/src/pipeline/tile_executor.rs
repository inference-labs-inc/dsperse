use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{DsperseError, Result};
use crate::schema::tiling::TilingInfo;
use crate::utils::paths::resolve_relative_path;

pub fn resolve_tile_circuit(
    tiling: &TilingInfo,
    tile_idx: usize,
    slices_dir: &Path,
    default_circuit: Option<&Path>,
) -> std::result::Result<Option<PathBuf>, String> {
    let from_tiles = tiling
        .tiles
        .as_deref()
        .and_then(|ts| ts.get(tile_idx))
        .and_then(|ti| ti.jstprove_circuit_path.as_deref());
    let from_single = tiling
        .tile
        .as_ref()
        .and_then(|ti| ti.jstprove_circuit_path.as_deref());
    let path_str = from_tiles.or(from_single);
    match path_str {
        Some(p) => match resolve_relative_path(slices_dir, p) {
            Ok(resolved) => Ok(Some(resolved)),
            Err(e) => Err(e.to_string()),
        },
        None => Ok(default_circuit.map(|p| p.to_path_buf())),
    }
}

pub fn execute_tiles<T, F>(parallel: usize, num_tiles: usize, op: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if num_tiles == 0 {
        return Err(DsperseError::Pipeline("num_tiles is 0".into()));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel)
        .build()
        .map_err(|e| DsperseError::Pipeline(format!("thread pool: {e}")))?;

    let results = pool.install(|| (0..num_tiles).into_par_iter().map(op).collect());

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::tiling::{TileInfo, TilingInfo};

    fn make_tiling() -> TilingInfo {
        TilingInfo {
            slice_idx: 0,
            tile_size: 4,
            num_tiles: 4,
            tiles_y: 2,
            tiles_x: 2,
            halo: [0, 0, 0, 0],
            out_tile: [4, 4],
            stride: [1, 1],
            c_in: 1,
            c_out: 1,
            input_name: "input".into(),
            output_name: "output".into(),
            tile: None,
            tiles: None,
        }
    }

    #[test]
    fn resolve_tile_circuit_no_info() {
        let tiling = make_tiling();
        let result = resolve_tile_circuit(&tiling, 0, Path::new("/tmp"), None);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn resolve_tile_circuit_with_default() {
        let tiling = make_tiling();
        let default = PathBuf::from("/tmp/circuit.bundle");
        let result = resolve_tile_circuit(&tiling, 0, Path::new("/tmp"), Some(&default));
        assert_eq!(result.unwrap(), Some(default));
    }

    #[test]
    fn resolve_tile_circuit_from_single_tile() {
        let mut tiling = make_tiling();
        tiling.tile = Some(TileInfo {
            path: "tile.onnx".into(),
            conv_out: [4, 4],
            jstprove_circuit_path: Some("jstprove/circuit.bundle".into()),
        });
        let result = resolve_tile_circuit(&tiling, 0, Path::new("/slices"), None);
        let resolved = result.unwrap().unwrap();
        assert!(resolved.to_string_lossy().contains("circuit.bundle"));
    }

    #[test]
    fn execute_tiles_collects_results() {
        let results = execute_tiles(2, 4, |i| i * 2).unwrap();
        assert_eq!(results.len(), 4);
        let mut sorted = results.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 2, 4, 6]);
    }

    #[test]
    fn execute_tiles_zero_tiles_errors() {
        let result = execute_tiles(1, 0, |i| i);
        assert!(result.is_err());
    }
}
