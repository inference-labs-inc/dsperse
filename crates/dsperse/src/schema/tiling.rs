use serde::{self, Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileInfo {
    #[serde(default)]
    pub path: String,
    #[serde(default = "default_pair_zero")]
    pub conv_out: [i64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jstprove_circuit_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilingInfo {
    #[serde(default)]
    pub slice_idx: usize,
    #[serde(default)]
    pub tile_size: usize,
    #[serde(default = "default_one")]
    pub num_tiles: usize,
    #[serde(default = "default_one")]
    pub tiles_y: usize,
    #[serde(default = "default_one")]
    pub tiles_x: usize,
    #[serde(default = "default_quad_zero", deserialize_with = "deserialize_halo")]
    pub halo: [i64; 4],
    #[serde(default = "default_pair_zero")]
    pub out_tile: [i64; 2],
    #[serde(default = "default_pair_one")]
    pub stride: [i64; 2],
    #[serde(default)]
    pub c_in: usize,
    #[serde(default)]
    pub c_out: usize,
    #[serde(default = "default_input_name")]
    pub input_name: String,
    #[serde(default = "default_output_name")]
    pub output_name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_names: Vec<String>,
    #[serde(default = "default_four")]
    pub ndim: usize,
    #[serde(default)]
    pub h: usize,
    #[serde(default)]
    pub w: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<Vec<TileInfo>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_elements: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub original_shape: Vec<i64>,
}

impl TilingInfo {
    pub fn all_input_names(&self) -> Vec<&str> {
        if self.input_names.is_empty() {
            vec![&self.input_name]
        } else {
            self.input_names.iter().map(|s| s.as_str()).collect()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelGroupInfo {
    #[serde(default)]
    pub group_idx: usize,
    #[serde(default)]
    pub c_start: usize,
    #[serde(default)]
    pub c_end: usize,
    #[serde(default)]
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jstprove_circuit_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jstprove_settings_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSplitInfo {
    #[serde(default)]
    pub slice_idx: usize,
    #[serde(default)]
    pub c_in: usize,
    #[serde(default)]
    pub c_out: usize,
    #[serde(default = "default_one")]
    pub num_groups: usize,
    #[serde(default)]
    pub channels_per_group: usize,
    #[serde(default = "default_input_name")]
    pub input_name: String,
    #[serde(default = "default_output_name")]
    pub output_name: String,
    #[serde(default)]
    pub h: usize,
    #[serde(default)]
    pub w: usize,
    #[serde(default)]
    pub out_h: usize,
    #[serde(default)]
    pub out_w: usize,
    #[serde(default)]
    pub groups: Vec<ChannelGroupInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias_path: Option<String>,
}

fn default_one() -> usize {
    1
}

fn default_four() -> usize {
    4
}

fn default_pair_zero() -> [i64; 2] {
    [0, 0]
}

fn default_pair_one() -> [i64; 2] {
    [1, 1]
}

fn default_quad_zero() -> [i64; 4] {
    [0, 0, 0, 0]
}

fn deserialize_halo<'de, D>(deserializer: D) -> std::result::Result<[i64; 4], D::Error>
where
    D: Deserializer<'de>,
{
    let v: Vec<i64> = Vec::deserialize(deserializer)?;
    match v.len() {
        2 => Ok([v[0], v[1], v[0], v[1]]),
        4 => Ok([v[0], v[1], v[2], v[3]]),
        _ => Err(serde::de::Error::custom(format!(
            "expected 2 or 4 elements for halo, got {}",
            v.len()
        ))),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimSplitKind {
    #[default]
    MatMulOutputDim,
    HeadDim,
    BatchDim,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DimSplitInfo {
    #[serde(default)]
    pub slice_idx: usize,
    #[serde(default)]
    pub split_kind: DimSplitKind,
    #[serde(default)]
    pub split_dim: usize,
    #[serde(default)]
    pub dim_size: usize,
    #[serde(default = "default_one")]
    pub num_groups: usize,
    #[serde(default)]
    pub elements_per_group: usize,
    #[serde(default = "default_input_name")]
    pub input_name: String,
    #[serde(default = "default_output_name")]
    pub output_name: String,
    #[serde(default)]
    pub concat_axis: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jstprove_circuit_path: Option<String>,
}

fn default_input_name() -> String {
    "input".to_string()
}

fn default_output_name() -> String {
    "output".to_string()
}
