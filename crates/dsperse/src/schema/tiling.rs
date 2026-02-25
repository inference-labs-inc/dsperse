use serde::{Deserialize, Serialize};

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
    #[serde(default = "default_pair_zero")]
    pub halo: [i64; 2],
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<Vec<TileInfo>>,
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
    pub settings_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vk_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pk_path: Option<String>,
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

fn default_pair_zero() -> [i64; 2] {
    [0, 0]
}

fn default_pair_one() -> [i64; 2] {
    [1, 1]
}

fn default_input_name() -> String {
    "input".to_string()
}

fn default_output_name() -> String {
    "output".to_string()
}
