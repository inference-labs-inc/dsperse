use std::collections::HashMap;

use super::onnx_proto::{ModelProto, TensorProto};

/// Graph rewrite placeholder: detecting `Div(X, X)` and collapsing it to a
/// constant-ones tensor is only sound when the element type is a floating
/// point dtype AND every element of X is finite AND non-zero.  Without a
/// traced-properties side channel carrying that guarantee the rewrite would
/// silently turn `0 / 0 = NaN` and integer underflow into `1`.
///
/// The earlier implementation rewrote unconditionally and is preserved here
/// as documentation so that a follow-up can plug it in once
/// `traced_dtypes` / `traced_all_finite_nonzero` maps are available.
pub fn rewrite_self_div_to_one(
    _model: &mut ModelProto,
    _traced_shapes: &mut HashMap<String, Vec<i64>>,
) -> usize {
    // Intentionally a no-op: see module doc.  Re-enable behind a proper
    // traced-properties guard once available.
    let _ = TensorProto::FLOAT;
    0
}
