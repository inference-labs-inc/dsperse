use std::collections::HashMap;

use ndarray::ArrayD;

use crate::error::{DsperseError, Result};

#[derive(Default)]
pub struct TensorStore {
    tensors: HashMap<String, ArrayD<f64>>,
}

impl TensorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Result<&ArrayD<f64>> {
        self.tensors
            .get(name)
            .ok_or_else(|| DsperseError::Pipeline(format!("tensor '{name}' not found in store")))
    }

    pub fn try_get(&self, name: &str) -> Option<&ArrayD<f64>> {
        self.tensors.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<ArrayD<f64>> {
        self.tensors.remove(name)
    }

    pub fn total_elements(&self) -> usize {
        self.tensors.values().map(|t| t.len()).sum()
    }

    pub fn put(&mut self, name: String, tensor: ArrayD<f64>) {
        self.tensors.insert(name, tensor);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn as_map(&self) -> &HashMap<String, ArrayD<f64>> {
        &self.tensors
    }

    pub fn gather(&self, names: &[String]) -> Result<ArrayD<f64>> {
        crate::utils::io::gather_inputs_from_cache(&self.tensors, names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn put_and_get() {
        let mut store = TensorStore::new();
        let arr = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        store.put("x".into(), arr.clone());
        assert_eq!(store.get("x").unwrap(), &arr);
    }

    #[test]
    fn get_missing_returns_error() {
        let store = TensorStore::new();
        assert!(store.get("missing").is_err());
    }

    #[test]
    fn try_get_missing_returns_none() {
        let store = TensorStore::new();
        assert!(store.try_get("missing").is_none());
    }

    #[test]
    fn contains_check() {
        let mut store = TensorStore::new();
        assert!(!store.contains("a"));
        store.put(
            "a".into(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        );
        assert!(store.contains("a"));
    }
}
