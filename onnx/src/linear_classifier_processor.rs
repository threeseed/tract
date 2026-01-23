//! Linear Classifier implementation with coefficient extraction and probability calculation.
//!
//! This module provides functionality to:
//! - Read coefficients from an ONNX LinearClassifier model
//! - Process sample inputs through the classifier
//! - Calculate individual class probabilities
//! - Apply Softmax to produce final normalized probabilities

use crate::pb::{ModelProto, NodeProto};
#[cfg(target_arch = "aarch64")]
use crate::softmax::softmax_neon_alloc;
use prost::Message;
use std::fs;
use std::path::Path;
use tract_hir::internal::*;

/// Represents the extracted coefficients and intercepts from a LinearClassifier ONNX model.
#[derive(Debug, Clone)]
pub struct LinearClassifierCoefficients {
    /// Coefficient matrix of shape [num_features, num_classes]
    pub coefficients: Vec<f32>,
    /// Intercept/bias vector of shape [num_classes]
    pub intercepts: Option<Vec<f32>>,
    /// Number of input features
    pub num_features: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Class labels (integer or string based)
    pub class_labels: ClassLabels,
}

/// Class label types supported by LinearClassifier
#[derive(Debug, Clone)]
pub enum ClassLabels {
    Integers(Vec<i64>),
    Strings(Vec<String>),
}

impl ClassLabels {
    pub fn len(&self) -> usize {
        match self {
            ClassLabels::Integers(v) => v.len(),
            ClassLabels::Strings(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Reads coefficients from an ONNX LinearClassifier model file.
///
/// # Arguments
/// * `path` - Path to the ONNX model file
///
/// # Returns
/// * `TractResult<LinearClassifierCoefficients>` - Extracted coefficients or error
pub fn read_coefficients_from_onnx(path: impl AsRef<Path>) -> TractResult<LinearClassifierCoefficients> {
    let path = path.as_ref();
    let data = fs::read(path).with_context(|| format!("Failed to read ONNX file: {:?}", path))?;
    let proto = ModelProto::decode(&*data).context("Failed to decode ONNX model")?;
    
    extract_linear_classifier_coefficients(&proto)
}

/// Extracts LinearClassifier coefficients from a parsed ONNX ModelProto.
fn extract_linear_classifier_coefficients(proto: &ModelProto) -> TractResult<LinearClassifierCoefficients> {
    let graph = proto.graph.as_ref().context("Model has no graph")?;
    
    // Find the LinearClassifier node
    let lc_node = graph
        .node
        .iter()
        .find(|n| n.op_type == "LinearClassifier")
        .context("No LinearClassifier node found in model")?;
    
    // Extract coefficients
    let coefficients: Vec<f32> = get_attr_floats(lc_node, "coefficients")?;
    ensure!(!coefficients.is_empty(), "coefficients attribute is empty");
    
    // Extract intercepts (optional)
    let intercepts: Option<Vec<f32>> = get_attr_floats_opt(lc_node, "intercepts")?;
    
    // Extract class labels
    let class_labels = extract_class_labels(lc_node)?;
    let num_classes = class_labels.len();
    
    // Calculate number of features from coefficients
    // coefficients has shape [num_classes * num_features] in row-major order
    let num_features = if let Some(ref intercepts) = intercepts {
        let num_models = intercepts.len();
        coefficients.len() / num_models
    } else {
        coefficients.len() / num_classes
    };
    
    Ok(LinearClassifierCoefficients {
        coefficients,
        intercepts,
        num_features,
        num_classes,
        class_labels,
    })
}

/// Helper function to extract floats attribute from a node.
fn get_attr_floats(node: &NodeProto, name: &str) -> TractResult<Vec<f32>> {
    let attr = node
        .attribute
        .iter()
        .find(|a| a.name == name)
        .with_context(|| format!("Attribute '{}' not found", name))?;
    Ok(attr.floats.clone())
}

/// Helper function to optionally extract floats attribute from a node.
fn get_attr_floats_opt(node: &NodeProto, name: &str) -> TractResult<Option<Vec<f32>>> {
    let attr = node.attribute.iter().find(|a| a.name == name);
    Ok(attr.map(|a| a.floats.clone()))
}

/// Extracts class labels from a LinearClassifier node.
fn extract_class_labels(node: &NodeProto) -> TractResult<ClassLabels> {
    // Try integer labels first
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "classlabels_ints") {
        return Ok(ClassLabels::Integers(attr.ints.clone()));
    }
    
    // Try string labels
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "classlabels_strings") {
        let strings: Vec<String> = attr
            .strings
            .iter()
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        return Ok(ClassLabels::Strings(strings));
    }
    
    bail!("No class labels found in LinearClassifier node")
}

/// Processor for running LinearClassifier inference with probability calculation.
pub struct LinearClassifierProcessor {
    coefficients: LinearClassifierCoefficients,
}

impl LinearClassifierProcessor {
    /// Creates a new processor from an ONNX file.
    pub fn from_onnx(path: impl AsRef<Path>) -> TractResult<Self> {
        let coefficients = read_coefficients_from_onnx(path)?;
        Ok(Self { coefficients })
    }
    
    /// Creates a new processor from pre-extracted coefficients.
    pub fn new(coefficients: LinearClassifierCoefficients) -> Self {
        Self { coefficients }
    }
    
    /// Returns the number of input features expected.
    pub fn num_features(&self) -> usize {
        self.coefficients.num_features
    }
    
    /// Returns the number of output classes.
    pub fn num_classes(&self) -> usize {
        self.coefficients.num_classes
    }
    
    /// Calculates raw scores (logits) for a single input sample.
    ///
    /// # Arguments
    /// * `input` - Input feature vector of length `num_features`
    ///
    /// # Returns
    /// * `TractResult<Vec<f32>>` - Raw scores for each class
    pub fn calculate_logits(&self, input: &[f32]) -> TractResult<Vec<f32>> {
        ensure!(
            input.len() == self.coefficients.num_features,
            "Input length {} doesn't match expected features {}",
            input.len(),
            self.coefficients.num_features
        );
        
        let num_classes = self.coefficients.num_classes;
        let num_features = self.coefficients.num_features;
        let mut logits = vec![0.0f32; num_classes];
        
        // Compute linear combination: logits[c] = sum_i(input[i] * coefficients[c * num_features + i])
        for c in 0..num_classes {
            let mut sum = 0.0f32;
            for i in 0..num_features {
                let coef_idx = c * num_features + i;
                if coef_idx < self.coefficients.coefficients.len() {
                    sum += input[i] * self.coefficients.coefficients[coef_idx];
                }
            }
            
            // Add intercept if available
            if let Some(ref intercepts) = self.coefficients.intercepts {
                if c < intercepts.len() {
                    sum += intercepts[c];
                }
            }
            
            logits[c] = sum;
        }
        
        Ok(logits)
    }
    
    /// Calculates probability for a single input using the specified class index.
    ///
    /// # Arguments
    /// * `input` - Input feature vector
    /// * `class_idx` - Class index to get probability for
    ///
    /// # Returns
    /// * `TractResult<f32>` - Probability for the specified class (after softmax)
    pub fn calculate_class_probability(&self, input: &[f32], class_idx: usize) -> TractResult<f32> {
        let probabilities = self.calculate_probabilities(input)?;
        ensure!(
            class_idx < probabilities.len(),
            "Class index {} out of bounds (num_classes={})",
            class_idx,
            probabilities.len()
        );
        Ok(probabilities[class_idx])
    }
    
    /// Calculates probabilities for all classes using Softmax.
    ///
    /// # Arguments
    /// * `input` - Input feature vector
    ///
    /// # Returns
    /// * `TractResult<Vec<f32>>` - Softmax probabilities for each class
    pub fn calculate_probabilities(&self, input: &[f32]) -> TractResult<Vec<f32>> {
        let logits = self.calculate_logits(input)?;
        Ok(softmax(&logits))
    }
    
    /// Processes multiple inputs and returns combined probabilities using Softmax.
    ///
    /// This method:
    /// 1. Calculates logits for each input sample
    /// 2. Applies Softmax to each sample's logits independently
    /// 3. Returns probabilities for all samples
    ///
    /// # Arguments
    /// * `inputs` - Slice of input samples, each with `num_features` elements
    ///
    /// # Returns
    /// * `TractResult<Vec<Vec<f32>>>` - Softmax probabilities for each input sample
    pub fn process_inputs(&self, inputs: &[Vec<f32>]) -> TractResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(inputs.len());
        
        for (idx, input) in inputs.iter().enumerate() {
            let probabilities = self.calculate_probabilities(input)
                .with_context(|| format!("Failed to process input sample {}", idx))?;
            results.push(probabilities);
        }
        
        Ok(results)
    }
    
    /// Processes multiple inputs and combines them into a final probability using ensemble averaging.
    ///
    /// This method:
    /// 1. Calculates Softmax probabilities for each input
    /// 2. Averages the probabilities across all inputs
    /// 3. Applies Softmax again to normalize the combined result
    ///
    /// # Arguments
    /// * `inputs` - Slice of input samples
    ///
    /// # Returns
    /// * `TractResult<Vec<f32>>` - Final combined probabilities
    pub fn combine_probabilities(&self, inputs: &[Vec<f32>]) -> TractResult<Vec<f32>> {
        ensure!(!inputs.is_empty(), "No inputs provided");
        
        let all_probs = self.process_inputs(inputs)?;
        let num_classes = self.num_classes();
        let num_inputs = all_probs.len() as f32;
        
        // Average probabilities across all inputs
        let mut combined = vec![0.0f32; num_classes];
        for probs in &all_probs {
            for (i, &p) in probs.iter().enumerate() {
                combined[i] += p / num_inputs;
            }
        }
        
        // Apply softmax to normalize the combined probabilities
        Ok(softmax(&combined))
    }
    
    /// Returns the predicted class index for an input.
    pub fn predict_class(&self, input: &[f32]) -> TractResult<usize> {
        let probabilities = self.calculate_probabilities(input)?;
        Ok(argmax(&probabilities))
    }
    
    /// Returns the class label for a predicted class index.
    pub fn get_class_label(&self, class_idx: usize) -> Option<String> {
        match &self.coefficients.class_labels {
            ClassLabels::Integers(ints) => ints.get(class_idx).map(|i| i.to_string()),
            ClassLabels::Strings(strings) => strings.get(class_idx).cloned(),
        }
    }
}

/// Computes the Softmax function over a vector of logits.
///
/// Softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
///
/// Uses the numerically stable version: exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
///
/// On aarch64/ARM platforms, this uses NEON SIMD intrinsics for optimal performance.
/// Falls back to a scalar implementation on other architectures.
#[inline]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    #[cfg(target_arch = "aarch64")]
    {
        softmax_neon_alloc(logits)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        softmax_scalar(logits)
    }
}

/// Scalar fallback implementation of softmax for non-ARM architectures.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn softmax_scalar(logits: &[f32]) -> Vec<f32> {
    // Find max for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

    // Compute sum of exp values
    let sum: f32 = exp_values.iter().sum();

    // Normalize
    let recip = sum.recip();
    exp_values.iter().map(|&e| e * recip).collect()
}

/// Returns the index of the maximum value in the slice.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Sample inputs for testing the LinearClassifier.
pub fn get_sample_inputs(num_features: usize, num_samples: usize) -> Vec<Vec<f32>> {
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let mut sample = Vec::with_capacity(num_features);
        for j in 0..num_features {
            // Generate diverse sample values using a simple deterministic pattern
            let value = ((i as f32 * 0.1) + (j as f32 * 0.3)).sin() * 10.0;
            sample.push(value);
        }
        samples.push(sample);
    }
    
    samples
}

/// Complete setup and inference pipeline.
///
/// This function demonstrates the full workflow:
/// 1. Reading coefficients from an ONNX file
/// 2. Creating sample inputs
/// 3. Calculating probabilities for each input
/// 4. Combining results with Softmax
pub fn setup_and_run(onnx_path: impl AsRef<Path>) -> TractResult<LinearClassifierResult> {
    // (a) Read coefficients from ONNX file
    let processor = LinearClassifierProcessor::from_onnx(&onnx_path)?;
    
    println!("Loaded LinearClassifier model:");
    println!("  - Number of features: {}", processor.num_features());
    println!("  - Number of classes: {}", processor.num_classes());
    
    // (b) Create sample inputs
    let sample_inputs = get_sample_inputs(processor.num_features(), 5);
    
    println!("\nProcessing {} sample inputs...", sample_inputs.len());
    
    // (c) Calculate probability for each input
    let mut individual_probabilities = Vec::with_capacity(sample_inputs.len());
    for (idx, input) in sample_inputs.iter().enumerate() {
        let probs = processor.calculate_probabilities(input)?;
        println!("  Sample {}: probabilities = {:?}", idx, probs);
        individual_probabilities.push(probs);
    }
    
    // (d) Combine probabilities using Softmax
    let final_probabilities = processor.combine_probabilities(&sample_inputs)?;
    
    println!("\nFinal combined probabilities (after Softmax): {:?}", final_probabilities);
    
    // Find predicted class
    let predicted_class_idx = argmax(&final_probabilities);
    let predicted_label = processor.get_class_label(predicted_class_idx);
    
    println!("Predicted class: {} (index: {})", 
             predicted_label.as_deref().unwrap_or("unknown"), 
             predicted_class_idx);
    
    Ok(LinearClassifierResult {
        individual_probabilities,
        final_probabilities,
        predicted_class_idx,
        predicted_label,
    })
}

/// Result of the LinearClassifier inference pipeline.
#[derive(Debug, Clone)]
pub struct LinearClassifierResult {
    /// Probabilities for each individual input sample
    pub individual_probabilities: Vec<Vec<f32>>,
    /// Final combined probabilities after Softmax
    pub final_probabilities: Vec<f32>,
    /// Index of the predicted class
    pub predicted_class_idx: usize,
    /// Label of the predicted class (if available)
    pub predicted_label: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        
        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that larger logits give larger probabilities
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
    
    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values that could cause overflow without stability fix
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        
        // Should still sum to 1 and not produce NaN/Inf
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|&p| p.is_finite()));
    }
    
    #[test]
    fn test_softmax_empty() {
        let logits: Vec<f32> = vec![];
        let probs = softmax(&logits);
        assert!(probs.is_empty());
    }
    
    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.5, 0.3, 0.1];
        assert_eq!(argmax(&values), 1);
    }
    
    #[test]
    fn test_sample_inputs_generation() {
        let samples = get_sample_inputs(12, 5);
        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|s| s.len() == 12));
    }
}
