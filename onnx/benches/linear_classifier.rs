use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tract_hir::internal::*;
use tract_onnx::linear_classifier_processor::{LinearClassifierProcessor, softmax};
use tract_onnx::tract_core::dims;

/// Compare outputs between Tract and custom processor implementations.
/// This function runs both approaches on the same inputs and compares results.
fn compare_implementations() {
    println!("\n=== Comparing Tract vs Custom Processor Implementations ===\n");

    let model_path = "test_cases/linear_classifier/model.onnx";
    let onnx_path = PathBuf::from(&model_path);

    // Setup Tract model
    let tract_model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();
    let n = tract_model.sym("N");
    let tract_model = tract_model
        .with_input_fact(0, f32::fact(dims!(n, 12)).into())
        .unwrap()
        .with_output_fact(0, i64::fact(dims!(n)).into())
        .unwrap()
        .with_output_fact(1, f32::fact(dims!(n, 14)).into())
        .unwrap()
        .into_optimized()
        .unwrap();
    let tract_runnable = tract_model.into_runnable().unwrap();

    // Setup custom processor
    let custom_processor = LinearClassifierProcessor::from_onnx(&onnx_path).unwrap();

    println!("Model info:");
    println!("  - Number of features: {}", custom_processor.num_features());
    println!("  - Number of classes: {}", custom_processor.num_classes());

    // Generate deterministic test inputs using seeded RNG
    let mut rng = StdRng::seed_from_u64(42);
    let num_features = custom_processor.num_features();
    let num_test_samples = 5;

    println!("\n--- Test Inputs ({} samples, {} features each) ---", num_test_samples, num_features);

    for sample_idx in 0..num_test_samples {
        let input_vec: Vec<f32> = (0..num_features)
            .map(|_| rng.gen_range(-10.0f32..10.0f32))
            .collect();

        println!("\nSample {}: {:?}", sample_idx, input_vec);

        // Run Tract inference
        let shape: TVec<usize> = tvec![1, num_features];
        let input_tensor = Tensor::from_shape(&shape, &input_vec).unwrap();
        let tract_outputs = tract_runnable.run(tvec!(input_tensor.into_tvalue())).unwrap();

        // Tract output 0: predicted class (i64)
        let tract_class = tract_outputs[0].to_array_view::<i64>().unwrap();
        let tract_class_val = tract_class.iter().next().unwrap();

        // Tract output 1: probabilities (f32) - shape [1, num_classes]
        let tract_probs = tract_outputs[1].to_array_view::<f32>().unwrap();
        let tract_probs_vec: Vec<f32> = tract_probs.iter().cloned().collect();

        // Run custom processor inference
        let custom_probs = custom_processor.calculate_probabilities(&input_vec).unwrap();
        let custom_class = custom_processor.predict_class(&input_vec).unwrap();

        println!("\n  Tract Results:");
        println!("    Predicted class: {}", tract_class_val);
        println!("    Probabilities: {:?}", tract_probs_vec);

        println!("\n  Custom Processor Results:");
        println!("    Predicted class: {}", custom_class);
        println!("    Probabilities: {:?}", custom_probs);

        // Compare outputs
        println!("\n  Comparison:");
        let class_match = *tract_class_val as usize == custom_class;
        println!("    Classes match: {}", if class_match { "✓ YES" } else { "✗ NO" });

        // Compare probabilities with tolerance
        let tolerance = 1e-5;
        let probs_match = tract_probs_vec.iter().zip(custom_probs.iter()).all(|(a, b)| (a - b).abs() < tolerance);
        println!("    Probabilities match (tolerance {}): {}", tolerance, if probs_match { "✓ YES" } else { "✗ NO" });

        if !probs_match {
            println!("    Probability differences:");
            for (i, (t, c)) in tract_probs_vec.iter().zip(custom_probs.iter()).enumerate() {
                let diff = (t - c).abs();
                if diff > tolerance {
                    println!("      Class {}: Tract={:.6}, Custom={:.6}, Diff={:.6}", i, t, c, diff);
                }
            }
        }
    }

    println!("\n=== Comparison Complete ===\n");
}

fn bench_linear_classifier(c: &mut Criterion) {
    // First, run the comparison to show output differences
    compare_implementations();

    let mut group = c.benchmark_group("onnx_linear_classifier");

    // Load model
    let model_path = "test_cases/linear_classifier/model.onnx";
    let onnx_path = PathBuf::from(&model_path);
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();

    // Configure dimensions
    let n = model.sym("N");
    let model = model
        .with_input_fact(0, f32::fact(dims!(n, 12)).into())
        .unwrap()
        .with_output_fact(0, i64::fact(dims!(n)).into())
        .unwrap()
        .with_output_fact(1, f32::fact(dims!(n, 14)).into())
        .unwrap()
        .into_optimized()
        .unwrap();

    let input_fact = model.input_fact(0).unwrap().clone();
    let shape: TVec<usize> = input_fact
        .shape
        .as_concrete()
        .map(|s| s.iter().copied().collect())
        .unwrap_or_else(|| tvec![1, 12]);
    let num_features = shape[1];

    // Pre-generate random input tensors for benchmarking
    let mut rng = rand::thread_rng();
    let input_tensors: Arc<Vec<Tensor>> = Arc::new(
        (0..1_000_000)
            .map(|_| {
                let sample: Vec<f32> =
                    (0..num_features).map(|_| rng.gen_range(-30.0f32..30.0f32)).collect();
                Tensor::from_shape(&shape, &sample).unwrap()
            })
            .collect(),
    );

    // Pre-generate random input vectors for custom processor
    let input_vecs: Arc<Vec<Vec<f32>>> = Arc::new(
        input_tensors
            .iter()
            .map(|t| t.as_slice::<f32>().unwrap().to_vec())
            .collect(),
    );

    let runnable = Arc::new(model.clone().into_runnable().unwrap());
    let processor = Arc::new(LinearClassifierProcessor::from_onnx(&onnx_path).unwrap());

    // Pre-compute logits for all inputs (for softmax-only benchmark)
    let precomputed_logits: Arc<Vec<Vec<f32>>> = Arc::new(
        input_vecs
            .iter()
            .map(|v| processor.calculate_logits(v).unwrap())
            .collect(),
    );

    let iteration_counts = vec![1_000_000];
 
    // Benchmark softmax-only with pre-computed logits (sequential)
    for &iterations in &iteration_counts {
        group.bench_function(BenchmarkId::new("softmax_only_sequential", iterations), |b| {
            let logits = Arc::clone(&precomputed_logits);

            b.iter_custom(|_| {
                let start = Instant::now();

                for i in 0..iterations {
                    let _ = softmax(&logits[i % 1_000_000]);
                }

                start.elapsed()
            });
        });
    }

    // Benchmark Tract implementation (parallel)
    for &iterations in &iteration_counts {
        group.bench_function(BenchmarkId::new("tract_parallel", iterations), |b| {
            let runnable = Arc::clone(&runnable);
            let tensors = Arc::clone(&input_tensors);

            b.iter_custom(|_| {
                let start = Instant::now();

                (0..iterations).into_par_iter().for_each(|i| {
                    let runnable = Arc::clone(&runnable);
                    let input_val = tensors[i % 1_000_000].clone().into_tvalue();
                    let _ = runnable.run(tvec!(input_val)).unwrap();
                });

                start.elapsed()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_linear_classifier);
criterion_main!(benches);
