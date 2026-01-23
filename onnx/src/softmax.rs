/// NEON-optimized softmax for ARMv8.2+ (M2 Mac, Graviton 2/3)
/// Uses standard library exp() for full accuracy.
///
/// NEON optimizations retained:
/// - `vmaxvq_f32`/`vaddvq_f32` for O(1) horizontal reductions
/// - 4x loop unrolling for max-finding and normalization passes
/// - Vectorized normalization (multiply by reciprocal)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Accurate softmax: writes result to `output` slice.
///
/// # Panics
/// Panics if `logits.len() != output.len()`.
#[cfg(target_arch = "aarch64")]
pub fn softmax_neon(logits: &[f32], output: &mut [f32]) {
    assert_eq!(logits.len(), output.len(), "length mismatch");

    let len = logits.len();
    if len == 0 {
        return;
    }

    unsafe {
        let inp = logits.as_ptr();
        let out = output.as_mut_ptr();

        let chunks = len / 16;
        let remainder_chunks = (len % 16) / 4;
        let scalar_start = (chunks * 16) + (remainder_chunks * 4);

        // ══════════════════════════════════════════════════════════════════
        // PASS 1: Find maximum (vectorized)
        // ══════════════════════════════════════════════════════════════════
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0usize;

        for _ in 0..chunks {
            let v0 = vld1q_f32(inp.add(i));
            let v1 = vld1q_f32(inp.add(i + 4));
            let v2 = vld1q_f32(inp.add(i + 8));
            let v3 = vld1q_f32(inp.add(i + 12));

            let m01 = vmaxq_f32(v0, v1);
            let m23 = vmaxq_f32(v2, v3);
            max_v = vmaxq_f32(max_v, vmaxq_f32(m01, m23));

            i += 16;
        }

        for _ in 0..remainder_chunks {
            max_v = vmaxq_f32(max_v, vld1q_f32(inp.add(i)));
            i += 4;
        }

        // ARMv8.2+ horizontal max
        let mut max_val = vmaxvq_f32(max_v);

        for j in scalar_start..len {
            max_val = max_val.max(*inp.add(j));
        }

        // ══════════════════════════════════════════════════════════════════
        // PASS 2: Compute exp(x - max) using std::exp, accumulate sum
        // ══════════════════════════════════════════════════════════════════
        let mut sum = 0.0f32;

        for j in 0..len {
            let e = (*inp.add(j) - max_val).exp();
            *out.add(j) = e;
            sum += e;
        }

        // ══════════════════════════════════════════════════════════════════
        // PASS 3: Normalize (vectorized multiply by 1/sum)
        // ══════════════════════════════════════════════════════════════════
        let inv_sum = 1.0 / sum;
        let inv_sum_v = vdupq_n_f32(inv_sum);

        i = 0;
        for _ in 0..chunks {
            let v0 = vmulq_f32(vld1q_f32(out.add(i)), inv_sum_v);
            let v1 = vmulq_f32(vld1q_f32(out.add(i + 4)), inv_sum_v);
            let v2 = vmulq_f32(vld1q_f32(out.add(i + 8)), inv_sum_v);
            let v3 = vmulq_f32(vld1q_f32(out.add(i + 12)), inv_sum_v);

            vst1q_f32(out.add(i), v0);
            vst1q_f32(out.add(i + 4), v1);
            vst1q_f32(out.add(i + 8), v2);
            vst1q_f32(out.add(i + 12), v3);

            i += 16;
        }

        for _ in 0..remainder_chunks {
            let v = vmulq_f32(vld1q_f32(out.add(i)), inv_sum_v);
            vst1q_f32(out.add(i), v);
            i += 4;
        }

        for j in scalar_start..len {
            *out.add(j) *= inv_sum;
        }
    }
}

/// Convenience wrapper that allocates output.
#[cfg(target_arch = "aarch64")]
pub fn softmax_neon_alloc(logits: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; logits.len()];
    softmax_neon(logits, &mut output);
    output
}

/// In-place softmax - modifies the input slice directly.
#[cfg(target_arch = "aarch64")]
pub fn softmax_neon_inplace(logits: &mut [f32]) {
    let len = logits.len();
    if len == 0 {
        return;
    }

    unsafe {
        let ptr = logits.as_mut_ptr();
        let chunks = len / 16;
        let remainder_chunks = (len % 16) / 4;
        let scalar_start = (chunks * 16) + (remainder_chunks * 4);

        // Pass 1: max (vectorized)
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0usize;

        for _ in 0..chunks {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            let v2 = vld1q_f32(ptr.add(i + 8));
            let v3 = vld1q_f32(ptr.add(i + 12));
            max_v = vmaxq_f32(max_v, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
            i += 16;
        }
        for _ in 0..remainder_chunks {
            max_v = vmaxq_f32(max_v, vld1q_f32(ptr.add(i)));
            i += 4;
        }
        let mut max_val = vmaxvq_f32(max_v);
        for j in scalar_start..len {
            max_val = max_val.max(*ptr.add(j));
        }

        // Pass 2: exp (scalar, accurate)
        let mut sum = 0.0f32;
        for j in 0..len {
            let e = (*ptr.add(j) - max_val).exp();
            *ptr.add(j) = e;
            sum += e;
        }

        // Pass 3: normalize (vectorized)
        let inv_sum = 1.0 / sum;
        let inv_sum_v = vdupq_n_f32(inv_sum);

        i = 0;
        for _ in 0..chunks {
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), inv_sum_v));
            vst1q_f32(ptr.add(i + 4), vmulq_f32(vld1q_f32(ptr.add(i + 4)), inv_sum_v));
            vst1q_f32(ptr.add(i + 8), vmulq_f32(vld1q_f32(ptr.add(i + 8)), inv_sum_v));
            vst1q_f32(ptr.add(i + 12), vmulq_f32(vld1q_f32(ptr.add(i + 12)), inv_sum_v));
            i += 16;
        }
        for _ in 0..remainder_chunks {
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), inv_sum_v));
            i += 4;
        }
        for j in scalar_start..len {
            *ptr.add(j) *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    #[test]
    fn test_accuracy() {
        let logits: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1) - 50.0).collect();
        let reference = reference_softmax(&logits);
        let mut output = vec![0.0f32; logits.len()];

        softmax_neon(&logits, &mut output);

        for (i, (a, b)) in output.iter().zip(reference.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(
                err < 1e-7,
                "mismatch at {}: got {}, expected {}, err={}",
                i,
                a,
                b,
                err
            );
        }
    }

    #[test]
    fn test_sum_to_one() {
        let logits: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1) - 50.0).collect();
        let mut output = vec![0.0f32; logits.len()];
        softmax_neon(&logits, &mut output);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {}", sum);
    }

    #[test]
    fn test_inplace() {
        let original: Vec<f32> = (0..128).map(|i| i as f32 * 0.5 - 32.0).collect();
        let reference = reference_softmax(&original);

        let mut inplace = original.clone();
        softmax_neon_inplace(&mut inplace);

        for (a, b) in inplace.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_small_sizes() {
        for size in [1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33] {
            let logits: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let reference = reference_softmax(&logits);
            let mut output = vec![0.0f32; size];
            softmax_neon(&logits, &mut output);

            let sum: f32 = output.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "sum != 1 for size {}", size);

            for (a, b) in output.iter().zip(reference.iter()) {
                assert!((a - b).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn test_extreme_values() {
        // Large positive values
        let logits = vec![100.0, 101.0, 102.0, 103.0];
        let mut output = vec![0.0f32; 4];
        softmax_neon(&logits, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Large negative values
        let logits = vec![-100.0, -101.0, -102.0, -103.0];
        let mut output = vec![0.0f32; 4];
        softmax_neon(&logits, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Mixed extreme
        let logits = vec![-1000.0, 0.0, 1000.0];
        let mut output = vec![0.0f32; 3];
        softmax_neon(&logits, &mut output);
        assert!(output[2] > 0.99); // Should be ~1.0
    }
}