//! Cross-validates fft-convolver-nostd output against the reference std implementation.

use fft_convolver::FFTConvolver as StdConvolver;
use fft_convolver_nostd::FFTConvolver32;
use fft_convolver_nostd::FFTConvolver128;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn identity_ir_matches_std() {
    let ir: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

    // std convolver
    let mut std_conv = StdConvolver::<f32>::default();
    std_conv.init(32, &ir).unwrap();
    let mut std_out = vec![0.0f32; 32];
    std_conv.process(&input, &mut std_out).unwrap();

    // no-std convolver
    let mut nostd_conv = FFTConvolver32::<1>::default();
    nostd_conv.init(&ir).unwrap();
    let mut nostd_out = [0.0f32; 32];
    nostd_conv.process(&input, &mut nostd_out).unwrap();

    assert!(
        approx_eq(&std_out, &nostd_out, 1e-4),
        "identity IR mismatch\nstd:   {std_out:.5?}\nnostd: {nostd_out:.5?}"
    );
}

#[test]
fn multi_segment_ir_matches_std() {
    // IR longer than one block → exercises multi-segment path
    let ir: Vec<f32> = {
        let mut v = vec![0.0f32; 64];
        v[0] = 0.8;
        v[16] = 0.4;
        v[48] = 0.2;
        v
    };
    let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();

    let mut std_conv = StdConvolver::<f32>::default();
    std_conv.init(32, &ir).unwrap();
    let mut std_out = vec![0.0f32; 128];
    std_conv.process(&input, &mut std_out).unwrap();

    let mut nostd_conv = FFTConvolver32::<2>::default();
    nostd_conv.init(&ir).unwrap();
    let mut nostd_out = [0.0f32; 128];
    nostd_conv.process(&input, &mut nostd_out).unwrap();

    assert!(
        approx_eq(&std_out, &nostd_out, 1e-3),
        "multi-segment mismatch at first difference: {}",
        std_out
            .iter()
            .zip(nostd_out.iter())
            .enumerate()
            .find(|(_, (a, b))| (*a - *b).abs() > 1e-3)
            .map(|(i, (a, b))| format!("index {i}: std={a:.6} nostd={b:.6}"))
            .unwrap_or_default()
    );
}

#[test]
fn impulse_response_shapes_output_correctly() {
    // Non-trivial IR: decaying exponential
    let n = 128usize;
    let ir: Vec<f32> = (0..n).map(|i| 0.9f32.powi(i as i32)).collect();
    let input: Vec<f32> = (0..256).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

    let mut std_conv = StdConvolver::<f32>::default();
    std_conv.init(128, &ir).unwrap();
    let mut std_out = vec![0.0f32; 256];
    std_conv.process(&input, &mut std_out).unwrap();

    let mut nostd_conv = FFTConvolver128::<1>::default();
    nostd_conv.init(&ir).unwrap();
    let mut nostd_out = [0.0f32; 256];
    nostd_conv.process(&input, &mut nostd_out).unwrap();

    // Check the first 128 samples (within the IR) match
    assert!(
        approx_eq(&std_out[..n], &nostd_out[..n], 1e-3),
        "IR shape mismatch"
    );
}
