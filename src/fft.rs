use microfft::Complex32;

/// Forward RFFT: fills `output` with `SEG_SIZE/2 + 1` unpacked complex bins.
///
/// The Nyquist bin (real-valued) is stored at `output[SEG_SIZE/2]` with `.im = 0`.
/// All other bins are positive-frequency terms.
///
/// microfft works in-place and clobbers `input` — do not read it after this call.
pub fn forward(input: &mut [f32], output: &mut [Complex32]) {
    debug_assert_eq!(input.len() / 2 + 1, output.len());

    let packed = rfft_dispatch(input);
    let nyquist_re = packed[0].im;
    packed[0].im = 0.0;

    let half = packed.len();
    output[..half].copy_from_slice(packed);
    output[half] = Complex32::new(nyquist_re, 0.0);
}

/// Inverse RFFT: reconstructs `SEG_SIZE` real samples from `FFT_CPLX_SIZE` unpacked bins.
///
/// `input` is the unpacked spectrum (DC, positive bins, Nyquist).
/// `scratch` must have length `SEG_SIZE` and is used to hold the full symmetric spectrum.
/// Result is written into `output` (length `SEG_SIZE`).
pub fn inverse(input: &[Complex32], scratch: &mut [Complex32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len() / 2 + 1);
    debug_assert_eq!(scratch.len(), output.len());

    let n = output.len();
    let half = input.len() - 1; // = SEG_SIZE / 2 = BLOCK_SIZE

    // DC bin
    scratch[0] = Complex32::new(input[0].re, 0.0);
    // Positive frequencies and their conjugate-symmetric negatives
    for k in 1..half {
        scratch[k] = input[k];
        scratch[n - k] = Complex32::new(input[k].re, -input[k].im);
    }
    // Nyquist bin
    scratch[half] = Complex32::new(input[half].re, 0.0);

    ifft_dispatch(scratch);

    for (out, c) in output.iter_mut().zip(scratch.iter()) {
        *out = c.re;
    }
}

fn rfft_dispatch(input: &mut [f32]) -> &mut [Complex32] {
    match input.len() {
        16 => {
            let arr: &mut [f32; 16] = input.try_into().unwrap();
            microfft::real::rfft_16(arr)
        }
        32 => {
            let arr: &mut [f32; 32] = input.try_into().unwrap();
            microfft::real::rfft_32(arr)
        }
        64 => {
            let arr: &mut [f32; 64] = input.try_into().unwrap();
            microfft::real::rfft_64(arr)
        }
        128 => {
            let arr: &mut [f32; 128] = input.try_into().unwrap();
            microfft::real::rfft_128(arr)
        }
        256 => {
            let arr: &mut [f32; 256] = input.try_into().unwrap();
            microfft::real::rfft_256(arr)
        }
        512 => {
            let arr: &mut [f32; 512] = input.try_into().unwrap();
            microfft::real::rfft_512(arr)
        }
        1024 => {
            let arr: &mut [f32; 1024] = input.try_into().unwrap();
            microfft::real::rfft_1024(arr)
        }
        2048 => {
            let arr: &mut [f32; 2048] = input.try_into().unwrap();
            microfft::real::rfft_2048(arr)
        }
        4096 => {
            let arr: &mut [f32; 4096] = input.try_into().unwrap();
            microfft::real::rfft_4096(arr)
        }
        n => panic!("unsupported FFT size: {n}"),
    }
}

fn ifft_dispatch(scratch: &mut [Complex32]) {
    // ifft_N returns &mut [Complex32; N] pointing into `scratch` — the result is
    // already reflected in `scratch`, so the return value is intentionally ignored.
    #[allow(unused_must_use)]
    match scratch.len() {
        16 => {
            let arr: &mut [Complex32; 16] = scratch.try_into().unwrap();
            microfft::inverse::ifft_16(arr);
        }
        32 => {
            let arr: &mut [Complex32; 32] = scratch.try_into().unwrap();
            microfft::inverse::ifft_32(arr);
        }
        64 => {
            let arr: &mut [Complex32; 64] = scratch.try_into().unwrap();
            microfft::inverse::ifft_64(arr);
        }
        128 => {
            let arr: &mut [Complex32; 128] = scratch.try_into().unwrap();
            microfft::inverse::ifft_128(arr);
        }
        256 => {
            let arr: &mut [Complex32; 256] = scratch.try_into().unwrap();
            microfft::inverse::ifft_256(arr);
        }
        512 => {
            let arr: &mut [Complex32; 512] = scratch.try_into().unwrap();
            microfft::inverse::ifft_512(arr);
        }
        1024 => {
            let arr: &mut [Complex32; 1024] = scratch.try_into().unwrap();
            microfft::inverse::ifft_1024(arr);
        }
        2048 => {
            let arr: &mut [Complex32; 2048] = scratch.try_into().unwrap();
            microfft::inverse::ifft_2048(arr);
        }
        4096 => {
            let arr: &mut [Complex32; 4096] = scratch.try_into().unwrap();
            microfft::inverse::ifft_4096(arr);
        }
        n => panic!("unsupported IFFT size: {n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn roundtrip_128() {
        let mut input = [0.0f32; 128];
        for i in 0..128 {
            input[i] = ((i * 7 + 13) % 50) as f32 / 25.0 - 1.0;
        }
        let original = input;

        let mut spectrum = [Complex32::default(); 65]; // 128/2 + 1
        let mut scratch = [Complex32::default(); 128];
        let mut recovered = [0.0f32; 128];

        forward(&mut input, &mut spectrum);
        inverse(&spectrum, &mut scratch, &mut recovered);

        for (&a, &b) in recovered.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-4, "roundtrip: {a} != {b}");
        }
    }

    #[test]
    fn sine_wave_frequency_256() {
        let n = 256usize;
        let freq_bin = 10;
        let mut input = [0.0f32; 256];
        for i in 0..n {
            input[i] = (2.0 * PI * freq_bin as f32 * i as f32 / n as f32).sin();
        }

        let mut spectrum = [Complex32::default(); 129]; // 256/2 + 1
        forward(&mut input, &mut spectrum);

        let magnitude = spectrum[freq_bin].norm();
        let expected = n as f32 / 2.0;
        assert!(
            (magnitude - expected).abs() < 0.5,
            "expected magnitude ~{expected} at bin {freq_bin}, got {magnitude}"
        );

        for (i, f) in spectrum.iter().enumerate() {
            if i != freq_bin {
                assert!(
                    f.norm() < 1.0,
                    "bin {i} should be near zero, got {}",
                    f.norm()
                );
            }
        }
    }
}
