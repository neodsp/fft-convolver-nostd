//! no_std FFT convolution for embedded audio DSP.
//!
//! Partitioned overlap-add FFT convolution backed by [microfft](https://docs.rs/microfft).
//! All buffers are stack-allocated fixed-size arrays; no heap required.
//!
//! # Type parameters
//!
//! - `BLOCK_SIZE`: partition size. Must be a power of two from 8 to 2048.
//! - `SEG_SIZE`: must equal `2 * BLOCK_SIZE`.
//! - `FFT_CPLX_SIZE`: must equal `BLOCK_SIZE + 1`.
//! - `SEG_COUNT`: maximum number of IR segments (`max_ir_len / BLOCK_SIZE`, rounded up).
//!
//! Use the provided type aliases (`FFTConvolver64`, `FFTConvolver128`, …) to avoid
//! spelling out all four parameters.
//!
//! # Example
//!
//! ```
//! use fft_convolver_nostd::FFTConvolver256;
//!
//! // BLOCK_SIZE=256, up to 4 segments (max IR length = 1024 samples)
//! let mut conv = FFTConvolver256::<4>::default();
//! let ir = [0.5f32, 0.3, 0.2, 0.1];
//! conv.init(&ir).unwrap();
//!
//! let input = [1.0f32; 256];
//! let mut output = [0.0f32; 256];
//! conv.process(&input, &mut output).unwrap();
//! ```

#![no_std]
#![deny(missing_debug_implementations)]

mod fft;
mod utilities;

use crate::utilities::{complex_multiply_accumulate, copy_and_pad, sum};
use microfft::Complex32;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FFTConvolverError {
    /// The impulse response is longer than the initial one passed to `init`.
    ImpulseResponseExceedsCapacity,
    /// `input` and `output` slices have different lengths.
    InputOutputLengthMismatch,
    /// The supplied block size is not a supported power of two (see supported sizes).
    UnsupportedBlockSize,
}

impl core::fmt::Display for FFTConvolverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ImpulseResponseExceedsCapacity => {
                f.write_str("impulse response exceeds configured capacity")
            }
            Self::InputOutputLengthMismatch => {
                f.write_str("input and output buffers must have the same length")
            }
            Self::UnsupportedBlockSize => {
                f.write_str("block size is not supported (must be power of 2, from 8 to 2048)")
            }
        }
    }
}

impl core::error::Error for FFTConvolverError {}

// ── FFTConvolver ─────────────────────────────────────────────────────────────

/// Partitioned FFT convolver with compile-time buffer sizes.
///
/// `BLOCK_SIZE`, `SEG_SIZE`, `FFT_CPLX_SIZE`, and `SEG_COUNT` are const parameters;
/// see the [module-level docs](crate) for what they mean and which ones are derived.
///
/// For easy construction, use the [`FFTConvolver64`] / [`FFTConvolver128`] / … aliases.
#[derive(Clone)]
pub struct FFTConvolver<
    const BLOCK_SIZE: usize,
    const SEG_SIZE: usize,
    const FFT_CPLX_SIZE: usize,
    const SEG_COUNT: usize,
> {
    ir_len: usize,
    active_seg_count: usize,

    // Frequency-domain segments: audio history and IR
    segments: [[Complex32; FFT_CPLX_SIZE]; SEG_COUNT],
    segments_ir: [[Complex32; FFT_CPLX_SIZE]; SEG_COUNT],

    // Real FFT I/O scratch. The forward FFT clobbers this in place (microfft).
    fft_buffer: [f32; SEG_SIZE],
    // Scratch for the full symmetric spectrum during IRFFT
    ifft_scratch: [Complex32; SEG_SIZE],

    // Convolution accumulator and pre-multiplied tail
    pre_multiplied: [Complex32; FFT_CPLX_SIZE],
    conv: [Complex32; FFT_CPLX_SIZE],

    // Overlap-add tail from the previous block
    overlap: [f32; BLOCK_SIZE],

    // Circular-buffer index into segments[]
    current: usize,

    // Sub-block input accumulator
    input_buffer: [f32; BLOCK_SIZE],
    input_buffer_fill: usize,
}

impl<
    const BLOCK_SIZE: usize,
    const SEG_SIZE: usize,
    const FFT_CPLX_SIZE: usize,
    const SEG_COUNT: usize,
> Default for FFTConvolver<BLOCK_SIZE, SEG_SIZE, FFT_CPLX_SIZE, SEG_COUNT>
{
    fn default() -> Self {
        Self::default_const()
    }
}

impl<
    const BLOCK_SIZE: usize,
    const SEG_SIZE: usize,
    const FFT_CPLX_SIZE: usize,
    const SEG_COUNT: usize,
> FFTConvolver<BLOCK_SIZE, SEG_SIZE, FFT_CPLX_SIZE, SEG_COUNT>
{
    /// Zero-initialized convolver suitable for `static` placement.
    ///
    /// Rust does not call `Default::default()` in `static` initializers, so use this
    /// `const fn` instead:
    ///
    /// ```rust
    /// use fft_convolver_nostd::FFTConvolver512;
    /// static CONV: FFTConvolver512<16> = FFTConvolver512::<16>::default_const();
    /// ```
    pub const fn default_const() -> Self {
        const ZERO: Complex32 = Complex32::new(0.0, 0.0);
        Self {
            ir_len: 0,
            active_seg_count: 0,
            segments: [[ZERO; FFT_CPLX_SIZE]; SEG_COUNT],
            segments_ir: [[ZERO; FFT_CPLX_SIZE]; SEG_COUNT],
            fft_buffer: [0.0; SEG_SIZE],
            ifft_scratch: [ZERO; SEG_SIZE],
            pre_multiplied: [ZERO; FFT_CPLX_SIZE],
            conv: [ZERO; FFT_CPLX_SIZE],
            overlap: [0.0; BLOCK_SIZE],
            current: 0,
            input_buffer: [0.0; BLOCK_SIZE],
            input_buffer_fill: 0,
        }
    }
}

impl<
    const BLOCK_SIZE: usize,
    const SEG_SIZE: usize,
    const FFT_CPLX_SIZE: usize,
    const SEG_COUNT: usize,
> core::fmt::Debug for FFTConvolver<BLOCK_SIZE, SEG_SIZE, FFT_CPLX_SIZE, SEG_COUNT>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FFTConvolver")
            .field("block_size", &BLOCK_SIZE)
            .field("seg_size", &SEG_SIZE)
            .field("seg_count", &SEG_COUNT)
            .field("ir_len", &self.ir_len)
            .field("active_seg_count", &self.active_seg_count)
            .finish_non_exhaustive()
    }
}

impl<
    const BLOCK_SIZE: usize,
    const SEG_SIZE: usize,
    const FFT_CPLX_SIZE: usize,
    const SEG_COUNT: usize,
> FFTConvolver<BLOCK_SIZE, SEG_SIZE, FFT_CPLX_SIZE, SEG_COUNT>
{
    /// Initializes the convolver with an impulse response.
    ///
    /// Must be called before `process`. All memory is already allocated (it lives in the
    /// struct); this just computes and stores the frequency-domain IR segments.
    ///
    /// Returns [`UnsupportedBlockSize`](FFTConvolverError::UnsupportedBlockSize) if
    /// `BLOCK_SIZE` is not a power-of-two size supported by microfft (8 to 2048), or if the
    /// const relationships `SEG_SIZE == 2 * BLOCK_SIZE` and
    /// `FFT_CPLX_SIZE == BLOCK_SIZE + 1` are violated.
    ///
    /// Returns [`ImpulseResponseExceedsCapacity`](FFTConvolverError::ImpulseResponseExceedsCapacity)
    /// if `impulse_response.len() > BLOCK_SIZE * SEG_COUNT`.
    pub fn init(&mut self, impulse_response: &[f32]) -> Result<(), FFTConvolverError> {
        // Verify the const relationships at runtime so misuse is caught immediately.
        const {
            assert!(
                SEG_SIZE == BLOCK_SIZE * 2,
                "SEG_SIZE must equal 2 * BLOCK_SIZE"
            )
        };
        const {
            assert!(
                FFT_CPLX_SIZE == BLOCK_SIZE + 1,
                "FFT_CPLX_SIZE must equal BLOCK_SIZE + 1"
            )
        };

        if !Self::block_size_supported() {
            return Err(FFTConvolverError::UnsupportedBlockSize);
        }

        *self = Self::default();

        self.ir_len = impulse_response.len();

        if self.ir_len == 0 {
            return Ok(());
        }

        let max_ir = BLOCK_SIZE * SEG_COUNT;
        if self.ir_len > max_ir {
            return Err(FFTConvolverError::ImpulseResponseExceedsCapacity);
        }

        self.active_seg_count = self.ir_len.div_ceil(BLOCK_SIZE);

        for i in 0..self.active_seg_count {
            let remaining = self.ir_len - i * BLOCK_SIZE;
            let size_copy = remaining.min(BLOCK_SIZE);
            copy_and_pad(
                &mut self.fft_buffer,
                &impulse_response[i * BLOCK_SIZE..],
                size_copy,
            );
            fft::forward(&mut self.fft_buffer, &mut self.segments_ir[i]);
        }

        self.input_buffer_fill = 0;
        self.current = 0;

        Ok(())
    }

    /// Updates the impulse response without reallocating.
    ///
    /// The new IR must not exceed the length passed to `init`.
    pub fn set_response(&mut self, impulse_response: &[f32]) -> Result<(), FFTConvolverError> {
        if impulse_response.len() > self.ir_len {
            return Err(FFTConvolverError::ImpulseResponseExceedsCapacity);
        }

        self.fft_buffer.fill(0.0);
        self.conv.fill(Complex32::default());
        self.pre_multiplied.fill(Complex32::default());
        self.overlap.fill(0.0);

        self.active_seg_count = if impulse_response.is_empty() {
            0
        } else {
            impulse_response.len().div_ceil(BLOCK_SIZE)
        };

        for i in 0..self.active_seg_count {
            let remaining = impulse_response.len() - i * BLOCK_SIZE;
            let size_copy = remaining.min(BLOCK_SIZE);
            copy_and_pad(
                &mut self.fft_buffer,
                &impulse_response[i * BLOCK_SIZE..],
                size_copy,
            );
            fft::forward(&mut self.fft_buffer, &mut self.segments_ir[i]);
        }

        for seg in self.segments_ir.iter_mut().skip(self.active_seg_count) {
            seg.fill(Complex32::default());
        }

        self.input_buffer.fill(0.0);
        self.input_buffer_fill = 0;
        self.current = 0;
        for seg in &mut self.segments {
            seg.fill(Complex32::default());
        }

        Ok(())
    }

    /// Convolves `input` with the impulse response, writing results to `output`.
    ///
    /// Real-time safe: no allocations. Handles arbitrary input lengths via internal
    /// sub-block buffering.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), FFTConvolverError> {
        if input.len() != output.len() {
            return Err(FFTConvolverError::InputOutputLengthMismatch);
        }

        if self.active_seg_count == 0 {
            output.fill(0.0);
            return Ok(());
        }

        let mut processed = 0;
        while processed < output.len() {
            let input_buffer_was_empty = self.input_buffer_fill == 0;
            let processing = (output.len() - processed).min(BLOCK_SIZE - self.input_buffer_fill);

            let pos = self.input_buffer_fill;
            self.input_buffer[pos..pos + processing]
                .copy_from_slice(&input[processed..processed + processing]);

            // Forward FFT of current input block
            copy_and_pad(&mut self.fft_buffer, &self.input_buffer, BLOCK_SIZE);
            fft::forward(&mut self.fft_buffer, &mut self.segments[self.current]);

            // Complex multiply and accumulate: tail segments
            if input_buffer_was_empty {
                self.pre_multiplied.fill(Complex32::default());
                for i in 1..self.active_seg_count {
                    let index_ir = i;
                    let index_audio = (self.current + i) % self.active_seg_count;
                    complex_multiply_accumulate(
                        &mut self.pre_multiplied,
                        &self.segments_ir[index_ir],
                        &self.segments[index_audio],
                    );
                }
            }
            self.conv.copy_from_slice(&self.pre_multiplied);
            complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
            );

            // Inverse FFT
            fft::inverse(&self.conv, &mut self.ifft_scratch, &mut self.fft_buffer);

            // Overlap-add
            sum(
                &mut output[processed..processed + processing],
                &self.fft_buffer[pos..pos + processing],
                &self.overlap[pos..pos + processing],
            );

            self.input_buffer_fill += processing;
            if self.input_buffer_fill == BLOCK_SIZE {
                self.input_buffer.fill(0.0);
                self.input_buffer_fill = 0;
                self.overlap
                    .copy_from_slice(&self.fft_buffer[BLOCK_SIZE..SEG_SIZE]);
                self.current = if self.current > 0 {
                    self.current - 1
                } else {
                    self.active_seg_count - 1
                };
            }
            processed += processing;
        }

        Ok(())
    }

    /// Resets convolution state while preserving the loaded impulse response.
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.input_buffer_fill = 0;
        self.fft_buffer.fill(0.0);
        for seg in &mut self.segments {
            seg.fill(Complex32::default());
        }
        self.conv.fill(Complex32::default());
        self.pre_multiplied.fill(Complex32::default());
        self.overlap.fill(0.0);
        self.current = 0;
    }

    fn block_size_supported() -> bool {
        matches!(BLOCK_SIZE, 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048)
    }
}

// ── Convenience aliases ───────────────────────────────────────────────────────

/// `FFTConvolver` with `BLOCK_SIZE=8` (SEG_SIZE=16, FFT_CPLX_SIZE=9).
pub type FFTConvolver8<const SEG_COUNT: usize> = FFTConvolver<8, 16, 9, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=16` (SEG_SIZE=32, FFT_CPLX_SIZE=17).
pub type FFTConvolver16<const SEG_COUNT: usize> = FFTConvolver<16, 32, 17, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=32` (SEG_SIZE=64, FFT_CPLX_SIZE=33).
pub type FFTConvolver32<const SEG_COUNT: usize> = FFTConvolver<32, 64, 33, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=64` (SEG_SIZE=128, FFT_CPLX_SIZE=65).
pub type FFTConvolver64<const SEG_COUNT: usize> = FFTConvolver<64, 128, 65, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=128` (SEG_SIZE=256, FFT_CPLX_SIZE=129).
pub type FFTConvolver128<const SEG_COUNT: usize> = FFTConvolver<128, 256, 129, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=256` (SEG_SIZE=512, FFT_CPLX_SIZE=257).
pub type FFTConvolver256<const SEG_COUNT: usize> = FFTConvolver<256, 512, 257, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=512` (SEG_SIZE=1024, FFT_CPLX_SIZE=513).
pub type FFTConvolver512<const SEG_COUNT: usize> = FFTConvolver<512, 1024, 513, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=1024` (SEG_SIZE=2048, FFT_CPLX_SIZE=1025).
pub type FFTConvolver1024<const SEG_COUNT: usize> = FFTConvolver<1024, 2048, 1025, SEG_COUNT>;
/// `FFTConvolver` with `BLOCK_SIZE=2048` (SEG_SIZE=4096, FFT_CPLX_SIZE=2049).
pub type FFTConvolver2048<const SEG_COUNT: usize> = FFTConvolver<2048, 4096, 2049, SEG_COUNT>;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;

    use super::*;

    #[test]
    fn process_identity_ir() {
        let mut conv = FFTConvolver64::<4>::default();
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        conv.init(&ir).unwrap();

        let input = [0.0, 1.0, 2.0, 3.0f32];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output).unwrap();

        for i in 0..4 {
            assert!((input[i] - output[i]).abs() < 1e-4, "mismatch at {i}");
        }
    }

    #[test]
    fn zero_latency() {
        let mut conv = FFTConvolver32::<1>::default();
        let ir = [0.5f32, 0.3, 0.2, 0.1];
        conv.init(&ir).unwrap();

        let mut input = [0.0f32; 16];
        input[0] = 1.0;
        let mut output = [0.0f32; 16];
        conv.process(&input, &mut output).unwrap();

        assert!((output[0] - 0.5).abs() < 1e-4, "output[0]={}", output[0]);
        assert!((output[1] - 0.3).abs() < 1e-4, "output[1]={}", output[1]);
        assert!((output[2] - 0.2).abs() < 1e-4, "output[2]={}", output[2]);
        assert!((output[3] - 0.1).abs() < 1e-4, "output[3]={}", output[3]);
    }

    #[test]
    fn reset_clears_state() {
        let ir = [0.5f32, 0.3, 0.2, 0.1];
        let mut conv1 = FFTConvolver32::<1>::default();
        conv1.init(&ir).unwrap();

        let history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let mut hist_out = [0.0f32; 8];
        conv1.process(&history, &mut hist_out).unwrap();
        conv1.reset();

        let test = [1.0, 1.0, 1.0, 1.0f32];
        let mut out1 = [0.0f32; 4];
        conv1.process(&test, &mut out1).unwrap();

        let mut conv2 = FFTConvolver32::<1>::default();
        conv2.init(&ir).unwrap();
        let mut out2 = [0.0f32; 4];
        conv2.process(&test, &mut out2).unwrap();

        for i in 0..4 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-4,
                "mismatch at {i}: reset={}, fresh={}",
                out1[i],
                out2[i]
            );
        }
    }

    #[test]
    fn set_response_matches_init() {
        let ir1 = [0.5f32, 0.3, 0.2, 0.1];
        let ir2 = [0.8f32, 0.6, 0.4, 0.2];

        let mut conv1 = FFTConvolver32::<1>::default();
        conv1.init(&ir1).unwrap();
        conv1.set_response(&ir2).unwrap();

        let mut conv2 = FFTConvolver32::<1>::default();
        conv2.init(&ir2).unwrap();

        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let mut out1 = [0.0f32; 8];
        let mut out2 = [0.0f32; 8];
        conv1.process(&input, &mut out1).unwrap();
        conv2.process(&input, &mut out2).unwrap();

        for i in 0..8 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-4,
                "mismatch at {i}: set_response={}, init={}",
                out1[i],
                out2[i]
            );
        }
    }

    #[test]
    fn set_response_too_long_errors() {
        let ir1 = [0.5f32, 0.3, 0.2, 0.1];
        let ir2 = [0.8f32, 0.6, 0.4, 0.2, 0.1, 0.05];
        let mut conv = FFTConvolver32::<1>::default();
        conv.init(&ir1).unwrap();
        assert!(matches!(
            conv.set_response(&ir2),
            Err(FFTConvolverError::ImpulseResponseExceedsCapacity)
        ));
    }

    #[test]
    fn mismatch_lengths_errors() {
        let mut conv = FFTConvolver32::<1>::default();
        conv.init(&[1.0f32]).unwrap();
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 8];
        assert!(matches!(
            conv.process(&input, &mut output),
            Err(FFTConvolverError::InputOutputLengthMismatch)
        ));
    }

    #[test]
    fn empty_ir_produces_silence() {
        let mut conv = FFTConvolver64::<4>::default();
        conv.init(&[]).unwrap();
        let input = [1.0f32; 64];
        let mut output = [0.0f32; 64];
        conv.process(&input, &mut output).unwrap();
        for &s in &output {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn large_ir_multi_segment() {
        let ir_len = 256usize;
        let mut ir = vec![0.0f32; ir_len];
        ir[0] = 1.0; // identity

        let mut conv = FFTConvolver64::<4>::default();
        conv.init(&ir).unwrap();

        let input: vec::Vec<f32> = (0..ir_len).map(|i| i as f32 * 0.001).collect();
        let mut output = vec![0.0f32; ir_len];
        conv.process(&input, &mut output).unwrap();

        for i in 0..ir_len {
            assert!(
                (output[i] - input[i]).abs() < 1e-3,
                "mismatch at {i}: input={}, output={}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn block_size_8_zero_latency() {
        let mut conv = FFTConvolver8::<1>::default();
        let ir = [0.5f32, 0.3, 0.2, 0.1];
        conv.init(&ir).unwrap();

        let mut input = [0.0f32; 8];
        input[0] = 1.0;
        let mut output = [0.0f32; 8];
        conv.process(&input, &mut output).unwrap();

        assert!((output[0] - 0.5).abs() < 1e-4, "output[0]={}", output[0]);
        assert!((output[1] - 0.3).abs() < 1e-4, "output[1]={}", output[1]);
        assert!((output[2] - 0.2).abs() < 1e-4, "output[2]={}", output[2]);
        assert!((output[3] - 0.1).abs() < 1e-4, "output[3]={}", output[3]);
    }
}
