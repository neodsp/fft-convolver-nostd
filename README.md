# fft-convolver-nostd

`no_std` FFT convolution for embedded audio DSP in Rust.

Companion to [fft-convolver](https://github.com/neodsp/fft-convolver). Uses [microfft](https://docs.rs/microfft) as the FFT backend. All buffers are compile-time sized stack arrays, zero heap allocation required.

## When to use this vs `fft-convolver`

| | `fft-convolver` | `fft-convolver-nostd` |
|---|---|---|
| Environment | `std` (desktop, server) | `no_std` (embedded, RTOS) |
| FFT backend | `realfft` / `rustfft` | `microfft` |
| Buffer sizes | Runtime (any) | Compile-time (`const` params) |
| Floating point | `f32` and `f64` | `f32` only |
| IR update at runtime | Yes (`set_response`) | Yes (`set_response`) |

## Features

- **`no_std`**: Runs on bare-metal targets (e.g. ARM Cortex-M4/M7)
- **Zero allocation**: All buffers live in the struct — place it `static` for embedded use
- **Real-time safe**: `process`, `set_response`, and `reset` perform no allocations
- **Zero latency**: Output is sample-aligned with input
- **Flexible input size**: Arbitrary block sizes handled via internal sub-block buffering

## Usage

### Type parameters

The convolver takes four const generic parameters:

```rust
FFTConvolver<BLOCK_SIZE, SEG_SIZE, FFT_CPLX_SIZE, SEG_COUNT>
```

- `BLOCK_SIZE` — partition size; must be a power of two (8–2048)
- `SEG_SIZE` — must equal `2 * BLOCK_SIZE`
- `FFT_CPLX_SIZE` — must equal `BLOCK_SIZE + 1`
- `SEG_COUNT` — maximum number of IR segments; determines max IR length (`BLOCK_SIZE * SEG_COUNT` samples)

Use the provided aliases to avoid spelling these out:

| Alias | `BLOCK_SIZE` | Max IR (with `SEG_COUNT=N`) |
|---|---|---|
| `FFTConvolver8<N>` | 8 | `8 * N` samples |
| `FFTConvolver16<N>` | 16 | `16 * N` samples |
| `FFTConvolver32<N>` | 32 | `32 * N` samples |
| `FFTConvolver64<N>` | 64 | `64 * N` samples |
| `FFTConvolver128<N>` | 128 | `128 * N` samples |
| `FFTConvolver256<N>` | 256 | `256 * N` samples |
| `FFTConvolver512<N>` | 512 | `512 * N` samples |
| `FFTConvolver1024<N>` | 1024 | `1024 * N` samples |
| `FFTConvolver2048<N>` | 2048 | `2048 * N` samples |

### Basic example

```rust
use fft_convolver_nostd::FFTConvolver256;

// Block size 256, max IR length = 256 * 4 = 1024 samples
let mut conv = FFTConvolver256::<4>::default();

let ir = [0.5f32, 0.3, 0.2, 0.1];
conv.init(&ir).unwrap();

let input = [1.0f32; 256];
let mut output = [0.0f32; 256];
conv.process(&input, &mut output).unwrap();
```

### Embedded / static placement

Large convolvers should live in a `static` to avoid stack overflows:

```rust
use fft_convolver_nostd::FFTConvolver512;

// 512-sample block, up to 8192-sample IR = ~134 KB in BSS (not on the stack)
static mut CONV: FFTConvolver512<16> = FFTConvolver512::<16>::default_const();

fn audio_init(ir: &[f32]) {
    unsafe { CONV.init(ir).unwrap() };
}

fn audio_callback(input: &[f32], output: &mut [f32]) {
    unsafe { CONV.process(input, output).unwrap() };
}
```

> **Note:** `default_const()` is a `const fn` that produces a zero-initialized convolver suitable for `static` placement, since Rust does not currently call `Default::default()` in `static` initializers.

### Updating the impulse response at runtime

```rust
use fft_convolver_nostd::FFTConvolver128;

let ir1 = [0.5f32, 0.3, 0.2, 0.1];
let mut conv = FFTConvolver128::<1>::default();
conv.init(&ir1).unwrap();

// swap to a different IR — no allocation, real-time safe
let ir2 = [0.8f32, 0.6, 0.4];
conv.set_response(&ir2).unwrap();
```

### Handling stream discontinuities

```rust
use fft_convolver_nostd::FFTConvolver128;

let mut conv = FFTConvolver128::<1>::default();
conv.init(&[0.5f32, 0.3, 0.2]).unwrap();

let input = [1.0f32; 128];
let mut output = [0.0f32; 128];
conv.process(&input, &mut output).unwrap();

// Seek / discontinuity: clear history but keep the loaded IR
conv.reset();
conv.process(&input, &mut output).unwrap();
```

## Memory footprint

The struct size is determined entirely by the const parameters. For reference:

| Config | `BLOCK_SIZE` | `SEG_COUNT` | Approx. struct size |
|---|---|---|---|
| `FFTConvolver64::<8>` | 64 | 8 | ~10 KB |
| `FFTConvolver256::<16>` | 256 | 16 | ~134 KB |
| `FFTConvolver512::<16>` | 512 | 16 | ~264 KB |
| `FFTConvolver1024::<8>` | 1024 | 8 | ~264 KB |

For anything above ~8 KB, use `static` placement.

## Supported block sizes

`BLOCK_SIZE` must be one of: **8, 16, 32, 64, 128, 256, 512, 1024, 2048**.

Other values will return `FFTConvolverError::UnsupportedBlockSize` from `init`.

## Real-time safety

Operations marked `#[nonblocking]` (checked by [rtsan](https://docs.rs/rtsan-standalone) in debug builds):

| Method | Real-time safe |
|---|---|
| `init` | No (resets all buffers) |
| `process` | Yes |
| `set_response` | Yes |
| `reset` | Yes |

## License

MIT
