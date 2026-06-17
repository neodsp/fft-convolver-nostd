use microfft::Complex32;

pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
    assert!(dst.len() >= src_size);
    dst[0..src_size].copy_from_slice(&src[0..src_size]);
    dst[src_size..].fill(0.0);
}

pub fn complex_multiply_accumulate(result: &mut [Complex32], a: &[Complex32], b: &[Complex32]) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 4 * (len / 4);
    #[allow(clippy::identity_op)]
    for i in (0..end4).step_by(4) {
        result[i + 0].re += a[i + 0].re * b[i + 0].re - a[i + 0].im * b[i + 0].im;
        result[i + 1].re += a[i + 1].re * b[i + 1].re - a[i + 1].im * b[i + 1].im;
        result[i + 2].re += a[i + 2].re * b[i + 2].re - a[i + 2].im * b[i + 2].im;
        result[i + 3].re += a[i + 3].re * b[i + 3].re - a[i + 3].im * b[i + 3].im;
        result[i + 0].im += a[i + 0].re * b[i + 0].im + a[i + 0].im * b[i + 0].re;
        result[i + 1].im += a[i + 1].re * b[i + 1].im + a[i + 1].im * b[i + 1].re;
        result[i + 2].im += a[i + 2].re * b[i + 2].im + a[i + 2].im * b[i + 2].re;
        result[i + 3].im += a[i + 3].re * b[i + 3].im + a[i + 3].im * b[i + 3].re;
    }
    for i in end4..len {
        result[i].re += a[i].re * b[i].re - a[i].im * b[i].im;
        result[i].im += a[i].re * b[i].im + a[i].im * b[i].re;
    }
}

pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 4 * (len / 4);
    #[allow(clippy::identity_op)]
    for i in (0..end4).step_by(4) {
        result[i + 0] = a[i + 0] + b[i + 0];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
    }
    for i in end4..len {
        result[i] = a[i] + b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_and_pad_test() {
        let mut dst = [1.0f32; 10];
        let src = [2.0, 3.0, 4.0, 5.0, 6.0f32];
        copy_and_pad(&mut dst, &src, src.len());

        assert_eq!(dst[0], 2.0);
        assert_eq!(dst[4], 6.0);
        for &v in &dst[5..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn complex_multiply_accumulate_test() {
        let mut result = [Complex32::new(0.0, 0.0); 10];
        let a: [Complex32; 10] = core::array::from_fn(|i| Complex32::new(i as f32, (9 - i) as f32));
        let b: [Complex32; 10] = core::array::from_fn(|i| Complex32::new((9 - i) as f32, i as f32));

        complex_multiply_accumulate(&mut result, &a, &b);

        for num in &result {
            assert_eq!(num.re, 0.0);
        }
        assert_eq!(result[0].im, 81.0);
        assert_eq!(result[9].im, 81.0);
    }

    #[test]
    fn sum_test() {
        let mut result = [0.0f32; 10];
        let a = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.0f32];
        let b = [0., 6., 3., 1., 5., 3., 5., 1., 4., 0.0f32];
        sum(&mut result, &a, &b);
        assert_eq!(result[1], 7.0);
        assert_eq!(result[4], 9.0);
    }
}
