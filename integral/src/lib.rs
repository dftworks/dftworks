pub fn simpson_rab(y: &[f64], rab: &[f64]) -> f64 {
    let mut n = y.len();

    if n.is_multiple_of(2) {
        n -= 3;
    }

    let r12 = 1.0 / 3.0;

    let mut t1;

    let mut t2;

    let mut t3 = y[0] * rab[0] * r12;

    let mut s = 0.0;

    for i in (0..n - 1).step_by(2) {
        t1 = t3;

        t2 = y[i + 1] * rab[i + 1] * r12;

        t3 = y[i + 2] * rab[i + 2] * r12;

        s += t1 + 4.0 * t2 + t3;
    }

    if y.len().is_multiple_of(2) {
        let n = y.len();

        let r12 = 3.0 / 8.0;

        s += y[n - 4] * rab[n - 4] * r12
            + 3.0 * y[n - 3] * rab[n - 3] * r12
            + 3.0 * y[n - 2] * rab[n - 2] * r12
            + y[n - 1] * rab[n - 1] * r12;
    }

    s
}

pub fn simpson(y: &[f64], dx: f64) -> f64 {
    let n = y.len();

    let mut t1;

    let mut t2;

    let mut t3 = y[0];

    let mut s = 0.0;

    for i in (0..n - 1).step_by(2) {
        t1 = t3;

        t2 = y[i + 1];

        t3 = y[i + 2];

        s += t1 + 4.0 * t2 + t3;
    }

    s * dx / 3.0
}

pub fn simpson_log(y: &[f64], r: &[f64]) -> f64 {
    let n = y.len();

    assert!(!n.is_multiple_of(2));

    let nf64 = n as f64;

    let dx = (r[n - 1] / r[0]).ln() / (nf64 - 1.0);

    let mut s = 0.0;

    for i in (0..n - 1).step_by(2) {
        s += 1.0 * y[i] * r[i];

        s += 4.0 * y[i + 1] * r[i + 1];

        s += 1.0 * y[i + 2] * r[i + 2];
    }

    s * dx / 3.0
}

#[test]
fn test_simpson() {
    let y: Vec<f64> = itertools_num::linspace::<f64>(0.0, 4.0, 5).collect();
    let sum = simpson(&y[..], 1.0);
    assert_eq!(sum, 8.0);
}

#[test]
fn test_simpson_rab_odd_count_matches_analytic() {
    // Integrate y = x^2 on [0, 4] with dx = 1.
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let rab = vec![1.0; y.len()];

    let got = simpson_rab(&y, &rab);
    let expected = 64.0 / 3.0;
    assert!((got - expected).abs() < 1.0e-12);
}

#[test]
fn test_simpson_rab_even_count_uses_38_tail() {
    // Integrate y = x^3 on [0, 5] with 6 points (dx = 1).
    // Composite Simpson 1/3 + 3/8 should be exact for cubic polynomials.
    let y = vec![0.0, 1.0, 8.0, 27.0, 64.0, 125.0];
    let rab = vec![1.0; y.len()];

    let got = simpson_rab(&y, &rab);
    let expected = 625.0 / 4.0;
    assert!((got - expected).abs() < 1.0e-12);
}

#[test]
fn test_simpson_log_constant_in_log_space() {
    // If y(r) = 1/r, then y(r)*r = 1 and the log-grid Simpson rule is exact.
    let n = 5;
    let r0 = 0.5f64;
    let ratio = 2.0f64;
    let r: Vec<f64> = (0..n).map(|i| r0 * ratio.powi(i as i32)).collect();
    let y: Vec<f64> = r.iter().map(|ri| 1.0 / ri).collect();

    let got = simpson_log(&y, &r);
    let expected = (r[n - 1] / r[0]).ln();
    assert!((got - expected).abs() < 1.0e-12);
}

#[test]
#[should_panic]
fn test_simpson_log_requires_odd_length() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let r = vec![1.0, 2.0, 4.0, 8.0];
    let _ = simpson_log(&y, &r);
}
