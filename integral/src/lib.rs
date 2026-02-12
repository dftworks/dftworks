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

    println!("{:?}", &y);

    let sum = simpson(&y[..], 1.0);

    assert_eq!(sum, 8.0);
}
