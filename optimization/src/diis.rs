#![allow(warnings)]

use fifo::*;
use matrix::*;
use utility::*;

pub struct DIIS {
    alpha: f64,
    vin: FIFO<Vec<f64>>,
    vout: FIFO<Vec<f64>>,
}

use crate::OptimizationDriver;

impl DIIS {
    pub fn new(alpha: f64, nstep: usize) -> DIIS {
        DIIS {
            alpha,
            vin: FIFO::new(nstep),
            vout: FIFO::new(nstep),
        }
    }
}

impl OptimizationDriver for DIIS {
    fn compute_next_input(&mut self, inp: &mut [f64], out: &[f64], mask: &[f64]) {
        self.vin.push(inp.to_vec());
        self.vout.push(out.to_vec());

        let coef = compute_coef(&self.vout);

        // println!("coef = {:?}", coef);

        for i in 0..inp.len() {
            inp[i] = 0.0;
        }

        for j in 0..coef.len() {
            let tin = &self.vin[j];
            let tout = &self.vout[j];

            for i in 0..inp.len() {
                inp[i] += coef[j] * (tin[i] - self.alpha * tout[i]);
            }
        }

        let vin_last = self.vin.last();
        for i in 0..mask.len() {
            inp[i] = inp[i] * mask[i] + vin_last[i] * (1.0 - mask[i]); // mask[i] ::  1.0: change; 0.0: no change
        }
    }
}

fn compute_coef(vout: &FIFO<Vec<f64>>) -> Vec<f64> {
    let n = vout.len();

    let mut a = Matrix::<f64>::new(n, n);

    for i in 0..n {
        let vi = &vout[i];

        for j in 0..n {
            let vj = &vout[j];

            a[[j, i]] = utility::ddot_product(vj, vi);
        }
    }

    let mut alpha = vec![0.0; n];

    let mut b = a.clone();

    a.pinv();
    
    let s = a.sum();

    for i in 0..n {
        alpha[i] = 0.0;

        for j in 0..n {
            alpha[i] += a[[j, i]] / s;
        }
    }

    alpha
}

fn df_xy(x: f64, y: f64) -> Vec<f64> {
    vec![-2.0 * x - y * y, -x * x - 2.0 * y]
}

/// cargo test test_diis --lib -- --nocapture
#[test]
fn test_diis() {
    let mut diis = DIIS::new(0.7, 6);

    let mut inp = vec![0.0; 2];

    inp[0] = 0.1;
    inp[1] = 0.1;

    let mut out = vec![0.0; 2];

    let mask = [1.0, 1.0];

    for i in 0..50 {
        println!("{:5?}   input : {:20.16?}", i, inp);
        out = df_xy(inp[0], inp[1]);
        println!("       output : {:20.16?}", out);

        let norm2 = out.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm2 < 1.0E-12 {
            break;
        }

        diis.compute_next_input(&mut inp, &out, &mask);
    }
}
