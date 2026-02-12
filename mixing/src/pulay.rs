//#![allow(warnings)]

use control::Control;
use fifo::*;
use matrix::*;
use num_traits::identities::Zero;
use types::*;

use std::fs::File;
use std::io::Write;

use crate::Mixing;

pub struct MixingPulay {
    metric_weight: f64,
    alpha: f64,
    dump_history: bool,

    vin: FIFO<Vec<c64>>,
    vout: FIFO<Vec<c64>>,

    niter: usize,
}

impl MixingPulay {
    pub fn new(control: &Control) -> MixingPulay {
        let alpha = control.get_scf_rho_mix_alpha();
        let nhistory = control.get_scf_rho_mix_history_steps();
        let metric_weight = control.get_scf_rho_mix_pulay_metric_weight();
        let dump_history = std::env::var_os("DFTWORKS_PULAY_DUMP").is_some();

        MixingPulay {
            metric_weight,
            alpha,
            dump_history,
            vin: FIFO::new(nhistory),
            vout: FIFO::new(nhistory),
            niter: 0,
        }
    }
}

impl Mixing for MixingPulay {
    fn compute_next_density(&mut self, gs: &[f64], inp: &mut [c64], out: &[c64]) {
        self.niter += 1;

        self.vin.push(inp.to_vec());
        self.vout.push(out.to_vec());

        let ng = inp.len();

        if self.niter == 1 {
            for ig in 0..ng {
                inp[ig] = inp[ig] + self.alpha * out[ig];
            }
        //println!("linear mixing -- {}", self.niter);
        } else {
            let coef = compute_coef(self.metric_weight, gs, &self.vout);

            for z in inp.iter_mut() {
                *z = c64::zero();
            }

            for j in 0..coef.len() {
                let tin = &self.vin[j];
                let tout = &self.vout[j];

                for ig in 0..ng {
                    inp[ig] += coef[j] * (tin[ig] + self.alpha * tout[ig]);
                }
            }
        }

        if self.dump_history {
            dump_profile("rho-res-", self.niter, gs, out);
            dump_profile("rho-g-", self.niter, gs, inp);
        }
    }
}

fn dump_profile(prefix: &str, niter: usize, gs: &[f64], v: &[c64]) {
    if let Ok(mut output) = File::create(format!("{prefix}{niter}")) {
        for (ig, (&g, val)) in gs.iter().zip(v.iter()).enumerate() {
            let _ = writeln!(&mut output, "{:5?}{:20.12?}{:20.12?}", ig, g, val.norm());
        }
    }
}

fn compute_coef(weight: f64, gs: &[f64], vout: &FIFO<Vec<c64>>) -> Vec<c64> {
    let n = vout.len();

    let mut a = Matrix::<c64>::new(n, n);

    let ng = gs.len();

    let mut metric = vec![0.0; ng];

    for ig in 1..ng {
        let q2 = gs[ig] * gs[ig];
        metric[ig] = (weight + q2) / q2;
    }

    metric[0] = 0.0; //metric[1];

    // println!("M0/MN = {}", metric[0] / metric.last().unwrap());

    //println!("metric_head = {:?}", &metric[0..10]);
    //println!("metric_tail = {:?}", &metric[metric.len() - 10..]);

    for i in 0..n {
        let vi = &vout[i];

        for j in 0..n {
            let vj = &vout[j];

            a[[j, i]] = utility::zdot_product_metric(vj, vi, &metric);
        }
    }

    let mut alpha = vec![c64::zero(); n];

    a.pinv();

    let s = a.sum();

    for i in 0..n {
        let mut f = c64::zero();

        for j in 0..n {
            f += a[[j, i]];
        }

        alpha[i] = f / s;
    }

    //println!(" sum of coef: {}", alpha.iter().sum::<c64>());

    alpha
}
