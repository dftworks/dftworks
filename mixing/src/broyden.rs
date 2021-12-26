// Phys. Rev. B 38, 12807 (1988)

use control::Control;
use fifo::*;
use matrix::*;
use num_traits::identities::Zero;
use types::*;

use crate::Mixing;

pub struct MixingBroyden {
    alpha: f64,
    vin: FIFO<Vec<c64>>,
    vout: FIFO<Vec<c64>>,

    niter: usize,
}

impl MixingBroyden {
    pub fn new(control: &Control) -> MixingBroyden {
        let alpha = control.get_scf_rho_mix_alpha();
        let nhistory = control.get_scf_rho_mix_history_steps();

        MixingBroyden {
            alpha,
            vin: FIFO::new(nhistory),
            vout: FIFO::new(nhistory),
            niter: 0,
        }
    }
}

impl Mixing for MixingBroyden {
    fn compute_next_density(&mut self, _gs: &[f64], inp: &mut [c64], out: &[c64]) {
        self.vin.push(inp.to_vec());
        self.vout.push(out.to_vec());

        let ng = inp.len();

        for ig in 0..ng {
            inp[ig] += self.alpha * out[ig];
        }

        if self.vin.len() > 1 {
            let m = self.vin.len() - 1;

            // println!("m = {}", m);

            let omega = vec![1.0; self.vin.len()];

            // for (i, v) in omega.iter_mut().enumerate() {
            //    *v = 1.0 / utility::l2_norm(&self.vout[i]);
            // }

            //println!("omega = {:?}", omega);

            let a = compute_a(&self.vout, &omega);
            let c = compute_c(&self.vout, &omega);
            let beta = compute_beta(&a);
            let gamma = c.dot(&beta);

            // println!("a = {}", a);
            //            println!("c = {}", c);
            //            println!("beta = {}", beta);
            //println!("gamma = {}", gamma);

            let mut dres = vec![c64::zero(); ng];
            let mut drho = vec![c64::zero(); ng];

            for n in 0..m {
                for ig in 0..ng {
                    dres[ig] = self.vout[n + 1][ig] - self.vout[n][ig];
                    drho[ig] = self.vin[n + 1][ig] - self.vin[n][ig];
                }

                let l2_norm = utility::l2_norm(&dres);

                for ig in 0..ng {
                    dres[ig] /= l2_norm;
                    drho[ig] /= l2_norm;
                }

                for ig in 0..ng {
                    inp[ig] -= omega[n] * gamma[[m, n]] * (self.alpha * dres[ig] + drho[ig]);
                }
            }
        }

        self.niter += 1;
    }
}

fn compute_beta(a: &Matrix<c64>) -> Matrix<c64> {
    let mut m = a.clone();

    let omega0 = 0.01; // the weight determining how close the subsequent inverser Jacobian

    for i in 0..m.nrow() {
        m[[i, i]] += omega0 * omega0;
    }

    m.inv();

    m
}

fn compute_a(res: &FIFO<Vec<c64>>, omega: &[f64]) -> Matrix<c64> {
    let m = res.len() - 1;

    let ng = res.last().len();

    let mut a = Matrix::<c64>::new(m, m);

    let mut dresj = vec![c64::zero(); ng];
    let mut dresi = vec![c64::zero(); ng];

    for i in 0..m {
        for ig in 0..ng {
            dresi[ig] = res[i + 1][ig] - res[i][ig];
        }

        utility::normalize_vector_c64(&mut dresi);

        for j in 0..m {
            for ig in 0..ng {
                dresj[ig] = res[j + 1][ig] - res[j][ig];
            }

            utility::normalize_vector_c64(&mut dresj);

            a[[i, j]] = utility::zdot_product(&dresj, &dresi) * omega[j] * omega[i];
        }
    }

    a
}

// dimension: (m+1) x m

fn compute_c(res: &FIFO<Vec<c64>>, omega: &[f64]) -> Matrix<c64> {
    let m = res.len() - 1;

    let ng = res.last().len();

    let mut c = Matrix::<c64>::new(m + 1, m);

    let mut dresk = vec![c64::zero(); ng];

    for i in 0..m + 1 {
        let resi = &res[i];

        for k in 0..m {
            for ig in 0..ng {
                dresk[ig] = res[k + 1][ig] - res[k][ig];
            }

            utility::normalize_vector_c64(&mut dresk);

            c[[i, k]] = utility::zdot_product(&dresk, &resi) * omega[k];
        }
    }

    c
}
