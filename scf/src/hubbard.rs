use control::Control;
use dfttypes::{VKEigenVector, VKSCF};
use kscf::KSCF;
use matrix::Matrix;
use mpi_sys::MPI_COMM_WORLD;
use types::c64;

fn occ_index(iat: usize, m: usize, mp: usize, n_m: usize) -> usize {
    iat * n_m * n_m + m + mp * n_m
}

fn dudarev_energy(
    u_eff: f64,
    n_atoms: usize,
    n_m: usize,
    occ_global: &[c64],
    spin_weight: f64,
) -> f64 {
    let mut energy = 0.0;

    for iat in 0..n_atoms {
        let mut tr_n = 0.0;
        let mut tr_n2 = 0.0;

        for m in 0..n_m {
            tr_n += occ_global[occ_index(iat, m, m, n_m)].re;

            for mp in 0..n_m {
                let a = occ_global[occ_index(iat, m, mp, n_m)];
                let b = occ_global[occ_index(iat, mp, m, n_m)];
                tr_n2 += (a * b).re;
            }
        }

        energy += 0.5 * u_eff * (tr_n - tr_n2);
    }

    spin_weight * energy
}

fn symmetrize_occupation(occ_global: &mut [c64], n_atoms: usize, n_m: usize) {
    for iat in 0..n_atoms {
        for m in 0..n_m {
            for mp in 0..n_m {
                if mp < m {
                    continue;
                }

                let a_idx = occ_index(iat, m, mp, n_m);
                let b_idx = occ_index(iat, mp, m, n_m);
                let avg = 0.5 * (occ_global[a_idx] + occ_global[b_idx].conj());

                occ_global[a_idx] = avg;
                occ_global[b_idx] = avg.conj();
            }
        }
    }
}

fn update_hubbard_for_channel<'a>(
    kscf_channel: &mut [KSCF<'a>],
    evecs_channel: &[Matrix<c64>],
    spin_weight: f64,
) -> f64 {
    if kscf_channel.is_empty() {
        return 0.0;
    }

    if !kscf_channel[0].hubbard_is_enabled() {
        return 0.0;
    }

    let occ_len = kscf_channel[0].hubbard_occ_len();
    let n_atoms = kscf_channel[0].hubbard_n_atoms();
    let n_m = kscf_channel[0].hubbard_n_m();
    let u_eff = kscf_channel[0].hubbard_u_eff();

    let mut occ_local = vec![c64::new(0.0, 0.0); occ_len];
    for (ik, kscf) in kscf_channel.iter().enumerate() {
        kscf.hubbard_accumulate_occupation(&evecs_channel[ik], &mut occ_local);
    }

    let mut occ_global = vec![c64::new(0.0, 0.0); occ_len];
    dwmpi::reduce_slice_sum(&occ_local, &mut occ_global, MPI_COMM_WORLD);
    dwmpi::bcast_slice(&mut occ_global, MPI_COMM_WORLD);
    symmetrize_occupation(&mut occ_global, n_atoms, n_m);

    for kscf in kscf_channel.iter_mut() {
        kscf.hubbard_set_global_occupation(&occ_global);
    }

    dudarev_energy(u_eff, n_atoms, n_m, &occ_global, spin_weight)
}

pub fn update_hubbard_nonspin(
    control: &Control,
    vkscf: &mut VKSCF,
    vkevecs: &VKEigenVector,
) -> f64 {
    if !control.get_hubbard_u_enabled() {
        return 0.0;
    }

    let t_vkscf = vkscf.as_non_spin_mut().unwrap();
    let t_vkevecs = vkevecs.as_non_spin().unwrap();

    // Non-spin Kohn-Sham states are doubly occupied in this code path.
    update_hubbard_for_channel(t_vkscf, t_vkevecs, 2.0)
}

pub fn update_hubbard_spin(control: &Control, vkscf: &mut VKSCF, vkevecs: &VKEigenVector) -> f64 {
    if !control.get_hubbard_u_enabled() {
        return 0.0;
    }

    let (vkscf_up, vkscf_dn) = vkscf.as_spin_mut().unwrap();
    let (vkevecs_up, vkevecs_dn) = vkevecs.as_spin().unwrap();

    let energy_up = update_hubbard_for_channel(vkscf_up, vkevecs_up, 1.0);
    let energy_dn = update_hubbard_for_channel(vkscf_dn, vkevecs_dn, 1.0);

    energy_up + energy_dn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dudarev_energy_increases_for_fractional_occupation() {
        let n_atoms = 1usize;
        let n_m = 2usize;
        let u_eff = 4.0;

        // n = diag(0.5, 0.5) => Tr n = 1, Tr n^2 = 0.5
        let mut occ = vec![c64::new(0.0, 0.0); n_atoms * n_m * n_m];
        occ[occ_index(0, 0, 0, n_m)] = c64::new(0.5, 0.0);
        occ[occ_index(0, 1, 1, n_m)] = c64::new(0.5, 0.0);

        let e = dudarev_energy(u_eff, n_atoms, n_m, &occ, 1.0);
        assert!((e - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_dudarev_energy_zero_for_idempotent_occupation() {
        let n_atoms = 1usize;
        let n_m = 2usize;
        let u_eff = 3.2;

        // n = diag(1, 0) => Tr n = 1, Tr n^2 = 1
        let mut occ = vec![c64::new(0.0, 0.0); n_atoms * n_m * n_m];
        occ[occ_index(0, 0, 0, n_m)] = c64::new(1.0, 0.0);

        let e = dudarev_energy(u_eff, n_atoms, n_m, &occ, 1.0);
        assert!(e.abs() < 1.0e-12);
    }
}
