#![allow(warnings)]

use atompsp::*;
use crystal::*;
use dfttypes::*;
use dwconsts::*;
use gvector::*;
use itertools::multizip;
use kgylm::KGYLM;
use kscf::*;
use lattice::*;
use matrix::*;
use ndarray::*;
use num_traits::identities::Zero;
use pspot::*;
use pwbasis::*;
use pwdensity::*;
use types::*;
use vector3::*;

// Phys. Rev. B 41, 7878 (1990)
pub fn nlcc_xc(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    vxcg: &Vec<c64>,
) -> Matrix<f64> {
    let mut stress_c64 = Matrix::<c64>::new(3, 3);

    let volume = crystal.get_latt().volume();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let gidx = pwden.get_gindex();

    let npw_rho = pwden.get_n_plane_waves();
    let mut rhocoreg = vec![0.0; npw_rho];
    let mut drhocoreg = vec![0.0; npw_rho];

    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();
    let species = crystal.get_atom_species();

    let unit_mat = Matrix::<f64>::unit(3);

    for iat in 0..natoms {
        let atom = atom_positions[iat];

        let atpsp = atpsps.get_psp(&species[iat]);

        rhocore_of_g_one_atom(atpsp, pwden, volume, &mut rhocoreg);
        drhocore_of_g_one_atom(atpsp, pwden, volume, &mut drhocoreg);

        for ipw in 0..npw_rho {
            let mill = miller[gidx[ipw]];

            //let ngd =
            //    -TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let ngd = -TWOPI * utility::dot_product_v3i32_v3f64(mill, atom);

            let comm = c64 {
                re: ngd.cos(),
                im: ngd.sin(),
            } * vxcg[ipw].conj();

            let gcoord = cart[gidx[ipw]].to_vec();

            for i in 0..3 {
                for j in 0..3 {
                    stress_c64[[j, i]] += comm
                        * (rhocoreg[ipw] * unit_mat[[j, i]]
                            + drhocoreg[ipw] * 2.0 * gcoord[j] * gcoord[i]);
                }
            }
        }
    }

    let mut stress = Matrix::<f64>::new(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            stress[[i, j]] = stress_c64[[i, j]].re;
        }
    }

    stress
}

pub fn vpsloc(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    rhog: &[c64],
) -> Matrix<f64> {
    let mut stress_c64 = Matrix::<c64>::new(3, 3);

    let volume = crystal.get_latt().volume();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let gidx = pwden.get_gindex();

    let npw_rho = pwden.get_n_plane_waves();
    let mut vatlocg = vec![0.0; npw_rho];
    let mut dvatlocg = vec![0.0; npw_rho];

    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();
    let species = crystal.get_atom_species();

    let unit_mat = Matrix::<f64>::unit(3);

    for iat in 0..natoms {
        let atom = atom_positions[iat];

        let atpsp = atpsps.get_psp(&species[iat]);

        vpsloc_of_g_one_atom(atpsp, pwden, volume, &mut vatlocg);
        dvpsloc_of_g_one_atom(atpsp, pwden, volume, &mut dvatlocg);

        for ipw in 0..npw_rho {
            let mill = miller[gidx[ipw]];

            //let ngd =
            //    -TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let ngd = -TWOPI * utility::dot_product_v3i32_v3f64(mill, atom);

            let comm = c64 {
                re: ngd.cos(),
                im: ngd.sin(),
            } * rhog[ipw].conj();

            let gcoord = cart[gidx[ipw]].to_vec();

            for i in 0..3 {
                for j in 0..3 {
                    stress_c64[[j, i]] += comm
                        * (vatlocg[ipw] * unit_mat[[j, i]]
                            + dvatlocg[ipw] * 2.0 * gcoord[j] * gcoord[i]);
                }
            }
        }
    }

    let mut stress = Matrix::<f64>::new(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            stress[[i, j]] = stress_c64[[i, j]].re;
        }
    }

    stress
}

pub fn xc(
    latt: &Lattice,
    rho_3d: &mut Array3<c64>,
    rhocore_3d: &Array3<c64>,
    vxc_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
) -> Matrix<f64> {
    let stress_vxc_rho = energy::vxc(
        latt,
        rho_3d.as_slice(),
        rhocore_3d.as_slice(),
        vxc_3d.as_slice(),
    ) / latt.volume();

    let stress_exc = energy::exc(latt, rho_3d, rhocore_3d, exc_3d) / latt.volume();

    let mut stress = Matrix::<f64>::new(3, 3);

    for i in 0..3 {
        stress[[i, i]] = -(stress_exc - stress_vxc_rho);
    }

    stress
}

pub fn xc_spin(
    latt: &Lattice,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    vxc_3d: &VXCR,
    exc_3d: &Array3<c64>,
) -> Matrix<f64> {
    let stress_vxc_rho =
        energy::vxc_spin(latt, rho_3d, rhocore_3d.as_slice(), vxc_3d) / latt.volume();

    let stress_exc = energy::exc_spin(latt, rho_3d, rhocore_3d, exc_3d) / latt.volume();

    let mut stress = Matrix::<f64>::new(3, 3);

    for i in 0..3 {
        stress[[i, i]] = -(stress_exc - stress_vxc_rho);
    }

    stress
}

pub fn hartree(gvec: &GVector, pwden: &PWDensity, rhog: &[c64]) -> Matrix<f64> {
    let cart = gvec.get_cart();

    let gnorm = pwden.get_g();
    let gidx = pwden.get_gindex();
    let npw = pwden.get_n_plane_waves();

    let unit_mat = Matrix::<f64>::unit(3);

    let mut stress = Matrix::<f64>::new(3, 3);

    for ig in 1..npw {
        let g = cart[gidx[ig]].to_vec();

        for i in 0..3 {
            for j in 0..3 {
                stress[[j, i]] += -FOURPI / 2.0 * rhog[ig].norm_sqr() / gnorm[ig] / gnorm[ig]
                    * (2.0 * g[i] * g[j] / gnorm[ig] / gnorm[ig] - unit_mat[[j, i]]);
            }
        }
    }

    stress
}

pub fn kinetic(crystal: &Crystal, vkscf: &[KSCF], vevecs: &Vec<Matrix<c64>>) -> Matrix<f64> {
    let cart = vkscf[0].get_gvec().get_cart();

    let mut stress = Matrix::<f64>::new(3, 3);

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evecs = &vevecs[ik];

        let gidx = kscf.get_pwwfc().get_gindex();
        let npw = kscf.get_pwwfc().get_n_plane_waves();
        let xk = kscf.get_pwwfc().get_k_cart().to_vec();
        let nbnd = kscf.get_nbands();
        let occ = kscf.get_occ();

        for ibnd in 0..nbnd {
            if occ[ibnd] < EPS20 {
                continue;
            }

            let cnk = evecs.get_col(ibnd);

            for ikg in 0..npw {
                let g = cart[gidx[ikg]].to_vec();

                for i in 0..3 {
                    for j in 0..3 {
                        stress[[j, i]] += cnk[ikg].norm_sqr()
                            * (xk[j] + g[j])
                            * (xk[i] + g[i])
                            * occ[ibnd]
                            * kscf.get_k_weight();
                    }
                }
            }
        }
    }

    let volume = crystal.get_latt().volume();

    stress.as_mut_slice().iter_mut().for_each(|v| *v /= volume);

    // let p_stress = stress.as_mut_slice();

    // for v in p_stress.iter_mut() {
    //     *v /= volume;
    // }

    stress
}

// This implementation does not include the prefractor 1/(2-\delta_{\alpha,\beta}) in Eq. 15 PRB 15, 14697 (1995).
// Surprisely, this is basically the same with the calculation result in QE.
// Since the off-diagonal parts are very small compared with the diagonals.
// So this may not be a big problem in QE.
// This implementation is the same with Eq. B2 in PRB 41, 1394 (1990) correction.

pub fn vnl(crystal: &Crystal, vkscf: &[KSCF], vevecs: &Vec<Matrix<c64>>) -> Matrix<f64> {
    let mut stress = Matrix::<c64>::new(3, 3);

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evecs = &vevecs[ik];

        let kylms = kscf.get_kgylm();

        let kgbeta_all = kscf.get_vnl().get_kgbeta_all();
        let dkgbeta_all = kscf.get_vnl().get_dkgbeta_all();

        for (isp, specie) in crystal.get_unique_species().iter().enumerate() {
            let kgbeta = kgbeta_all.get(specie).unwrap();

            let dkgbeta = dkgbeta_all.get(specie).unwrap();

            let atom_positions = crystal.get_atom_positions_of_specie(isp);

            let atompsp = kscf.get_pspot().get_psp(specie);

            stress += vnl_of_one_specie_one_k(
                atompsp,
                &atom_positions,
                kscf.get_gvec(),
                kscf.get_pwwfc(),
                kgbeta,
                dkgbeta,
                &kylms,
                kscf,
                evecs,
            ) * kscf.get_k_weight();
        }
    }

    let mut stress_f64 = Matrix::<f64>::new(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            stress_f64[[j, i]] = 2.0 * stress[[j, i]].re;
        }
    }

    let volume = crystal.get_latt().volume();

    stress_f64
        .as_mut_slice()
        .iter_mut()
        .for_each(|v| *v /= volume);

    // let p_stress = stress_f64.as_mut_slice();

    // for v in p_stress.iter_mut() {
    //     *v /= volume;
    // }

    stress_f64
}

pub fn vnl_of_one_specie_one_k(
    atpsp: &dyn AtomPSP,
    atom_positions: &[Vector3f64],
    gvec: &GVector,
    pwwfc: &PWBasis,
    vnlbeta: &Vec<Vec<f64>>,
    dvnlbeta: &Vec<Vec<f64>>,
    ylms: &KGYLM,
    kscf: &KSCF,
    evecs: &Matrix<c64>,
) -> Matrix<c64> {
    let xk = pwwfc.get_k_cart().to_vec();

    let kg = pwwfc.get_kg();

    let miller = gvec.get_miller();

    let cart = gvec.get_cart();

    let gidx = pwwfc.get_gindex();

    let nbeta = atpsp.get_nbeta();

    let npw = pwwfc.get_n_plane_waves();

    let occ = kscf.get_occ();

    let unit_mat = Matrix::<f64>::unit(3);
    let mut stress = Matrix::<c64>::new(3, 3);

    for ibnd in 0..kscf.get_nbands() {
        if occ[ibnd] < EPS20 {
            continue;
        }

        let cnk = evecs.get_col(ibnd);

        let mut stress_band = Matrix::<c64>::new(3, 3);

        for atom_position in atom_positions.iter() {
            let sfact =
                fhkl::compute_structure_factor_for_many_g_one_atom(miller, gidx, *atom_position);

            for ibeta in 0..nbeta {
                let l = atpsp.get_lbeta(ibeta);

                let beta = &vnlbeta[ibeta];
                let dbeta = &dvnlbeta[ibeta];

                let dfact = atpsp.get_dfact(ibeta);

                for m in utility::get_quant_num_m(l) {
                    let ylm = ylms.get_data(l, m);
                    let dylm = ylms.get_data_derivatives(l, m);

                    let mut beta_kg_cnk = c64::zero();

                    for ipw in 0..npw {
                        beta_kg_cnk += cnk[ipw].conj() * beta[ipw] * ylm[ipw] * sfact[ipw];
                    }

                    for ipw in 0..npw {
                        let p_dylm = dylm[ipw].to_vec();

                        let g = cart[gidx[ipw]].to_vec();

                        for i in 0..3 {
                            for j in 0..3 {
                                stress_band[[j, i]] += dfact
                                    * beta_kg_cnk
                                    * cnk[ipw]
                                    * sfact[ipw].conj()
                                    * (beta[ipw] * (xk[i] + g[i]) * p_dylm[j]
                                        + 0.5 * unit_mat[[i, j]] * beta[ipw] * ylm[ipw]
                                        - dbeta[ipw] * ylm[ipw] * (xk[i] + g[i]) * (xk[j] + g[j])
                                            / kg[ipw].max(EPS20));
                            }
                        }
                    }
                }
            }
        }

        stress += stress_band * occ[ibnd];
    }

    stress
}

pub fn get_max_stress(stress: &Matrix<f64>) -> f64 {
    stress
        .as_slice()
        .iter()
        .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
        .unwrap()
        .abs()

    // let p_data = stress.as_slice();

    // let nlen = p_data.len();

    // let mut v_max = 0.0;

    // for i in 0..nlen {
    //     if v_max < p_data[i].abs() {
    //         v_max = p_data[i].abs();
    //     }
    // }

    // v_max
}

fn frac_to_cart(latt: &Lattice, mf: &Matrix<f64>, mc: &mut Matrix<f64>) {
    let mat = latt.as_matrix().clone().transpose();

    for c in mc.as_mut_slice().iter_mut() {
        *c = 0.0;
    }

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    mc[[i, j]] += mat[[i, k]] * mf[[k, l]] * mat[[j, l]];
                }
            }
        }
    }
}

fn cart_to_frac(latt: &Lattice, mc: &Matrix<f64>, mf: &mut Matrix<f64>) {
    let mut mat = latt.as_matrix().clone().transpose();

    mat.inv();

    for f in mf.as_mut_slice().iter_mut() {
        *f = 0.0;
    }

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    mf[[i, j]] += mat[[i, k]] * mc[[k, l]] * mat[[j, l]];
                }
            }
        }
    }
}

fn vpsloc_of_g_one_atom(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64, vlocg: &mut [f64]) {
    let ffact_vloc = vloc::vloc_of_g_on_shells(atompsp, pwden, volume);

    let npw_rho = pwden.get_n_plane_waves();

    let gshell_index = pwden.get_gshell_index();

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        vlocg[i] = ffact_vloc[ish];
    }
}

fn dvpsloc_of_g_one_atom(
    atompsp: &dyn AtomPSP,
    pwden: &PWDensity,
    volume: f64,
    dvlocg: &mut [f64],
) {
    let ffact_dvloc = vloc::dvloc_of_g_on_shells(atompsp, pwden, volume);

    let npw_rho = pwden.get_n_plane_waves();

    let gshell_index = pwden.get_gshell_index();

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        dvlocg[i] = ffact_dvloc[ish];
    }
}

fn rhocore_of_g_one_atom(
    atompsp: &dyn AtomPSP,
    pwden: &PWDensity,
    volume: f64,
    rhocoreg: &mut [f64],
) {
    let ffact_rhocore = nlcc::rhocore_of_g_on_shells(atompsp, pwden, volume);

    let npw_rho = pwden.get_n_plane_waves();

    let gshell_index = pwden.get_gshell_index();

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        rhocoreg[i] = ffact_rhocore[ish];
    }
}

fn drhocore_of_g_one_atom(
    atompsp: &dyn AtomPSP,
    pwden: &PWDensity,
    volume: f64,
    drhocoreg: &mut [f64],
) {
    let ffact_drhocore = nlcc::drhocore_of_g_on_shells(atompsp, pwden, volume);

    let npw_rho = pwden.get_n_plane_waves();

    let gshell_index = pwden.get_gshell_index();

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        drhocoreg[i] = ffact_drhocore[ish];
    }
}

pub fn disp_stress(stress: &Matrix<f64>) {
    for i in 0..3 {
        print!("                  | ");

        for j in 0..3 {
            print!("{:20.6} ", stress[[i, j]] * STRESS_HA_TO_KB);
        }

        println!("  |");
    }
}

pub fn display_stress_by_parts(
    s_kin: &Matrix<f64>,
    s_ha: &Matrix<f64>,
    s_xc: &Matrix<f64>,
    s_xc_nlcc: &Matrix<f64>,
    s_vpsloc: &Matrix<f64>,
    s_vnl: &Matrix<f64>,
    s_ew: &Matrix<f64>,
    s_tot: &Matrix<f64>,
) {
    println!("\n   {:-^88}", " stress (kbar) ");

    println!("     total");
    disp_stress(s_tot);

    println!("     kinetic");
    disp_stress(s_kin);

    println!("     Hartree");
    disp_stress(s_ha);

    println!("     xc");
    disp_stress(s_xc);

    println!("     xc_nlcc");
    disp_stress(s_xc_nlcc);

    println!("     local");
    disp_stress(s_vpsloc);

    println!("     non-local");
    disp_stress(s_vnl);

    println!("     Ewald");
    disp_stress(s_ew);
}

pub fn stress_to_force_on_cell(latt: &Lattice, stress: &Matrix<f64>) -> Matrix<f64> {
    let volume = latt.volume();

    let mut cell_force = Matrix::<f64>::new(3, 3);

    let mut ainv = Matrix::<f64>::new(3, 3);

    // follow the derivation in "Analytical stress tensor and pressure calculations with the CRYSTAL code", Molecular Physics, 108, 223 (2010)
    // but here we use column vectors for lattice vectors instead of row vectors in the paper

    let latt_a = latt.get_vector_a();
    let latt_b = latt.get_vector_b();
    let latt_c = latt.get_vector_c();

    ainv[[0, 0]] = latt_a.x;
    ainv[[1, 0]] = latt_a.y;
    ainv[[2, 0]] = latt_a.z;

    ainv[[0, 1]] = latt_b.x;
    ainv[[1, 1]] = latt_b.y;
    ainv[[2, 1]] = latt_b.z;

    ainv[[0, 2]] = latt_c.x;
    ainv[[1, 2]] = latt_c.y;
    ainv[[2, 2]] = latt_c.z;

    ainv.inv();

    cell_force.set_zeros();

    for l in 0..3 {
        for i in 0..3 {
            //cell_force[[l, i]] = 0.0;

            for m in 0..3 {
                cell_force[[l, i]] += stress[[l, m]] * ainv[[i, m]];
            }

            //cell_force[[l, i]] *= volume;
        }
    }

    cell_force
        .as_mut_slice()
        .iter_mut()
        .for_each(|x| *x *= volume);

    cell_force
}
