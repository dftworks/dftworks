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
use types::*;
use ndarray::*;
use nalgebra::Matrix3;
use num_traits::identities::Zero;
use pspot::*;
use pwbasis::*;
use pwdensity::*;
use types::*;
use std::collections::HashMap;

pub mod symmetry;

pub struct SpectralWorkspace {
    npw_rho: usize,
    volume: f64,
    atom_species_stamp: Vec<String>,
    vpsloc_by_species: HashMap<String, Vec<f64>>,
    dvpsloc_by_species: HashMap<String, Vec<f64>>,
    rhocore_by_species: HashMap<String, Vec<f64>>,
    drhocore_by_species: HashMap<String, Vec<f64>>,
}

impl SpectralWorkspace {
    pub fn new() -> Self {
        Self {
            npw_rho: 0,
            volume: f64::NAN,
            atom_species_stamp: Vec::new(),
            vpsloc_by_species: HashMap::new(),
            dvpsloc_by_species: HashMap::new(),
            rhocore_by_species: HashMap::new(),
            drhocore_by_species: HashMap::new(),
        }
    }

    fn prepare(&mut self, atpsps: &PSPot, crystal: &Crystal, pwden: &PWDensity) {
        let volume = crystal.get_latt().volume();
        let npw_rho = pwden.get_n_plane_waves();
        let atom_species = crystal.get_atom_species();

        let needs_refresh = self.npw_rho != npw_rho
            || (self.volume - volume).abs() > 1.0e-12
            || self.atom_species_stamp.len() != atom_species.len()
            || !self
                .atom_species_stamp
                .iter()
                .zip(atom_species.iter())
                .all(|(lhs, rhs)| lhs == rhs);
        if !needs_refresh {
            return;
        }

        self.vpsloc_by_species.clear();
        self.dvpsloc_by_species.clear();
        self.rhocore_by_species.clear();
        self.drhocore_by_species.clear();

        let mut unique_species: Vec<String> = Vec::new();
        for sp in atom_species.iter() {
            if unique_species.iter().any(|seen| seen == sp) {
                continue;
            }
            unique_species.push(sp.clone());
        }

        for sp in unique_species.iter() {
            let atpsp = atpsps.get_psp(sp);

            let mut vpslocg = vec![0.0; npw_rho];
            vpsloc_of_g_one_atom(atpsp, pwden, volume, &mut vpslocg);
            self.vpsloc_by_species.insert(sp.clone(), vpslocg);

            let mut dvpslocg = vec![0.0; npw_rho];
            dvpsloc_of_g_one_atom(atpsp, pwden, volume, &mut dvpslocg);
            self.dvpsloc_by_species.insert(sp.clone(), dvpslocg);

            let mut rhocoreg = vec![0.0; npw_rho];
            rhocore_of_g_one_atom(atpsp, pwden, volume, &mut rhocoreg);
            self.rhocore_by_species.insert(sp.clone(), rhocoreg);

            let mut drhocoreg = vec![0.0; npw_rho];
            drhocore_of_g_one_atom(atpsp, pwden, volume, &mut drhocoreg);
            self.drhocore_by_species.insert(sp.clone(), drhocoreg);
        }

        self.npw_rho = npw_rho;
        self.volume = volume;
        self.atom_species_stamp = atom_species.to_vec();
    }
}

// Stress decomposition utilities.
//
// Similar to forces, total stress is assembled from multiple physically
// distinct pieces (kinetic, Hartree, XC, local, non-local, Ewald, NLCC).
// Keeping pieces separate makes convergence/debug output interpretable.

// Phys. Rev. B 41, 7878 (1990)
pub fn nlcc_xc(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    vxcg: &Vec<c64>,
) -> Matrix<f64> {
    let mut workspace = SpectralWorkspace::new();
    let mut out = Matrix::<f64>::new(3, 3);
    nlcc_xc_with_workspace(atpsps, crystal, gvec, pwden, &mut workspace, vxcg, &mut out);
    out
}

pub fn nlcc_xc_with_workspace(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    workspace: &mut SpectralWorkspace,
    vxcg: &[c64],
    out: &mut Matrix<f64>,
) {
    // NLCC stress correction from core-charge response to XC potential.
    let mut stress_c64 = [[c64 { re: 0.0, im: 0.0 }; 3]; 3];

    let volume = crystal.get_latt().volume();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let gidx = pwden.get_gindex();

    workspace.prepare(atpsps, crystal, pwden);
    let npw_rho = workspace.npw_rho;

    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();
    let species = crystal.get_atom_species();

    for iat in 0..natoms {
        let atom = atom_positions[iat];

        let rhocoreg = workspace
            .rhocore_by_species
            .get(&species[iat])
            .unwrap_or_else(|| panic!("missing NLCC spectral cache for species '{}'", species[iat]));
        let drhocoreg = workspace
            .drhocore_by_species
            .get(&species[iat])
            .unwrap_or_else(|| panic!("missing NLCC derivative spectral cache for species '{}'", species[iat]));

        for ipw in 0..npw_rho {
            let mill = miller[gidx[ipw]];

            //let ngd =
            //    -TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let ngd = -TWOPI * utility::dot_product_v3i32_v3f64(mill, atom);

            let comm = c64 {
                re: ngd.cos(),
                im: ngd.sin(),
            } * vxcg[ipw].conj();
            let gcoord = cart[gidx[ipw]];

            for i in 0..3 {
                for j in 0..3 {
                    // Isotropic + anisotropic pieces from radial derivative term.
                    let unit = if i == j { 1.0 } else { 0.0 };
                    let gi = match i {
                        0 => gcoord.x,
                        1 => gcoord.y,
                        _ => gcoord.z,
                    };
                    let gj = match j {
                        0 => gcoord.x,
                        1 => gcoord.y,
                        _ => gcoord.z,
                    };
                    stress_c64[j][i] +=
                        comm * (rhocoreg[ipw] * unit + drhocoreg[ipw] * 2.0 * gj * gi);
                }
            }
        }
    }

    out.set_zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = stress_c64[i][j].re;
        }
    }
}

pub fn vpsloc(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    rhog: &[c64],
) -> Matrix<f64> {
    let mut workspace = SpectralWorkspace::new();
    let mut out = Matrix::<f64>::new(3, 3);
    vpsloc_with_workspace(
        atpsps,
        crystal,
        gvec,
        pwden,
        &mut workspace,
        rhog,
        &mut out,
    );
    out
}

pub fn vpsloc_with_workspace(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    workspace: &mut SpectralWorkspace,
    rhog: &[c64],
    out: &mut Matrix<f64>,
) {
    // Local ionic stress from reciprocal-space local potential form factor.
    let mut stress_c64 = [[c64 { re: 0.0, im: 0.0 }; 3]; 3];

    let volume = crystal.get_latt().volume();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let gidx = pwden.get_gindex();

    workspace.prepare(atpsps, crystal, pwden);
    let npw_rho = workspace.npw_rho;

    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();
    let species = crystal.get_atom_species();

    for iat in 0..natoms {
        let atom = atom_positions[iat];

        let vatlocg = workspace
            .vpsloc_by_species
            .get(&species[iat])
            .unwrap_or_else(|| panic!("missing local-potential spectral cache for species '{}'", species[iat]));
        let dvatlocg = workspace
            .dvpsloc_by_species
            .get(&species[iat])
            .unwrap_or_else(|| panic!("missing local-potential derivative spectral cache for species '{}'", species[iat]));

        for ipw in 0..npw_rho {
            let mill = miller[gidx[ipw]];

            //let ngd =
            //    -TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let ngd = -TWOPI * utility::dot_product_v3i32_v3f64(mill, atom);

            let comm = c64 {
                re: ngd.cos(),
                im: ngd.sin(),
            } * rhog[ipw].conj();
            let gcoord = cart[gidx[ipw]];

            for i in 0..3 {
                for j in 0..3 {
                    // Isotropic + anisotropic pieces from radial derivative term.
                    let unit = if i == j { 1.0 } else { 0.0 };
                    let gi = match i {
                        0 => gcoord.x,
                        1 => gcoord.y,
                        _ => gcoord.z,
                    };
                    let gj = match j {
                        0 => gcoord.x,
                        1 => gcoord.y,
                        _ => gcoord.z,
                    };
                    stress_c64[j][i] +=
                        comm * (vatlocg[ipw] * unit + dvatlocg[ipw] * 2.0 * gj * gi);
                }
            }
        }
    }

    out.set_zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = stress_c64[i][j].re;
        }
    }
}

pub fn xc(
    latt: &Lattice,
    rho_3d: &mut Array3<c64>,
    rhocore_3d: &Array3<c64>,
    vxc_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
) -> Matrix<f64> {
    // In this implementation XC stress uses diagonal thermodynamic form:
    // sigma_xc,ii = -(E_xc - \int rho v_xc)/Omega
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
    // Spin-resolved analogue of diagonal XC stress.
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
    // Reciprocal-space Hartree stress, excluding G=0 term.
    let cart = gvec.get_cart();

    let gnorm = pwden.get_g();
    let gidx = pwden.get_gindex();
    let npw = pwden.get_n_plane_waves();

    let unit_mat = Matrix::<f64>::identity(3);

    let mut stress = Matrix::<f64>::new(3, 3);

    for ig in 1..npw {
        let g = cart[gidx[ig]].as_slice().to_vec();

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
    // Kinetic stress from occupied states and plane-wave coefficients.
    let cart = vkscf[0].get_gvec().get_cart();

    let mut stress = Matrix::<f64>::new(3, 3);

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evecs = &vevecs[ik];

        let gidx = kscf.get_pwwfc().get_gindex();
        let npw = kscf.get_pwwfc().get_n_plane_waves();
        let xk = kscf.get_pwwfc().get_k_cart().as_slice().to_vec();
        let nbnd = kscf.get_nbands();
        let occ = kscf.get_occ();

        for ibnd in 0..nbnd {
            if occ[ibnd] < EPS20 {
                continue;
            }

            let cnk = evecs.get_col(ibnd);

            for ikg in 0..npw {
                let g = cart[gidx[ikg]].as_slice().to_vec();

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
    let xk = pwwfc.get_k_cart().as_slice().to_vec();

    let kg = pwwfc.get_kg();

    let miller = gvec.get_miller();

    let cart = gvec.get_cart();

    let gidx = pwwfc.get_gindex();

    let nbeta = atpsp.get_nbeta();

    let npw = pwwfc.get_n_plane_waves();

    let occ = kscf.get_occ();

    let unit_mat = Matrix::<f64>::identity(3);
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
                        let p_dylm = dylm[ipw].as_slice().to_vec();

                        let g = cart[gidx[ipw]].as_slice().to_vec();

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

fn matrix_to_matrix3(m: &Matrix<f64>) -> Matrix3<f64> {
    let mut out = Matrix3::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = m[[i, j]];
        }
    }
    out
}

fn matrix3_to_matrix(m3: &Matrix3<f64>, out: &mut Matrix<f64>) {
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = m3[(i, j)];
        }
    }
}

fn frac_to_cart(latt: &Lattice, mf: &Matrix<f64>, mc: &mut Matrix<f64>) {
    let lat_t = matrix_to_matrix3(&latt.as_matrix().transpose());
    let m_frac = matrix_to_matrix3(mf);
    let m_cart = lat_t * m_frac * lat_t.transpose();
    matrix3_to_matrix(&m_cart, mc);
}

fn cart_to_frac(latt: &Lattice, mc: &Matrix<f64>, mf: &mut Matrix<f64>) {
    let lat_t = matrix_to_matrix3(&latt.as_matrix().transpose());
    let lat_t_inv = lat_t
        .try_inverse()
        .expect("lattice matrix transpose is singular in cart_to_frac");
    let m_cart = matrix_to_matrix3(mc);
    let m_frac = lat_t_inv * m_cart * lat_t_inv.transpose();
    matrix3_to_matrix(&m_frac, mf);
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
    s_vdw: &Matrix<f64>,
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

    println!("     vdW");
    disp_stress(s_vdw);
}

pub fn vdw(
    control: &control::Control,
    crystal: &Crystal,
    stress_vdw: &mut Matrix<f64>,
) {
    use vdw::{VdwCorrection, VdwMethod, VdwDamping};

    stress_vdw.set_zeros();

    if !control.get_vdw_correction() {
        return;
    }

    let damping = match control.get_vdw_damping() {
        "zero" => VdwDamping::Zero,
        "bj" => VdwDamping::BJ,
        _ => VdwDamping::BJ,
    };

    let xc_scheme = control.get_xc_scheme();
    let vdw_calc = VdwCorrection::new(VdwMethod::D3, damping, xc_scheme);

    match vdw_calc.stress(crystal) {
        Ok(stress_array) => {
            for i in 0..3 {
                for j in 0..3 {
                    stress_vdw[[i, j]] = stress_array[i][j];
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: vdW stress calculation failed: {}", e);
        }
    }
}

pub fn stress_to_force_on_cell(latt: &Lattice, stress: &Matrix<f64>) -> Matrix<f64> {
    let volume = latt.volume();
    let a_inv = matrix_to_matrix3(latt.as_matrix())
        .try_inverse()
        .expect("lattice matrix is singular in stress_to_force_on_cell");
    let stress_mat = matrix_to_matrix3(stress);
    let cell_force_mat = (stress_mat * a_inv.transpose()) * volume;

    let mut cell_force = Matrix::<f64>::new(3, 3);
    matrix3_to_matrix(&cell_force_mat, &mut cell_force);
    cell_force
}
