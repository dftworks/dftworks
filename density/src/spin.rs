use crate::Density;
use atompsp::AtomPSP;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use fhkl;
use gvector::GVector;
use itertools::multizip;
use magmom::*;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::*;
use num_traits::Zero;
use pspot::PSPot;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::*;
use vector3::*;

pub struct DensitySpin {
    // Per-species starting magnetization used to split initial atomic density
    // into up/down channels.
    starting_mag: MagMoment,
}

impl DensitySpin {
    pub fn new() -> DensitySpin {
        let mut magmom = MagMoment::new();
        magmom.read_file("in.magmom");
        DensitySpin {
            starting_mag: magmom,
        }
    }
}

impl Density for DensitySpin {
    fn from_atomic_super_position(
        &self,
        pspot: &PSPot,
        crystal: &Crystal,
        rgtrans: &RGTransform,
        gvec: &GVector,
        pwden: &PWDensity,
        rhog: &mut RHOG,
        rho_3d: &mut RHOR,
    ) {
        let (rhog_up, rhog_dn) = rhog.as_spin_mut().unwrap();
        let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin_mut().unwrap();

        // Build spin-resolved rho(G) from atomic superposition plus
        // species-dependent initial magnetic moments.

        self.atomic_super_position(pspot, crystal, pwden, gvec, rhog_up, rhog_dn);

        // Transform reciprocal densities to real-space mesh.

        rgtrans.g1d_to_r3d(gvec, pwden, rhog_up, rho_3d_up.as_mut_slice());
        rgtrans.g1d_to_r3d(gvec, pwden, rhog_dn, rho_3d_dn.as_mut_slice());
    }

    fn get_change_in_density(&self, rhog: &[c64], rhog_new: &[c64]) -> f64 {
        // L2-norm squared in reciprocal space.
        let mut delta_rho = 0.0;

        for (&x, &y) in multizip((rhog.iter(), rhog_new.iter())) {
            delta_rho += (x - y).norm_sqr();
        }

        delta_rho
        // delta_rho.sqrt()
    }

    fn compute_charge_density(
        &self,
        vkscf: &VKSCF,
        rgtrans: &RGTransform,
        vkevecs: &VKEigenVector,
        volume: f64,
        rho_3d: &mut RHOR,
    ) {
        let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin_mut().unwrap();

        // Reset accumulators each SCF iteration.
        rho_3d_up.set_value(c64::zero());
        rho_3d_dn.set_value(c64::zero());

        let [n1, n2, n3] = rho_3d_up.shape();

        let (vkscf_up, vkscf_dn) = vkscf.as_spin().unwrap();
        let (vkevecs_up, vkevecs_dn) = vkevecs.as_spin().unwrap();

        let mut rho_3d_up_local = Array3::<c64>::new([n1, n2, n3]);
        let mut rho_3d_dn_local = Array3::<c64>::new([n1, n2, n3]);
        rho_3d_up_local.set_value(c64::zero());
        rho_3d_dn_local.set_value(c64::zero());

        let mut unk = Array3::<c64>::new([n1, n2, n3]);

        // FFT workspace reused while converting band coefficients to u_nk(r).

        let mut fft_work = Array3::<c64>::new([n1, n2, n3]);

        // Spin-up channel:
        // rho_up(r) = sum_{n,k} f_up(n,k) w_k |u_up(n,k,r)|^2

        for (ik, kscf) in vkscf_up.iter().enumerate() {
            let occ = kscf.get_occ();

            let nev = kscf.get_nbands();

            for ib in 0..nev {
                if occ[ib] > EPS20 {
                    // Plane-wave coefficients -> u_nk(r).

                    kscf.get_unk(
                        rgtrans,
                        &vkevecs_up[ik],
                        volume,
                        ib,
                        &mut unk,
                        &mut fft_work,
                    );

                    // Add weighted orbital density.

                    let factor = occ[ib] * kscf.get_k_weight();

                    for (y, x) in multizip((
                        rho_3d_up_local.as_mut_slice().iter_mut(),
                        unk.as_slice().iter(),
                    )) {
                        *y += x.norm_sqr() * factor;
                    }
                } else {
                    // Occupations are ordered; remaining bands are empty.
                    break;
                }
            }
        }

        // Spin-down channel:
        // rho_dn(r) = sum_{n,k} f_dn(n,k) w_k |u_dn(n,k,r)|^2

        for (ik, kscf) in vkscf_dn.iter().enumerate() {
            let occ = kscf.get_occ();

            let nev = kscf.get_nbands();

            for ib in 0..nev {
                if occ[ib] > EPS20 {
                    // Plane-wave coefficients -> u_nk(r).

                    kscf.get_unk(
                        rgtrans,
                        &vkevecs_dn[ik],
                        volume,
                        ib,
                        &mut unk,
                        &mut fft_work,
                    );

                    // Add weighted orbital density.

                    let factor = occ[ib] * kscf.get_k_weight();

                    for (y, x) in multizip((
                        rho_3d_dn_local.as_mut_slice().iter_mut(),
                        unk.as_slice().iter(),
                    )) {
                        *y += x.norm_sqr() * factor;
                    }
                } else {
                    // Occupations are ordered; remaining bands are empty.
                    break;
                }
            }
        }

        // MPI reduction + broadcast so all ranks share identical rho_up/rho_dn.
        dwmpi::reduce_slice_sum(
            rho_3d_up_local.as_slice(),
            rho_3d_up.as_mut_slice(),
            MPI_COMM_WORLD,
        );
        dwmpi::reduce_slice_sum(
            rho_3d_dn_local.as_slice(),
            rho_3d_dn.as_mut_slice(),
            MPI_COMM_WORLD,
        );

        dwmpi::bcast_slice(rho_3d_up.as_mut_slice(), MPI_COMM_WORLD);
        dwmpi::bcast_slice(rho_3d_dn.as_mut_slice(), MPI_COMM_WORLD);
    }
}

impl DensitySpin {
    fn atom_super_pos_one_specie(
        &self,
        atompsp: &dyn AtomPSP,
        atom_positions: &[Vector3f64],
        pwden: &PWDensity,
        gvec: &GVector,
        volume: f64,
    ) -> Vec<c64> {
        let miller = gvec.get_miller();

        let gindex = pwden.get_gindex();

        let gshell_index = pwden.get_gshell_index();

        let npw_rho = pwden.get_n_plane_waves();

        // Structure factor for atomic positions of this species.

        let sfact = fhkl::compute_structure_factor(miller, gindex, atom_positions);

        // Radial atomic form factor sampled on |G| shells.

        let ffact_rho = rho_of_g_on_shells(atompsp, &pwden, volume);

        // Combine radial form factor and structure factor.

        let mut rhog = vec![c64::zero(); npw_rho];

        for i in 0..npw_rho {
            let ish = gshell_index[i];

            rhog[i] = ffact_rho[ish] * sfact[i];
        }

        rhog
    }

    fn atomic_super_position(
        &self,
        atpsps: &PSPot,
        crystal: &Crystal,
        pwden: &PWDensity,
        gvec: &GVector,
        rhog_up: &mut [c64],
        rhog_dn: &mut [c64],
    ) {
        let volume = crystal.get_latt().volume();

        let species = crystal.get_unique_species();

        let npw_rho = pwden.get_n_plane_waves();

        let mag = self.starting_mag.get_starting_moment();

        // Build species contribution then split into up/down using
        // initial magnetization m:
        //   rho_up = rho * (1 + m) / 2
        //   rho_dn = rho * (1 - m) / 2
        for (isp, sp) in species.iter().enumerate() {
            let atpsp = atpsps.get_psp(sp);

            let atom_positions = crystal.get_atom_positions_of_specie(isp);

            let rhog_one =
                self.atom_super_pos_one_specie(atpsp, &atom_positions, pwden, gvec, volume);

            // spin up

            for i in 0..npw_rho {
                rhog_up[i] += rhog_one[i] * (1.0 + mag[isp]) / 2.0;
            }

            // spin down

            for i in 0..npw_rho {
                rhog_dn[i] += rhog_one[i] * (1.0 - mag[isp]) / 2.0;
            }
        }
    }
}

fn rho_of_g_on_shells(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64) -> Vec<f64> {
    let gshell = pwden.get_gshell_norms();
    let atrho = atompsp.get_rho();
    let rad = atompsp.get_rad();
    let rab = atompsp.get_rab();

    compute_rho_of_g(atrho, rad, rab, gshell, volume)
}

//https://blog.cupcakephysics.com/electromagnetism/math%20methods/2014/10/04/the-fourier-transform-of-the-coulomb-potential.html

// 3d integration: \frac{1}{\Omega} \int_V \rho_{at}^\tau (\vec{r}) e^{-i G \cdot r} dr
// 1d radial integration: \frac{1}{\Omega} \int_0^\infty \rho_{at}^\tau(r) \frac{sin(Gr)}{Gr} dr

fn compute_rho_of_g(
    rho: &[f64],
    rad: &[f64],
    rab: &[f64],
    gshell: &[f64],
    volume: f64,
) -> Vec<f64> {
    let nshell = gshell.len();

    let mut rhog = vec![0.0; nshell];

    let mmax = rho.len();

    let mut work = vec![0.0; mmax];

    // G = 0

    rhog[0] = integral::simpson_rab(rho, rab);

    // G > 0

    for iw in 1..nshell {
        for i in 0..mmax {
            if rad[i] < EPS8 {
                work[i] = rho[i];
            } else {
                let gr = gshell[iw] * rad[i];
                work[i] = rho[i] * gr.sin() / gr;
            }
        }

        rhog[iw] = integral::simpson_rab(&work, rab);
    }

    for v in rhog.iter_mut() {
        *v /= volume;
    }

    rhog
}
