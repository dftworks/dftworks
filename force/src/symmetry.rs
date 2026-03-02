#![allow(warnings)]

use control::Control;
use crystal::Crystal;
use lattice::Lattice;
use symmetry::SymmetryDriver;
use types::Vector3f64;

fn transpose_i32_3x3(m: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut out = [[0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    out
}

fn matvec_3x3(a: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2];
    }
    out
}

fn rotation_cart_from_fractional(latt: &Lattice, rotation_frac: &[[i32; 3]; 3]) -> [[f64; 3]; 3] {
    let mut rot_cart = [[0.0; 3]; 3];

    for col in 0..3 {
        let mut e_cart = [0.0; 3];
        e_cart[col] = 1.0;

        let mut e_frac = [0.0; 3];
        latt.cart_to_frac(&e_cart, &mut e_frac);

        let mut e_frac_rot = [0.0; 3];
        for i in 0..3 {
            e_frac_rot[i] = rotation_frac[i][0] as f64 * e_frac[0]
                + rotation_frac[i][1] as f64 * e_frac[1]
                + rotation_frac[i][2] as f64 * e_frac[2];
        }

        let mut e_cart_rot = [0.0; 3];
        latt.frac_to_cart(&e_frac_rot, &mut e_cart_rot);

        for row in 0..3 {
            rot_cart[row][col] = e_cart_rot[row];
        }
    }

    rot_cart
}

pub fn project_force_by_symmetry(
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    force: &mut [Vector3f64],
) {
    let natoms = force.len();
    let n_sym = symdrv.get_n_sym_ops();
    if natoms == 0 || n_sym == 0 {
        return;
    }

    let sym_atom = symdrv.get_sym_atom();
    if sym_atom.len() != natoms || sym_atom.iter().any(|row| row.len() != n_sym) {
        return;
    }

    let latt = crystal.get_latt();
    let mut rot_inv_cart = Vec::<[[f64; 3]; 3]>::with_capacity(n_sym);
    for isym in 0..n_sym {
        let rot_inv_frac = transpose_i32_3x3(symdrv.get_rotation(isym));
        rot_inv_cart.push(rotation_cart_from_fractional(latt, &rot_inv_frac));
    }

    let force_raw = force.to_vec();
    for iat in 0..natoms {
        let mut sum = [0.0; 3];
        let mut n_accum = 0usize;

        for isym in 0..n_sym {
            let jat = sym_atom[iat][isym];
            if jat >= natoms {
                continue;
            }

            let fj = [force_raw[jat].x, force_raw[jat].y, force_raw[jat].z];
            let fi = matvec_3x3(&rot_inv_cart[isym], &fj);

            sum[0] += fi[0];
            sum[1] += fi[1];
            sum[2] += fi[2];
            n_accum += 1;
        }

        if n_accum > 0 {
            let inv = 1.0 / n_accum as f64;
            force[iat].x = sum[0] * inv;
            force[iat].y = sum[1] * inv;
            force[iat].z = sum[2] * inv;
        }
    }
}

pub fn finalize_force_by_parts(
    control: &Control,
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    force_total: &mut [Vector3f64],
    force_ewald: &mut [Vector3f64],
    force_loc: &mut [Vector3f64],
    force_vnl: &mut [Vector3f64],
    force_nlcc: &mut [Vector3f64],
    force_vdw: &mut [Vector3f64],
) {
    if control.get_symmetry() {
        project_force_by_symmetry(crystal, symdrv, force_loc);
        project_force_by_symmetry(crystal, symdrv, force_vnl);
        project_force_by_symmetry(crystal, symdrv, force_nlcc);
        project_force_by_symmetry(crystal, symdrv, force_ewald);
        project_force_by_symmetry(crystal, symdrv, force_vdw);
    }

    for iat in 0..force_total.len() {
        force_total[iat] = force_ewald[iat] + force_loc[iat] + force_vnl[iat] + force_nlcc[iat] + force_vdw[iat];
    }

    if dwmpi::is_root() {
        crate::display(
            crystal,
            force_total,
            force_ewald,
            force_loc,
            force_vnl,
            force_nlcc,
            force_vdw,
        );
    }
}
