#![allow(warnings)]

use control::Control;
use crystal::Crystal;
use lattice::Lattice;
use types::Matrix;
use types::MatrixF64Ext;
use nalgebra::Matrix3;
use symmetry::SymmetryDriver;

fn array_to_matrix3(m: &[[f64; 3]; 3]) -> Matrix3<f64> {
    let mut out = Matrix3::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = m[i][j];
        }
    }
    out
}

fn matrix3_to_matrix(m3: &Matrix3<f64>, m: &mut Matrix<f64>) {
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = m3[(i, j)];
        }
    }
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

fn matrix_to_matrix3(m: &Matrix<f64>) -> Matrix3<f64> {
    let mut out = Matrix3::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = m[(i, j)];
        }
    }
    out
}

pub fn project_stress_by_symmetry(
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    stress: &mut Matrix<f64>,
) {
    let n_sym = symdrv.get_n_sym_ops();
    if n_sym == 0 {
        return;
    }

    let latt = crystal.get_latt();
    let stress_raw = matrix_to_matrix3(stress);
    let mut stress_proj = Matrix3::<f64>::zeros();

    for isym in 0..n_sym {
        let rot_cart = array_to_matrix3(&rotation_cart_from_fractional(
            latt,
            symdrv.get_rotation(isym),
        ));
        stress_proj += rot_cart * stress_raw * rot_cart.transpose();
    }

    let inv = 1.0 / n_sym as f64;
    stress_proj *= inv;

    matrix3_to_matrix(&stress_proj, stress);
    stress.symmetrize();
}

pub fn finalize_stress_by_parts(
    control: &Control,
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    stress_total: &mut Matrix<f64>,
    stress_kin: &mut Matrix<f64>,
    stress_hartree: &mut Matrix<f64>,
    stress_xc: &mut Matrix<f64>,
    stress_xc_nlcc: &mut Matrix<f64>,
    stress_loc: &mut Matrix<f64>,
    stress_vnl: &mut Matrix<f64>,
    stress_ewald: &mut Matrix<f64>,
    stress_vdw: &mut Matrix<f64>,
) {
    if control.get_symmetry() {
        project_stress_by_symmetry(crystal, symdrv, stress_kin);
        project_stress_by_symmetry(crystal, symdrv, stress_hartree);
        project_stress_by_symmetry(crystal, symdrv, stress_xc);
        project_stress_by_symmetry(crystal, symdrv, stress_xc_nlcc);
        project_stress_by_symmetry(crystal, symdrv, stress_loc);
        project_stress_by_symmetry(crystal, symdrv, stress_vnl);
        project_stress_by_symmetry(crystal, symdrv, stress_ewald);
        project_stress_by_symmetry(crystal, symdrv, stress_vdw);
    }

    for i in 0..3 {
        for j in 0..3 {
            stress_total[(i, j)] = stress_kin[(i, j)]
                + stress_hartree[(i, j)]
                + stress_xc[(i, j)]
                + stress_xc_nlcc[(i, j)]
                + stress_loc[(i, j)]
                + stress_vnl[(i, j)]
                + stress_ewald[(i, j)]
                + stress_vdw[(i, j)];
        }
    }

    if dwmpi::is_root() {
        crate::display_stress_by_parts(
            stress_kin,
            stress_hartree,
            stress_xc,
            stress_xc_nlcc,
            stress_loc,
            stress_vnl,
            stress_ewald,
            stress_vdw,
            stress_total,
        );
    }
}
