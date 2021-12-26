use dwconsts::*;
use num_traits::Zero;
use types::c64;
use vector3::*;

pub fn compute_structure_factor(
    miller: &[Vector3i32],
    gindex: &[usize],
    atom_positions: &[Vector3f64],
) -> Vec<c64> {
    let nsize = gindex.len();

    let mut sfact = vec![c64::zero(); nsize];

    for (i, ig) in gindex.iter().enumerate() {
        let g = miller[*ig];

        let mut t = c64::zero();

        for at in atom_positions {
            let gr = utility::dot_product_v3i32_v3f64(g, *at);

            t += (-I_C64 * TWOPI * gr).exp();
        }

        sfact[i] = t;
    }

    sfact
}

pub fn compute_structure_factor_for_many_g_one_atom(
    miller: &[Vector3i32],
    gindex: &[usize],
    atom_position: Vector3f64,
) -> Vec<c64> {
    let nsize = gindex.len();

    let mut sfact = vec![c64::zero(); nsize];

    for (i, ig) in gindex.iter().enumerate() {
        let g = miller[*ig];

        let gr = utility::dot_product_v3i32_v3f64(g, atom_position);

        sfact[i] = (-I_C64 * TWOPI * gr).exp();
    }

    sfact
}
