use dwconsts::*;
use num_traits::identities::Zero;
use types::c64;

pub fn potential(g: &[f64], rhog: &[c64], vhg: &mut [c64]) {
    // Neutral-cell convention: remove the G=0 Hartree component.
    // This avoids a divergent constant shift in periodic boundary conditions.
    vhg[0] = c64::zero();
    let n = g.len();

    for i in 1..n {
        // Reciprocal-space Poisson solve:
        //   v_H(G) = 4*pi * rho(G) / |G|^2
        vhg[i] = FOURPI * rhog[i] / (g[i] * g[i])
    }
}
