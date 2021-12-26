use dwconsts::*;
use num_traits::identities::Zero;
use types::c64;

pub fn potential(g: &[f64], rhog: &[c64], vhg: &mut [c64]) {
    vhg[0] = c64::zero();
    let n = g.len();

    for i in 1..n {
        vhg[i] = FOURPI * rhog[i] / (g[i] * g[i])
    }
}
