//#![allow(warnings)]

use dwconsts;
use std::f64::consts;
use types::*;
use vector3::*;

pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

// https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions:_jn,_yn

pub fn spherical_bessel_jn(n: usize, x: f64) -> f64 {
    if x < dwconsts::EPS6 {
        match n {
            0 => {
                let x2 = x * x;
                let x4 = x2 * x2;
                let x6 = x2 * x4;
                let x8 = x2 * x6;
                let x10 = x2 * x8;
                let x12 = x2 * x10;
                let x14 = x2 * x12;
                let x16 = x2 * x14;
                let x18 = x2 * x16;

                1.0 - x2 / 6.0 + x4 / 120.0 - x6 / 5040.0 + x8 / 362880.0 - x10 / 39916800.0
                    + x12 / 6227020800.0
                    - x14 / 1307674368000.0
                    + x16 / 355687428096000.0
                    - x18 / 121645100408832000.0

                // 1 - x**2/6 + x**4/120 - x**6/5040 + x**8/362880 - x**10/39916800 + x**12/6227020800 - x**14/1307674368000 + x**16/355687428096000 - x**18/121645100408832000 + O(x**20)
            }

            1 => {
                let x2 = x * x;
                let x3 = x * x2;
                let x5 = x2 * x3;
                let x7 = x2 * x5;
                let x9 = x2 * x7;
                let x11 = x2 * x9;

                x / 3.0 - x3 / 30.0 + x5 / 840.0 - x7 / 45360.0 + x9 / 3991680.0 - x11 / 518918400.0
                // x/3 - x**3/30 + x**5/840 - x**7/45360 + x**9/3991680 - x**11/518918400 + O(x**12)
            }

            2 => {
                let x2 = x * x;
                let x4 = x2 * x2;
                let x6 = x2 * x4;
                let x8 = x2 * x6;
                let x10 = x2 * x8;

                x2 / 15.0 - x4 / 210.0 + x6 / 7560.0 - x8 / 498960.0 + x10 / 51891840.0
                // x**2/15 - x**4/210 + x**6/7560 - x**8/498960 + x**10/51891840 + O(x**12)
            }

            3 => {
                let x2 = x * x;
                let x3 = x * x2;
                let x5 = x2 * x3;
                let x7 = x2 * x5;
                let x9 = x2 * x7;
                let x11 = x2 * x9;

                x3 / 105.0 - x5 / 1890.0 + x7 / 83160.0 - x9 / 6486480.0 + x11 / 778377600.0
                // x**3/105 - x**5/1890 + x**7/83160 - x**9/6486480 + x**11/778377600 + O(x**12)
            }

            4 => {
                let x2 = x * x;
                let x4 = x2 * x2;
                let x6 = x2 * x4;
                let x8 = x4 * x4;
                let x10 = x2 * x8;

                x4 / 945.0 - x6 / 20790.0 + x8 / 1081080.0 - x10 / 97297200.0
                // x**4/945 - x**6/20790 + x**8/1081080 - x**10/97297200 + O(x**12)
            }

            _ => {
                println!("spherical bessel function for n = {} is not implemented", n);
                std::process::exit(-1);
            }
        }
    } else {
        match n {
            0 => x.sin() / x,

            1 => x.sin() / x / x - x.cos() / x,

            2 => (3.0 / x / x - 1.0) * x.sin() / x - 3.0 * x.cos() / x / x,

            3 => {
                (15.0 / x.powf(4.0) - 6.0 / x.powf(2.0)) * x.sin()
                    - (15.0 / x.powf(3.0) - 1.0 / x) * x.cos()
            }

            4 => {
                (105.0 / x.powf(5.0) - 45.0 / x.powf(3.0) + 1.0 / x) * x.sin()
                    - (105.0 / x.powf(4.0) - 10.0 / x.powf(2.0)) * x.cos()
            }

            _ => {
                panic!("spherical bessel function for n = {} is not implemented", n);
            }
        }
    }
}

// from https://en.wikipedia.org/wiki/Spherical_harmonics
// Ylm is called a spherical harmonic function of degree l and order m.
// theta: polar angle, ranges from 0 at the north pole, to pi at the south pole
// phi  : azimuth, assumes all values from 0 to 2pi
//
// (x,y,z) - (r,theta, phi)
// r     = sqrt( x*x + y*y + z*z )
// theta = arccos(z/r)
// phi   = arctan(y/x)
//
// x = r sin(theta) cos(phi)
// y = r sin(theta) sin(phi)
// z = r cos(theta)
//
pub fn spherical_harmonics(l: usize, m: i32, theta: f64, phi: f64) -> c64 {
    match (l, m) {
        (0, 0) => {
            let f = 0.5 / (consts::PI).sqrt();

            c64 { re: f, im: 0.0 }
        }

        (1, -1) => {
            let f = 0.5 * (1.5 / consts::PI).sqrt();

            c64 {
                re: f * theta.sin() * phi.cos(),
                im: -f * theta.sin() * phi.sin(),
            }
        }

        (1, 0) => {
            let f = 0.5 * (3.0 / consts::PI).sqrt();

            c64 {
                re: f * theta.cos(),
                im: 0.0,
            }
        }

        (1, 1) => {
            let f = -0.5 * (1.5 / consts::PI).sqrt();

            c64 {
                re: f * theta.sin() * phi.cos(),
                im: f * theta.sin() * phi.sin(),
            }
        }

        (2, -2) => {
            let f = 0.25 * (7.5 / consts::PI).sqrt();
            let sn = theta.sin();
            let sn2 = sn * sn;

            c64 {
                re: f * sn2 * (2.0 * phi).cos(),
                im: -f * sn2 * (2.0 * phi).sin(),
            }
        }

        (2, -1) => {
            let f = 0.5 * (7.5 / consts::PI).sqrt();
            let sn = theta.sin();
            let cs = theta.cos();

            c64 {
                re: f * sn * cs * phi.cos(),
                im: -f * sn * cs * phi.sin(),
            }
        }

        (2, 0) => {
            let f = 0.25 * (5.0 / consts::PI).sqrt();
            let cs = theta.cos();

            c64 {
                re: f * (3.0 * cs * cs - 1.0),
                im: 0.0,
            }
        }

        (2, 1) => {
            let f = -0.5 * (7.5 / consts::PI).sqrt();
            let sn = theta.sin();
            let cs = theta.cos();

            c64 {
                re: f * sn * cs * phi.cos(),
                im: f * sn * cs * phi.sin(),
            }
        }

        (2, 2) => {
            let f = 0.25 * (7.5 / consts::PI).sqrt();
            let sn = theta.sin();

            c64 {
                re: f * sn * sn * (2.0 * phi).cos(),
                im: f * sn * sn * (2.0 * phi).sin(),
            }
        }

        (3, -3) => {
            let f = 1.0 / 8.0 * (35.0 / consts::PI).sqrt();
            let sn = theta.sin();

            c64 {
                re: f * sn * sn * sn * (3.0 * phi).cos(),
                im: -f * sn * sn * sn * (3.0 * phi).sin(),
            }
        }

        (3, -2) => {
            let f = 1.0 / 4.0 * (105.0 / 2.0 / consts::PI).sqrt();
            let sn = theta.sin();
            let cs = theta.cos();

            c64 {
                re: f * sn * sn * cs * (2.0 * phi).cos(),
                im: -f * sn * sn * cs * (2.0 * phi).sin(),
            }
        }

        (3, -1) => {
            let f = 1.0 / 8.0 * (21.0 / consts::PI).sqrt();
            let sn = theta.sin();
            let cs = theta.cos();

            c64 {
                re: f * sn * (5.0 * cs * cs - 1.0) * phi.cos(),
                im: -f * sn * (5.0 * cs * cs - 1.0) * phi.sin(),
            }
        }

        (3, 0) => {
            let f = 1.0 / 4.0 * (7.0 / consts::PI).sqrt();
            let cs = theta.cos();

            c64 {
                re: f * (5.0 * cs * cs * cs - 3.0 * cs),
                im: 0.0,
            }
        }

        (3, 1) => {
            let f = -1.0 / 8.0 * (21.0 / consts::PI).sqrt();
            let cs = theta.cos();
            let sn = theta.sin();

            c64 {
                re: f * sn * (5.0 * cs * cs - 1.0) * phi.cos(),
                im: f * sn * (5.0 * cs * cs - 1.0) * phi.sin(),
            }
        }

        (3, 2) => {
            let f = 1.0 / 4.0 * (105.0 / 2.0 / consts::PI).sqrt();
            let cs = theta.cos();
            let sn = theta.sin();

            c64 {
                re: f * sn * sn * cs * (2.0 * phi).cos(),
                im: f * sn * sn * cs * (2.0 * phi).sin(),
            }
        }

        (3, 3) => {
            let f = -1.0 / 8.0 * (35.0 / consts::PI).sqrt();
            let sn = theta.sin();

            c64 {
                re: f * sn * sn * sn * (3.0 * phi).cos(),
                im: f * sn * sn * sn * (3.0 * phi).sin(),
            }
        }

        _ => panic!(),
    }
}

pub fn real_spherical_harmonics(l: usize, lm: i32, v: Vector3f64) -> f64 {
    let x = v.x;
    let y = v.y;
    let z = v.z;

    let rnorm = v.norm2();

    match (l, lm) {
        (0, 0) => 0.5 / (consts::PI).sqrt(),

        (1, -1) => -1.0 * (3.0 / 4.0 / consts::PI).sqrt() * y / rnorm,

        (1, 0) => (3.0 / 4.0 / consts::PI).sqrt() * z / rnorm,

        (1, 1) => -1.0 * (3.0 / 4.0 / consts::PI).sqrt() * x / rnorm,

        (2, -2) => 1.0 / 2.0 * (15.0 / consts::PI).sqrt() * x * y / rnorm / rnorm,

        (2, -1) => 1.0 / 2.0 * (15.0 / consts::PI).sqrt() * y * z / rnorm / rnorm,

        (2, 0) => {
            1.0 / 4.0 * (5.0 / consts::PI).sqrt() * (-x * x - y * y + 2.0 * z * z) / rnorm / rnorm
        }

        (2, 1) => 1.0 / 2.0 * (15.0 / consts::PI).sqrt() * z * x / (rnorm * rnorm),

        (2, 2) => 1.0 / 4.0 * (15.0 / consts::PI).sqrt() * (x * x - y * y) / (rnorm * rnorm),

        (3, -3) => {
            -1.0 / 4.0 * (35.0 / 2.0 / consts::PI).sqrt() * (3.0 * x * x - y * y) * y
                / (rnorm * rnorm * rnorm)
        }

        (3, -2) => 1.0 / 2.0 * (105.0 / consts::PI).sqrt() * x * y * z / (rnorm * rnorm * rnorm),

        (3, -1) => {
            -1.0 / 4.0 * (21.0 / 2.0 / consts::PI).sqrt() * y * (4.0 * z * z - x * x - y * y)
                / (rnorm * rnorm * rnorm)
        }

        (3, 0) => {
            1.0 / 4.0 * (7.0 / consts::PI).sqrt() * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
                / (rnorm * rnorm * rnorm)
        }

        (3, 1) => {
            -1.0 / 4.0 * (21.0 / 2.0 / consts::PI).sqrt() * x * (4.0 * z * z - x * x - y * y)
                / (rnorm * rnorm * rnorm)
        }

        (3, 2) => {
            1.0 / 4.0 * (105.0 / consts::PI).sqrt() * (x * x - y * y) * z / rnorm / rnorm / rnorm
        }

        (3, 3) => {
            -1.0 / 4.0 * (35.0 / 2.0 / consts::PI).sqrt() * (x * x - 3.0 * y * y) * x
                / rnorm
                / rnorm
                / rnorm
        }

        _ => panic!(),
    }
}

pub fn real_spherical_harmonics_diff(l: usize, lm: i32, v: Vector3f64) -> Vector3f64 {
    let dx = dwconsts::EPS8;

    let x = v.x;
    let y = v.y;
    let z = v.z;

    let x2 = real_spherical_harmonics(l, lm, Vector3f64 { x: x + dx, y, z });
    let x1 = real_spherical_harmonics(l, lm, Vector3f64 { x: x - dx, y, z });

    let y2 = real_spherical_harmonics(l, lm, Vector3f64 { x, y: y + dx, z });
    let y1 = real_spherical_harmonics(l, lm, Vector3f64 { x, y: y - dx, z });

    let z2 = real_spherical_harmonics(l, lm, Vector3f64 { x, y, z: z + dx });
    let z1 = real_spherical_harmonics(l, lm, Vector3f64 { x, y, z: z - dx });

    Vector3f64 {
        x: (x2 - x1) / dx / 2.0,
        y: (y2 - y1) / dx / 2.0,
        z: (z2 - z1) / dx / 2.0,
    }
}

#[test]
fn test_spherical_harmonics() {
    let theta = consts::PI / 180.0 * 11.0;
    let phi = consts::PI / 180.0 * 22.0;

    for l in 0..4 {
        for pm in 0..2 * l + 1 {
            let m = pm as i32 - l as i32;
            println!(
                "({:1},{:2}) Ylm = {:+20.12}",
                l,
                m,
                spherical_harmonics(l, m, theta, phi)
            );
        }
    }
}

#[test]
fn test_spherical_bessel_jn() {
    let x = dwconsts::EPS3 * 1.0;

    for n in 0..5 {
        let y = spherical_bessel_jn(n, x);
        println!("n = {}\t x = {}\t y = {:.30E}", n, x, y);
    }
}
