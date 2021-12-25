use std::f64;
use types::c64;

// units : stress

pub const STRESS_HA_TO_GP: f64 = 29421.02648438959;
pub const STRESS_HA_TO_KB: f64 = 294210.2648438959;
pub const STRESS_KB_TO_HA: f64 = 1.0 / 294210.2648438959;

// units : force

pub const FORCE_HA_TO_EV: f64 = 51.42208619083232;
pub const FORCE_EV_TO_HA: f64 = 1.0 / 51.42208619083232;
pub const FORCE_HA_TO_RY: f64 = 2.0;

// units : length

pub const BOHR_TO_ANG: f64 = 0.529177249;
pub const ANG_TO_BOHR: f64 = 1.0 / BOHR_TO_ANG;

// units : volume

pub const BOHR3_TO_ANG3: f64 = BOHR_TO_ANG * BOHR_TO_ANG * BOHR_TO_ANG;
pub const ANG3_TO_BOHR3: f64 = ANG_TO_BOHR * ANG_TO_BOHR * ANG_TO_BOHR;

// units : energy

pub const RY_TO_EV: f64 = 13.605698066;
pub const HA_TO_EV: f64 = 2.0 * RY_TO_EV;
pub const HA_TO_RY: f64 = 2.0;
pub const EV_TO_HA: f64 = 1.0 / HA_TO_EV;
pub const RY_TO_HA: f64 = 1.0 / HA_TO_RY;

// Boltzmann constant

pub const BOLTZMANN_CONSTANT: f64 = 8.617333262145E-5 * EV_TO_HA; // Hartree K^-1

//

pub const ONE_C64: c64 = c64 { re: 1.0, im: 0.0 };
pub const I_C64: c64 = c64 { re: 0.0, im: 1.0 };

// pi

pub const PI: f64 = f64::consts::PI;
pub const TWOPI: f64 = 2.0 * f64::consts::PI;
pub const THREEPI: f64 = 3.0 * f64::consts::PI;
pub const FOURPI: f64 = 4.0 * f64::consts::PI;

// numerical convergence

pub const EPS0: f64 = 1E0;
pub const EPS1: f64 = 1E-1;
pub const EPS2: f64 = 1E-2;
pub const EPS3: f64 = 1E-3;
pub const EPS4: f64 = 1E-4;
pub const EPS5: f64 = 1E-5;
pub const EPS6: f64 = 1E-6;
pub const EPS7: f64 = 1E-7;
pub const EPS8: f64 = 1E-8;
pub const EPS9: f64 = 1E-9;
pub const EPS10: f64 = 1E-10;
pub const EPS11: f64 = 1E-11;
pub const EPS12: f64 = 1E-12;
pub const EPS13: f64 = 1E-13;
pub const EPS14: f64 = 1E-14;
pub const EPS15: f64 = 1E-15;
pub const EPS16: f64 = 1E-16;
pub const EPS17: f64 = 1E-17;
pub const EPS18: f64 = 1E-18;
pub const EPS19: f64 = 1E-19;
pub const EPS20: f64 = 1E-20;
pub const EPS21: f64 = 1E-21;
pub const EPS22: f64 = 1E-22;
pub const EPS23: f64 = 1E-23;
pub const EPS24: f64 = 1E-24;
pub const EPS25: f64 = 1E-25;
pub const EPS26: f64 = 1E-26;
pub const EPS27: f64 = 1E-27;
pub const EPS28: f64 = 1E-28;
pub const EPS29: f64 = 1E-29;
pub const EPS30: f64 = 1E-30;
pub const EPS31: f64 = 1E-31;
pub const EPS32: f64 = 1E-32;
pub const EPS33: f64 = 1E-33;
pub const EPS34: f64 = 1E-34;
pub const EPS35: f64 = 1E-35;
pub const EPS50: f64 = 1E-50;
pub const EPS80: f64 = 1E-80;

