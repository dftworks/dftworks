mod fd;
use fd::*;
mod gs;
use gs::*;
mod mp1;
use mp1::*;
mod mp2;
use mp2::*;
mod mv;
use mv::*;
use control::SmearingScheme;

pub trait Smearing {
    fn get_occupation_number(
        &self,
        fermi_level: f64,
        temperature: f64,
        electron_energy: f64,
    ) -> f64;
}

pub fn new(smearing_scheme: SmearingScheme) -> Box<dyn Smearing> {
    match smearing_scheme {
        SmearingScheme::Fd => Box::new(SmearingFD {}),
        SmearingScheme::Gs => Box::new(SmearingGS {}),
        SmearingScheme::Mp1 => Box::new(SmearingMP1 {}),
        SmearingScheme::Mp2 => Box::new(SmearingMP2 {}),
        SmearingScheme::Mv => Box::new(SmearingMV {}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control::SmearingScheme;

    #[test]
    fn test_mv_smearing_limits() {
        let s = new(SmearingScheme::Mv);
        let f_low = s.get_occupation_number(0.0, 300.0, -10.0);
        let f_high = s.get_occupation_number(0.0, 300.0, 10.0);
        assert!(f_low > 0.999_999);
        assert!(f_high < 1.0e-6);
    }

    #[test]
    fn test_mv_smearing_alias_parse() {
        assert_eq!(SmearingScheme::parse("mv"), Some(SmearingScheme::Mv));
        assert_eq!(SmearingScheme::parse("cold"), Some(SmearingScheme::Mv));
        assert_eq!(
            SmearingScheme::parse("marzari-vanderbilt"),
            Some(SmearingScheme::Mv)
        );
    }
}
