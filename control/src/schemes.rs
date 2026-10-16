// Enumerated in.ctrl values.
//
// Each enum is declared with `scheme_enum!`, one line per variant:
//
//     Variant => "canonical" | "alias" | ...,
//
// The canonical spelling is what `as_str` returns and what the startup
// display prints; aliases are accepted on input only. `parse`, `ALL` and the
// "expected one of" message are all generated from the same list, so they
// cannot drift apart. Matching is case-insensitive and ignores surrounding
// whitespace.

/// Common interface of the generated enums, used by `params::parse_scheme`.
pub trait SchemeEnum: Sized + Copy + 'static {
    /// Every variant, in declaration order.
    const ALL: &'static [Self];

    fn as_str(self) -> &'static str;

    fn parse(s: &str) -> Option<Self>;

    /// Canonical spellings, plus aliases when there are any, for error messages.
    fn accepted_values() -> String;
}

macro_rules! scheme_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $( $variant:ident => $canonical:literal $(| $alias:literal)* ),+ $(,)?
        }
        default = $default:ident;
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $( $variant, )+
        }

        impl Default for $name {
            fn default() -> Self {
                $name::$default
            }
        }

        impl $name {
            pub const ALL: &'static [$name] = &[$( $name::$variant ),+];

            // (spelling, variant, is_canonical)
            const NAMES: &'static [(&'static str, $name, bool)] = &[
                $(
                    ($canonical, $name::$variant, true),
                    $( ($alias, $name::$variant, false), )*
                )+
            ];

            pub fn as_str(self) -> &'static str {
                match self {
                    $( $name::$variant => $canonical, )+
                }
            }

            pub fn parse(s: &str) -> Option<Self> {
                let s = s.trim().to_lowercase();
                Self::NAMES
                    .iter()
                    .find(|(name, _, _)| *name == s)
                    .map(|(_, variant, _)| *variant)
            }

            pub fn accepted_values() -> String {
                let pick = |canonical: bool| {
                    Self::NAMES
                        .iter()
                        .filter(|(_, _, is_canonical)| *is_canonical == canonical)
                        .map(|(name, _, _)| *name)
                        .collect::<Vec<_>>()
                };
                let names = pick(true).join(", ");
                let aliases = pick(false);
                match aliases.len() {
                    0 => names,
                    1 => format!("{} (alias: {})", names, aliases[0]),
                    _ => format!("{} (aliases: {})", names, aliases.join(", ")),
                }
            }
        }

        // Prints the canonical spelling; `pad` keeps width/alignment specifiers working.
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.pad(self.as_str())
            }
        }

        impl SchemeEnum for $name {
            const ALL: &'static [Self] = $name::ALL;

            fn as_str(self) -> &'static str {
                $name::as_str(self)
            }

            fn parse(s: &str) -> Option<Self> {
                $name::parse(s)
            }

            fn accepted_values() -> String {
                $name::accepted_values()
            }
        }
    };
}

scheme_enum! {
    /// Run mode. Both modes currently run the same SCF path; band-structure
    /// runs are driven by `kpts_scheme = kline` with an `in.kline` file.
    pub enum Task {
        Scf => "scf",
        Band => "band",
    }
    default = Scf;
}

scheme_enum! {
    pub enum SpinScheme {
        NonSpin => "nonspin",
        Spin => "spin",
        Ncl => "ncl",
    }
    default = NonSpin;
}

scheme_enum! {
    #[derive(PartialOrd, Ord)]
    pub enum VerbosityLevel {
        Quiet => "quiet",
        Normal => "normal" | "medium",
        Verbose => "verbose" | "high",
        Debug => "debug",
    }
    default = Normal;
}

scheme_enum! {
    pub enum XcScheme {
        LdaPz => "lda-pz",
        LsdaPz => "lsda-pz",
        Pbe => "pbe",
        Hse06 => "hse06",
    }
    default = LdaPz;
}

scheme_enum! {
    pub enum SmearingScheme {
        Fd => "fd",
        Gs => "gs",
        Mp1 => "mp1",
        Mp2 => "mp2",
        Mv => "mv" | "cold" | "marzari-vanderbilt" | "marzari_vanderbilt",
    }
    default = Mp2;
}

scheme_enum! {
    /// Only implemented solvers are listed; add a variant together with its
    /// implementation in the `eigensolver` crate.
    pub enum EigenSolverScheme {
        Pcg => "pcg",
        Lobpcg => "lobpcg",
    }
    default = Pcg;
}

scheme_enum! {
    pub enum PotScheme {
        Upf => "upf",
        UpfFr => "upf-fr",
    }
    default = Upf;
}

scheme_enum! {
    pub enum KptsScheme {
        Kmesh => "kmesh",
        Kline => "kline",
    }
    default = Kmesh;
}

scheme_enum! {
    pub enum MixingScheme {
        Pulay => "pulay",
        Broyden => "broyden",
    }
    default = Pulay;
}

scheme_enum! {
    pub enum GeomOptScheme {
        Bfgs => "bfgs",
        Diis => "diis",
    }
    default = Bfgs;
}

scheme_enum! {
    pub enum FftPlannerScheme {
        Estimate => "estimate",
        Measure => "measure",
    }
    default = Estimate;
}

scheme_enum! {
    pub enum ScfLogFormat {
        None => "none",
        Jsonl => "jsonl",
        Csv => "csv",
    }
    default = None;
}

scheme_enum! {
    pub enum ElectricFieldAxis {
        A => "a" | "x" | "1",
        B => "b" | "y" | "2",
        C => "c" | "z" | "3",
    }
    default = C;
}

impl ElectricFieldAxis {
    pub fn axis_index(self) -> usize {
        match self {
            ElectricFieldAxis::A => 0,
            ElectricFieldAxis::B => 1,
            ElectricFieldAxis::C => 2,
        }
    }
}

scheme_enum! {
    pub enum KPointScheduleScheme {
        Contiguous => "contiguous",
        CostAware => "cost_aware" | "cost-aware" | "costaware",
        Dynamic => "dynamic",
    }
    default = CostAware;
}

scheme_enum! {
    /// DOS broadening shape.
    pub enum DosScheme {
        Gauss => "gauss",
    }
    default = Gauss;
}

scheme_enum! {
    /// DFT-D damping function.
    pub enum VdwDampingScheme {
        Zero => "zero",
        Bj => "bj",
    }
    default = Bj;
}
