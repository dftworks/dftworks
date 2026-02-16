#[derive(Debug, Default)]
pub struct ExportSummary {
    pub written_files: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Neighbor {
    pub(crate) ikb: usize,       // zero-based
    pub(crate) gshift: [i32; 3], // reciprocal lattice shift to map k+b to ikb
}

#[derive(Debug, Clone)]
pub(crate) struct MeshTopology {
    pub(crate) neighbors: Vec<Vec<Neighbor>>,
    pub(crate) nntot: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TrialOrbital {
    pub(crate) atom_index: usize,
    pub(crate) species: String,
    pub(crate) l: usize,
    pub(crate) m: i32,
    pub(crate) hybrid_kind: Option<HybridKind>,
    pub(crate) hybrid_group: Option<usize>,
    pub(crate) hybrid_component: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HybridKind {
    Sp3,
}
