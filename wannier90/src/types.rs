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
}
