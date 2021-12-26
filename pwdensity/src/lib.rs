use gvector::GVector;
use vector3::Vector3f64;

pub struct PWDensity {
    npw: usize, // number of plane waves
    gindex: Vec<usize>,
    g: Vec<f64>, // storing the norm of the vectors G
    nshell: usize,
    gshell: Vec<f64>,         //the norm of unique Gs size
    gshell_index: Vec<usize>, // each G vector size's index in gshell
}

impl PWDensity {
    pub fn get_g(&self) -> &[f64] {
        self.g.as_slice()
    }

    pub fn get_gmax(&self) -> f64 {
        *self.g.last().unwrap()
    }

    pub fn get_gindex(&self) -> &[usize] {
        self.gindex.as_slice()
    }

    pub fn get_n_plane_waves(&self) -> usize {
        self.npw
    }

    pub fn get_n_gshell(&self) -> usize {
        self.nshell
    }

    pub fn get_gshell_norms(&self) -> &[f64] {
        self.gshell.as_slice()
    }

    pub fn get_gshell_index(&self) -> &[usize] {
        self.gshell_index.as_slice()
    }

    pub fn new(ecut: f64, gvec: &GVector) -> PWDensity {
        let k_gamma = Vector3f64::zeros();

        let npw = gvec.get_n_plane_waves(ecut, k_gamma);

        let mut gindex = vec![0usize; npw];

        gvec.set_g_vector_index(ecut, k_gamma, gindex.as_mut_slice());

        let mut g = vec![0.0; npw];

        compute_g(gvec, gindex.as_slice(), g.as_mut_slice());

        let nshell = get_n_g_shell(g.as_slice());

        let mut gshell = vec![0.0; nshell];

        let mut gshell_index = vec![0usize; npw];

        set_g_shell_and_index(
            g.as_slice(),
            gshell.as_mut_slice(),
            gshell_index.as_mut_slice(),
        );

        PWDensity {
            npw,
            gindex,
            g,
            nshell,
            gshell,
            gshell_index,
        }
    }
}

fn compute_g(gvec: &GVector, gindex: &[usize], gnorm: &mut [f64]) {
    let gcart = gvec.get_cart();

    for (i, j) in gindex.iter().enumerate() {
        gnorm[i] = gcart[*j].norm2();
    }
}

fn get_n_g_shell(g: &[f64]) -> usize {
    let mut ng: usize = 0;

    let mut glen: f64 = -1.0;

    for x in g.iter() {
        if x - glen > 1.0E-10 {
            glen = *x;
            ng += 1;
        }
    }

    ng
}

fn set_g_shell_and_index(g: &[f64], shell: &mut [f64], index: &mut [usize]) {
    let mut ng: usize = 0;

    let mut glen: f64 = -1.0;

    for (i, x) in g.iter().enumerate() {
        if x - glen > 1.0E-10 {
            glen = *x;
            shell[ng] = glen;
            ng += 1;
        }

        index[i] = ng - 1;
    }
}
