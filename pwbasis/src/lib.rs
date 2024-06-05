//#![allow(warnings)]
use gvector::GVector;
use utility;
use vector3::Vector3f64;

pub struct PWBasis {
    k_cart: Vector3f64, // in cartesian coordinates
    k_index: usize,     // index of this xk in all xks
    npw: usize,         // number of plane waves
    gindex: Vec<usize>,
    kg: Vec<f64>, // norms of the vectors xk+gvec
}

impl PWBasis {
    pub fn get_kg(&self) -> &[f64] {
        self.kg.as_slice()
    }

    pub fn get_k_cart(&self) -> Vector3f64 {
        self.k_cart
    }

    pub fn get_k_index(&self) -> usize {
        self.k_index
    }

    pub fn get_gindex(&self) -> &[usize] {
        self.gindex.as_slice()
    }

    pub fn get_n_plane_waves(&self) -> usize {
        self.npw
    }

    pub fn new(k_cart: Vector3f64, k_index: usize, ecut: f64, gvec: &GVector) -> PWBasis {
        let npw = gvec.get_n_plane_waves(ecut, k_cart);

        let mut t_gindex: Vec<usize> = vec![0; npw];

        gvec.set_g_vector_index(ecut, k_cart, t_gindex.as_mut_slice());

        let mut t_kg = vec![0.0; npw];

        compute_kg(gvec, k_cart, t_gindex.as_slice(), t_kg.as_mut_slice());

        // sort |k+G|

        let ordered_index = utility::argsort(&t_kg);

        let mut gindex: Vec<usize> = vec![0; npw];

        let mut kg = vec![0.0; npw];

        for (i, &j) in ordered_index.iter().enumerate() {
            kg[i] = t_kg[j];
            gindex[i] = t_gindex[j];
        }

        PWBasis {
            k_cart,
            k_index,
            npw,
            gindex,
            kg,
        }
    }

    /// Save the PWBasis to a HDF5 file.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        let dataset_pwbasis = group
            .new_dataset_builder()
            .with_data(&self.get_kg())
            .create("kg")
            .unwrap();

        dataset_pwbasis
            .new_attr_builder()
            .with_data(&self.get_k_cart().to_vec())
            .create("k_cart")
            .unwrap();

        let int_size = match usize::BITS {
            32 => hdf5::types::IntSize::U4,
            64 => hdf5::types::IntSize::U8,
            _ => panic!("Unknown usize size"),
        };

        dataset_pwbasis
            .new_attr_builder()
            .empty_as(&hdf5::types::TypeDescriptor::Unsigned(int_size))
            .create("k_index")
            .unwrap()
            .write_scalar(&self.get_k_index())
            .unwrap();

        dataset_pwbasis
            .new_attr_builder()
            .empty_as(&hdf5::types::TypeDescriptor::Unsigned(int_size))
            .create("n_pw")
            .unwrap()
            .write_scalar(&self.get_n_plane_waves())
            .unwrap();
    }
}

fn compute_kg(gvec: &GVector, xk: Vector3f64, gindex: &[usize], kg: &mut [f64]) {
    let gcart = gvec.get_cart();

    for (i, &j) in gindex.iter().enumerate() {
        let xkg = xk + gcart[j];

        kg[i] = xkg.norm2();
    }
}
