#![allow(warnings)]
use gvector::GVector;
use utility;
use vector3::Vector3f64;

/// A plane wave basis set for a specific k-point
///
/// PWBasis represents a set of plane wave basis functions for a given k-point,
/// containing the G-vector indices and kinetic energies |k+G|² for each plane wave.
#[derive(Default)]
pub struct PWBasis {
    k_cart: Vector3f64, // in cartesian coordinates
    k_index: usize,     // index of this xk in all xks
    npw: usize,         // number of plane waves
    gindex: Vec<usize>, // indices of G vectors used in this set of plane wave basis
    kg: Vec<f64>,       // norms of the vectors xk+gvec
}

impl PWBasis {
    /// Returns the kinetic energies |k+G|² for all plane waves
    pub fn get_kg(&self) -> &[f64] {
        &self.kg
    }

    /// Returns the k-point in Cartesian coordinates
    pub fn get_k_cart(&self) -> Vector3f64 {
        self.k_cart
    }

    /// Returns the index of this k-point in the full k-point list
    pub fn get_k_index(&self) -> usize {
        self.k_index
    }

    /// Returns the G-vector indices used in this plane wave basis
    pub fn get_gindex(&self) -> &[usize] {
        &self.gindex
    }

    /// Returns the number of plane waves in this basis
    pub fn get_n_plane_waves(&self) -> usize {
        self.npw
    }

    /// Creates a new PWBasis for the given k-point and energy cutoff
    ///
    /// # Arguments
    /// * `k_cart` - k-point in Cartesian coordinates
    /// * `k_index` - index of this k-point in the full k-point list
    /// * `ecut` - energy cutoff in Hartree
    /// * `gvec` - G-vector grid
    ///
    /// # Returns
    /// A new PWBasis with plane waves sorted by |k+G|²
    pub fn new(k_cart: Vector3f64, k_index: usize, ecut: f64, gvec: &GVector) -> PWBasis {
        let npw = gvec.get_n_plane_waves(ecut, k_cart);

        // Pre-allocate vectors with known capacity
        let mut gindex = Vec::with_capacity(npw);
        gindex.resize(npw, 0);

        gvec.set_g_vector_index(ecut, k_cart, &mut gindex);

        // Compute kinetic energies directly without temporary allocation
        let mut kg = Vec::with_capacity(npw);
        kg.resize(npw, 0.0);

        compute_kg_optimized(gvec, k_cart, &gindex, &mut kg);

        // Sort by kinetic energy using argsort
        let ordered_indices = utility::argsort(&kg);

        // Apply sorting more efficiently
        let sorted_gindex = ordered_indices.iter().map(|&i| gindex[i]).collect();
        let sorted_kg = ordered_indices.iter().map(|&i| kg[i]).collect();

        PWBasis {
            k_cart,
            k_index,
            npw,
            gindex: sorted_gindex,
            kg: sorted_kg,
        }
    }

    /// Save the PWBasis to a HDF5 file.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - HDF5 dataset creation fails
    /// - HDF5 attribute creation fails
    /// - HDF5 write operations fail
    /// - usize has an unsupported bit width
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        let dataset_pwbasis = group
            .new_dataset_builder()
            .with_data(&self.get_kg())
            .create("kg")
            .expect("Failed to create kg dataset");

        dataset_pwbasis
            .new_attr_builder()
            .with_data(&self.get_k_cart().to_vec())
            .create("k_cart")
            .expect("Failed to create k_cart attribute");

        let int_size = match usize::BITS {
            32 => hdf5::types::IntSize::U4,
            64 => hdf5::types::IntSize::U8,
            _ => panic!("Unsupported usize size: {} bits", usize::BITS),
        };

        dataset_pwbasis
            .new_attr_builder()
            .empty_as(&hdf5::types::TypeDescriptor::Unsigned(int_size))
            .create("k_index")
            .expect("Failed to create k_index attribute")
            .write_scalar(&self.get_k_index())
            .expect("Failed to write k_index value");

        dataset_pwbasis
            .new_attr_builder()
            .empty_as(&hdf5::types::TypeDescriptor::Unsigned(int_size))
            .create("n_pw")
            .expect("Failed to create n_pw attribute")
            .write_scalar(&self.get_n_plane_waves())
            .expect("Failed to write n_pw value");

        group
            .new_dataset_builder()
            .with_data(&self.get_gindex())
            .create("gindex")
            .expect("Failed to create gindex dataset");
    }

    /// Load a PWBasis from a HDF5 file.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - HDF5 dataset reading fails
    /// - HDF5 attribute reading fails
    /// - Required datasets or attributes are missing
    /// - Data format is invalid (e.g., k_cart doesn't have 3 elements)
    /// - Data dimensions are inconsistent
    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        let kg_dataset = group.dataset("kg").expect("Failed to find kg dataset");

        let kg: Vec<f64> = kg_dataset
            .read()
            .expect("Failed to read kg dataset")
            .to_vec();

        let gindex: Vec<usize> = group
            .dataset("gindex")
            .expect("Failed to find gindex dataset")
            .read()
            .expect("Failed to read gindex dataset")
            .to_vec();

        // New files store metadata on the "kg" dataset; keep a fallback to group-level attrs
        // for compatibility with any legacy files.
        let k_index: usize = kg_dataset
            .attr("k_index")
            .or_else(|_| group.attr("k_index"))
            .expect("Failed to find k_index attribute")
            .read_scalar()
            .expect("Failed to read k_index attribute");

        let npw: usize = kg_dataset
            .attr("n_pw")
            .or_else(|_| group.attr("n_pw"))
            .expect("Failed to find n_pw attribute")
            .read_scalar()
            .expect("Failed to read n_pw attribute");

        let k_cart_vec: Vec<f64> = kg_dataset
            .attr("k_cart")
            .or_else(|_| group.attr("k_cart"))
            .expect("Failed to find k_cart attribute")
            .read()
            .expect("Failed to read k_cart attribute")
            .to_vec();

        // Validate k_cart vector has exactly 3 elements
        if k_cart_vec.len() != 3 {
            panic!(
                "k_cart attribute must have exactly 3 elements, found {}",
                k_cart_vec.len()
            );
        }

        let k_cart = Vector3f64::new(k_cart_vec[0], k_cart_vec[1], k_cart_vec[2]);

        // Validate data consistency
        if kg.len() != npw {
            panic!(
                "Inconsistent kg data: expected {} elements, found {}",
                npw,
                kg.len()
            );
        }
        if gindex.len() != npw {
            panic!(
                "Inconsistent gindex data: expected {} elements, found {}",
                npw,
                gindex.len()
            );
        }

        PWBasis {
            k_cart,
            k_index,
            npw,
            gindex,
            kg,
        }
    }
}

// Optimized version of compute_kg that avoids unnecessary vector operations
fn compute_kg_optimized(gvec: &GVector, k_cart: Vector3f64, gindex: &[usize], kg: &mut [f64]) {
    let gcart = gvec.get_cart();

    // Use iterators for better performance and avoid bounds checking
    kg.iter_mut()
        .zip(gindex.iter())
        .for_each(|(kg_val, &g_idx)| {
            let k_plus_g = k_cart + gcart[g_idx];
            *kg_val = k_plus_g.norm2();
        });
}

// Legacy function maintained for compatibility
fn compute_kg(gvec: &GVector, xk: Vector3f64, gindex: &[usize], kg: &mut [f64]) {
    compute_kg_optimized(gvec, xk, gindex, kg);
}
