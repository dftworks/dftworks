use crate::Array3;

use itertools::multizip;

use types::*;

use hdf5::File as File_hdf5;
use std::convert::TryInto;

impl Array3<c64> {
    pub fn scale(&mut self, f: f64) {
        self.data.iter_mut().for_each(|x| *x *= f);
    }

    pub fn norm2(&self) -> f64 {
        self.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
    }

    pub fn scaled_assign(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = *s * factor;
        }
    }

    pub fn diff_norm_sum(&self, rhs: &Array3<c64>) -> f64 {
        let prhs = rhs.as_slice();
        let plhs = self.as_slice();

        assert_eq!(prhs.len(), plhs.len());

        let mut sum = 0.0;

        for (&l, &r) in multizip((plhs.iter(), prhs.iter())) {
            sum += (l - r).norm_sqr();
        }

        sum.sqrt()
    }

    pub fn sqr_add(&mut self, rhs: &Array3<c64>) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d += s.norm_sqr();
        }
    }

    pub fn scaled_sqr_add(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d += s.norm_sqr() * factor;
        }
    }

    pub fn scaled_add(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d += (*s) * factor;
        }
    }

    pub fn mix(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = (*d) * factor + (*s) * (1.0 - factor);
        }
    }

    pub fn add(&mut self, addition: f64) {
        let pdst = self.as_mut_slice();

        for d in pdst.iter_mut() {
            *d = *d + addition;
        }
    }

    /// Save the array to a HDF5 file. The shape, and the real and the imaginary parts of the data are saved in their respective datasets.
    pub fn save_hdf5(&self, filename: &str) {
        let file = File_hdf5::create(filename).unwrap();

        // Write shape
        let _dataset_shape = file
            .new_dataset_builder()
            .with_data(&self.shape)
            .create("shape")
            .unwrap();

        let real_data: Vec<f64> = self.data.iter().map(|&c| c.re.into()).collect();
        let imag_data: Vec<f64> = self.data.iter().map(|&c| c.im.into()).collect();

        // Write real part
        let _dataset_real = file
            .new_dataset_builder()
            .with_data(&real_data)
            .create("real")
            .unwrap();

        // Write imaginary part
        let _dataset_imag = file
            .new_dataset_builder()
            .with_data(&imag_data)
            .create("imag")
            .unwrap();
    }

    /// Load the array from a HDF5 file as saved by the save_hdf5 function.
    pub fn load_hdf5(&mut self, filename: &str) {
        let file = File_hdf5::open(filename).unwrap();

        // Read shape
        let shape: Vec<usize> = file.dataset("shape").unwrap().read().unwrap().to_vec();
        self.shape = shape.try_into().unwrap();

        // Read data
        let real_data: Vec<f64> = file.dataset("real").unwrap().read().unwrap().to_vec();
        let imag_data: Vec<f64> = file.dataset("imag").unwrap().read().unwrap().to_vec();
        self.data = real_data
            .iter()
            .zip(imag_data)
            .map(|(&r, i)| c64::new(r, i))
            .collect();
    }
}

#[test]
fn test_array3_c64() {
    let mut m: Array3<c64> = Array3::<c64>::new([4, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                m[[k, j, i]] = c64 {
                    re: (i * j * k) as f64,
                    im: 0.0,
                };
            }
        }
    }

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                // check if memory is contiguous and value is correct

                println!("{:p} {} {}", &m[[k, j, i]], m[[k, j, i]], i * j * k);
            }
        }
    }

    println!("sum of m = {}", m.sum());

    m.save("a3.dat");

    let mut m_reload = Array3::<c64>::new([4, 3, 2]);
    m_reload.load("a3.dat");
    println!("{:?}", m_reload);

    m_reload.add_from(&m);

    println!("sum of m_reload = {}", m_reload.sum());
}
