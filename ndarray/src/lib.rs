use hdf5::File as File_hdf5;
use hdf5::types::H5Type;
use num::complex::Complex;

pub mod array2;
pub use array2::*;

mod array3_c64;
pub use array3_c64::*;

use num::traits::Zero;
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::ops::{Index, IndexMut};

use itertools::multizip;

#[derive(Default, Debug, Clone)]
pub struct Array3<T> {
    shape: [usize; 3],
    data: Vec<T>,
}

impl<T: Default + Copy + Clone + Zero + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>>
    Array3<T>
{
    pub fn new(shape: [usize; 3]) -> Array3<T> {
        let nlen = shape[0] * shape[1] * shape[2];

        Array3 {
            shape,
            data: vec![T::default(); nlen],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn sum(&self) -> T {
        let mut s = T::zero();
        for v in self.data.iter() {
            s = s + *v;
        }

        s
    }

    pub fn set_value(&mut self, value: T)
    where
        T: Clone + Copy,
    {
        self.data.iter_mut().for_each(|x| *x = value);
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn as_slice(&self) -> &Vec<T> {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    pub fn hadamard_product(src1: &Array3<T>, src2: &Array3<T>, dst: &mut Array3<T>) {
        let a = src1.as_slice();
        let b = src2.as_slice();
        let c = dst.as_mut_slice();

        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());

        for (x, y, z) in multizip((a.iter(), b.iter(), c.iter_mut())) {
            *z = (*x) * (*y);
        }
    }

    pub fn assign(&mut self, rhs: &Array3<T>) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        pdst[..psrc.len()].copy_from_slice(psrc);
    }

    pub fn add_from(&mut self, rhs: &Array3<T>) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());
        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = *d + *s;
        }
    }

    pub fn substract(&mut self, rhs: &Array3<T>) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());
        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = *d - *s;
        }
    }

    pub fn save(&self, filename: &str) {
        let mut f = File::create(filename).unwrap();

        let n0: &[u8] = &self.shape[0].to_be_bytes();

        let n1: &[u8] = &self.shape[1].to_be_bytes();

        let n2: &[u8] = &self.shape[2].to_be_bytes();

        let data: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                std::mem::size_of::<T>() * self.data.len(),
            )
        };

        f.write_all(n0).unwrap();

        f.write_all(n1).unwrap();

        f.write_all(n2).unwrap();

        f.write_all(data).unwrap();
    }

    pub fn load(&mut self, filename: &str) {
        let mut f = match File::open(filename) {
            Err(why) => panic!("couldn't open {}: {}", filename, why.to_string()),

            Ok(f) => f,
        };

        let mut buf = Vec::new();

        f.read_to_end(&mut buf).unwrap();

        let element_size = std::mem::size_of::<usize>();

        let mut p0 = 0;
        let mut p1 = element_size;

        let n0 = usize::from_be_bytes(buf[p0..p1].try_into().unwrap());

        p0 = p1;
        p1 = p0 + element_size;

        let n1 = usize::from_be_bytes(buf[p0..p1].try_into().unwrap());

        p0 = p1;
        p1 = p0 + element_size;

        let n2 = usize::from_be_bytes(buf[p0..p1].try_into().unwrap());

        p0 = p1;
        p1 = buf.len();

        let data: Vec<T> = unsafe {
            std::slice::from_raw_parts(buf[p0..p1].as_ptr() as *const T, n0 * n1 * n2).to_vec()
        };

        for (s, d) in multizip((data.iter(), self.data.iter_mut())) {
            *d = *s;
        }
    }
}


impl<T: Default + Copy + Clone + H5Type + Into<f64>> Array3<Complex<T>> {
    /// Save the array to a HDF5 file. The shape, and the real and the imaginary parts of the data are saved in their respective datasets.
    pub fn save_hdf5(&self, filename: &str) {
        let file = File_hdf5::create(filename).unwrap();

        // Write shape
        let _dataset_shape = file.new_dataset_builder()
            .with_data(&self.shape)
            .create("shape").unwrap();

        let real_data: Vec<f64> = self.data.iter().map(|&c| c.re.into()).collect();
        let imag_data: Vec<f64> = self.data.iter().map(|&c| c.im.into()).collect();
        
        // Write real part
        let _dataset_real = file.new_dataset_builder()
            .with_data(&real_data)
            .create("real").unwrap();
        
        // Write imaginary part
        let _dataset_imag = file.new_dataset_builder()
            .with_data(&imag_data)
            .create("imag").unwrap();
    }

    /// Load the array from a HDF5 file as saved by the save_hdf5 function.
    pub fn load_hdf5(&mut self, filename: &str) {
        let file = File_hdf5::open(filename).unwrap();
        
        // Read shape
        let shape: Vec<usize> = file.dataset("shape").unwrap().read().unwrap().to_vec();
        self.shape = shape.try_into().unwrap();

        // Read data
        let real_data: Vec<T> = file.dataset("real").unwrap().read().unwrap().to_vec();
        let imag_data: Vec<T> = file.dataset("imag").unwrap().read().unwrap().to_vec();
        self.data = real_data.iter().zip(imag_data).map(|(&r, i)| Complex::new(r, i)).collect();
    }
}

impl<T> Index<[usize; 3]> for Array3<T> {
    type Output = T;

    fn index(&self, idx: [usize; 3]) -> &T {
        let n0 = self.shape[0];
        let n1 = self.shape[1];

        let pos = idx[0] + idx[1] * n0 + idx[2] * n0 * n1;

        &self.data[pos]
    }
}

impl<T> IndexMut<[usize; 3]> for Array3<T> {
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut Self::Output {
        let n0 = self.shape[0];
        let n1 = self.shape[1];

        let pos = idx[0] + idx[1] * n0 + idx[2] * n0 * n1;

        &mut self.data[pos]
    }
}
