//pub mod array2;
//pub use array2::*;

mod array3_c64;

use ndarray_crate::{Array3 as NdArray3, ShapeBuilder, Zip};
use num::traits::Zero;
use std::convert::TryInto;
use std::fs::File;
use std::io::{Read, Write};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Array3<T> {
    shape: [usize; 3],
    data: NdArray3<T>,
}

impl<T: Default + Clone> Default for Array3<T> {
    fn default() -> Self {
        Self {
            shape: [0, 0, 0],
            data: NdArray3::from_elem((0, 0, 0).f(), T::default()),
        }
    }
}

impl<T: Default + Copy + Clone + Zero + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>>
    Array3<T>
{
    pub fn new(shape: [usize; 3]) -> Array3<T> {
        Array3 {
            shape,
            // Keep first-index-fastest layout to preserve FFT/indexing behavior.
            data: NdArray3::from_elem((shape[0], shape[1], shape[2]).f(), T::default()),
        }
    }

    pub fn from_vec(shape: [usize; 3], data: Vec<T>) -> Array3<T> {
        let nlen = shape[0] * shape[1] * shape[2];
        assert_eq!(data.len(), nlen);

        let data = NdArray3::from_shape_vec((shape[0], shape[1], shape[2]).f(), data)
            .expect("invalid Array3 shape/data length");

        Array3 { shape, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
        self.data.fill(value);
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
            .as_slice_memory_order()
            .expect("Array3 is not contiguous in memory order")
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
            .as_slice_memory_order_mut()
            .expect("Array3 is not contiguous in memory order")
    }

    pub fn hadamard_product(src1: &Array3<T>, src2: &Array3<T>, dst: &mut Array3<T>) {
        assert_eq!(src1.shape, src2.shape);
        assert_eq!(src1.shape, dst.shape);

        Zip::from(dst.data.view_mut())
            .and(src1.data.view())
            .and(src2.data.view())
            .for_each(|z, &x, &y| *z = x * y);
    }

    pub fn assign(&mut self, rhs: &Array3<T>) {
        assert_eq!(self.shape, rhs.shape);
        self.data.assign(&rhs.data);
    }

    pub fn add_from(&mut self, rhs: &Array3<T>) {
        assert_eq!(self.shape, rhs.shape);
        Zip::from(self.data.view_mut())
            .and(rhs.data.view())
            .for_each(|d, &s| *d = *d + s);
    }

    pub fn substract(&mut self, rhs: &Array3<T>) {
        assert_eq!(self.shape, rhs.shape);
        Zip::from(self.data.view_mut())
            .and(rhs.data.view())
            .for_each(|d, &s| *d = *d - s);
    }

    pub fn save(&self, filename: &str) {
        let mut f = File::create(filename).unwrap();

        let n0: &[u8] = &self.shape[0].to_be_bytes();

        let n1: &[u8] = &self.shape[1].to_be_bytes();

        let n2: &[u8] = &self.shape[2].to_be_bytes();

        let slice = self.as_slice();
        let data: &[u8] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
        };

        f.write_all(n0).unwrap();

        f.write_all(n1).unwrap();

        f.write_all(n2).unwrap();

        f.write_all(data).unwrap();
    }

    pub fn load(&mut self, filename: &str) {
        let mut f = match File::open(filename) {
            Err(why) => panic!("couldn't open {}: {}", filename, why),

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

        let shape = [n0, n1, n2];
        let nlen = n0 * n1 * n2;
        let nbytes = std::mem::size_of::<T>() * nlen;
        assert_eq!(p1 - p0, nbytes, "corrupted Array3 binary payload");

        let mut data = vec![T::default(); nlen];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf[p0..p1].as_ptr(),
                data.as_mut_ptr() as *mut u8,
                nbytes,
            );
        }

        *self = Array3::from_vec(shape, data);
    }
}

impl<T> Index<[usize; 3]> for Array3<T> {
    type Output = T;

    fn index(&self, idx: [usize; 3]) -> &T {
        &self.data[[idx[0], idx[1], idx[2]]]
    }
}

impl<T> IndexMut<[usize; 3]> for Array3<T> {
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut Self::Output {
        &mut self.data[[idx[0], idx[1], idx[2]]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        path.push(format!(
            "dftworks_ndarray_{name}_{}_{}.bin",
            std::process::id(),
            ts
        ));
        path
    }

    #[test]
    fn test_array3_indexing_and_shape_contract() {
        let a = Array3::from_vec([2, 2, 2], vec![0i32, 1, 2, 3, 4, 5, 6, 7]);

        // First-index-fastest memory order.
        assert_eq!(a[[0, 0, 0]], 0);
        assert_eq!(a[[1, 0, 0]], 1);
        assert_eq!(a[[0, 1, 0]], 2);
        assert_eq!(a[[1, 1, 0]], 3);
        assert_eq!(a[[0, 0, 1]], 4);
        assert_eq!(a[[1, 1, 1]], 7);
        assert_eq!(a.shape(), [2, 2, 2]);
        assert_eq!(a.len(), 8);
        assert!(!a.is_empty());
    }

    #[test]
    fn test_array3_arithmetic_helpers() {
        let mut a = Array3::from_vec([2, 2, 1], vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array3::from_vec([2, 2, 1], vec![10.0f64, 20.0, 30.0, 40.0]);
        let mut dst = Array3::new([2, 2, 1]);

        Array3::hadamard_product(&a, &b, &mut dst);
        assert_eq!(dst.as_slice(), &[10.0, 40.0, 90.0, 160.0]);

        a.add_from(&b);
        assert_eq!(a.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

        a.substract(&b);
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        let mut assigned = Array3::new([2, 2, 1]);
        assigned.assign(&b);
        assert_eq!(assigned.as_slice(), b.as_slice());

        assigned.set_value(5.0);
        assert_eq!(assigned.sum(), 20.0);
    }

    #[test]
    fn test_array3_save_load_roundtrip() {
        let original = Array3::from_vec([2, 1, 2], vec![1.5f64, -2.0, 3.25, 4.75]);
        let filename = unique_temp_file("roundtrip");
        let filename_str = filename.to_str().expect("non-utf8 temp path");

        original.save(filename_str);

        let mut restored = Array3::<f64>::default();
        restored.load(filename_str);

        std::fs::remove_file(&filename).expect("failed to remove temp file");

        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.as_slice(), original.as_slice());
    }
}
