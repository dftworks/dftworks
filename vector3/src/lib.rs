mod vector3_f64;
pub use vector3_f64::*;

mod vector3_i32;
pub use vector3_i32::*;

use types::c64;
pub type Vector3c64 = Vector3<c64>;

///////////////////////////////////////////////////

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: num_traits::identities::Zero + Copy + std::ops::Mul<Output = T>> Vector3<T> {
    #[inline]
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x, y, z }
    }

    #[inline]
    pub fn zeros() -> Vector3<T> {
        Vector3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        vec![self.x, self.y, self.z]
    }

    /// Get a slice view of the vector components
    /// 
    /// # Safety
    /// This relies on the struct layout. The struct should be #[repr(C)] for guaranteed layout.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(&self.x as *const T, 3) }
    }

    /// Get a mutable slice view of the vector components
    /// 
    /// # Safety
    /// This relies on the struct layout. The struct should be #[repr(C)] for guaranteed layout.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(&mut self.x as *mut T, 3) }
    }

    #[inline]
    pub fn set_zeros(&mut self) {
        self.x = T::zero();
        self.y = T::zero();
        self.z = T::zero();
    }
}

pub fn as_mut_slice_of_element<T>(v: &mut [Vector3<T>]) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut T, v.len() * 3) }
}

pub fn as_slice_of_element<T>(v: &[Vector3<T>]) -> &[T] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const T, v.len() * 3) }
}

#[cfg(test)]
mod tests;
