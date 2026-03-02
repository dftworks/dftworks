use nalgebra as na;
use types::c64;

pub type Vector3<T> = na::Vector3<T>;
pub type Vector3f64 = Vector3<f64>;
pub type Vector3i32 = Vector3<i32>;
pub type Vector3c64 = Vector3<c64>;

pub fn as_mut_slice_of_element<T>(v: &mut [Vector3<T>]) -> &mut [T] {
    debug_assert_eq!(
        std::mem::size_of::<Vector3<T>>(),
        std::mem::size_of::<[T; 3]>()
    );
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut T, v.len() * 3) }
}

pub fn as_slice_of_element<T>(v: &[Vector3<T>]) -> &[T] {
    debug_assert_eq!(
        std::mem::size_of::<Vector3<T>>(),
        std::mem::size_of::<[T; 3]>()
    );
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const T, v.len() * 3) }
}

#[test]
fn test_vector3() {
    let mut v = vec![Vector3f64::new(1.0, 2.0, 3.0); 3];

    let v_f64 = as_mut_slice_of_element(&mut v);

    let mut v2 = vec![0.0; 9];
    for i in 0..9 {
        v2[i] = v_f64[i];
    }

    assert_eq!(v2, v_f64);
}
