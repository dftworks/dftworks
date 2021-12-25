use crate::Vector3;

pub type Vector3i32 = Vector3<i32>;

use std::fmt;
use std::ops::Add;

impl Add<Vector3i32> for Vector3i32 {
    type Output = Vector3i32;

    fn add(self, rhs: Vector3i32) -> Vector3i32 {
        Vector3i32 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl fmt::Display for Vector3i32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}
