use std::fmt;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;

use crate::Vector3;

pub type Vector3f64 = Vector3<f64>;

impl Vector3f64 {
    pub fn dot_product(&self, other: &Vector3f64) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    //
    // c.x = a.y * b.z - a.z * b.y
    // c.y = a.z * b.x - a.x * b.z
    // c.z = a.x * b.y - a.y * b.x
    //
    pub fn cross_product(&self, other: &Vector3f64) -> Vector3f64 {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;

        Vector3f64::new(x, y, z)
    }

    pub fn norm2(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl Add<Vector3f64> for Vector3f64 {
    type Output = Vector3f64;

    fn add(self, rhs: Vector3f64) -> Vector3f64 {
        Vector3f64 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Mul<Vector3f64> for Vector3f64 {
    type Output = f64;

    fn mul(self, rhs: Vector3f64) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl Mul<f64> for Vector3f64 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Vector3f64::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vector3f64> for f64 {
    type Output = Vector3f64;

    fn mul(self, rhs: Vector3f64) -> Vector3f64 {
        Vector3f64::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl Div<f64> for Vector3f64 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Vector3f64::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl fmt::Display for Vector3f64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}
