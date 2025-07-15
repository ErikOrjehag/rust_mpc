
use std::ops::{Add, Sub, Mul, Div};



#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    value: f64,
    derivative: f64,
}

impl Dual {
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Dual {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Dual {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Dual {
            value: self.value * rhs.value,
            derivative: self.value * rhs.derivative + self.derivative * rhs.value,
        }
    }
}

impl Div for Dual
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Dual {
            value: self.value / rhs.value,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) / (rhs.value * rhs.value)
        }
    }
}

impl Dual {
    pub fn sin(self) -> Self {
        Dual {
            value: self.value.sin(),
            derivative: self.value.cos() * self.derivative,
        }
    }

    pub fn cos(self) -> Self {
        Dual {
            value: self.value.cos(),
            derivative: -self.value.sin() * self.derivative,
        }
    }

    pub fn exp(self) -> Self {
        let val = self.value.exp();
        Dual {
            value: val,
            derivative: val * self.derivative,
        }
    }

    pub fn ln(self) -> Self {
        Dual {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }
}
