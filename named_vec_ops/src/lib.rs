

pub trait NamedVecOps<T, const N: usize>:
    Copy
  + Clone
  + std::ops::Add<Self, Output = Self>
  + std::ops::Sub<Self, Output = Self>
  + std::ops::Mul<T, Output = Self>
  + std::ops::AddAssign<Self>
  + std::ops::SubAssign<Self>
{
    const SIZE: usize = N;

    fn to_svector(&self) -> nalgebra::SVector<T, N>;
    fn from_svector(v: &nalgebra::SVector<T, N>) -> Self;
}
