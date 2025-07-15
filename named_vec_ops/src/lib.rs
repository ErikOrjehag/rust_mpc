

pub trait NamedVecOps<T, const N: usize>: Copy + Clone {
    const SIZE: usize = N;
    fn to_svector(&self) -> nalgebra::SVector<T, N>;
    fn from_svector(v: &nalgebra::SVector<T, N>) -> Self;
    
}
