
#![feature(generic_const_exprs)]

use named_vec_ops::NamedVecOps;
use named_vec_ops_derive::NamedVecOps;
use num_dual::{Dual64, DualNum};
use nalgebra::{SMatrix, SVector};

pub trait Model<const XN: usize, const UN: usize>
{
    type State<T>: NamedVecOps<T, XN> + Clone + Copy;
    type Control<T>: NamedVecOps<T, UN> + Clone + Copy;
    type Parameters: Default + Clone + Copy;

    fn dynamics<T: DualNum<f64> + Clone>(
        state: &Self::State<T>,
        control: &Self::Control<T>,
        params: &Self::Parameters,
    ) -> Self::State<T>;
}

fn linearize<
    M: Model<XN, UN>,
    const XN: usize,
    const UN: usize,
>(
    x: &M::State<f64>,
    u: &M::Control<f64>,
    p: &M::Parameters,
    a: &mut SMatrix<f64, XN, XN>,
    b: &mut SMatrix<f64, XN, UN>,
)
{
    let x_svec = x.to_svector();
    let u_svec = u.to_svector();

    let x_dual_svec  = x_svec.map(Dual64::from);
    let u_dual_svec  = u_svec.map(Dual64::from);

    let x_dual = M::State::<Dual64>::from_svector(&x_dual_svec);
    let u_dual = M::Control::<Dual64>::from_svector(&u_dual_svec);

    let make_perturbed_dual_x = |i| {
        let mut perturbed= x_dual_svec.clone();
        perturbed[(i, 0)].eps = 1.0;
        M::State::<Dual64>::from_svector(&perturbed)
    };

    let make_perturbed_dual_u = |i| {
        let mut perturbed: SVector<Dual64, UN> = u_dual_svec.clone();
        perturbed[(i, 0)].eps = 1.0;
        M::Control::<Dual64>::from_svector(&perturbed)
    };

    for i in 0..XN {
        let dfds = M::dynamics::<Dual64>(
                &make_perturbed_dual_x(i),
                &u_dual,
                p,
        ).to_svector();
        for j in 0..XN {
            a[(j, i)] = dfds[j].eps;
        }
    }

    for i in 0..UN {
        let dfdc = M::dynamics::<Dual64>(
            &x_dual,
            &make_perturbed_dual_u(i),
            p,
        ).to_svector();
        for j in 0..XN {
            b[(j, i)] = dfdc[j].eps;
        }
    }
}

fn cg_solve <
    const UH: usize,
>(
    h: &SMatrix<f64, UH, UH>,
    f: &SVector<f64, UH>,
    x: &mut SVector<f64, UH>,
    r: &mut SVector<f64, UH>,
    p: &mut SVector<f64, UH>,
    hp: &mut SVector<f64, UH>,
    tol: f64,
    max_iter: usize,
)
{
    if f.norm_squared() < tol * tol {
        println!("Cost gradient near zero, skipping CG");
        return;
    }

    *r = -f - h * *x;
    *p = r.clone();

    let mut rs_old = r.dot(r);

    for _ in 0..max_iter {
        *hp = h * *p;
        let denom = p.dot(hp);
        // if denom.abs() < 1e-12 {
        //     println!("hp ≈ 0 -> either \n 1) search direction hit null space of H\n 2) we have converged\n 3) H is not strictly positive definite (e.g due to bad scaling, model linearization issues)\n 4) numerical errors accumulate and cause loss of percision");
        //     break;
        // }
        if denom.abs() < 1e-12 {
            println!("CG exited early: no significant gradient direction (pᵀHp ≈ 0)");
            break;
        }
        let alpha = rs_old / denom;

        *x += alpha * *p;
        *r -= alpha * *hp;

        if r.norm_squared() < tol * tol {
            break;
        }

        let rs_new = r.dot(r);
        *p = *r + (rs_new / rs_old) * *p;
        rs_old = rs_new;
    }
}

// // Support code for assert
// pub struct Assert<const CHECK: bool>;
// pub trait IsTrue {}
// impl IsTrue for Assert<true> {}

pub struct MPC<
    M: Model<XN, UN>,
    const XN: usize,
    const UN: usize,
    const HORIZON: usize,
    const XH: usize = {XN * HORIZON},
    const UH: usize = {UN * HORIZON},
>
{
    pub dt: f64,               // time step
    pub cg_max_iter: usize,    // maximum number of iterations for the CG solver
    pub cg_tol: f64,           // tolerance for CG solver convergence

    pub x_ref: [M::State::<f64>; HORIZON],    // reference states over the horizon
    pub u_ref: [M::Control::<f64>; HORIZON],  // reference controls over the horizon
    pub p: [M::Parameters; HORIZON],          // parameters over the horizon
    pub q: [SMatrix<f64, XN, XN>; HORIZON],   // state cost matrices over the horizon
    pub r: [SMatrix<f64, UN, UN>; HORIZON],   // control cost matrices over the horizon

    delta_u_prev: SVector<f64, UH>,  // previous control input guess
    h: SMatrix<f64, UH, UH>,   // hessian of the cost function
    f: SVector<f64, UH>,       // gradient vector of the cost function
    r_buf: SVector<f64, UH>,   // residual buffer for CG
    p_buf: SVector<f64, UH>,   // search direction buffer for CG
    hp_buf: SVector<f64, UH>,  // hessian-vector product buffer for CG
    su: SMatrix<f64, XH, UH>,  // state-control matrix for the MPC problem
    a_cache: [SMatrix<f64, XN, XN>; HORIZON],            // cache for linearized dynamics A matrices
    b_cache: [SMatrix<f64, XN, UN>; HORIZON],            // cache for linearized dynamics B matrices
    a_prod: [[SMatrix<f64, XN, XN>; HORIZON]; HORIZON],  // cumulative product of A matrices
    delta_x_ref: SVector<f64, XH>,       // difference between current/predicted state and reference state over the horizon
    x_pred: SVector<f64, XH>,            // predicted states over the horizon

    xnxni: SMatrix<f64, XN, XN>, // identity matrix of size XN
}

impl<
    M: Model<XN, UN>,
    const XN: usize,
    const UN: usize,
    const HORIZON: usize,
    const XH: usize,
    const UH: usize,
> MPC<M, XN, UN, HORIZON, XH, UH>
{
    pub fn new(
        dt: f64,
        cg_max_iter: usize,
        cg_tol: f64,
    ) -> Self
    {
        assert!(cg_max_iter > 0, "max_iter must be greater than 0");
        assert!(cg_tol > 0.0, "tol must be greater than 0.0");
        Self {
            dt,
            cg_max_iter,
            cg_tol,
            delta_u_prev: SVector::<f64, UH>::zeros(),
            h: SMatrix::<f64, UH, UH>::zeros(),
            f: SVector::<f64, UH>::zeros(),
            r_buf: SVector::<f64, UH>::zeros(),
            p_buf: SVector::<f64, UH>::zeros(),
            hp_buf: SVector::<f64, UH>::zeros(),
            su: SMatrix::<f64, XH, UH>::zeros(),
            a_cache: [SMatrix::<f64, XN, XN>::zeros(); HORIZON],
            b_cache: [SMatrix::<f64, XN, UN>::zeros(); HORIZON],
            a_prod: [[SMatrix::<f64, XN, XN>::zeros(); HORIZON]; HORIZON],
            delta_x_ref: SVector::<f64, XH>::zeros(),
            x_pred: SVector::<f64, XH>::zeros(),
            x_ref: [M::State::<f64>::from_svector(&SVector::<f64, XN>::zeros()); HORIZON],
            u_ref: [M::Control::<f64>::from_svector(&SVector::<f64, UN>::zeros()); HORIZON],
            p: [M::Parameters::default(); HORIZON],
            q: [SMatrix::<f64, XN, XN>::identity(); HORIZON],
            r: [SMatrix::<f64, UN, UN>::identity(); HORIZON],
            xnxni: SMatrix::<f64, XN, XN>::identity(),
        }
    }

    pub fn solve
    (
        &mut self,
        x0: &M::State::<f64>,
    ) -> M::Control::<f64>
    {
        self.x_pred.fixed_rows_mut::<XN>(0).copy_from(&x0.to_svector());

        let mut q_mat = SMatrix::<f64, XH, XH>::zeros();
        let mut r_mat = SMatrix::<f64, UH, UH>::zeros();

        for i in 0..HORIZON {
            q_mat.view_mut((XN * i, XN * i), (XN, XN)).copy_from(&self.q[i]);
            r_mat.view_mut((UN * i, UN * i), (UN, UN)).copy_from(&self.r[i]);
    
            // TODO: Add option to linearize around
            // 1) reference trajectory
            // 2) predicted trajectory
            linearize::<M, XN, UN>(
                &self.x_ref[i],
                &self.u_ref[i],
                &self.p[i],
                &mut self.a_cache[i],
                &mut self.b_cache[i],
            );

            // Build su matrix
            self.a_prod[i][i] = self.xnxni;
            for j in (0..i).rev() {
                self.a_prod[i][j] = self.a_cache[i - 1] * self.a_prod[i - 1][j];
                let block = self.a_prod[i][j] * self.b_cache[j];
                self.su.view_mut((XN * i, UN * j), (XN, UN)).copy_from(&block);
            }

            let x_pred_i = self.x_pred.fixed_rows::<XN>(i * XN).into_owned();
            let x_error = x_pred_i - self.x_ref[i].to_svector();
            self.delta_x_ref.fixed_rows_mut::<XN>(i * XN).copy_from(&x_error);

            if i == HORIZON - 1 {
                break;
            }

            let u_i = M::Control::<f64>::from_svector(&self.delta_u_prev.fixed_rows::<UN>(i * UN).into_owned()) + self.u_ref[i];
            let dx = M::dynamics::<f64>(&M::State::<f64>::from_svector(&x_pred_i), &u_i, &self.p[i]);

            let x_pred_i_plus_1 = M::State::<f64>::from_svector(&x_pred_i) + dx * self.dt;
            self.x_pred.fixed_rows_mut::<XN>((i + 1) * XN).copy_from(&x_pred_i_plus_1.to_svector());
        }

        self.h.copy_from(&(self.su.transpose() * q_mat * self.su + r_mat));
        self.f.copy_from(&(self.su.transpose() * q_mat * self.delta_x_ref));

        cg_solve::<UH>(&self.h, &self.f, &mut self.delta_u_prev, &mut self.r_buf, &mut self.p_buf, &mut self.hp_buf, self.cg_tol, self.cg_max_iter);

        let u0 = M::Control::<f64>::from_svector(&(self.delta_u_prev.fixed_rows::<UN>(0) + self.u_ref[0].to_svector()));
        
        u0
    }
}
