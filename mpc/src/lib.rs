
#![feature(generic_const_exprs)]

use named_vec_ops::NamedVecOps;
use named_vec_ops_derive::NamedVecOps;
use num_dual::{Dual64, DualNum};
use nalgebra::{SMatrix, SVector};

fn linearize<
    F,
    XDT,
    UDT,
    XT,
    UT,
    PT,
    const XN: usize,
    const UN: usize,
>(
    dynamics_f: F,
    x: &XT,
    u: &UT,
    p: &PT,
    a: &mut SMatrix<f64, XN, XN>,
    b: &mut SMatrix<f64, XN, UN>,
)
where
    XDT: NamedVecOps<Dual64, XN>,
    UDT: NamedVecOps<Dual64, UN>,
    XT: NamedVecOps<f64, XN>,
    UT: NamedVecOps<f64, UN>,
    F: Fn(&XDT, &UDT, &PT) -> XDT,
{
    let x_svec = x.to_svector();
    let u_svec = u.to_svector();

    let x_dual_svec  = x_svec.map(Dual64::from);
    let u_dual_svec  = u_svec.map(Dual64::from);

    let x_dual = XDT::from_svector(&x_dual_svec);
    let u_dual = UDT::from_svector(&u_dual_svec);

    let make_perturbed_dual_x = |i| {
        let mut perturbed= x_dual_svec.clone();
        perturbed[(i, 0)].eps = 1.0;
        XDT::from_svector(&perturbed)
    };

    let make_perturbed_dual_u = |i| {
        let mut perturbed: SVector<Dual64, UN> = u_dual_svec.clone();
        perturbed[(i, 0)].eps = 1.0;
        UDT::from_svector(&perturbed)
    };

    for i in 0..XN {
        let dfds = dynamics_f(
                &make_perturbed_dual_x(i),
                &u_dual,
                p,
        ).to_svector();
        for j in 0..XN {
            a[(j, i)] = dfds[j].eps;
        }
    }

    for i in 0..UN {
        let dfdc = dynamics_f(
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

// Support code for assert
pub struct Assert<const CHECK: bool>;
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

pub struct MPC<
    const XN: usize,
    const UN: usize,
    const HORIZON: usize,
    XT: NamedVecOps<f64, XN>,
    UT: NamedVecOps<f64, UN>,
    PT: Copy + Clone + Default,
    const XH: usize = {XN * HORIZON},
    const UH: usize = {UN * HORIZON},
>
{
    pub dt: f64,               // time step
    pub cg_max_iter: usize,    // maximum number of iterations for the CG solver
    pub cg_tol: f64,           // tolerance for CG solver convergence

    pub x_ref: [XT; HORIZON],                // reference states over the horizon
    pub u_ref: [UT; HORIZON],                // reference controls over the horizon
    pub p: [PT; HORIZON],                    // parameters over the horizon
    pub q: [SMatrix<f64, XN, XN>; HORIZON],  // state cost matrices over the horizon
    pub r: [SMatrix<f64, UN, UN>; HORIZON],  // control cost matrices over the horizon

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

    XNXNI: SMatrix<f64, XN, XN>, // identity matrix of size XN
}

impl<
    const XN: usize,
    const UN: usize,
    const HORIZON: usize,
    XT: NamedVecOps<f64, XN>,
    UT: NamedVecOps<f64, UN>,
    PT: Copy + Clone + Default,
    const XH: usize,
    const UH: usize,
> MPC<XN, UN, HORIZON, XT, UT, PT, XH, UH>
where
    Assert<{XN > 0}>: IsTrue,
    Assert<{UN > 0}>: IsTrue,
    Assert<{HORIZON > 0}>: IsTrue,
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
            x_ref: [XT::from_svector(&SVector::<f64, XN>::zeros()); HORIZON],
            u_ref: [UT::from_svector(&SVector::<f64, UN>::zeros()); HORIZON],
            p: [PT::default(); HORIZON],
            q: [SMatrix::<f64, XN, XN>::identity(); HORIZON],
            r: [SMatrix::<f64, UN, UN>::identity(); HORIZON],
            XNXNI: SMatrix::<f64, XN, XN>::identity(),
        }
    }

    pub fn solve<
        F64,
        FD,
        XDT: NamedVecOps<Dual64, XN>,
        UDT: NamedVecOps<Dual64, UN>,
    >
    (
        &mut self,
        dynamics_f: &F64,
        dynamics_d: &FD,
        x0: &XT,
    ) -> UT
    where
        F64: Fn(&XT, &UT, &PT) -> XT,
        FD: Fn(&XDT, &UDT, &PT) -> XDT,
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
            linearize(
                dynamics_d,
                &self.x_ref[i],
                &self.u_ref[i],
                &self.p[i],
                &mut self.a_cache[i],
                &mut self.b_cache[i],
            );

            // Build su matrix
            self.a_prod[i][i] = self.XNXNI;
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

            let u_i = UT::from_svector(&self.delta_u_prev.fixed_rows::<UN>(i * UN).into_owned()) + self.u_ref[i];
            let dx = dynamics_f(&XT::from_svector(&x_pred_i), &u_i, &self.p[i]);

            let x_pred_i_plus_1 = XT::from_svector(&x_pred_i) + dx * self.dt;
            self.x_pred.fixed_rows_mut::<XN>((i + 1) * XN).copy_from(&x_pred_i_plus_1.to_svector());
        }

        self.h.copy_from(&(self.su.transpose() * q_mat * self.su + r_mat));
        self.f.copy_from(&(self.su.transpose() * q_mat * self.delta_x_ref));

        cg_solve::<UH>(&self.h, &self.f, &mut self.delta_u_prev, &mut self.r_buf, &mut self.p_buf, &mut self.hp_buf, self.cg_tol, self.cg_max_iter);

        let u0 = UT::from_svector(&(self.delta_u_prev.fixed_rows::<UN>(0) + self.u_ref[0].to_svector()));
        
        u0
    }
}

// fn mpc() {
//     let mut trajectory: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(sim_steps);

//     println!("\n\n- - - - - - - - - - - - - -\nStarting MPC simulation with {} steps \n\n", sim_steps);

//     for t in 0..sim_steps {

//         // user space simulator
//         for i in 0..HORIZON {
//             x_ref[i] = State64 { x: 0.2 + 0.1 * (t + i) as f64, y: 0.1 * (t + i) as f64, angle: 45.0 * std::f64::consts::PI / 180.0 };
//             u_ref[i] = Control64 { v: 1.0, w: 0.0 };

//             // x_ref[i] -= x;
//         }

//         let mut x_lin = x;

//         // mpc space solver
//         for i in 0..HORIZON {
//             linearize(
//                 dynamics,
//                 &x_lin,
//                 &u_ref[i],
//                 &params,
//                 &mut a_cache[i],
//                 &mut b_cache[i],
//             );

//             let dx = dynamics(&x_lin, &u_ref[i], &params);
//             x_lin += dx * dt;
//         }

//         for i in 0..HORIZON {
//             for j in (0..=i).rev() {
//                 if j == i {
//                     a_prod[i][j] = SMatrix::<f64, XN, XN>::identity();
//                 } else {
//                     a_prod[i][j] = a_cache[i - 1] * a_prod[i - 1][j];
//                 }
//             }
//         }

//         su.fill(0.0);
//         for i in 0..HORIZON {
//             for j in 0..i {
//                 let block = a_prod[i][j] * b_cache[j];
//                 su.view_mut((XN * i, UN * j), (XN, UN)).copy_from(&block);
//             }
//         }

//         // delta_x0.fill(0.0);
//         // delta_x0.fixed_rows_mut::<XN>(0).copy_from(&(x - x_ref[0]).to_svector());

//         let mut delta_x_ref = SVector::<f64, XH>::zeros();
//         let mut x_pred = x;

//         for i in 0..HORIZON {
//             let x_err = x_pred - x_ref[i];
//             delta_x_ref.fixed_rows_mut::<XN>(i * XN).copy_from(&x_err.to_svector());

//             // Predict next state using current control guess (e.g. delta_u_prev segment)
//             let u_i = Control64::from_svector(&(delta_u_prev.fixed_rows::<UN>(i * UN) + u_ref[i].to_svector()));
//             x_pred += dynamics(&x_pred, &u_i, &params) * dt;
//         }

//         // let q_mat = SMatrix::<f64, XH, XH>::from_diagonal_element(q);
//         // let r_mat = SMatrix::<f64, UH, UH>::from_diagonal_element(r);

//         let mut q_mat = SMatrix::<f64, XH, XH>::zeros();
//         for i in 0..HORIZON {
//             let offset = i * XN;
//             q_mat[(offset, offset)] = 1.0;
//             q_mat[(offset + 1, offset + 1)] = 1.0;
//             q_mat[(offset + 2, offset + 2)] = 0.1;
//         }
//         let mut r_mat = SMatrix::<f64, UH, UH>::zeros();
//         for i in 0..HORIZON {
//             let offset = i * UN;
//             r_mat[(offset, offset)] = 0.01;
//             r_mat[(offset + 1, offset + 1)] = 0.01;
//         }

//         h.copy_from(&(su.transpose() * q_mat * su + r_mat));
//         // f.copy_from(&(su.transpose() * q_mat * delta_x0));
//         f.copy_from(&(su.transpose() * q_mat * delta_x_ref));

//         let h_cond = h.norm() / h.try_inverse().map(|inv| inv.norm()).unwrap_or(1e12);
//         println!("Condition number estimate of H: {:.2e}", h_cond);

//         cg_solve::<UH>(&h, &f, &mut delta_u_prev, &mut r_buf, &mut p_buf, &mut hp_buf, cg_tol, max_iter);

//         let u0 = Control64::from_svector(&(delta_u_prev.fixed_rows::<2>(0) + u_ref[0].to_svector()));
        
//         let dx = dynamics(&x, &u0, &params);
//         x += dx * dt;

//         println!("Step {}: x = {:?}, u = {:?}", t, x, u0);

//         trajectory.push((x.x, x.y, x_ref[0].x, x_ref[0].y));
//     }

//     let mut traj_file = File::create("trajectory.csv").expect("Could not create file");
//     writeln!(traj_file, "step,x,y,rx,ry").unwrap();

//     for (i, (x, y, rx, ry)) in trajectory.iter().enumerate() {
//         writeln!(traj_file, "{},{},{},{},{}", i, x, y, rx, ry).unwrap();
//     }
    
// }

// fn main() {
//     mpc1();
// }