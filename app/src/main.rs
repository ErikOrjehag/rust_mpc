
use named_vec_ops::NamedVecOps;
use named_vec_ops_derive::NamedVecOps;
use num_dual::{Dual64, DualNum};
use nalgebra::{SMatrix, SVector};

#[derive(Debug, Copy, Clone, NamedVecOps)]
struct State<T>
{
    x: T,
    y: T,
    angle: T,
}

type State64 = State<f64>;
type StateDual = State<Dual64>;

#[derive(Debug, Copy, Clone, NamedVecOps)]
struct Control<T>
{
    v: T,
    w: T,
}

type Control64 = Control<f64>;
type ControlDual = Control<Dual64>;

#[derive(Debug, Copy, Clone)]
struct Parameters
{
    ref_x: f64,
    ref_y: f64,
}

fn dynamics<T: DualNum<f64> + Clone>(
    state: &State<T>,
    control: &Control<T>,
    _params: &Parameters,
) -> State<T> {
    State {
        x: state.angle.cos() * control.v.clone(),
        y: state.angle.sin() * control.v.clone(),
        angle: control.w.clone(),
    }
}

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
    const XN: usize,
    const UN: usize,
    const XH: usize,
    const UH: usize,
>(
    h: &SMatrix<f64, UH, UH>,
    f: &SVector<f64, UH>,
    x: &mut SVector<f64, UH>,
    r: &SVector<f64, UH>,
    p: &mut SVector<f64, UH>,
    hp: &mut SVector<f64, UH>,
    tol: f64,
    max_iter: usize,
)
{
    // todo
}

fn mpc1() {
    const HORIZON: usize = 10;
    const XN: usize = State64::SIZE;
    const UN: usize = Control64::SIZE;

    const XH: usize = XN * HORIZON;
    const UH: usize = UN * HORIZON;

    let sim_steps: usize = 30;
    let max_iter: usize = 50;
    let tol = 1e-6;
    let q = 1.0;
    let r = 0.1;

    let mut x: State64 = State { x: 0.0, y: 0.0, angle: 0.0 };
    let mut u_prev = SVector::<f64, UH>::zeros();
    let mut delta_x0 = SVector::<f64, XH>::zeros();
    let mut h = SMatrix::<f64, UH, UH>::zeros();
    let mut f = SVector::<f64, UH>::zeros();
    let mut r_buf = SVector::<f64, UH>::zeros();
    let mut p_buf = SVector::<f64, UH>::zeros();
    let mut hp_buf = SVector::<f64, UH>::zeros();
    let mut su = SMatrix::<f64, XH, UH>::zeros();

    let mut a_cache = [SMatrix::<f64, XN, XN>::zeros(); HORIZON];
    let mut b_cache = [SMatrix::<f64, XN, UN>::zeros(); HORIZON];
    let mut a_prod = [[SMatrix::<f64, XN, XN>::zeros(); HORIZON]; HORIZON];

    let mut x_ref = [State64 { x: 0.0, y: 0.0, angle: 0.0 }; HORIZON];
    let mut u_ref = [Control64 { v: 0.0, w: 0.0 }; HORIZON];
    let params = Parameters { ref_x: 0.0, ref_y: 0.0 };

    for t in 0..sim_steps {

        // user space simulator
        for i in 0..HORIZON {
            x_ref[i] = State64 { x: 0.1 * (t + i) as f64, y: 0.1 * (t + i) as f64, angle: 45.0 * std::f64::consts::PI / 180.0 };
            u_ref[i] = Control64 { v: 0.1, w: 0.0 };
        }

        // mpc space solver
        for i in 0..HORIZON {
            linearize(
                dynamics,
                &x_ref[i],
                &u_ref[i],
                &params,
                &mut a_cache[i],
                &mut b_cache[i],
            );
        }

        for i in 0..HORIZON {
            for j in (0..=i).rev() {
                if j == i {
                    a_prod[i][j] = SMatrix::<f64, XN, XN>::identity();
                } else {
                    a_prod[i][j] = a_cache[i - 1] * a_prod[i - 1][j];
                }
            }
        }

        // su.fill(0.0);
        for i in 0..HORIZON {
            for j in 0..i {
                let block = a_prod[i][j] * b_cache[i];
                su.view_mut((XN * i, UN * j), (XN, UN)).copy_from(&block);
            }
        }

        delta_x0.fill(0.0);
        delta_x0.fixed_rows_mut::<XN>(0).copy_from(&(x - x_ref[0]).to_svector());

        let q_mat = SMatrix::<f64, XH, XH>::from_diagonal_element(q);
        let r_mat = SMatrix::<f64, UH, UH>::from_diagonal_element(r);

        h.copy_from(&(su.transpose() * q_mat * su + r_mat));
        f.copy_from(&(su.transpose() * q_mat * delta_x0));

        cg_solve()

    }
    

    // for t in 0..sim_steps {
    //     let x_ref: Vec<State64> = (0..HORIZON)
    //         .map(|k| State64 { x: 0.1 * (t + k) as f64, y: 0.1 * (t + k) as f64, angle: 0.0 })
    //         .collect();

    //     let u_ref: Vec<Control64> = (0..HORIZON)
    //         .map(|k| Control64 { v: 0.1, w: 0.0 })
    //         .collect();

    //     let p = Parameters {
    //         ref_x: 0.0,
    //         ref_y: 0.0,
    //     };

    //     let mut a = SMatrix::<f64, {XH}, {State64::SIZE}>::zeros();
    //     let mut b = SMatrix::<f64, {XH}, {Control64::SIZE}>::zeros();

    //     for k in 0..HORIZON {
    //         let (a_k, b_k) = linearize(
    //             dynamics,
    //             &x_ref[k],
    //             &u_ref[k],
    //             &p,
    //         );
    //         a.view_mut((State64::SIZE * k, 0), (State64::SIZE, State64::SIZE)).copy_from(&a_k);
    //         b.view_mut((State64::SIZE * k, 0), (State64::SIZE, Control64::SIZE)).copy_from(&b_k);
    //     }

    //     let su = build_time_varying_su::<{State64::SIZE}, {Control64::SIZE}, HORIZON, XH, UH>(&a, &b);

    //     println!("{:?}", su);

    // }
}

fn main() {
    mpc1();
}