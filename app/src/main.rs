#![feature(generic_const_exprs)]

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
    SDT,
    CDT,
    ST,
    CT,
    PT,
    const SN: usize,
    const CN: usize,
>(
    dynamics_f: F,
    state: &ST,
    control: &CT,
    params: &PT,
) -> (SMatrix<f64, SN, SN>, SMatrix<f64, SN, CN>)
where
    SDT: NamedVecOps<Dual64, SN>,
    CDT: NamedVecOps<Dual64, CN>,
    ST: NamedVecOps<f64, SN>,
    CT: NamedVecOps<f64, CN>,
    F: Fn(&SDT, &CDT, &PT) -> SDT,
{
    let mut a = SMatrix::zeros();
    let mut b = SMatrix::zeros();

    let state_svec = state.to_svector();
    let control_svec = control.to_svector();

    let state_dual_svec  = state_svec.map(Dual64::from);
    let control_dual_svec  = control_svec.map(Dual64::from);

    let state_dual = SDT::from_svector(&state_dual_svec);
    let control_dual = CDT::from_svector(&control_dual_svec);

    let make_perturbed_dual_state = |i| {
        let mut perturbed= state_dual_svec.clone();
        perturbed[(i, 0)].eps = 1.0;
        SDT::from_svector(&perturbed)
    };

    let make_perturbed_dual_control = |i| {
        let mut perturbed: SVector<Dual64, CN> = control_dual_svec.clone();
        perturbed[(i, 0)].eps = 1.0;
        CDT::from_svector(&perturbed)
    };

    for i in 0..SN {
        let dfds = dynamics_f(
                &make_perturbed_dual_state(i),
                &control_dual,
                params,
        ).to_svector();
        for j in 0..SN {
            a[(j, i)] = dfds[j].eps;
        }
    }

    for i in 0..CN {
        let dfdc = dynamics_f(
            &state_dual,
            &make_perturbed_dual_control(i),
            params,
        ).to_svector();
        for j in 0..SN {
            b[(j, i)] = dfdc[j].eps;
        }
    }

    (a, b)
}

fn build_time_varying_su<const SN: usize, const CN: usize, const HORIZON: usize>(
    a: &SMatrix<f64, {SN * HORIZON}, SN>,
    b: &SMatrix<f64, {SN * HORIZON}, CN>,
) -> SMatrix<f64, {SN * HORIZON}, {CN * HORIZON}> {
    let mut su = SMatrix::<f64, {SN * HORIZON}, {CN * HORIZON}>::zeros();
    for i in 0..HORIZON {
        let mut prod = SMatrix::<f64, SN, SN>::identity();
        for j in 0..i {
            if j > 0 {
                let block = prod * a.view((SN*(i - j + 1), 0), (SN, SN));
                prod.view_mut((0, 0), (SN, SN)).copy_from(&block);
            }
            let block = prod * b.view((SN*(i - j), 0), (SN, CN));
            su.view_mut((SN * i, CN * j), (SN, CN)).copy_from(&block);
        }
    }
    su
}

fn main() {

    const HORIZON: usize = 10;
    let sim_steps = 30;
    let q = 1.0;
    let r = 0.1;
    let tol = 1e-6;
    let max_iter = 50;

    let mut x: State<f64> = State { x: 0.0, y: 0.0, angle: 0.0 };
    let mut u: Control<f64> = Control { v: 0.0, w: 0.0 };

    let mut u_prev = SVector::<f64, {Control64::SIZE * HORIZON}>::zeros();

    for t in 0..sim_steps {
        let x_ref: Vec<State64> = (0..HORIZON)
            .map(|k| State64 { x: 0.1 * (t + k) as f64, y: 0.1 * (t + k) as f64, angle: 0.0 })
            .collect();

        let u_ref: Vec<Control64> = (0..HORIZON)
            .map(|k| Control64 { v: 0.1, w: 0.0 })
            .collect();

        let p = Parameters {
            ref_x: 0.0,
            ref_y: 0.0,
        };

        let mut a = SMatrix::<f64, {State64::SIZE * HORIZON}, {State64::SIZE}>::zeros();
        let mut b = SMatrix::<f64, {State64::SIZE * HORIZON}, {Control64::SIZE}>::zeros();

        for k in 0..HORIZON {
            let (a_k, b_k) = linearize(
                dynamics,
                &x_ref[k],
                &u_ref[k],
                &p,
            );
            a.view_mut((State64::SIZE * k, 0), (State64::SIZE, State64::SIZE)).copy_from(&a_k);
            b.view_mut((State64::SIZE * k, 0), (State64::SIZE, Control64::SIZE)).copy_from(&b_k);
        }

        let su = build_time_varying_su::<{State64::SIZE}, {Control64::SIZE}, HORIZON>(&a, &b);

        println!("{:?}", su);

    }
}