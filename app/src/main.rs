
use nalgebra::SMatrix;
use named_vec_ops::NamedVecOps;
use named_vec_ops_derive::NamedVecOps;
use num_dual::{DualNum, Dual64};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Copy, Clone, NamedVecOps)]
struct State<T>
{
    x: T,
    y: T,
    angle: T,
}

type State64 = State<f64>;

#[derive(Debug, Copy, Clone, NamedVecOps)]
struct Control<T>
{
    v: T,
    w: T,
}

type Control64 = Control<f64>;

#[derive(Debug, Copy, Clone, Default)]
struct Parameters
{
    // ref_x: f64,
    // ref_y: f64,
}

fn dynamics<T: DualNum<f64> + Clone>(
    state: &State<T>,
    control: &Control<T>,
    _params: &Parameters,
) -> State<T>
{
    State {
        x: state.angle.cos() * control.v.clone(),
        y: state.angle.sin() * control.v.clone(),
        angle: control.w.clone(),
    }
}

fn mpc1()
{
    let mut traj_file = File::create("trajectory.csv").expect("Could not create file");
    writeln!(traj_file, "step,x,y,rx,ry").unwrap();

    const HORIZON: usize = 50;

    let dt = 0.1;
    let max_iter: usize = 50;
    let tol = 1e-6;

    let sim_steps: usize = 200;

    let mut solver = mpc::MPC::<{State64::SIZE}, {Control64::SIZE}, HORIZON, State64, Control64, Parameters>::new(dt, max_iter, tol);

    for i in 0..HORIZON {
        solver.q[i][(0, 0)] = 1.0; // State cost for x
        solver.q[i][(1, 1)] = 1.0; // State cost for y
        solver.q[i][(2, 2)] = 0.1; // State cost for angle
        solver.r[i][(0, 0)] = 0.01; // Control cost for v
        solver.r[i][(1, 1)] = 2.0; // Control cost for w
    }

    let mut x0 = State64 { x: 0.0, y: 0.0, angle: 20.0 * std::f64::consts::PI / 180.0 };
    let p = Parameters::default();

    for t in 0..sim_steps {

        for i in 0..HORIZON {
            solver.x_ref[i] = State64 { x: 0.1 * ((t + i) as f64), y: 0.1 * ((t + i) as f64), angle: 45.0 * std::f64::consts::PI / 180.0 };
            solver.u_ref[i] = Control64 { v: 1.0, w: 0.0 };
            solver.p[i] = Parameters::default();
        }

        writeln!(traj_file, "{},{},{},{},{}", t, x0.x, x0.y, solver.x_ref[0].x, solver.x_ref[0].y).unwrap();

        let u0 = solver.solve(&dynamics::<f64>, &dynamics::<Dual64>, &x0);

        let dx = dynamics(&x0, &u0, &p);
        x0 += dx * dt;

        println!("Step {}: x = {:?}, u = {:?}", t, x0, u0);    

    }
}

fn main()
{
    mpc1();
}