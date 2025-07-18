
use named_vec_ops::NamedVecOps;
use named_vec_ops_derive::NamedVecOps;
use mpc::Model;
use num_dual::DualNum;
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

struct DiffDriveModel;

impl Model<{State64::SIZE}, {Control64::SIZE}> for DiffDriveModel
{
    type State<T: DualNum<f64> + Copy> = State<T>;
    type Control<T: DualNum<f64> + Copy> = Control<T>;
    type Parameters = Parameters;

    fn dynamics<T: DualNum<f64> + Copy>(
        state: &Self::State<T>,
        control: &Self::Control<T>,
        params: &Self::Parameters,
    ) -> Self::State<T>
    {
        State {
            x: state.angle.cos() * control.v.clone(),
            y: state.angle.sin() * control.v.clone(),
            angle: control.w.clone(),
        }
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

    let mut solver = mpc::MPC::<DiffDriveModel, {State64::SIZE}, {Control64::SIZE}, HORIZON>::new(dt, max_iter, tol);

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
            let angle = 45.0 * std::f64::consts::PI / 180.0;
            let v = 1.0;
            solver.u_ref[i] = Control64 { v: v, w: 0.0 };
            solver.x_ref[i] = State64 { x: dt * v * angle.cos() * ((t + i) as f64), y: dt * v * angle.sin() * ((t + i) as f64), angle: angle };
            solver.p[i] = Parameters::default();
        }

        writeln!(traj_file, "{},{},{},{},{}", t, x0.x, x0.y, solver.x_ref[0].x, solver.x_ref[0].y).unwrap();

        let u0 = solver.solve(&x0);

        let dx = DiffDriveModel::dynamics::<f64>(&x0, &u0, &p);
        x0 += dx * dt;

        println!("Step {}: x = {:?}, u = {:?}", t, x0, u0);    

    }
}

fn main()
{
    mpc1();
}