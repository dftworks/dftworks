use control::Control;
use dwconsts::*;

pub(crate) enum ScfStopReason {
    Converged,
    MaxIter,
}

pub(crate) trait ScfIterationAdapter {
    fn prepare_potential_in_rspace(&mut self);
    fn solve_eigen_equations(&mut self, scf_iter: usize, energy_diff: f64) -> f64;
    fn update_occupations(&mut self) -> f64;
    fn compute_harris_energy(&mut self) -> f64;
    fn rebuild_density(&mut self);
    fn compute_charge(&mut self) -> f64;
    fn refresh_energy_terms(&mut self);
    fn compute_scf_energy(&mut self) -> f64;
    fn mix_and_rebuild_potential(&mut self);
}

fn print_iteration_header() {
    if dwmpi::is_root() {
        println!(
            "    {:>3}  {:>10} {:>10} {:>16} {:>25} {:>25} {:>12}",
            "", "eps(eV)", "Fermi(eV)", "charge", "Eharris(Ry)", "Escf(Ry)", "dE(eV)"
        );
    }
}

fn print_iteration_row(
    scf_iter: usize,
    eigvalue_epsilon: f64,
    fermi_level: f64,
    charge: f64,
    energy_harris: f64,
    energy_scf: f64,
    energy_diff: f64,
) {
    if dwmpi::is_root() {
        println!(
            "    {:>3}: {:>10.3E} {:>10.3E} {:>16.6E} {:>25.12E} {:>25.12E} {:>12.3E}",
            scf_iter,
            eigvalue_epsilon * HA_TO_EV,
            fermi_level * HA_TO_EV,
            charge,
            energy_harris * HA_TO_RY,
            energy_scf * HA_TO_RY,
            energy_diff * HA_TO_EV
        );
    }
}

fn print_stop_reason(stop_reason: &ScfStopReason) {
    if !dwmpi::is_root() {
        return;
    }

    let msg = match stop_reason {
        ScfStopReason::Converged => "scf_convergence_success",
        ScfStopReason::MaxIter => "scf_convergence_failure",
    };

    println!("\n     {:<width1$}", msg, width1 = OUT_WIDTH1);
}

pub(crate) fn run_scf_iteration_engine(control: &Control, adapter: &mut dyn ScfIterationAdapter) {
    print_iteration_header();

    let mut scf_iter = 1;
    let mut energy_diff = 0.0;

    loop {
        adapter.prepare_potential_in_rspace();

        let eigvalue_epsilon = adapter.solve_eigen_equations(scf_iter, energy_diff);
        let fermi_level = adapter.update_occupations();
        let energy_harris = adapter.compute_harris_energy();

        adapter.rebuild_density();
        let charge = adapter.compute_charge();

        adapter.refresh_energy_terms();
        let energy_scf = adapter.compute_scf_energy();

        energy_diff = (energy_scf - energy_harris).abs();

        print_iteration_row(
            scf_iter,
            eigvalue_epsilon,
            fermi_level,
            charge,
            energy_harris,
            energy_scf,
            energy_diff,
        );

        if energy_diff < control.get_energy_epsilon() {
            let stop_reason = ScfStopReason::Converged;
            print_stop_reason(&stop_reason);
            break;
        }

        if scf_iter == control.get_scf_max_iter() {
            let stop_reason = ScfStopReason::MaxIter;
            print_stop_reason(&stop_reason);
            break;
        }

        adapter.mix_and_rebuild_potential();
        scf_iter += 1;
    }
}
