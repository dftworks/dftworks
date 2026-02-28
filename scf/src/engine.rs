use control::{Control, ScfLogFormat, VerbosityLevel};
use dwconsts::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::Instant;

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

fn escape_json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

struct IterationTimings {
    t_prepare_s: f64,
    t_solve_s: f64,
    t_occ_s: f64,
    t_harris_s: f64,
    t_density_s: f64,
    t_charge_s: f64,
    t_refresh_s: f64,
    t_scf_s: f64,
    t_mix_s: f64,
    t_iter_s: f64,
}

struct StructuredScfLogger {
    format: ScfLogFormat,
    writer: BufWriter<std::fs::File>,
}

impl StructuredScfLogger {
    fn new(control: &Control) -> Option<Self> {
        if !dwmpi::is_root() {
            return None;
        }

        let format = control.get_scf_log_format_enum();
        if matches!(format, ScfLogFormat::None) {
            return None;
        }

        let path = control.get_scf_log_file().trim();
        if path.is_empty() {
            return None;
        }

        let file = match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("warning: failed to open structured SCF log '{}': {}", path, err);
                return None;
            }
        };
        let write_csv_header = file.metadata().map(|m| m.len() == 0).unwrap_or(true);

        let mut logger = Self {
            format,
            writer: BufWriter::new(file),
        };

        if matches!(format, ScfLogFormat::Csv) && write_csv_header {
            let header = "iter,eps_ev,fermi_ev,charge,eharris_ry,escf_ry,de_ev,converged,stop_reason,t_prepare_s,t_solve_s,t_occ_s,t_harris_s,t_density_s,t_charge_s,t_refresh_s,t_scf_s,t_mix_s,t_iter_s\n";
            if let Err(err) = logger.writer.write_all(header.as_bytes()) {
                eprintln!(
                    "warning: failed to write structured SCF CSV header '{}': {}",
                    path, err
                );
                return None;
            }
        }

        Some(logger)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_iteration(
        &mut self,
        scf_iter: usize,
        eigvalue_epsilon: f64,
        fermi_level: f64,
        charge: f64,
        energy_harris: f64,
        energy_scf: f64,
        energy_diff: f64,
        converged: bool,
        stop_reason: Option<&str>,
        timings: &IterationTimings,
    ) {
        let write_result = match self.format {
            ScfLogFormat::None => Ok(()),
            ScfLogFormat::Jsonl => {
                let reason = stop_reason.unwrap_or("");
                let line = format!(
                    "{{\"iter\":{},\"eps_ev\":{:.16e},\"fermi_ev\":{:.16e},\"charge\":{:.16e},\"eharris_ry\":{:.16e},\"escf_ry\":{:.16e},\"de_ev\":{:.16e},\"converged\":{},\"stop_reason\":\"{}\",\"t_prepare_s\":{:.6e},\"t_solve_s\":{:.6e},\"t_occ_s\":{:.6e},\"t_harris_s\":{:.6e},\"t_density_s\":{:.6e},\"t_charge_s\":{:.6e},\"t_refresh_s\":{:.6e},\"t_scf_s\":{:.6e},\"t_mix_s\":{:.6e},\"t_iter_s\":{:.6e}}}\n",
                    scf_iter,
                    eigvalue_epsilon * HA_TO_EV,
                    fermi_level * HA_TO_EV,
                    charge,
                    energy_harris * HA_TO_RY,
                    energy_scf * HA_TO_RY,
                    energy_diff * HA_TO_EV,
                    converged,
                    escape_json_string(reason),
                    timings.t_prepare_s,
                    timings.t_solve_s,
                    timings.t_occ_s,
                    timings.t_harris_s,
                    timings.t_density_s,
                    timings.t_charge_s,
                    timings.t_refresh_s,
                    timings.t_scf_s,
                    timings.t_mix_s,
                    timings.t_iter_s
                );
                self.writer.write_all(line.as_bytes())
            }
            ScfLogFormat::Csv => {
                let reason = stop_reason.unwrap_or("");
                let line = format!(
                    "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},\"{}\",{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
                    scf_iter,
                    eigvalue_epsilon * HA_TO_EV,
                    fermi_level * HA_TO_EV,
                    charge,
                    energy_harris * HA_TO_RY,
                    energy_scf * HA_TO_RY,
                    energy_diff * HA_TO_EV,
                    converged,
                    reason.replace('"', "'"),
                    timings.t_prepare_s,
                    timings.t_solve_s,
                    timings.t_occ_s,
                    timings.t_harris_s,
                    timings.t_density_s,
                    timings.t_charge_s,
                    timings.t_refresh_s,
                    timings.t_scf_s,
                    timings.t_mix_s,
                    timings.t_iter_s
                );
                self.writer.write_all(line.as_bytes())
            }
        };

        if let Err(err) = write_result {
            eprintln!("warning: failed to write structured SCF log record: {}", err);
            return;
        }

        if let Err(err) = self.writer.flush() {
            eprintln!("warning: failed to flush structured SCF log: {}", err);
        }
    }
}

pub(crate) fn run_scf_iteration_engine(control: &Control, adapter: &mut dyn ScfIterationAdapter) {
    let verbosity = control.get_verbosity_enum();
    let show_table = verbosity >= VerbosityLevel::Normal;
    let show_timing_rows = verbosity >= VerbosityLevel::Debug;

    if show_table {
        print_iteration_header();
    }

    let mut structured_logger = StructuredScfLogger::new(control);

    let mut scf_iter = 1;
    let mut energy_diff = 0.0;

    loop {
        let t_iter = Instant::now();

        let t0 = Instant::now();
        adapter.prepare_potential_in_rspace();
        let t_prepare_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let eigvalue_epsilon = adapter.solve_eigen_equations(scf_iter, energy_diff);
        let t_solve_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let fermi_level = adapter.update_occupations();
        let t_occ_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let energy_harris = adapter.compute_harris_energy();
        let t_harris_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        adapter.rebuild_density();
        let t_density_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let charge = adapter.compute_charge();
        let t_charge_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        adapter.refresh_energy_terms();
        let t_refresh_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let energy_scf = adapter.compute_scf_energy();
        let t_scf_s = t0.elapsed().as_secs_f64();

        energy_diff = (energy_scf - energy_harris).abs();

        if show_table {
            print_iteration_row(
                scf_iter,
                eigvalue_epsilon,
                fermi_level,
                charge,
                energy_harris,
                energy_scf,
                energy_diff,
            );
        }

        let converged = energy_diff < control.get_energy_epsilon();
        let hit_max_iter = scf_iter == control.get_scf_max_iter();
        let (stop_reason_opt, t_mix_s) = if converged {
            (Some(ScfStopReason::Converged), 0.0)
        } else if hit_max_iter {
            (Some(ScfStopReason::MaxIter), 0.0)
        } else {
            let t0 = Instant::now();
            adapter.mix_and_rebuild_potential();
            (None, t0.elapsed().as_secs_f64())
        };

        let timings = IterationTimings {
            t_prepare_s,
            t_solve_s,
            t_occ_s,
            t_harris_s,
            t_density_s,
            t_charge_s,
            t_refresh_s,
            t_scf_s,
            t_mix_s,
            t_iter_s: t_iter.elapsed().as_secs_f64(),
        };

        if show_timing_rows && dwmpi::is_root() {
            println!(
                "      timing(iter={}): prep={:.3e}s solve={:.3e}s occ={:.3e}s harris={:.3e}s dens={:.3e}s charge={:.3e}s refresh={:.3e}s escf={:.3e}s mix={:.3e}s total={:.3e}s",
                scf_iter,
                timings.t_prepare_s,
                timings.t_solve_s,
                timings.t_occ_s,
                timings.t_harris_s,
                timings.t_density_s,
                timings.t_charge_s,
                timings.t_refresh_s,
                timings.t_scf_s,
                timings.t_mix_s,
                timings.t_iter_s
            );
        }

        if let Some(logger) = structured_logger.as_mut() {
            let stop_reason_text = match stop_reason_opt {
                Some(ScfStopReason::Converged) => Some("converged"),
                Some(ScfStopReason::MaxIter) => Some("max_iter"),
                None => None,
            };
            logger.write_iteration(
                scf_iter,
                eigvalue_epsilon,
                fermi_level,
                charge,
                energy_harris,
                energy_scf,
                energy_diff,
                converged,
                stop_reason_text,
                &timings,
            );
        }

        if let Some(stop_reason) = stop_reason_opt {
            if verbosity >= VerbosityLevel::Normal {
                print_stop_reason(&stop_reason);
            }
            break;
        } else {
            scf_iter += 1;
        }
    }
}
