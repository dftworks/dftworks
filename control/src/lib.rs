use dwconsts::*;

use std::{
    fs::File,
    io::{BufRead, BufReader},
    ops::Not,
};

#[derive(Debug, Default)]
pub struct Control {
    verbosity: String,
    ecut_wfc: f64,
    ecut_rho: f64,

    geom_optim_cell: bool,
    geom_optim_scheme: String,
    geom_optim_max_steps: usize,
    geom_optim_history_steps: usize,
    geom_optim_alpha: f64,
    geom_optim_force_tolerance: f64,
    geom_optim_stress_tolerance: f64,

    scf_harris: bool,
    scf_max_iter: usize,
    scf_min_iter: usize,
    scf_max_iter_rand_wfc: usize,
    scf_max_iter_wfc: usize,
    scf_rho_mix_scheme: String,
    scf_rho_mix_alpha: f64, // old*alpha + new*(1-alpha)
    scf_rho_mix_beta: f64,  // alpha * G2/(G2+beta)
    scf_rho_mix_history_steps: usize,
    scf_rho_mix_pulay_metric_weight: f64,
    rho_epsilon: f64,
    energy_epsilon: f64,

    eigval_epsilon: f64,

    eigval_same_epsilon: bool,

    nband: usize,

    spin_scheme: String, // nonspin, spin, ncl
    task: String,        // scf, band
    restart: bool,
    save_rho: bool,
    save_wfc: bool,
    eigen_solver: String, // sd, psd, cg, pcg, arpack, davidson

    davidson_ndim: usize,

    temperature: f64,
    smearing_scheme: String,
    xc_scheme: String,
    pot_scheme: String,
    output_atomic_density: bool,

    dos_scheme: String,
    dos_sigma: f64,
    dos_ne: usize,

    kpts_scheme: String,

    symmetry: bool,

    occ_inversion: f64,
}

impl Control {
    pub fn new() -> Control {
        Control::default()
    }

    pub fn get_symmetry(&self) -> bool {
        self.symmetry
    }

    pub fn is_scf_harris(&self) -> bool {
        self.scf_harris
    }

    pub fn get_output_atomic_density(&self) -> bool {
        self.output_atomic_density
    }

    //pub fn is_scf(&self) -> bool {
    //    "scf" == self.get_task()
    //}

    //pub fn is_band(&self) -> bool {
    //    "band" == self.get_task()
    //}

    pub fn is_spin(&self) -> bool {
        "spin" == self.get_spin_scheme()
    }

    //pub fn is_nscf(&self) -> bool {
    //    "nscf" == self.get_task()
    //}

    pub fn get_dos_scheme(&self) -> &str {
        &self.dos_scheme
    }

    pub fn get_dos_sigma(&self) -> f64 {
        self.dos_sigma
    }

    pub fn get_dos_ne(&self) -> usize {
        self.dos_ne
    }

    pub fn get_kpts_scheme(&self) -> &str {
        &self.kpts_scheme
    }

    pub fn get_scf_min_iter(&self) -> usize {
        self.scf_min_iter
    }

    pub fn get_scf_max_iter_wfc(&self) -> usize {
        self.scf_max_iter_wfc
    }

    pub fn get_scf_max_iter_rand_wfc(&self) -> usize {
        self.scf_max_iter_rand_wfc
    }

    pub fn get_pot_scheme(&self) -> &str {
        &self.pot_scheme
    }

    pub fn get_xc_scheme(&self) -> &str {
        &self.xc_scheme
    }

    pub fn get_geom_optim_cell(&self) -> bool {
        self.geom_optim_cell
    }

    pub fn get_geom_optim_scheme(&self) -> &str {
        &self.geom_optim_scheme
    }

    pub fn get_geom_optim_max_steps(&self) -> usize {
        self.geom_optim_max_steps
    }

    pub fn get_geom_optim_history_steps(&self) -> usize {
        self.geom_optim_history_steps
    }

    pub fn get_geom_optim_alpha(&self) -> f64 {
        self.geom_optim_alpha
    }

    pub fn get_geom_optim_force_tolerance(&self) -> f64 {
        self.geom_optim_force_tolerance
    }

    pub fn get_geom_optim_stress_tolerance(&self) -> f64 {
        self.geom_optim_stress_tolerance
    }

    pub fn get_smearing_scheme(&self) -> &str {
        &self.smearing_scheme
    }

    pub fn get_temperature(&self) -> f64 {
        self.temperature
    }

    pub fn get_scf_max_iter(&self) -> usize {
        self.scf_max_iter
    }

    pub fn get_ecut(&self) -> f64 {
        self.ecut_wfc
    }

    pub fn get_ecutrho(&self) -> f64 {
        self.ecut_rho
    }

    pub fn get_nband(&self) -> usize {
        self.nband
    }

    pub fn get_scf_rho_mix_scheme(&self) -> &str {
        &self.scf_rho_mix_scheme
    }

    pub fn get_scf_rho_mix_history_steps(&self) -> usize {
        self.scf_rho_mix_history_steps
    }

    pub fn get_scf_rho_mix_pulay_metric_weight(&self) -> f64 {
        self.scf_rho_mix_pulay_metric_weight
    }

    pub fn get_scf_rho_mix_beta(&self) -> f64 {
        self.scf_rho_mix_beta
    }

    pub fn get_scf_rho_mix_alpha(&self) -> f64 {
        self.scf_rho_mix_alpha
    }

    pub fn get_energy_epsilon(&self) -> f64 {
        self.energy_epsilon
    }

    pub fn get_rho_epsilon(&self) -> f64 {
        self.rho_epsilon
    }

    pub fn get_eigval_epsilon(&self) -> f64 {
        self.eigval_epsilon
    }

    pub fn get_spin_scheme(&self) -> &str {
        &self.spin_scheme
    }

    pub fn get_task(&self) -> &str {
        &self.task
    }

    pub fn get_eigen_solver(&self) -> &str {
        &self.eigen_solver
    }

    pub fn get_davidson_ndim(&self) -> usize {
        self.davidson_ndim
    }

    // the setting of task will override some parameters read from in.ctrl

    pub fn set_task(&mut self, task: &str) {
        self.task = task.to_string();

        match task {
            "scf" => {
                //self.restart = false;
            }
            "band" => {
                self.restart = true;
                self.save_rho = false;
                self.save_wfc = false;
                //self.scf_max_iter = 1;
            }

            _ => {}
        }
    }

    pub fn get_verbosity(&self) -> &str {
        &self.verbosity
    }

    pub fn get_restart(&self) -> bool {
        self.restart
    }

    pub fn get_save_rho(&self) -> bool {
        self.save_rho
    }

    pub fn get_save_wfc(&self) -> bool {
        self.save_wfc
    }

    pub fn get_occ_inversion(&self) -> f64 {
        self.occ_inversion
    }

    pub fn read_file(&mut self, inpfile: &str) {
        self.task = "scf".to_string();

        self.spin_scheme = "nonspin".to_string();

        self.geom_optim_cell = false;
        self.geom_optim_scheme = "bfgs".to_string();
        self.geom_optim_max_steps = 1;
        self.geom_optim_history_steps = 4;
        self.geom_optim_alpha = 0.8;
        self.geom_optim_force_tolerance = 0.01; // eV / A
        self.geom_optim_stress_tolerance = 0.05; // kbar

        self.save_rho = true;
        self.save_wfc = true;

        self.ecut_wfc = 400.0 * EV_TO_HA; // ev in in.ctrl, need to convert to HA
        self.ecut_rho = 4.0 * self.ecut_wfc;

        self.eigval_same_epsilon = false;
        self.eigval_epsilon = 1.0E-6; // ev
        self.rho_epsilon = 1.0E-5;

        self.energy_epsilon = EPS6; //eV

        self.eigen_solver = "pcg".to_string();

        self.davidson_ndim = 6;

        self.scf_rho_mix_scheme = "simple".to_string();
        self.scf_rho_mix_alpha = 0.7;
        self.scf_rho_mix_beta = 0.5;
        self.scf_rho_mix_history_steps = 4;
        self.scf_rho_mix_pulay_metric_weight = 20.0;

        self.scf_harris = false;
        self.scf_min_iter = 1;
        self.scf_max_iter = 60;
        self.scf_max_iter_rand_wfc = 1;
        self.scf_max_iter_wfc = 30;

        self.dos_scheme = "gauss".to_string();
        self.dos_ne = 500;
        self.dos_sigma = 0.1; // eV

        self.kpts_scheme = "kmesh".to_string(); // "kmesh", "kpath", "klist"

        self.verbosity = "high".to_string();

        self.occ_inversion = 0.0;

        let mut b_has_invalid_parameter = false;

        let mut b_ecut_rho_set = false;

        let lines = self.read_file_data_to_vec(inpfile);

        for line in lines.iter() {
            let s: Vec<&str> = line.split('=').map(|x| x.trim()).collect();

            match s[0] {
                "verbosity" => {
                    self.verbosity = s[1].parse().unwrap();
                }

                "scf_harris" => {
                    self.scf_harris = s[1].parse().unwrap();
                }

                "scf_max_iter_wfc" => {
                    self.scf_max_iter_wfc = s[1].parse().unwrap();
                }

                "scf_max_iter_rand_wfc" => {
                    self.scf_max_iter_rand_wfc = s[1].parse().unwrap();
                }

                "pot_scheme" => {
                    self.pot_scheme = s[1].parse().unwrap();
                }

                "output_atomic_density" => {
                    self.output_atomic_density = s[1].parse().unwrap();
                }

                "xc_scheme" => {
                    self.xc_scheme = s[1].parse().unwrap();
                }

                "geom_optim_cell" => {
                    self.geom_optim_cell = s[1].parse().unwrap();
                }

                "geom_optim_scheme" => {
                    self.geom_optim_scheme = s[1].parse().unwrap();
                }

                "geom_optim_max_steps" => {
                    self.geom_optim_max_steps = s[1].parse().unwrap();
                }

                "geom_optim_history_steps" => {
                    self.geom_optim_history_steps = s[1].parse().unwrap();
                }

                "geom_optim_alpha" => {
                    self.geom_optim_alpha = s[1].parse().unwrap();
                }

                "geom_optim_force_tolerance" => {
                    self.geom_optim_force_tolerance = s[1].parse().unwrap();
                }

                "geom_optim_stress_tolerance" => {
                    self.geom_optim_stress_tolerance = s[1].parse().unwrap();
                }

                "smearing_scheme" => {
                    self.smearing_scheme = s[1].parse().unwrap();
                }

                "temperature" => {
                    self.temperature = s[1].parse().unwrap();
                }

                "ecut_wfc" => {
                    self.ecut_wfc = s[1].parse::<f64>().unwrap() * EV_TO_HA;
                }

                "ecut_rho" => {
                    self.ecut_rho = s[1].parse::<f64>().unwrap() * EV_TO_HA;
                    b_ecut_rho_set = true;
                }

                "scf_min_iter" => {
                    self.scf_min_iter = s[1].parse().unwrap();
                }

                "scf_max_iter" => {
                    self.scf_max_iter = s[1].parse().unwrap();
                }

                "nband" => {
                    self.nband = s[1].parse().unwrap();
                }

                "scf_rho_mix_alpha" => {
                    self.scf_rho_mix_alpha = s[1].parse().unwrap();
                }

                "scf_rho_mix_beta" => {
                    self.scf_rho_mix_beta = s[1].parse().unwrap();
                }

                "energy_epsilon" => {
                    self.energy_epsilon = s[1].parse::<f64>().unwrap() * EV_TO_HA;
                }

                "rho_epsilon" => {
                    self.rho_epsilon = s[1].parse().unwrap();
                }

                "scf_rho_mix_scheme" => {
                    self.scf_rho_mix_scheme = s[1].parse().unwrap();
                }

                "scf_rho_mix_history_steps" => {
                    self.scf_rho_mix_history_steps = s[1].parse().unwrap();
                }

                "scf_rho_mix_pulay_metric_weight" => {
                    self.scf_rho_mix_pulay_metric_weight = s[1].parse().unwrap();
                }

                "eigval_epsilon" => {
                    self.eigval_epsilon = s[1].parse::<f64>().unwrap() * EV_TO_HA;
                }

                "eigval_same_epsilon" => {
                    self.eigval_same_epsilon = s[1].parse().unwrap();
                }

                "spin_scheme" => {
                    self.spin_scheme = s[1].parse().unwrap();
                }

                "task" => {
                    self.task = s[1]
                        .parse::<String>()
                        .unwrap()
                        .trim()
                        .to_lowercase()
                        .to_string();
                }

                "restart" => {
                    self.restart = s[1].parse().unwrap();
                }

                "save_rho" => {
                    self.save_rho = s[1].parse().unwrap();
                }

                "save_wfc" => {
                    self.save_wfc = s[1].parse().unwrap();
                }

                "eigen_solver" => {
                    self.eigen_solver = s[1].parse().unwrap();
                }

                "davidson_ndim" => {
                    self.davidson_ndim = s[1].parse().unwrap();
                }

                "dos_scheme" => {
                    self.dos_scheme = s[1].parse().unwrap();
                }

                "dos_sigma" => {
                    self.dos_sigma = s[1].parse::<f64>().unwrap() * EV_TO_HA;
                }

                "dos_ne" => {
                    self.dos_ne = s[1].parse().unwrap();
                }

                "kpts_scheme" => {
                    self.kpts_scheme = s[1].parse().unwrap();
                }

                "occ_inversion" => {
                    self.occ_inversion = s[1].parse::<f64>().unwrap();
                }

                "symmetry" => {
                    self.symmetry = s[1].parse().unwrap();
                    //println!(" symmetry = {}", self.get_symmetry());
                    //println!(" {:?}", s);
                }

                "" => {}

                _ => {
                    println!("unknown parameter : {}", line);
                    b_has_invalid_parameter = true;
                }
            }
        }

        if b_has_invalid_parameter {
            println!("Program exited abnormally");

            std::process::exit(-1);
        }

        // if the ecut_rho is not set, 4.0 * ecut_wfc is ok with norm-conserving pseudopotentials

        if b_ecut_rho_set.not() {
            self.ecut_rho = 4.0 * self.ecut_wfc;
        }
    }

    pub fn read_file_data_to_vec(&mut self, pspfile: &str) -> Vec<String> {
        let file = File::open(pspfile).unwrap();

        let lines = BufReader::new(file).lines();

        let lines: Vec<String> = lines.filter_map(std::io::Result::ok).collect();

        lines
    }

    pub fn display(&self) {
        const OUT_WIDTH1: usize = 28;
        const OUT_WIDTH2: usize = 18;

        println!("   {:-^80}", " control parameters ");
        println!();

        println!(
            "   {:<width1$} = {:>width2$}",
            "restart",
            self.get_restart(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "spin_scheme",
            self.get_spin_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "pot_scheme",
            self.get_pot_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} ",
            "eigen_solver",
            self.get_eigen_solver(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.3E} eV",
            "energy_conv_eps",
            self.get_energy_epsilon() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$.3E} eV",
            "eig_conv_eps",
            self.get_eigval_epsilon() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_harris",
            self.scf_harris,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_max_iter",
            self.get_scf_max_iter(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_min_iter",
            self.get_scf_min_iter(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "smearing_scheme",
            self.get_smearing_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$} K",
            "temperature",
            self.get_temperature(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.3} eV",
            "ecut",
            self.get_ecut() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$.3} eV",
            "ecutrho",
            self.get_ecutrho() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "nband",
            self.get_nband(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_rho_mix_scheme",
            self.get_scf_rho_mix_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_rho_mix_alpha",
            self.get_scf_rho_mix_alpha(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_rho_mix_beta",
            self.get_scf_rho_mix_beta(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_cell",
            self.get_geom_optim_cell(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_scheme",
            self.get_geom_optim_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_history_steps",
            self.get_geom_optim_history_steps(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_max_steps",
            self.get_geom_optim_max_steps(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_alpha",
            self.get_geom_optim_alpha(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} eV/A",
            "geom_optim_force_tolerance",
            self.get_geom_optim_force_tolerance(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} kbar",
            "geom_optim_stress_tolerance",
            self.get_geom_optim_stress_tolerance(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!();
    }
}
