use control::{Control, FftPlannerScheme, SpinScheme, VerbosityLevel};
use crystal::Crystal;
use kpts::KPTS;
use mpi_sys::MPI_COMM_WORLD;
use pspot::PSPot;

pub(crate) struct BootstrapData {
    pub control: Control,
    pub crystal: Crystal,
    pub pots: PSPot,
    pub kpts: Box<dyn KPTS>,
    pub zions: Vec<f64>,
    pub verbosity: VerbosityLevel,
    pub spin_scheme: SpinScheme,
}

pub(crate) fn load_bootstrap_inputs() -> Result<BootstrapData, String> {
    let control =
        Control::from_file("in.ctrl").map_err(|err| format!("failed to load control file: {}", err))?;
    let verbosity = control.get_verbosity_enum();

    dwfft3d::init_backend();
    let fft_planning_mode = match control.get_fft_planner_enum() {
        FftPlannerScheme::Estimate => dwfft3d::FftPlanningMode::Estimate,
        FftPlannerScheme::Measure => dwfft3d::FftPlanningMode::Measure,
    };
    let fft_wisdom_file = control.get_fft_wisdom_file().trim();
    dwfft3d::configure_runtime(dwfft3d::BackendOptions {
        threads: control.get_fft_threads(),
        planning_mode: fft_planning_mode,
        wisdom_file: if fft_wisdom_file.is_empty() {
            None
        } else {
            Some(fft_wisdom_file.to_string())
        },
    });

    if dwmpi::is_root() && verbosity >= VerbosityLevel::Normal {
        crate::runtime_display::display_program_header();
        crate::runtime_display::display_system_information();
        control.display();
    }

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let pots = PSPot::new(control.get_pot_scheme_enum());
    if dwmpi::is_root() && verbosity >= VerbosityLevel::Normal {
        pots.display();
    }

    let zions = crystal.get_zions(&pots);

    let kpts = kpts::try_new(
        control.get_kpts_scheme_enum(),
        &crystal,
        control.get_symmetry(),
    )
    .map_err(|err| format!("failed to initialize k-points: {}", err))?;

    if dwmpi::is_root() && verbosity >= VerbosityLevel::Normal {
        kpts.display();
    }

    let mut provenance_status: i32 = 0;
    if dwmpi::is_root() {
        if let Err(err) = crate::provenance::emit_run_provenance_manifest(&control, kpts.as_ref()) {
            eprintln!("failed to write/verify provenance manifest: {}", err);
            provenance_status = 1;
        } else if verbosity >= VerbosityLevel::Normal {
            println!("   provenance_manifest = {}", control.get_provenance_manifest());
        }
    }
    dwmpi::bcast_scalar(&mut provenance_status, MPI_COMM_WORLD);
    if provenance_status != 0 {
        return Err("failed to initialize run provenance manifest".to_string());
    }

    let spin_scheme = control.get_spin_scheme_enum();

    Ok(BootstrapData {
        control,
        crystal,
        pots,
        kpts,
        zions,
        verbosity,
        spin_scheme,
    })
}
