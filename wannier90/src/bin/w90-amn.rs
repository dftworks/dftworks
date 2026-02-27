use control::Control;
use crystal::Crystal;
use mpi_sys::MPI_COMM_WORLD;

fn main() {
    dwmpi::init();

    let mut exit_code = 0;

    let control = match Control::from_file("in.ctrl") {
        Ok(control) => control,
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("failed to load control file: {}", err);
            }
            dwmpi::barrier(MPI_COMM_WORLD);
            dwmpi::finalize();
            std::process::exit(1);
        }
    };

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let kpts = match kpts::try_new(
        control.get_kpts_scheme_enum(),
        &crystal,
        control.get_symmetry(),
    ) {
        Ok(kpts) => Some(kpts),
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("failed to initialize k-points: {}", err);
            }
            exit_code = 1;
            None
        }
    };

    if let Some(kpts) = kpts.as_ref() {
        if exit_code == 0 {
            match wannier90::write_overlap_inputs(&control, &crystal, kpts.as_ref()) {
                Ok(summary) => {
                    if dwmpi::is_root() {
                        for filename in summary.written_files.iter() {
                            println!("wrote {}", filename);
                        }
                    }
                }
                Err(err) => {
                    if dwmpi::is_root() {
                        eprintln!("failed to write Wannier90 overlap files: {}", err);
                    }
                    exit_code = 1;
                }
            }
        }
    }

    dwmpi::barrier(MPI_COMM_WORLD);
    dwmpi::finalize();

    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
