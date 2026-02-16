use control::Control;
use crystal::Crystal;
use mpi_sys::MPI_COMM_WORLD;

fn main() {
    dwmpi::init();

    let mut exit_code = 0;

    let mut control = Control::new();
    control.read_file("in.ctrl");

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let kpts = kpts::new(control.get_kpts_scheme(), &crystal, control.get_symmetry());

    match wannier90::write_win_inputs(&control, &crystal, kpts.as_ref()) {
        Ok(summary) => {
            if dwmpi::is_root() {
                for filename in summary.written_files.iter() {
                    println!("wrote {}", filename);
                }
            }
        }
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("failed to write Wannier90 .win input: {}", err);
            }
            exit_code = 1;
        }
    }

    dwmpi::barrier(MPI_COMM_WORLD);
    dwmpi::finalize();

    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
