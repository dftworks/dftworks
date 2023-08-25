use dwmpi::*;
use types::*;
use mpi_sys::*;

fn main() {
    let mut rank = 0;
    let mut size = 0;

    init();

    comm_rank(MPI_COMM_WORLD, &mut rank);
    comm_size(MPI_COMM_WORLD, &mut size);

    if rank == MPI_ROOT {
        println!("send and receive scalar values");

        for dest in 1..size {
            send_scalar(&(dest as f64 * 1.1), dest, 1, MPI_COMM_WORLD);
        }
    } else {
        let mut buf: f64 = 1.0;

        recv_scalar(&mut buf, 0, 1, MPI_COMM_WORLD);

        println!("node {:3} received {:6.3} from 0", rank, buf);
    }

    barrier(MPI_COMM_WORLD);

    if rank == MPI_ROOT {
        println!("send and receive slice");

        for dest in 1..size {
            let buf: [c64; 10] = [c64 {
                re: 3.0,
                im: dest as f64,
            }; 10];

            send_slice(&buf[1..], dest, 1, MPI_COMM_WORLD);
        }
    } else {
        let mut buf: [c64; 10] = [c64 { re: 0.0, im: 0.0 }; 10];

        recv_slice(&mut buf[1..], 0, 1, MPI_COMM_WORLD);

        println!("node {:3} received {:#?} from 0", rank, buf);
    }

    let mut bb: f64 = 0.0;

    if rank == MPI_ROOT {
        bb = 15.0;
    }

    bcast_scalar(&bb, MPI_COMM_WORLD);

    println!(" bb = {}", bb);

    barrier(MPI_COMM_WORLD);

    let mut xx: [f64; 10] = [0.0; 10];
    if rank == MPI_ROOT {
        for i in 0..10 {
            xx[i] = i as f64;
        }
    }

    bcast_slice(&xx, MPI_COMM_WORLD);

    barrier(MPI_COMM_WORLD);

    println!(" xx = {:#?}", xx);

    finalize();
}
