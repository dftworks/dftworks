use dwmpi::*;
use types::*;

fn main() {
    init();

    let rank = get_comm_world_rank();
    let size = get_comm_world_size();

    if is_root() {
        println!("send and receive scalar values");

        for dest in 1..size {
            send_scalar(&(dest as f64 * 1.1), dest, 1, comm_world());
        }
    } else {
        let mut buf: f64 = 1.0;

        recv_scalar(&mut buf, 0, 1, comm_world());

        println!("node {:3} received {:6.3} from 0", rank, buf);
    }

    barrier(comm_world());

    if is_root() {
        println!("send and receive slice");

        for dest in 1..size {
            let buf: [c64; 10] = [c64 {
                re: 3.0,
                im: dest as f64,
            }; 10];

            send_slice(&buf[1..], dest, 1, comm_world());
        }
    } else {
        let mut buf: [c64; 10] = [c64 { re: 0.0, im: 0.0 }; 10];

        recv_slice(&mut buf[1..], 0, 1, comm_world());

        println!("node {:3} received {:#?} from 0", rank, buf);
    }

    let mut bb: f64 = 0.0;

    if is_root() {
        bb = 15.0;
    }

    bcast_scalar(&mut bb, comm_world());

    println!(" bb = {}", bb);

    barrier(comm_world());

    let mut xx: [f64; 10] = [0.0; 10];
    if is_root() {
        for i in 0..10 {
            xx[i] = i as f64;
        }
    }

    bcast_slice(&mut xx, comm_world());

    barrier(comm_world());

    println!(" xx = {:#?}", xx);

    finalize();
}
