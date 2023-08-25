
use mpi_sys::*;
use std::{os::raw::*, ptr};
use types::*;

pub trait MPIDataType {
    fn get_mpi_data_type(&self) -> i32;
}

impl MPIDataType for c64 {
    fn get_mpi_data_type(&self) -> i32 {
        MPI_DOUBLE_COMPLEX
    }
}

impl MPIDataType for f64 {
    fn get_mpi_data_type(&self) -> i32 {
        MPI_DOUBLE
    }
}

impl MPIDataType for i32 {
    fn get_mpi_data_type(&self) -> i32 {
        MPI_INT
    }
}

impl MPIDataType for char {
    fn get_mpi_data_type(&self) -> i32 {
        MPI_CHAR
    }
}

pub fn init() -> i32 {
    unsafe { MPI_Init(ptr::null(), ptr::null()) }
}

pub fn finalize() -> i32 {
    unsafe { MPI_Finalize() }
}

pub fn comm_rank(comm: MpiComm, rank: &mut i32) -> i32 {
    unsafe { MPI_Comm_rank(comm, rank) }
}

pub fn comm_size(comm: MpiComm, size: &mut i32) -> i32 {
    unsafe { MPI_Comm_size(comm, size) }
}

pub fn send_scalar<T: MPIDataType>(buf: &T, dest: i32, tag: i32, comm: MpiComm) -> i32 {
    unsafe {
        MPI_Send(
            buf as *const T as *const c_void,
            1,
            buf.get_mpi_data_type(),
            dest,
            tag,
            comm,
        )
    }
}

pub fn recv_scalar<T: MPIDataType>(buf: &mut T, source: i32, tag: i32, comm: MpiComm) -> i32 {
    unsafe {
        let mut status: MPI_Status = Default::default();

        MPI_Recv(
            buf as *mut T as *mut c_void,
            1,
            buf.get_mpi_data_type(),
            source,
            tag,
            comm,
            &mut status,
        )
    }
}

pub fn send_slice<T: MPIDataType + Default>(buf: &[T], dest: i32, tag: i32, comm: MpiComm) -> i32 {
    unsafe {
        let t: T = Default::default();

        MPI_Send(
            buf.as_ptr() as *const c_void,
            buf.len() as i32,
            t.get_mpi_data_type(),
            dest,
            tag,
            comm,
        )
    }
}

pub fn recv_slice<T: MPIDataType + Default>(
    buf: &mut [T],
    source: i32,
    tag: i32,
    comm: MpiComm,
) -> i32 {
    unsafe {
        let mut status: MPI_Status = Default::default();
        let t: T = Default::default();

        MPI_Recv(
            buf.as_mut_ptr() as *mut c_void,
            buf.len() as i32,
            t.get_mpi_data_type(),
            source,
            tag,
            comm,
            &mut status,
        )
    }
}

pub fn barrier(comm: MpiComm) -> i32 {
    unsafe { MPI_Barrier(comm) }
}

pub fn bcast_scalar<T: MPIDataType>(buf: &T, comm: MpiComm) -> i32 {
    unsafe {
        MPI_Bcast(
            buf as *const T as *const c_void,
            1,
            buf.get_mpi_data_type(),
            MPI_ROOT,
            comm,
        )
    }
}

pub fn bcast_slice<T: MPIDataType + Default>(buf: &[T], comm: MpiComm) -> i32 {
    unsafe {
        let t: T = Default::default();

        MPI_Bcast(
            buf.as_ptr() as *const c_void,
            buf.len() as i32,
            t.get_mpi_data_type(),
            MPI_ROOT,
            comm,
        )
    }
}

pub fn reduce_sum<T: MPIDataType + Default>(sbuf: &[T], dbuf: &mut [T], comm: MpiComm) -> i32 {
    unsafe {
        let t: T = Default::default();

        MPI_Reduce(
            sbuf.as_ptr() as *const c_void,
            dbuf.as_mut_ptr() as *mut c_void,
            sbuf.len() as i32,
            t.get_mpi_data_type(),
            MPI_SUM,
            MPI_ROOT,
            comm,
        )
    }
}

#[test]
fn test_mpi() {
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
