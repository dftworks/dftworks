#[cfg(feature = "rsmpi_backend")]
mod backend {
    use mpi::collective::SystemOperation;
    use mpi::datatype::Equivalence;
    use mpi::environment::Universe;
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::*;
    use std::cell::RefCell;
    use types::*;

    pub type MpiCommHandle = i32;

    pub const MPI_ROOT: i32 = 0;
    pub const MPI_COMM_WORLD: MpiCommHandle = 0;

    thread_local! {
        static MPI_UNIVERSE: RefCell<Option<Universe>> = const { RefCell::new(None) };
    }

    #[inline]
    pub fn comm_world() -> MpiCommHandle {
        MPI_COMM_WORLD
    }

    pub trait MPIDataType: Equivalence {}

    impl MPIDataType for c64 {}
    impl MPIDataType for f64 {}
    impl MPIDataType for bool {}
    impl MPIDataType for i32 {}

    pub fn init() -> i32 {
        MPI_UNIVERSE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                if let Some(universe) = mpi::initialize() {
                    *slot = Some(universe);
                }
            }
        });

        0
    }

    pub fn finalize() -> i32 {
        MPI_UNIVERSE.with(|slot| {
            slot.borrow_mut().take();
        });

        0
    }

    fn world_for(comm: MpiCommHandle) -> SimpleCommunicator {
        let _ = init();
        debug_assert_eq!(
            comm,
            MPI_COMM_WORLD,
            "dwmpi currently supports only MPI_COMM_WORLD"
        );
        SimpleCommunicator::world()
    }

    pub fn get_comm_world_rank() -> i32 {
        world_for(MPI_COMM_WORLD).rank()
    }

    pub fn get_comm_world_size() -> i32 {
        world_for(MPI_COMM_WORLD).size()
    }

    pub fn send_scalar<T: MPIDataType>(
        buf: &T,
        dest: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
        world_for(comm).process_at_rank(dest).send_with_tag(buf, tag);
        0
    }

    pub fn recv_scalar<T: MPIDataType>(
        buf: &mut T,
        source: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
        let _status = world_for(comm)
            .process_at_rank(source)
            .receive_into_with_tag(buf, tag);
        0
    }

    pub fn send_slice<T: MPIDataType + Default>(
        buf: &[T],
        dest: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
        world_for(comm)
            .process_at_rank(dest)
            .send_with_tag(buf, tag);
        0
    }

    pub fn recv_slice<T: MPIDataType + Default>(
        buf: &mut [T],
        source: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
        let _status = world_for(comm)
            .process_at_rank(source)
            .receive_into_with_tag(buf, tag);
        0
    }

    pub fn is_root() -> bool {
        get_comm_world_rank() == MPI_ROOT
    }

    pub fn barrier(comm: MpiCommHandle) -> i32 {
        world_for(comm).barrier();
        0
    }

    pub fn bcast_scalar<T: MPIDataType>(buf: &mut T, comm: MpiCommHandle) -> i32 {
        world_for(comm)
            .process_at_rank(MPI_ROOT)
            .broadcast_into(buf);
        0
    }

    pub fn bcast_slice<T: MPIDataType + Default>(buf: &mut [T], comm: MpiCommHandle) -> i32 {
        world_for(comm)
            .process_at_rank(MPI_ROOT)
            .broadcast_into(buf);
        0
    }

    pub fn reduce_sum<T: MPIDataType + Default>(
        sbuf: &[T],
        dbuf: &mut [T],
        comm: MpiCommHandle,
    ) -> i32 {
        let world = world_for(comm);
        let root = world.process_at_rank(MPI_ROOT);

        if world.rank() == MPI_ROOT {
            root.reduce_into_root(sbuf, dbuf, SystemOperation::sum());
        } else {
            root.reduce_into(sbuf, SystemOperation::sum());
        }

        0
    }

    pub fn reduce_slice_sum<T: MPIDataType + Default>(
        sbuf: &[T],
        dbuf: &mut [T],
        comm: MpiCommHandle,
    ) -> i32 {
        reduce_sum(sbuf, dbuf, comm)
    }

    pub fn reduce_scalar_sum<T: MPIDataType + Default>(
        sbuf: &T,
        dbuf: &mut T,
        comm: MpiCommHandle,
    ) -> i32 {
        let world = world_for(comm);
        let root = world.process_at_rank(MPI_ROOT);

        if world.rank() == MPI_ROOT {
            root.reduce_into_root(sbuf, dbuf, SystemOperation::sum());
        } else {
            root.reduce_into(sbuf, SystemOperation::sum());
        }

        0
    }

    pub fn reduce_scalar_max<T: MPIDataType + Default>(
        sbuf: &T,
        dbuf: &mut T,
        comm: MpiCommHandle,
    ) -> i32 {
        let world = world_for(comm);
        let root = world.process_at_rank(MPI_ROOT);

        if world.rank() == MPI_ROOT {
            root.reduce_into_root(sbuf, dbuf, SystemOperation::max());
        } else {
            root.reduce_into(sbuf, SystemOperation::max());
        }

        0
    }

    pub fn reduce_scalar_min<T: MPIDataType + Default>(
        sbuf: &T,
        dbuf: &mut T,
        comm: MpiCommHandle,
    ) -> i32 {
        let world = world_for(comm);
        let root = world.process_at_rank(MPI_ROOT);

        if world.rank() == MPI_ROOT {
            root.reduce_into_root(sbuf, dbuf, SystemOperation::min());
        } else {
            root.reduce_into(sbuf, SystemOperation::min());
        }

        0
    }
}

#[cfg(not(feature = "rsmpi_backend"))]
mod backend {
    use mpi_sys::*;
    use std::{os::raw::*, ptr};
    use types::*;

    pub type MpiCommHandle = MpiComm;

    pub const MPI_ROOT: i32 = mpi_sys::MPI_ROOT;
    pub const MPI_COMM_WORLD: MpiCommHandle = mpi_sys::MPI_COMM_WORLD;

    #[inline]
    pub fn comm_world() -> MpiCommHandle {
        MPI_COMM_WORLD
    }

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

    impl MPIDataType for bool {
        fn get_mpi_data_type(&self) -> i32 {
            MPI_INT
        }
    }

    impl MPIDataType for i32 {
        fn get_mpi_data_type(&self) -> i32 {
            MPI_INT
        }
    }

    pub fn init() -> i32 {
        unsafe { MPI_Init(ptr::null(), ptr::null()) }
    }

    pub fn finalize() -> i32 {
        unsafe { MPI_Finalize() }
    }

    pub fn get_comm_world_rank() -> i32 {
        let mut rank = 0;
        comm_rank(MPI_COMM_WORLD, &mut rank);
        rank
    }

    pub fn get_comm_world_size() -> i32 {
        let mut size = 0;
        comm_size(MPI_COMM_WORLD, &mut size);
        size
    }

    pub fn send_scalar<T: MPIDataType>(
        buf: &T,
        dest: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
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

    pub fn recv_scalar<T: MPIDataType>(
        buf: &mut T,
        source: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
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

    pub fn send_slice<T: MPIDataType + Default>(
        buf: &[T],
        dest: i32,
        tag: i32,
        comm: MpiCommHandle,
    ) -> i32 {
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
        comm: MpiCommHandle,
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

    pub fn is_root() -> bool {
        get_comm_world_rank() == MPI_ROOT
    }

    pub fn barrier(comm: MpiCommHandle) -> i32 {
        unsafe { MPI_Barrier(comm) }
    }

    pub fn bcast_scalar<T: MPIDataType>(buf: &mut T, comm: MpiCommHandle) -> i32 {
        unsafe {
            MPI_Bcast(
                buf as *mut T as *mut c_void,
                1,
                buf.get_mpi_data_type(),
                MPI_ROOT,
                comm,
            )
        }
    }

    pub fn bcast_slice<T: MPIDataType + Default>(buf: &mut [T], comm: MpiCommHandle) -> i32 {
        unsafe {
            let t: T = Default::default();

            MPI_Bcast(
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as i32,
                t.get_mpi_data_type(),
                MPI_ROOT,
                comm,
            )
        }
    }

    pub fn reduce_sum<T: MPIDataType + Default>(
        sbuf: &[T],
        dbuf: &mut [T],
        comm: MpiCommHandle,
    ) -> i32 {
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

    pub fn reduce_slice_sum<T: MPIDataType + Default>(
        sbuf: &[T],
        dbuf: &mut [T],
        comm: MpiCommHandle,
    ) -> i32 {
        reduce_sum(sbuf, dbuf, comm)
    }

    pub fn reduce_scalar_sum<T: MPIDataType + Default>(
        sbuf: &T,
        dbuf: &mut T,
        comm: MpiCommHandle,
    ) -> i32 {
        unsafe {
            let t: T = Default::default();

            MPI_Reduce(
                sbuf as *const T as *const c_void,
                dbuf as *mut T as *mut c_void,
                1,
                t.get_mpi_data_type(),
                MPI_SUM,
                MPI_ROOT,
                comm,
            )
        }
    }

    pub fn reduce_scalar_max<T: MPIDataType + Default>(
        sbuf: &T,
        dbuf: &mut T,
        comm: MpiCommHandle,
    ) -> i32 {
        unsafe {
            let t: T = Default::default();

            MPI_Reduce(
                sbuf as *const T as *const c_void,
                dbuf as *mut T as *mut c_void,
                1,
                t.get_mpi_data_type(),
                MPI_MAX,
                MPI_ROOT,
                comm,
            )
        }
    }

    pub fn reduce_scalar_min<T: MPIDataType + Default>(
        sbuf: &T,
        dbuf: &mut T,
        comm: MpiCommHandle,
    ) -> i32 {
        unsafe {
            let t: T = Default::default();

            MPI_Reduce(
                sbuf as *const T as *const c_void,
                dbuf as *mut T as *mut c_void,
                1,
                t.get_mpi_data_type(),
                MPI_MIN,
                MPI_ROOT,
                comm,
            )
        }
    }

    fn comm_rank(comm: MpiCommHandle, rank: &mut i32) -> i32 {
        unsafe { MPI_Comm_rank(comm, rank) }
    }

    fn comm_size(comm: MpiCommHandle, size: &mut i32) -> i32 {
        unsafe { MPI_Comm_size(comm, size) }
    }
}

pub use backend::*;

// mpirun --np 3 target/debug/test_mpi
#[test]
fn test_mpi() {
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
            let buf: [types::c64; 10] = [types::c64 {
                re: 3.0,
                im: dest as f64,
            }; 10];

            send_slice(&buf[1..], dest, 1, comm_world());
        }
    } else {
        let mut buf: [types::c64; 10] = [types::c64 { re: 0.0, im: 0.0 }; 10];

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
        for (i, xi) in xx.iter_mut().enumerate() {
            *xi = i as f64;
        }
    }

    bcast_slice(&mut xx, comm_world());
    barrier(comm_world());
    println!(" xx = {:#?}", xx);

    finalize();
}
