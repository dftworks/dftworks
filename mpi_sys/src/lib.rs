#![allow(warnings)]
use libc::ptrdiff_t;
use std::os::raw::*;

#[repr(C)]
pub struct ompi_info_t;
#[repr(C)]
pub struct ompi_win_t;

pub type MpiInfo = *mut ompi_info_t;
pub type MpiWin = *mut ompi_win_t;
pub type MpiAint = ptrdiff_t;
pub type MpiComm = c_int;
pub type MpiDatatype = c_int;
pub type MpiOp = c_int;

pub static MPI_MAX: MpiOp = 0x58000001;
pub static MPI_MIN: MpiOp = 0x58000002;
pub static MPI_SUM: MpiOp = 0x58000003;
pub static MPI_PROD: MpiOp = 0x58000004;
pub static MPI_LAND: MpiOp = 0x58000005;
pub static MPI_BAND: MpiOp = 0x58000006;
pub static MPI_LOR: MpiOp = 0x58000007;
pub static MPI_BOR: MpiOp = 0x58000008;
pub static MPI_LXOR: MpiOp = 0x58000009;
pub static MPI_BXOR: MpiOp = 0x5800000a;
pub static MPI_MINLOC: MpiOp = 0x5800000b;
pub static MPI_MAXLOC: MpiOp = 0x5800000c;
pub static MPI_REPLACE: MpiOp = 0x5800000d;

pub static MPI_ROOT: i32 = 0;

pub static MPI_COMM_WORLD: MpiComm = 0x44000000;

pub static MPI_CHAR: MpiDatatype = 0x4c000101;
pub static MPI_SIGNED_CHAR: MpiDatatype = 0x4c000118;
pub static MPI_UNSIGNED_CHAR: MpiDatatype = 0x4c000102;
pub static MPI_BYTE: MpiDatatype = 0x4c00010d;
pub static MPI_WCHAR: MpiDatatype = 0x4c00040e;
pub static MPI_SHORT: MpiDatatype = 0x4c000203;
pub static MPI_UNSIGNED_SHORT: MpiDatatype = 0x4c000204;
pub static MPI_INT: MpiDatatype = 0x4c000405;
pub static MPI_UNSIGNED: MpiDatatype = 0x4c000406;
pub static MPI_LONG: MpiDatatype = 0x4c000407;
pub static MPI_UNSIGNED_LONG: MpiDatatype = 0x4c000408;
pub static MPI_FLOAT: MpiDatatype = 0x4c00040a;
pub static MPI_DOUBLE: MpiDatatype = 0x4c00080b;
pub static MPI_LONG_DOUBLE: MpiDatatype = 0x4c00080c;
pub static MPI_LONG_LONG_INT: MpiDatatype = 0x4c000809;
pub static MPI_UNSIGNED_LONG_LONG: MpiDatatype = 0x4c000819;
pub static MPI_LONG_LONG: MpiDatatype = MPI_LONG_LONG_INT;
pub static MPI_DOUBLE_COMPLEX: MpiDatatype = 1275072546;

#[repr(C)]
#[derive(Default)]
pub struct MPI_Status {
    count: c_int,
    cancelled: c_int,
    mpi_source: c_int,
    mpi_tag: c_int,
    mpi_error: c_int,
}

#[link(name = "mpich", kind = "dylib")]
extern "C" {
    pub fn MPI_Init(argc: *const c_int, argv: *const c_char) -> c_int;

    pub fn MPI_Finalize() -> c_int;

    pub fn MPI_Comm_rank(comm: MpiComm, rank: *mut c_int) -> c_int;

    pub fn MPI_Comm_size(comm: MpiComm, size: *mut c_int) -> c_int;

    pub fn MPI_Send(
        buf: *const c_void,
        count: c_int,
        datatype: MpiDatatype,
        dest: c_int,
        tag: c_int,
        comm: c_int,
    ) -> c_int;

    pub fn MPI_Recv(
        buf: *mut c_void,
        count: c_int,
        datatype: MpiDatatype,
        source: c_int,
        tag: c_int,
        comm: MpiComm,
        status: *mut MPI_Status,
    ) -> c_int;

    pub fn MPI_Barrier(comm: MpiComm) -> c_int;

    pub fn MPI_Bcast(
        buf: *const c_void,
        count: c_int,
        datatype: MpiDatatype,
        root: c_int,
        comm: MpiComm,
    ) -> i32;

    pub fn MPI_Reduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: c_int,
        datatype: MpiDatatype,
        op: MpiOp,
        root: c_int,
        comm: MpiComm,
    ) -> i32;

    pub fn MPI_Scatter(
        sendbuf: *const c_void,
        sendcount: c_int,
        sendtype: MpiDatatype,
        recvbuf: *mut c_void,
        recvcount: c_int,
        recvtype: MpiDatatype,
        root: c_int,
        comm: MpiComm,
    ) -> i32;

    pub fn MPI_Scatterv(
        sendbuf: *const c_void,
        sendcounts: *const c_void,
        displs: *const c_void,
        sendtype: MpiDatatype,
        recvbuf: *const c_void,
        recvcount: c_int,
        recvtype: MpiDatatype,
        root: c_int,
        comm: MpiComm,
    ) -> i32;

    pub fn MPI_Win_create(
        base: *mut c_void,
        size: MpiAint,
        disp_unit: c_int,
        info: MpiInfo,
        comm: MpiComm,
        win: *mut MpiWin,
    ) -> c_int;

    pub fn MPI_Win_free(win: *mut MpiWin) -> c_int;

    pub fn MPI_Get(
        origin_addr: *mut c_void,
        origin_count: c_int,
        origin_datatype: MpiDatatype,
        target_rank: c_int,
        target_disp: MpiAint,
        target_count: c_int,
        target_datatype: MpiDatatype,
        win: MpiWin,
    ) -> c_int;

    pub fn MPI_Put(
        origin_add: *const c_void,
        origin_count: c_int,
        origin_datatype: MpiDatatype,
        target_rank: c_int,
        target_disp: MpiAint,
        target_count: c_int,
        target_datatype: MpiDatatype,
        win: MpiWin,
    ) -> c_int;

    pub fn MPI_Accumulate(
        origin_add: *const c_void,
        origin_count: c_int,
        origin_datatype: MpiDatatype,
        target_rank: c_int,
        target_disp: MpiAint,
        target_count: c_int,
        target_datatype: MpiDatatype,
        op: MpiOp,
        win: MpiWin,
    ) -> c_int;

    pub fn MPI_Get_accumulate(
        origin_add: *const c_void,
        origin_count: c_int,
        origin_datatype: MpiDatatype,
        result_addr: *mut c_void,
        result_count: c_int,
        result_datatype: MpiDatatype,
        target_rank: c_int,
        target_disp: MpiAint,
        target_count: c_int,
        target_datatype: MpiDatatype,
        op: MpiOp,
        win: MpiWin,
    ) -> c_int;

    pub fn MPI_Win_fence(assert: c_int, win: MpiWin) -> c_int;

    pub fn MPI_Get_processor_name(name: *mut c_char, resultlen: *mut c_int) -> c_int;

    pub fn MPI_Win_allocate(
        size: MpiAint,
        disp_unit: c_int,
        info: MpiInfo,
        comm: MpiComm,
        baseptr: *mut c_void,
        win: *mut MpiWin,
    ) -> c_int;

    pub fn MPI_Win_allocate_shared(
        size: MpiAint,
        disp_unit: c_int,
        info: MpiInfo,
        comm: MpiComm,
        baseptr: *mut c_void,
        win: *mut MpiWin,
    ) -> c_int;

    pub fn MPI_Comm_split_type(
        comm: MpiComm,
        split_type: c_int,
        key: c_int,
        info: MpiInfo,
        newcomm: *mut MpiComm,
    ) -> c_int;

    pub fn MPI_Win_shared_query(
        win: MpiWin,
        rank: c_int,
        size: *mut MpiAint,
        disp_unit: *mut c_int,
        baseptr: *mut c_void,
    ) -> c_int;
}
