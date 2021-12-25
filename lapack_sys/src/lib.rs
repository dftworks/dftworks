use std::os::raw::*;
use types::c64;

extern "C" {
    pub fn dsygvd_(
        itype: *const c_int,
        jobz: *const u8,
        uplo: *const u8,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        w: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        iwork: *mut c_int,
        liwork: *const c_int,
        info: *mut c_int,
    );

    pub fn zhegvd_(
        itype: *const c_int,
        jobz: *const u8,
        uplo: *const u8,
        n: *const c_int,
        a: *mut c64,
        lda: *const c_int,
        b: *mut c64,
        ldb: *const c_int,
        w: *mut c_double,
        work: *mut c64,
        lwork: *const c_int,
        rwork: *mut c_double,
        lrwork: *const c_int,
        iwork: *mut c_int,
        liwork: *const c_int,
        info: *mut c_int,
    );

    pub fn dgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *const c_double,
        lda: *const c_int,
        ipiv: *const c_int,
        info: *mut c_int,
    );

    pub fn dgetri_(
        n: *const c_int,
        a: *mut c_double,
        lda: *const c_int,
        ipiv: *const c_int,
        work: *mut c_double,
        lwork: *const c_int,
        info: *mut c_int,
    );

    pub fn zgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *const c64,
        lda: *const c_int,
        ipiv: *const c_int,
        info: *mut c_int,
    );

    pub fn zgetri_(
        n: *const c_int,
        a: *mut c64,
        lda: *const c_int,
        ipiv: *const c_int,
        work: *mut c64,
        lwork: *const c_int,
        info: *mut c_int,
    );

    pub fn dgelss_(
        m: *const c_int,
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        s: *mut f64,
        rcond: *const f64,
        rank: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );

    pub fn zgelss_(
        m: *const c_int,
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut c64,
        lda: *const c_int,
        b: *mut c64,
        ldb: *const c_int,
        s: *mut f64,
        rcond: *const f64,
        rank: *const c_int,
        work: *mut c64,
        lwork: *const c_int,
        rwork: *mut f64,
        info: *mut c_int,
    );
}
