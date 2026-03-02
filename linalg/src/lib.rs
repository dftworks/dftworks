use lapack_sys::*;
use nalgebra::DMatrix;
use types::c64;

pub fn eig(mat: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    assert_eq!(mat.nrows(), mat.ncols(), "linalg::eig requires square matrix");
    let nn = mat.nrows();
    let n = nn as i32;
    let mut sk = DMatrix::<f64>::identity(nn, nn);

    let lwork: i32 = 1 + 6 * n + 2 * n * n;
    let liwork: i32 = 3 + 5 * n;

    let mut work = vec![0.0; lwork as usize];
    let mut iwork = vec![0; liwork as usize];

    let mut info = 0i32;
    let mut eigval = vec![0.0; nn];
    let mut eigvec = mat.clone();
    let itype = 1i32; // A*X = (lambda)*B*x
    let jobz = b'V';
    let uplo = b'L';

    unsafe {
        dsygvd_(
            &itype,
            &jobz,
            &uplo,
            &n,
            eigvec.as_mut_slice().as_mut_ptr(),
            &n,
            sk.as_mut_slice().as_mut_ptr(),
            &n,
            eigval.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }

    if info != 0 {
        panic!("dsygvd_ exit code: {}", info);
    }

    (eigval, eigvec)
}

pub fn eigh(mat: &DMatrix<c64>) -> (Vec<f64>, DMatrix<c64>) {
    assert_eq!(mat.nrows(), mat.ncols(), "linalg::eigh requires square matrix");
    let nn = mat.nrows();
    let n = nn as i32;
    let mut sk = DMatrix::<c64>::identity(nn, nn);

    let lwork: i32 = 2 * n + n * n;
    let lrwork: i32 = 1 + 5 * n + 2 * n * n;
    let liwork: i32 = 3 + 5 * n;

    let mut work = vec![c64 { re: 0.0, im: 0.0 }; lwork as usize];
    let mut rwork = vec![0.0; lrwork as usize];
    let mut iwork = vec![0; liwork as usize];

    let mut info = 0i32;
    let mut eigval = vec![0.0; nn];
    let mut eigvec = mat.clone();
    let itype = 1i32;
    let jobz = b'V';
    let uplo = b'L';

    unsafe {
        zhegvd_(
            &itype,
            &jobz,
            &uplo,
            &n,
            eigvec.as_mut_slice().as_mut_ptr(),
            &n,
            sk.as_mut_slice().as_mut_ptr(),
            &n,
            eigval.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            rwork.as_mut_ptr(),
            &lrwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }

    if info != 0 {
        panic!("zhegvd_ exit code: {}", info);
    }

    (eigval, eigvec)
}

#[test]
fn test_eigh() {
    use nalgebra::DVector;

    let cm = DMatrix::<c64>::from_row_slice(
        2,
        2,
        &[
            c64 { re: 1.0, im: 0.0 },
            c64 { re: 0.0, im: 0.01 },
            c64 { re: 0.0, im: -0.01 },
            c64 { re: 1.0, im: 0.0 },
        ],
    );

    let (e, v) = eigh(&cm);
    println!("matrix = \n{}", cm);
    println!("eigval = \n{:?}", e);
    println!("eigvec = \n{}", v);

    let mut v0: Vec<c64> = v.column(0).iter().copied().collect();

    println!("vec0 = \n{:?}", v0);

    let mv = &cm * DVector::from_column_slice(&v0);
    println!("M.V = \n{:?}", mv.as_slice());

    for i in 0..v0.len() {
        v0[i] *= e[0];
    }
    println!("e*V = \n{:?}", v0);
}
