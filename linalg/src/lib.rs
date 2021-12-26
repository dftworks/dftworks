use lapack_sys::*;
use matrix::Matrix;
use types::c64;

pub fn eig(mat: &Matrix<f64>) -> (Vec<f64>, Matrix<f64>) {
    let nn = mat.nrow();
    let n = nn as i32;
    let mut sk = Matrix::<f64>::new(nn, nn);

    for i in 0..nn {
        sk[[i, i]] = 1.0;
    }

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
            eigvec.as_mut_ptr(),
            &n,
            sk.as_mut_ptr(),
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

pub fn eigh(mat: &Matrix<c64>) -> (Vec<f64>, Matrix<c64>) {
    let nn = mat.nrow();
    let n = nn as i32;
    let mut sk = Matrix::new(nn, nn);

    for i in 0..nn {
        sk[[i, i]] = c64 { re: 1.0, im: 0.0 };
    }

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
            eigvec.as_mut_ptr(),
            &n,
            sk.as_mut_ptr(),
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
    use matrix::Dot;

    let cm = Matrix::<c64>::from_row_slice(
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

    let mut v0 = v.get_col(0).to_vec();

    println!("vec0 = \n{:?}", v0);

    println!("M.V = \n{:?}", cm.dot(&v0));

    for i in 0..v0.len() {
        v0[i] *= e[0];
    }
    println!("e*V = \n{:?}", v0);
}
