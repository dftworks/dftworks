use crate::{c64, Matrix, MatrixExt};

pub fn save_matrix_f64_hdf5(matrix: &Matrix<f64>, group: &mut hdf5::Group) -> Result<(), String> {
    group
        .new_dataset_builder()
        .with_data(&[matrix.nrow(), matrix.ncol()])
        .create("shape")
        .map_err(|e| format!("failed to create dataset 'shape': {}", e))?;

    group
        .new_dataset_builder()
        .with_data(matrix.as_slice())
        .create("data")
        .map_err(|e| format!("failed to create dataset 'data': {}", e))?;

    Ok(())
}

pub fn load_matrix_f64_hdf5(group: &hdf5::Group) -> Result<Matrix<f64>, String> {
    let shape: Vec<usize> = group
        .dataset("shape")
        .map_err(|e| format!("failed to open dataset 'shape': {}", e))?
        .read()
        .map_err(|e| format!("failed to read dataset 'shape': {}", e))?
        .to_vec();

    if shape.len() != 2 {
        return Err(format!(
            "invalid matrix shape length: expected 2, got {}",
            shape.len()
        ));
    }

    let nrow = shape[0];
    let ncol = shape[1];
    let data: Vec<f64> = group
        .dataset("data")
        .map_err(|e| format!("failed to open dataset 'data': {}", e))?
        .read()
        .map_err(|e| format!("failed to read dataset 'data': {}", e))?
        .to_vec();

    let expected = nrow * ncol;
    if data.len() != expected {
        return Err(format!(
            "invalid matrix payload length: expected {}, got {}",
            expected,
            data.len()
        ));
    }

    Ok(Matrix::<f64>::from_column_slice(nrow, ncol, &data))
}

pub fn save_matrix_c64_hdf5(matrix: &Matrix<c64>, group: &mut hdf5::Group) -> Result<(), String> {
    group
        .new_dataset_builder()
        .with_data(&[matrix.nrow(), matrix.ncol()])
        .create("shape")
        .map_err(|e| format!("failed to create dataset 'shape': {}", e))?;

    let real_data: Vec<f64> = matrix.as_slice().iter().map(|&z| z.re).collect();
    let imag_data: Vec<f64> = matrix.as_slice().iter().map(|&z| z.im).collect();

    group
        .new_dataset_builder()
        .with_data(&real_data)
        .create("real")
        .map_err(|e| format!("failed to create dataset 'real': {}", e))?;
    group
        .new_dataset_builder()
        .with_data(&imag_data)
        .create("imag")
        .map_err(|e| format!("failed to create dataset 'imag': {}", e))?;

    Ok(())
}

pub fn load_matrix_c64_hdf5(group: &hdf5::Group) -> Result<Matrix<c64>, String> {
    let shape: Vec<usize> = group
        .dataset("shape")
        .map_err(|e| format!("failed to open dataset 'shape': {}", e))?
        .read()
        .map_err(|e| format!("failed to read dataset 'shape': {}", e))?
        .to_vec();

    if shape.len() != 2 {
        return Err(format!(
            "invalid matrix shape length: expected 2, got {}",
            shape.len()
        ));
    }

    let nrow = shape[0];
    let ncol = shape[1];
    let real_data: Vec<f64> = group
        .dataset("real")
        .map_err(|e| format!("failed to open dataset 'real': {}", e))?
        .read()
        .map_err(|e| format!("failed to read dataset 'real': {}", e))?
        .to_vec();
    let imag_data: Vec<f64> = group
        .dataset("imag")
        .map_err(|e| format!("failed to open dataset 'imag': {}", e))?
        .read()
        .map_err(|e| format!("failed to read dataset 'imag': {}", e))?
        .to_vec();

    if real_data.len() != imag_data.len() {
        return Err(format!(
            "invalid matrix payload: real_len={} imag_len={}",
            real_data.len(),
            imag_data.len()
        ));
    }

    let expected = nrow * ncol;
    if real_data.len() != expected {
        return Err(format!(
            "invalid matrix payload length: expected {}, got {}",
            expected,
            real_data.len()
        ));
    }

    let data: Vec<c64> = real_data
        .iter()
        .zip(imag_data.iter())
        .map(|(&re, &im)| c64::new(re, im))
        .collect();

    Ok(Matrix::<c64>::from_column_slice(nrow, ncol, &data))
}
