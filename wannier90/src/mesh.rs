use std::io;

pub(crate) fn validate_k_mesh(k_mesh: [i32; 3]) -> io::Result<()> {
    if k_mesh.iter().any(|&x| x <= 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wannier90 export requires kpts_scheme = kmesh",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_k_mesh_rejects_non_mesh_input() {
        assert!(validate_k_mesh([4, 4, 4]).is_ok());
        assert!(validate_k_mesh([0, 0, 0]).is_err());
    }
}
