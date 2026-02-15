use crate::types::{MeshTopology, Neighbor};
use kpts::KPTS;
use std::collections::HashMap;
use std::io;
use std::io::BufRead;

const NEIGHBOR_DIRS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

pub(crate) fn validate_k_mesh(k_mesh: [i32; 3]) -> io::Result<()> {
    if k_mesh.iter().any(|&x| x <= 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wannier90 export requires kpts_scheme = kmesh",
        ));
    }

    Ok(())
}

pub(crate) fn read_k_shift(in_kmesh_file: &str) -> io::Result<[i32; 3]> {
    let file = std::fs::File::open(in_kmesh_file)?;
    let lines: Vec<String> = std::io::BufReader::new(file)
        .lines()
        .map_while(std::io::Result::ok)
        .collect();

    if lines.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "in.kmesh must contain at least two lines",
        ));
    }

    let fields: Vec<&str> = lines[1].split_whitespace().collect();
    if fields.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "second line of in.kmesh must contain three shift values",
        ));
    }

    Ok([
        fields[0].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid k-shift value in in.kmesh",
            )
        })?,
        fields[1].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid k-shift value in in.kmesh",
            )
        })?,
        fields[2].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid k-shift value in in.kmesh",
            )
        })?,
    ])
}

pub(crate) fn build_mesh_topology(
    kpts: &dyn KPTS,
    k_mesh: [i32; 3],
    k_shift: [i32; 3],
) -> io::Result<MeshTopology> {
    let nkpt = kpts.get_n_kpts();
    let nk_expected = (k_mesh[0] * k_mesh[1] * k_mesh[2]) as usize;
    if nkpt != nk_expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "k-point count mismatch for Wannier90 export: got {}, expected {} from in.kmesh",
                nkpt, nk_expected
            ),
        ));
    }

    let mut grid_indices = Vec::with_capacity(nkpt);
    let mut lookup = HashMap::with_capacity(nkpt * 2);

    for ik in 0..nkpt {
        let k = kpts.get_k_frac(ik);
        let ix = frac_to_grid_index(k.x, k_mesh[0], k_shift[0])?;
        let iy = frac_to_grid_index(k.y, k_mesh[1], k_shift[1])?;
        let iz = frac_to_grid_index(k.z, k_mesh[2], k_shift[2])?;
        let key = (ix, iy, iz);

        if lookup.insert(key, ik).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate k-point detected while constructing Wannier90 mesh topology",
            ));
        }

        grid_indices.push([ix, iy, iz]);
    }

    let mut neighbors = vec![Vec::with_capacity(NEIGHBOR_DIRS.len()); nkpt];
    for (ik, ijk) in grid_indices.iter().enumerate() {
        for dir in NEIGHBOR_DIRS.iter() {
            let (jx, gx) = advance_with_wrap(ijk[0], dir[0], k_mesh[0]);
            let (jy, gy) = advance_with_wrap(ijk[1], dir[1], k_mesh[1]);
            let (jz, gz) = advance_with_wrap(ijk[2], dir[2], k_mesh[2]);

            let Some(ikb) = lookup.get(&(jx, jy, jz)) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "failed to locate neighboring k-point while constructing Wannier90 nn list",
                ));
            };

            neighbors[ik].push(Neighbor {
                ikb: *ikb,
                gshift: [gx, gy, gz],
            });
        }
    }

    Ok(MeshTopology {
        neighbors,
        nntot: NEIGHBOR_DIRS.len(),
    })
}

fn advance_with_wrap(i: i32, step: i32, n: i32) -> (i32, i32) {
    let mut j = i + step;
    let mut g = 0;
    if j < 0 {
        j += n;
        g = -1;
    } else if j >= n {
        j -= n;
        g = 1;
    }
    (j, g)
}

fn frac_to_grid_index(frac: f64, n: i32, shift: i32) -> io::Result<i32> {
    let mut wrapped = frac - frac.floor();
    if wrapped >= 1.0 - 1.0e-10 {
        wrapped = 0.0;
    }

    let target = wrapped * n as f64 - 0.5 * shift as f64;
    let nearest = target.round();

    if (target - nearest).abs() > 1.0e-6 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k-point fractional coordinate {} is inconsistent with mesh n={} shift={}",
                frac, n, shift
            ),
        ));
    }

    let mut idx = nearest as i32 % n;
    if idx < 0 {
        idx += n;
    }
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_k_mesh_rejects_non_mesh_input() {
        assert!(validate_k_mesh([4, 4, 4]).is_ok());
        assert!(validate_k_mesh([0, 0, 0]).is_err());
    }

    #[test]
    fn test_frac_to_grid_index_handles_wrapping_and_shift() {
        assert_eq!(frac_to_grid_index(0.0, 4, 0).unwrap(), 0);
        assert_eq!(frac_to_grid_index(0.75, 4, 0).unwrap(), 3);
        assert_eq!(frac_to_grid_index(-0.25, 4, 0).unwrap(), 3);
        assert_eq!(frac_to_grid_index(0.125, 4, 1).unwrap(), 0);
        assert_eq!(frac_to_grid_index(0.875, 4, 1).unwrap(), 3);
    }
}
