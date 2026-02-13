use control::{Control, SpinScheme};
use crystal::Crystal;
use dfttypes::{VKEigenValue, VKEigenVector};
use dwconsts::{BOHR_TO_ANG, HA_TO_EV};
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use matrix::Matrix;
use mpi_sys::MPI_COMM_WORLD;
use pwbasis::PWBasis;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use types::c64;

const NEIGHBOR_DIRS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

#[derive(Debug, Default)]
pub struct ExportSummary {
    pub written_files: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct Neighbor {
    ikb: usize,       // zero-based
    gshift: [i32; 3], // reciprocal lattice shift to map k+b to ikb
}

#[derive(Debug)]
struct MeshTopology {
    neighbors: Vec<Vec<Neighbor>>,
    nntot: usize,
}

pub fn export(
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    vkevals: &VKEigenValue,
    ik_first: usize,
) -> io::Result<ExportSummary> {
    let seedname = control.get_wannier90_seedname().trim();
    if seedname.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wannier90_seedname must not be empty",
        ));
    }

    let k_mesh = kpts.get_k_mesh();
    validate_k_mesh(k_mesh)?;
    let k_shift = read_k_shift("in.kmesh")?;
    let topology = build_mesh_topology(kpts, k_mesh, k_shift)?;

    // Ensure all ranks have reached export after writing wavefunctions/eigenvalues.
    dwmpi::barrier(MPI_COMM_WORLD);

    let mut summary = ExportSummary::default();

    if dwmpi::is_root() {
        match control.get_spin_scheme_enum() {
            SpinScheme::NonSpin => {
                let files = write_full_channel_data(
                    seedname,
                    control,
                    crystal,
                    kpts,
                    &topology,
                    control.is_spin(),
                    None,
                )?;
                summary.written_files.extend(files);
            }
            SpinScheme::Spin => {
                let up_seed = format!("{}.up", seedname);
                let dn_seed = format!("{}.dn", seedname);

                let up_files = write_full_channel_data(
                    &up_seed,
                    control,
                    crystal,
                    kpts,
                    &topology,
                    true,
                    Some("up"),
                )?;
                let dn_files = write_full_channel_data(
                    &dn_seed,
                    control,
                    crystal,
                    kpts,
                    &topology,
                    true,
                    Some("down"),
                )?;

                summary.written_files.extend(up_files);
                summary.written_files.extend(dn_files);
            }
            SpinScheme::Ncl => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "wannier90 export currently supports nonspin/spin only",
                ));
            }
        }
    }

    write_local_eig_part_files(seedname, vkevals, ik_first)?;
    dwmpi::barrier(MPI_COMM_WORLD);

    if dwmpi::is_root() {
        match control.get_spin_scheme_enum() {
            SpinScheme::NonSpin => {
                let eig_file = format!("{}.eig", seedname);
                merge_rank_parts(
                    &eig_file,
                    &|rank| format!("{}.eig.part.rank{}", seedname, rank),
                    dwmpi::get_comm_world_size(),
                )?;
                summary.written_files.push(eig_file);
            }
            SpinScheme::Spin => {
                let up_eig_file = format!("{}.up.eig", seedname);
                let dn_eig_file = format!("{}.dn.eig", seedname);
                merge_rank_parts(
                    &up_eig_file,
                    &|rank| format!("{}.up.eig.part.rank{}", seedname, rank),
                    dwmpi::get_comm_world_size(),
                )?;
                merge_rank_parts(
                    &dn_eig_file,
                    &|rank| format!("{}.dn.eig.part.rank{}", seedname, rank),
                    dwmpi::get_comm_world_size(),
                )?;
                summary.written_files.push(up_eig_file);
                summary.written_files.push(dn_eig_file);
            }
            SpinScheme::Ncl => {}
        }
    }

    dwmpi::barrier(MPI_COMM_WORLD);
    Ok(summary)
}

fn write_full_channel_data(
    channel_seed: &str,
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    topology: &MeshTopology,
    is_spin: bool,
    spin_channel: Option<&str>,
) -> io::Result<Vec<String>> {
    let nkpt = kpts.get_n_kpts();
    ensure_wfc_files_present(is_spin, nkpt)?;

    let win_file = format!("{}.win", channel_seed);
    write_win_file(&win_file, control, crystal, kpts, spin_channel)?;

    let nnkp_file = format!("{}.nnkp", channel_seed);
    write_nnkp_file(&nnkp_file, crystal, kpts, topology)?;

    let amn_file = format!("{}.amn", channel_seed);
    write_amn_file(
        &amn_file,
        control.get_nband(),
        nkpt,
        control.get_wannier90_num_wann(),
    )?;

    let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
    let [n1, n2, n3] = fftgrid.get_size();
    let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);

    let mmn_file = format!("{}.mmn", channel_seed);
    let (all_pwbasis, _blatt, all_eigvecs) = VKEigenVector::load_hdf5(is_spin, 0, nkpt - 1);
    let eigvecs = select_spin_channel(&all_eigvecs, spin_channel)?;
    write_mmn_file(
        &mmn_file,
        control.get_nband(),
        topology,
        eigvecs,
        &all_pwbasis,
        &gvec,
    )?;

    Ok(vec![win_file, nnkp_file, amn_file, mmn_file])
}

fn select_spin_channel<'a>(
    all_eigvecs: &'a VKEigenVector,
    spin_channel: Option<&str>,
) -> io::Result<&'a [Matrix<c64>]> {
    match (all_eigvecs, spin_channel) {
        (VKEigenVector::NonSpin(v), None) => Ok(v.as_slice()),
        (VKEigenVector::Spin(up, _), Some("up")) => Ok(up.as_slice()),
        (VKEigenVector::Spin(_, dn), Some("down")) => Ok(dn.as_slice()),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inconsistent spin channel selection for Wannier90 export",
        )),
    }
}

fn ensure_wfc_files_present(is_spin: bool, nkpt: usize) -> io::Result<()> {
    for ik in 0..nkpt {
        let filename = if is_spin {
            format!("out.wfc.up.k.{}.hdf5", ik)
        } else {
            format!("out.wfc.k.{}.hdf5", ik)
        };

        if !Path::new(&filename).exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "missing required wavefunction file '{}'; set save_wfc=true or enable wannier90_export during SCF",
                    filename
                ),
            ));
        }
    }
    Ok(())
}

fn write_local_eig_part_files(
    seedname: &str,
    vkevals: &VKEigenValue,
    ik_first: usize,
) -> io::Result<()> {
    let rank = dwmpi::get_comm_world_rank();

    match vkevals {
        VKEigenValue::NonSpin(vkevals) => {
            let filename = format!("{}.eig.part.rank{}", seedname, rank);
            write_eig_part_file(&filename, vkevals, ik_first)?;
        }
        VKEigenValue::Spin(vkevals_up, vkevals_dn) => {
            let up_filename = format!("{}.up.eig.part.rank{}", seedname, rank);
            let dn_filename = format!("{}.dn.eig.part.rank{}", seedname, rank);
            write_eig_part_file(&up_filename, vkevals_up, ik_first)?;
            write_eig_part_file(&dn_filename, vkevals_dn, ik_first)?;
        }
    }

    Ok(())
}

fn write_eig_part_file(filename: &str, eigvals: &[Vec<f64>], ik_first: usize) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for (ik_local, evals_at_k) in eigvals.iter().enumerate() {
        let ik_global = ik_first + ik_local + 1;
        for (iband, eval) in evals_at_k.iter().enumerate() {
            writeln!(
                writer,
                "{:6} {:6} {:22.14E}",
                iband + 1,
                ik_global,
                eval * HA_TO_EV
            )?;
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_win_file(
    filename: &str,
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    spin_channel: Option<&str>,
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "! Generated by dftworks. Edit projections/advanced options as needed."
    )?;
    writeln!(writer, "num_bands = {}", control.get_nband())?;
    writeln!(writer, "num_wann = {}", control.get_wannier90_num_wann())?;
    writeln!(writer, "num_iter = {}", control.get_wannier90_num_iter())?;

    if let Some(channel) = spin_channel {
        writeln!(writer, "spin = {}", channel)?;
    }

    let k_mesh = kpts.get_k_mesh();
    writeln!(
        writer,
        "mp_grid = {} {} {}",
        k_mesh[0], k_mesh[1], k_mesh[2]
    )?;
    writeln!(writer)?;

    writeln!(writer, "begin unit_cell_cart")?;
    writeln!(writer, "ang")?;
    let a = crystal.get_latt().get_vector_a();
    let b = crystal.get_latt().get_vector_b();
    let c = crystal.get_latt().get_vector_c();
    writeln!(
        writer,
        "  {:>18.10} {:>18.10} {:>18.10}",
        a.x * BOHR_TO_ANG,
        a.y * BOHR_TO_ANG,
        a.z * BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        "  {:>18.10} {:>18.10} {:>18.10}",
        b.x * BOHR_TO_ANG,
        b.y * BOHR_TO_ANG,
        b.z * BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        "  {:>18.10} {:>18.10} {:>18.10}",
        c.x * BOHR_TO_ANG,
        c.y * BOHR_TO_ANG,
        c.z * BOHR_TO_ANG
    )?;
    writeln!(writer, "end unit_cell_cart")?;
    writeln!(writer)?;

    writeln!(writer, "begin atoms_frac")?;
    for (species, position) in crystal
        .get_atom_species()
        .iter()
        .zip(crystal.get_atom_positions().iter())
    {
        writeln!(
            writer,
            "  {:<4} {:>16.10} {:>16.10} {:>16.10}",
            species, position.x, position.y, position.z
        )?;
    }
    writeln!(writer, "end atoms_frac")?;
    writeln!(writer)?;

    writeln!(writer, "! begin projections")?;
    writeln!(writer, "! <SPECIES>:s;p;d")?;
    writeln!(writer, "! end projections")?;
    writeln!(writer)?;

    writeln!(writer, "begin kpoints")?;
    for ik in 0..kpts.get_n_kpts() {
        let k = kpts.get_k_frac(ik);
        writeln!(writer, "  {:>16.10} {:>16.10} {:>16.10}", k.x, k.y, k.z)?;
    }
    writeln!(writer, "end kpoints")?;

    writer.flush()?;
    Ok(())
}

fn write_nnkp_file(
    filename: &str,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    topology: &MeshTopology,
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    let a = crystal.get_latt().get_vector_a();
    let b = crystal.get_latt().get_vector_b();
    let c = crystal.get_latt().get_vector_c();
    let blatt = crystal.get_latt().reciprocal();
    let ba = blatt.get_vector_a();
    let bb = blatt.get_vector_b();
    let bc = blatt.get_vector_c();

    writeln!(writer, "begin real_lattice")?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        a.x * BOHR_TO_ANG,
        a.y * BOHR_TO_ANG,
        a.z * BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        b.x * BOHR_TO_ANG,
        b.y * BOHR_TO_ANG,
        b.z * BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        c.x * BOHR_TO_ANG,
        c.y * BOHR_TO_ANG,
        c.z * BOHR_TO_ANG
    )?;
    writeln!(writer, "end real_lattice")?;
    writeln!(writer)?;

    writeln!(writer, "begin recip_lattice")?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        ba.x / BOHR_TO_ANG,
        ba.y / BOHR_TO_ANG,
        ba.z / BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        bb.x / BOHR_TO_ANG,
        bb.y / BOHR_TO_ANG,
        bb.z / BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        bc.x / BOHR_TO_ANG,
        bc.y / BOHR_TO_ANG,
        bc.z / BOHR_TO_ANG
    )?;
    writeln!(writer, "end recip_lattice")?;
    writeln!(writer)?;

    writeln!(writer, "begin kpoints")?;
    writeln!(writer, "{}", kpts.get_n_kpts())?;
    for ik in 0..kpts.get_n_kpts() {
        let k = kpts.get_k_frac(ik);
        writeln!(writer, " {:18.12E} {:18.12E} {:18.12E}", k.x, k.y, k.z)?;
    }
    writeln!(writer, "end kpoints")?;
    writeln!(writer)?;

    writeln!(writer, "begin nnkpts")?;
    writeln!(writer, "{}", topology.nntot)?;
    for (ik, knn) in topology.neighbors.iter().enumerate() {
        for nn in knn.iter() {
            writeln!(
                writer,
                "{:8} {:8} {:4} {:4} {:4}",
                ik + 1,
                nn.ikb + 1,
                nn.gshift[0],
                nn.gshift[1],
                nn.gshift[2]
            )?;
        }
    }
    writeln!(writer, "end nnkpts")?;

    writer.flush()?;
    Ok(())
}

fn write_amn_file(
    filename: &str,
    num_bands: usize,
    num_kpts: usize,
    num_wann: usize,
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "Generated by dftworks (identity gauge guess for first num_wann bands)"
    )?;
    writeln!(writer, "{:8} {:8} {:8}", num_bands, num_kpts, num_wann)?;

    for ik in 1..=num_kpts {
        for n in 1..=num_wann {
            for m in 1..=num_bands {
                let val = if m == n { 1.0 } else { 0.0 };
                writeln!(
                    writer,
                    "{:5} {:5} {:5} {:18.12E} {:18.12E}",
                    m, n, ik, val, 0.0
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_mmn_file(
    filename: &str,
    num_bands: usize,
    topology: &MeshTopology,
    eigvecs: &[Matrix<c64>],
    all_pwbasis: &[PWBasis],
    gvec: &GVector,
) -> io::Result<()> {
    let nkpt = eigvecs.len();
    if nkpt != all_pwbasis.len() || nkpt != topology.neighbors.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inconsistent dimensions for Wannier90 mmn export",
        ));
    }

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "Generated by dftworks")?;
    writeln!(writer, "{:8} {:8} {:8}", num_bands, nkpt, topology.nntot)?;

    let miller = gvec.get_miller();
    let mut k_lookup_maps = Vec::with_capacity(nkpt);
    for pw in all_pwbasis.iter() {
        let mut map = HashMap::with_capacity(pw.get_n_plane_waves() * 2);
        for (row, gidx) in pw.get_gindex().iter().enumerate() {
            let m = miller[*gidx];
            map.insert((m.x, m.y, m.z), row);
        }
        k_lookup_maps.push(map);
    }

    for ik in 0..nkpt {
        if eigvecs[ik].ncol() != num_bands {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "number of bands in eigenvectors does not match control.nband",
            ));
        }

        for nn in topology.neighbors[ik].iter() {
            let ikb = nn.ikb;

            writeln!(
                writer,
                "{:8} {:8} {:4} {:4} {:4}",
                ik + 1,
                ikb + 1,
                nn.gshift[0],
                nn.gshift[1],
                nn.gshift[2]
            )?;

            let block = compute_mmn_block(
                &eigvecs[ik],
                &eigvecs[ikb],
                all_pwbasis[ik].get_gindex(),
                &k_lookup_maps[ikb],
                miller,
                nn.gshift,
                num_bands,
            );

            // Wannier90 expects n outer, m inner.
            for n in 0..num_bands {
                for m in 0..num_bands {
                    let v = block[m + n * num_bands];
                    writeln!(writer, " {:18.12E} {:18.12E}", v.re, v.im)?;
                }
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn compute_mmn_block(
    c_k: &Matrix<c64>,
    c_kb: &Matrix<c64>,
    gindex_k: &[usize],
    lookup_kb: &HashMap<(i32, i32, i32), usize>,
    miller: &[vector3::Vector3i32],
    gshift: [i32; 3],
    num_bands: usize,
) -> Vec<c64> {
    let mut out = vec![c64::new(0.0, 0.0); num_bands * num_bands];

    for (row_k, gidx) in gindex_k.iter().enumerate() {
        let m = miller[*gidx];
        let key = (m.x + gshift[0], m.y + gshift[1], m.z + gshift[2]);

        let Some(&row_kb) = lookup_kb.get(&key) else {
            continue;
        };

        for ib in 0..num_bands {
            let c1_conj = c_k[[row_k, ib]].conj();
            for jb in 0..num_bands {
                out[ib + jb * num_bands] += c1_conj * c_kb[[row_kb, jb]];
            }
        }
    }

    out
}

fn merge_rank_parts<F>(out_file: &str, part_file_for_rank: &F, nrank: i32) -> io::Result<()>
where
    F: Fn(i32) -> String,
{
    let out = File::create(out_file)?;
    let mut writer = BufWriter::new(out);

    for rank in 0..nrank {
        let part_file = part_file_for_rank(rank);
        if Path::new(&part_file).exists() {
            let data = fs::read(&part_file)?;
            writer.write_all(&data)?;
        }
    }

    writer.flush()?;

    for rank in 0..nrank {
        let part_file = part_file_for_rank(rank);
        if Path::new(&part_file).exists() {
            fs::remove_file(part_file)?;
        }
    }

    Ok(())
}

fn validate_k_mesh(k_mesh: [i32; 3]) -> io::Result<()> {
    if k_mesh.iter().any(|&x| x <= 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wannier90 export requires kpts_scheme = kmesh",
        ));
    }

    Ok(())
}

fn read_k_shift(in_kmesh_file: &str) -> io::Result<[i32; 3]> {
    let file = File::open(in_kmesh_file)?;
    let lines: Vec<String> = BufReader::new(file)
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

fn build_mesh_topology(
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
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_write_eig_part_file_converts_to_ev() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = std::env::temp_dir()
            .join(format!("wannier90_test_{}.eig.part", now))
            .display()
            .to_string();

        let eigvals = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        write_eig_part_file(&filename, &eigvals, 2).unwrap();

        let content = fs::read_to_string(&filename).unwrap();
        fs::remove_file(&filename).unwrap();

        assert!(content.contains("     1      3"));
        assert!(content.contains("     2      4"));

        let first_line = content.lines().next().unwrap();
        let cols: Vec<&str> = first_line.split_whitespace().collect();
        let energy_ev = cols[2].parse::<f64>().unwrap();
        assert!((energy_ev - dwconsts::HA_TO_EV).abs() < 1.0e-10);
    }

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
