use crate::types::{HybridKind, TrialOrbital};
use crystal::Crystal;
use dwconsts::TWOPI;
use gvector::GVector;
use kgylm::KGYLM;
use kpts::KPTS;
use matrix::Matrix;
use pspot::PSPot;
use pwbasis::PWBasis;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use types::c64;

pub(crate) fn build_trial_orbitals(
    num_wann: usize,
    crystal: &Crystal,
    pots: &PSPot,
) -> io::Result<Vec<TrialOrbital>> {
    // Build an ordered list of candidate projector orbitals from PP_CHI
    // channels in pseudopotentials, then truncate to num_wann.
    let mut all_trial_orbitals = Vec::new();

    for (iat, species) in crystal.get_atom_species().iter().enumerate() {
        let atpsp = pots.get_psp(species);
        let l_channels = atpsp.get_wfc_channels();
        if l_channels.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "no pseudo-atomic wavefunctions (PP_CHI) found for species '{}'",
                    species
                ),
            ));
        }

        for l in l_channels {
            // Include all m for each angular momentum channel.
            for m in utility::get_quant_num_m(l) {
                all_trial_orbitals.push(TrialOrbital {
                    atom_index: iat,
                    species: species.clone(),
                    l,
                    m,
                    hybrid_kind: None,
                    hybrid_group: None,
                    hybrid_component: None,
                });
            }
        }
    }

    if all_trial_orbitals.len() < num_wann {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "not enough trial orbitals from pseudo-atomic wavefunctions: requested num_wann={}, available={}",
                num_wann,
                all_trial_orbitals.len()
            ),
        ));
    }

    all_trial_orbitals.truncate(num_wann);
    Ok(all_trial_orbitals)
}

pub(crate) fn build_trial_orbitals_from_win(
    win_file: &str,
    num_wann: usize,
    crystal: &Crystal,
    pots: &PSPot,
) -> io::Result<Vec<TrialOrbital>> {
    let file = File::open(win_file)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;

    let mut inside_projection_block = false;
    let mut found_projection_block = false;
    let mut projection_entries: Vec<String> = Vec::new();

    for line in lines.iter() {
        let trimmed = line.trim();
        let marker = trimmed.trim_start_matches('!').trim();

        if marker.eq_ignore_ascii_case("begin projections") {
            inside_projection_block = true;
            found_projection_block = true;
            continue;
        }

        if inside_projection_block && marker.eq_ignore_ascii_case("end projections") {
            break;
        }

        if !inside_projection_block {
            continue;
        }

        let line_wo_bang = trimmed.split('!').next().unwrap_or("").trim();
        let line_wo_comment = line_wo_bang.split('#').next().unwrap_or("").trim();
        if line_wo_comment.is_empty() {
            continue;
        }

        projection_entries.push(line_wo_comment.to_string());
    }

    if !found_projection_block {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "'{}' does not contain a 'begin projections' block",
                win_file
            ),
        ));
    }

    if projection_entries.is_empty() {
        // Keep compatibility with old behavior when projections are not set.
        return build_trial_orbitals(num_wann, crystal, pots);
    }

    let atom_species = crystal.get_atom_species();
    let mut trial_orbitals = Vec::new();
    let mut next_hybrid_group = 0usize;

    for entry in projection_entries.iter() {
        let mut fields = entry.splitn(2, ':').map(|x| x.trim());
        let target = fields.next().unwrap_or("");
        let projector_spec = fields.next().unwrap_or("");

        if target.is_empty() || projector_spec.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid projection entry in '{}': '{}'", win_file, entry),
            ));
        }

        let atom_indices = resolve_projection_target(target, atom_species)?;
        let projection_tokens = parse_projection_tokens(projector_spec)?;

        if projection_tokens.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("projection '{}' does not request any orbitals", entry),
            ));
        }

        for &iat in atom_indices.iter() {
            let species = atom_species[iat].clone();
            let atpsp = pots.get_psp(&species);
            for token in projection_tokens.iter() {
                match token {
                    ProjectionToken::AngularChannel(l) => {
                        if !atpsp.has_wfc(*l) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "projection '{}' requires species '{}' l={} but pseudopotential lacks this PP_CHI channel",
                                    entry, species, l
                                ),
                            ));
                        }

                        for m in utility::get_quant_num_m(*l) {
                            trial_orbitals.push(TrialOrbital {
                                atom_index: iat,
                                species: species.clone(),
                                l: *l,
                                m,
                                hybrid_kind: None,
                                hybrid_group: None,
                                hybrid_component: None,
                            });
                        }
                    }
                    ProjectionToken::Sp3 => {
                        if !atpsp.has_wfc(0) || !atpsp.has_wfc(1) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "projection '{}' requires species '{}' to provide both s(l=0) and p(l=1) PP_CHI channels",
                                    entry, species
                                ),
                            ));
                        }

                        let hybrid_group = next_hybrid_group;
                        next_hybrid_group += 1;

                        // Internal ordering for one sp3 set:
                        // [s, p(m=-1), p(m=0), p(m=1)].
                        trial_orbitals.push(TrialOrbital {
                            atom_index: iat,
                            species: species.clone(),
                            l: 0,
                            m: 0,
                            hybrid_kind: Some(HybridKind::Sp3),
                            hybrid_group: Some(hybrid_group),
                            hybrid_component: Some(0),
                        });
                        for (component, m) in [-1, 0, 1].iter().enumerate() {
                            trial_orbitals.push(TrialOrbital {
                                atom_index: iat,
                                species: species.clone(),
                                l: 1,
                                m: *m,
                                hybrid_kind: Some(HybridKind::Sp3),
                                hybrid_group: Some(hybrid_group),
                                hybrid_component: Some(component + 1),
                            });
                        }
                    }
                }
            }
        }
    }

    if trial_orbitals.len() != num_wann {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "projection block in '{}' expands to {} trial orbitals, but wannier90_num_wann is {}",
                win_file,
                trial_orbitals.len(),
                num_wann
            ),
        ));
    }

    Ok(trial_orbitals)
}

pub(crate) fn build_trial_orbitals_from_nnkp(
    nnkp_file: &str,
    num_wann: usize,
    crystal: &Crystal,
    pots: &PSPot,
) -> io::Result<Vec<TrialOrbital>> {
    let file = File::open(nnkp_file)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;

    let mut cursor = 0usize;
    while cursor < lines.len() {
        if lines[cursor]
            .trim()
            .eq_ignore_ascii_case("begin projections")
        {
            cursor += 1;
            break;
        }
        cursor += 1;
    }

    if cursor >= lines.len() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "'{}' does not contain a 'begin projections' block",
                nnkp_file
            ),
        ));
    }

    let nproj_line = next_data_line(&lines, &mut cursor).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "'{}' has an incomplete 'begin projections' block (missing projector count)",
                nnkp_file
            ),
        )
    })?;
    let nproj = nproj_line
        .split_whitespace()
        .next()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid projector-count line in '{}': '{}'",
                    nnkp_file, nproj_line
                ),
            )
        })?
        .parse::<usize>()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid projector-count value in '{}': '{}'",
                    nnkp_file, nproj_line
                ),
            )
        })?;

    let mut trial_orbitals = Vec::with_capacity(nproj);
    let atom_species = crystal.get_atom_species();
    let mut sp3_groups: HashMap<(usize, i32, String), usize> = HashMap::new();
    let mut next_sp3_group = 0usize;

    for iproj in 0..nproj {
        let header = next_data_line(&lines, &mut cursor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "projection #{} in '{}' is missing its header line",
                    iproj + 1,
                    nnkp_file
                ),
            )
        })?;
        let header_fields: Vec<&str> = header.split_whitespace().collect();
        if header_fields.len() < 6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid projection header in '{}': '{}'", nnkp_file, header),
            ));
        }

        let center_x = parse_projection_float(header_fields[0], nnkp_file, "x-coordinate")?;
        let center_y = parse_projection_float(header_fields[1], nnkp_file, "y-coordinate")?;
        let center_z = parse_projection_float(header_fields[2], nnkp_file, "z-coordinate")?;
        let l_raw = header_fields[3].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid angular momentum in '{}': '{}'", nnkp_file, header),
            )
        })?;
        let mr = header_fields[4].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid m_r index in '{}': '{}'", nnkp_file, header),
            )
        })?;
        let radial_channel =
            parse_nnkp_radial_channel(header_fields[5], nnkp_file, header_fields[5])?;

        let orientation = next_data_line(&lines, &mut cursor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "projection #{} in '{}' is missing its orientation line",
                    iproj + 1,
                    nnkp_file
                ),
            )
        })?;

        let atom_index =
            resolve_projection_center_to_atom([center_x, center_y, center_z], crystal)?;
        let species = atom_species[atom_index].clone();
        let atpsp = pots.get_psp(&species);

        match l_raw {
            l if l >= 0 => {
                let l = l as usize;
                if !atpsp.has_wfc(l) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "projection in '{}' requires species '{}' l={} but pseudopotential lacks this PP_CHI channel",
                            nnkp_file, species, l
                        ),
                    ));
                }

                let m_list = utility::get_quant_num_m(l);
                if mr == 0 || mr > m_list.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "projection in '{}' has out-of-range m_r={} for l={}",
                            nnkp_file, mr, l
                        ),
                    ));
                }

                trial_orbitals.push(TrialOrbital {
                    atom_index,
                    species,
                    l,
                    m: m_list[mr - 1],
                    hybrid_kind: None,
                    hybrid_group: None,
                    hybrid_component: None,
                });
            }
            -3 => {
                if !atpsp.has_wfc(0) || !atpsp.has_wfc(1) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "projection in '{}' requires species '{}' to provide both s(l=0) and p(l=1) PP_CHI channels",
                            nnkp_file, species
                        ),
                    ));
                }

                let (l, m, component) = match mr {
                    1 => (0usize, 0i32, 0usize),
                    2 => (1usize, -1i32, 1usize),
                    3 => (1usize, 0i32, 2usize),
                    4 => (1usize, 1i32, 3usize),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "projection in '{}' has out-of-range sp3 m_r={} (expected 1..4)",
                                nnkp_file, mr
                            ),
                        ))
                    }
                };

                let sp3_key = (
                    atom_index,
                    radial_channel,
                    orientation.trim().to_ascii_lowercase(),
                );
                let group = *sp3_groups.entry(sp3_key).or_insert_with(|| {
                    let group = next_sp3_group;
                    next_sp3_group += 1;
                    group
                });

                trial_orbitals.push(TrialOrbital {
                    atom_index,
                    species,
                    l,
                    m,
                    hybrid_kind: Some(HybridKind::Sp3),
                    hybrid_group: Some(group),
                    hybrid_component: Some(component),
                });
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "unsupported hybrid/angular projection l={} in '{}' (currently supports l>=0 and l=-3/sp3)",
                        l_raw, nnkp_file
                    ),
                ));
            }
        }
    }

    if trial_orbitals.len() != num_wann {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "projection block in '{}' expands to {} trial orbitals, but wannier90_num_wann is {}",
                nnkp_file,
                trial_orbitals.len(),
                num_wann
            ),
        ));
    }

    Ok(trial_orbitals)
}

fn resolve_projection_target(target: &str, atom_species: &[String]) -> io::Result<Vec<usize>> {
    if target == "*" || target.eq_ignore_ascii_case("all") {
        return Ok((0..atom_species.len()).collect());
    }

    if let Ok(iat_1based) = target.parse::<usize>() {
        if iat_1based == 0 || iat_1based > atom_species.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "projection target '{}' is out of atom index range 1..{}",
                    target,
                    atom_species.len()
                ),
            ));
        }
        return Ok(vec![iat_1based - 1]);
    }

    let mut exact_matches = Vec::new();
    for (iat, species) in atom_species.iter().enumerate() {
        if species.eq_ignore_ascii_case(target) {
            exact_matches.push(iat);
        }
    }
    if !exact_matches.is_empty() {
        return Ok(exact_matches);
    }

    let target_base = strip_species_suffix(target);
    let mut base_matches = Vec::new();
    for (iat, species) in atom_species.iter().enumerate() {
        if strip_species_suffix(species).eq_ignore_ascii_case(&target_base) {
            base_matches.push(iat);
        }
    }
    if !base_matches.is_empty() {
        return Ok(base_matches);
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("projection target '{}' matches no atoms in crystal", target),
    ))
}

fn strip_species_suffix(species: &str) -> String {
    species
        .trim()
        .trim_end_matches(|ch: char| ch.is_ascii_digit())
        .to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionToken {
    AngularChannel(usize),
    Sp3,
}

fn parse_projection_tokens(spec: &str) -> io::Result<Vec<ProjectionToken>> {
    let normalized = spec.trim().to_lowercase().replace(' ', "");
    let mut tokens = Vec::new();

    for token in normalized.split(|c| c == ';' || c == ',' || c == '+') {
        if token.is_empty() {
            continue;
        }

        match token {
            "s" => tokens.push(ProjectionToken::AngularChannel(0)),
            "p" => tokens.push(ProjectionToken::AngularChannel(1)),
            "d" => tokens.push(ProjectionToken::AngularChannel(2)),
            "f" => tokens.push(ProjectionToken::AngularChannel(3)),
            "sp3" => tokens.push(ProjectionToken::Sp3),
            "sp" => {
                tokens.push(ProjectionToken::AngularChannel(0));
                tokens.push(ProjectionToken::AngularChannel(1));
            }
            "spd" => {
                tokens.push(ProjectionToken::AngularChannel(0));
                tokens.push(ProjectionToken::AngularChannel(1));
                tokens.push(ProjectionToken::AngularChannel(2));
            }
            "spdf" => {
                tokens.push(ProjectionToken::AngularChannel(0));
                tokens.push(ProjectionToken::AngularChannel(1));
                tokens.push(ProjectionToken::AngularChannel(2));
                tokens.push(ProjectionToken::AngularChannel(3));
            }
            _ => {
                if token.chars().all(|ch| matches!(ch, 's' | 'p' | 'd' | 'f')) {
                    for ch in token.chars() {
                        match ch {
                            's' => tokens.push(ProjectionToken::AngularChannel(0)),
                            'p' => tokens.push(ProjectionToken::AngularChannel(1)),
                            'd' => tokens.push(ProjectionToken::AngularChannel(2)),
                            'f' => tokens.push(ProjectionToken::AngularChannel(3)),
                            _ => {}
                        }
                    }
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "unsupported projection spec '{}'; use combinations of s/p/d/f (e.g. s;p, sp3, spd)",
                            spec
                        ),
                    ));
                }
            }
        }
    }

    Ok(tokens)
}

fn parse_projection_float(token: &str, filename: &str, name: &str) -> io::Result<f64> {
    token.parse::<f64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid projection {} in '{}': '{}'", name, filename, token),
        )
    })
}

fn parse_nnkp_radial_channel(token: &str, filename: &str, raw_line_value: &str) -> io::Result<i32> {
    if let Ok(v) = token.parse::<i32>() {
        return Ok(v);
    }

    let as_float = token.parse::<f64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid radial channel in '{}': '{}'",
                filename, raw_line_value
            ),
        )
    })?;
    let rounded = as_float.round();
    if (as_float - rounded).abs() > 1.0e-8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid non-integer radial channel in '{}': '{}'",
                filename, raw_line_value
            ),
        ));
    }

    Ok(rounded as i32)
}

fn resolve_projection_center_to_atom(center: [f64; 3], crystal: &Crystal) -> io::Result<usize> {
    let mut best_iat = 0usize;
    let mut best_dist2 = f64::MAX;

    for (iat, atom_pos) in crystal.get_atom_positions().iter().enumerate() {
        let dx = wrap_frac_delta(center[0] - atom_pos.x);
        let dy = wrap_frac_delta(center[1] - atom_pos.y);
        let dz = wrap_frac_delta(center[2] - atom_pos.z);
        let dist2 = dx * dx + dy * dy + dz * dz;
        if dist2 < best_dist2 {
            best_dist2 = dist2;
            best_iat = iat;
        }
    }

    // `wannier90.x -pp` writes centers in fractional coordinates with finite
    // precision, so a small tolerance is expected.
    let tol = 5.0e-3f64;
    if best_dist2.sqrt() > tol {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "projection center ({:.6}, {:.6}, {:.6}) is not close to any atom position",
                center[0], center[1], center[2]
            ),
        ));
    }

    Ok(best_iat)
}

fn wrap_frac_delta(delta: f64) -> f64 {
    delta - delta.round()
}

fn next_data_line<'a>(lines: &'a [String], cursor: &mut usize) -> Option<&'a str> {
    while *cursor < lines.len() {
        let line = lines[*cursor].trim();
        *cursor += 1;

        let no_bang = line.split('!').next().unwrap_or("").trim();
        let no_hash = no_bang.split('#').next().unwrap_or("").trim();
        if no_hash.is_empty() {
            continue;
        }

        return Some(no_hash);
    }

    None
}

pub(crate) fn write_amn_file(
    filename: &str,
    num_bands: usize,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    eigvecs: &[Matrix<c64>],
    all_pwbasis: &[PWBasis],
    gvec: &GVector,
    pots: &PSPot,
    trial_orbitals: &[TrialOrbital],
) -> io::Result<()> {
    // AMN contains overlaps:
    //   A_{m n}^{(k)} = <psi_{m k} | g_n>
    // where g_n are trial orbitals (projectors).
    let nkpt = eigvecs.len();
    if nkpt != all_pwbasis.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inconsistent dimensions for Wannier90 amn export",
        ));
    }

    let num_wann = trial_orbitals.len();
    if num_wann == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "trial orbital list for AMN must not be empty",
        ));
    }

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "Generated by dftworks (Bloch-trial overlaps from pseudo-atomic wavefunctions)"
    )?;
    writeln!(writer, "{:8} {:8} {:8}", num_bands, nkpt, num_wann)?;

    let lmax_trial = trial_orbitals.iter().map(|orb| orb.l).max().unwrap_or(0);
    let atom_positions = crystal.get_atom_positions();
    let miller = gvec.get_miller();

    for ik in 0..nkpt {
        if eigvecs[ik].ncol() != num_bands {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "number of bands in eigenvectors does not match control.nband",
            ));
        }

        let pwwfc = &all_pwbasis[ik];
        let npw = pwwfc.get_n_plane_waves();
        // Y_lm(k+G) values reused for all bands at this k.
        let kgylm = KGYLM::new(pwwfc.get_k_cart(), lmax_trial, gvec, pwwfc);
        let k_frac = kpts.get_k_frac(ik);

        // Per-atom structure factors include additional k·tau phase needed
        // by Wannier90 AMN convention for Bloch functions.
        let mut atom_structure_factors = Vec::with_capacity(atom_positions.len());
        for atom_pos in atom_positions.iter() {
            let mut sfact = fhkl::compute_structure_factor_for_many_g_one_atom(
                miller,
                pwwfc.get_gindex(),
                *atom_pos,
            );

            let k_dot_tau = k_frac.x * atom_pos.x + k_frac.y * atom_pos.y + k_frac.z * atom_pos.z;
            let k_phase = c64::new(0.0, -TWOPI * k_dot_tau).exp();
            for s in sfact.iter_mut() {
                *s *= k_phase;
            }

            atom_structure_factors.push(sfact);
        }

        let mut radial_lookup = HashMap::new();
        for orb in trial_orbitals.iter() {
            let key = (orb.species.clone(), orb.l);
            if radial_lookup.contains_key(&key) {
                // Reuse radial transform for same (species,l).
                continue;
            }

            let atpsp = pots.get_psp(&orb.species);
            if !atpsp.has_wfc(orb.l) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "missing pseudo-atomic wavefunction for species '{}' and l={}",
                        orb.species, orb.l
                    ),
                ));
            }

            let chi = atpsp.get_wfc(orb.l);
            // Transform pseudo-atomic radial orbital to |k+G| representation.
            let chi_kg = compute_atomic_wfc_of_kg(
                pwwfc.get_kg(),
                orb.l,
                chi,
                atpsp.get_rad(),
                atpsp.get_rab(),
                crystal.get_latt().volume(),
            )?;
            radial_lookup.insert(key, chi_kg);
        }

        let mut overlaps = vec![c64::new(0.0, 0.0); num_bands * num_wann];

        for (n, orb) in trial_orbitals.iter().enumerate() {
            let ylm = kgylm.get_data(orb.l, orb.m);
            let chi_kg = radial_lookup
                .get(&(orb.species.clone(), orb.l))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "failed to locate precomputed radial orbital in AMN export",
                    )
                })?;
            let sfact = &atom_structure_factors[orb.atom_index];

            for mband in 0..num_bands {
                // Overlap in PW basis:
                // <psi_mk|g_n> = sum_G c_mk^*(G) g_n(k+G)
                let mut overlap = c64::new(0.0, 0.0);
                for ig in 0..npw {
                    let g_n = ylm[ig] * chi_kg[ig] * sfact[ig];
                    overlap += eigvecs[ik][[ig, mband]].conj() * g_n;
                }

                overlaps[mband + n * num_bands] = overlap;
            }
        }

        apply_hybrid_transformations(num_bands, trial_orbitals, &mut overlaps)?;

        for n in 0..num_wann {
            for mband in 0..num_bands {
                let overlap = overlaps[mband + n * num_bands];
                writeln!(
                    writer,
                    "{:5} {:5} {:5} {:18.12E} {:18.12E}",
                    mband + 1,
                    n + 1,
                    ik + 1,
                    overlap.re,
                    overlap.im
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn apply_hybrid_transformations(
    num_bands: usize,
    trial_orbitals: &[TrialOrbital],
    overlaps: &mut [c64],
) -> io::Result<()> {
    let mut sp3_groups: HashMap<usize, [Option<usize>; 4]> = HashMap::new();

    for (n, orb) in trial_orbitals.iter().enumerate() {
        if orb.hybrid_kind != Some(HybridKind::Sp3) {
            continue;
        }

        let group = orb.hybrid_group.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "missing hybrid_group metadata for sp3 trial orbital",
            )
        })?;
        let component = orb.hybrid_component.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "missing hybrid_component metadata for sp3 trial orbital",
            )
        })?;
        if component >= 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sp3 hybrid component index",
            ));
        }

        let entry = sp3_groups.entry(group).or_insert([None, None, None, None]);
        if entry[component].is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate orbital assigned to the same sp3 hybrid component",
            ));
        }
        entry[component] = Some(n);
    }

    for components in sp3_groups.values() {
        let [Some(i_s), Some(i_py), Some(i_pz), Some(i_px)] = *components else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "incomplete sp3 hybrid group in trial-orbital list",
            ));
        };

        for mband in 0..num_bands {
            let s = overlaps[mband + i_s * num_bands];
            let py = overlaps[mband + i_py * num_bands];
            let pz = overlaps[mband + i_pz * num_bands];
            let px = overlaps[mband + i_px * num_bands];

            // Tetrahedral sp3 combinations, using internal order
            // [s, p(m=-1), p(m=0), p(m=1)] = [s, py~, pz~, px~].
            // The signs map the real-spherical basis to cartesian-like hybrids.
            let h1 = 0.5 * (s - px - py + pz);
            let h2 = 0.5 * (s - px + py - pz);
            let h3 = 0.5 * (s + px - py - pz);
            let h4 = 0.5 * (s + px + py + pz);

            overlaps[mband + i_s * num_bands] = h1;
            overlaps[mband + i_py * num_bands] = h2;
            overlaps[mband + i_pz * num_bands] = h3;
            overlaps[mband + i_px * num_bands] = h4;
        }
    }

    Ok(())
}

fn compute_atomic_wfc_of_kg(
    kg: &[f64],
    l: usize,
    chi: &[f64],
    rad: &[f64],
    rab: &[f64],
    volume: f64,
) -> io::Result<Vec<f64>> {
    if chi.len() != rad.len() || rad.len() != rab.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inconsistent radial-grid dimensions while building trial orbitals",
        ));
    }

    let npw = kg.len();
    let mut chi_kg = vec![0.0; npw];
    let mut work = vec![0.0; rad.len()];
    let prefactor = dwconsts::FOURPI / volume.sqrt();

    for ig in 0..npw {
        // Radial integral of chi(r) * r * j_l(|k+G|r).
        for ir in 0..rad.len() {
            let r = rad[ir];
            work[ir] = chi[ir] * r * special::spherical_bessel_jn(l, kg[ig] * r);
        }
        // Simpson integration over radial mesh.
        chi_kg[ig] = prefactor * integral::simpson_rab(&work, rab);
    }

    Ok(chi_kg)
}

#[cfg(test)]
mod tests {
    use super::{parse_projection_tokens, resolve_projection_target, ProjectionToken};

    #[test]
    fn test_parse_projector_channels_accepts_common_specs() {
        assert_eq!(
            parse_projection_tokens("s").unwrap(),
            vec![ProjectionToken::AngularChannel(0)]
        );
        assert_eq!(
            parse_projection_tokens("sp3").unwrap(),
            vec![ProjectionToken::Sp3]
        );
        assert_eq!(
            parse_projection_tokens("s;p;d").unwrap(),
            vec![
                ProjectionToken::AngularChannel(0),
                ProjectionToken::AngularChannel(1),
                ProjectionToken::AngularChannel(2)
            ]
        );
        assert_eq!(
            parse_projection_tokens("spdf").unwrap(),
            vec![
                ProjectionToken::AngularChannel(0),
                ProjectionToken::AngularChannel(1),
                ProjectionToken::AngularChannel(2),
                ProjectionToken::AngularChannel(3)
            ]
        );
        assert_eq!(
            parse_projection_tokens("s, p + d").unwrap(),
            vec![
                ProjectionToken::AngularChannel(0),
                ProjectionToken::AngularChannel(1),
                ProjectionToken::AngularChannel(2)
            ]
        );
    }

    #[test]
    fn test_parse_projector_channels_rejects_unsupported_spec() {
        assert!(parse_projection_tokens("sp2").is_err());
        assert!(parse_projection_tokens("foo").is_err());
    }

    #[test]
    fn test_resolve_projection_target_supports_species_and_atom_index() {
        let atom_species = vec!["Si1".to_string(), "Si1".to_string(), "O1".to_string()];

        assert_eq!(
            resolve_projection_target("Si1", &atom_species).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            resolve_projection_target("Si", &atom_species).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            resolve_projection_target("2", &atom_species).unwrap(),
            vec![1]
        );
        assert_eq!(
            resolve_projection_target("*", &atom_species).unwrap(),
            vec![0, 1, 2]
        );
    }
}
