#![allow(warnings)]
use control::{Control, FftPlannerScheme, SpinScheme};
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use kpts_distribution::KPointDomain;
use kscf::KSCF;
use matrix::*;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::*;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::*;
use rayon;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;
use vnl::VNL;

const RESTART_LATTICE_TOL: f64 = 1.0e-8;
const RESTART_META_TOL: f64 = 1.0e-12;
const PROVENANCE_SCHEMA_VERSION: &str = "provenance-v1";

fn display_program_header() {
    println!();
    println!("   {:=^88}", "");
    println!("   {:^88}", "DFTWorks");
    println!(
        "   {:^88}",
        "Self-Consistent Plane-Wave Density Functional Theory"
    );
    println!("   {:=^88}", "");
    println!();
}

fn format_unix_seconds_as_utc_iso(timestamp_unix_s: u64) -> String {
    // Convert Unix epoch seconds to an ISO-like UTC timestamp without external crates.
    // Algorithm adapted from civil date conversion by Howard Hinnant.
    let days = (timestamp_unix_s / 86_400) as i64;
    let seconds_of_day = timestamp_unix_s % 86_400;

    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = mp + if mp < 10 { 3 } else { -9 }; // [1, 12]
    if month <= 2 {
        year += 1;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

fn display_system_information() {
    const OUT_WIDTH1: usize = 28;
    const OUT_WIDTH2: usize = 18;

    let backend = dwfft3d::backend_name();
    let fft_runtime = dwfft3d::backend_options();
    let mpi_rank = dwmpi::get_comm_world_rank();
    let mpi_ranks = dwmpi::get_comm_world_size();
    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp_unix_s = now.as_secs();
    let timestamp_utc = format_unix_seconds_as_utc_iso(timestamp_unix_s);

    println!("   {:-^88}", " system information ");
    println!();
    println!(
        "   {:<width1$} = {:>width2$}",
        "backend",
        backend,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "fft_threads",
        fft_runtime.threads,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "fft_planner",
        fft_runtime.planning_mode.as_str(),
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {}",
        "fft_wisdom_file",
        fft_runtime.wisdom_file.as_deref().unwrap_or("(none)"),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "hostname",
        hostname,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "os",
        std::env::consts::OS,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "arch",
        std::env::consts::ARCH,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "mpi_rank",
        mpi_rank,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "mpi_ranks",
        mpi_ranks,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {}",
        "timestamp_utc",
        timestamp_utc,
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:>4} (RAYON_NUM_THREADS={}, host_threads={})",
        "rayon_threads",
        rayon_threads,
        rayon_env,
        host_threads,
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {}",
        "working_directory",
        cwd,
        width1 = OUT_WIDTH1
    );
    println!();
}

#[derive(Clone, Debug)]
struct InputHashEntry {
    path: String,
    fnv1a64: String,
}

#[derive(Clone, Debug)]
struct RunProvenanceManifest {
    generated_utc: String,
    git_commit: String,
    build_features: Vec<String>,
    fft_backend: String,
    mpi_ranks: i32,
    rayon_threads: usize,
    rayon_env: String,
    spin_scheme: String,
    kpts_scheme: String,
    k_mesh: [i32; 3],
    random_seed: Option<u64>,
    input_hashes: Vec<InputHashEntry>,
    replay_fingerprint: String,
}

fn escape_json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes.iter() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

fn hash_file_fnv1a64(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read '{}': {}", path.display(), err))?;
    Ok(fnv1a64_hex(&bytes))
}

fn parse_in_pot_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }

    let file =
        File::open(path).map_err(|err| format!("failed to open '{}': {}", path.display(), err))?;
    let mut out = Vec::new();

    for (line_idx, line_res) in BufReader::new(file).lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line_res.map_err(|err| {
            format!(
                "failed to read '{}', line {}: {}",
                path.display(),
                line_no,
                err
            )
        })?;

        let line = line.split('#').next().unwrap_or(line.as_str());
        let line = line.split('!').next().unwrap_or(line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 2 {
            return Err(format!(
                "'{}' line {}: expected '<species> <filename>'",
                path.display(),
                line_no
            ));
        }

        out.push(PathBuf::from("pot").join(tokens[1]));
    }

    out.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    out.dedup();

    Ok(out)
}

fn collect_input_hashes(control: &Control) -> Result<Vec<InputHashEntry>, String> {
    let mut files = vec![
        PathBuf::from("in.ctrl"),
        PathBuf::from("in.crystal"),
        PathBuf::from("in.pot"),
    ];

    match control.get_kpts_scheme_enum() {
        control::KptsScheme::Kmesh => files.push(PathBuf::from("in.kmesh")),
        control::KptsScheme::Kline => files.push(PathBuf::from("in.kline")),
    }

    files.extend(parse_in_pot_paths(Path::new("in.pot"))?);

    files.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    files.dedup();

    let mut out = Vec::with_capacity(files.len());
    for path in files.iter() {
        let path_text = path.to_string_lossy().to_string();
        let hash = if path.is_file() {
            hash_file_fnv1a64(path)?
        } else {
            "missing".to_string()
        };
        out.push(InputHashEntry {
            path: path_text,
            fnv1a64: hash,
        });
    }

    Ok(out)
}

fn split_csv_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}

fn compute_replay_fingerprint(manifest: &RunProvenanceManifest) -> String {
    let mut canonical = String::new();
    canonical.push_str(PROVENANCE_SCHEMA_VERSION);
    canonical.push('\n');
    canonical.push_str(manifest.spin_scheme.as_str());
    canonical.push('\n');
    canonical.push_str(manifest.kpts_scheme.as_str());
    canonical.push('\n');
    canonical.push_str(&format!(
        "{},{},{}\n",
        manifest.k_mesh[0], manifest.k_mesh[1], manifest.k_mesh[2]
    ));
    canonical.push_str(
        manifest
            .random_seed
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
            .as_str(),
    );
    canonical.push('\n');
    canonical.push_str(&format!("{}\n", manifest.mpi_ranks));
    canonical.push_str(&format!("{}\n", manifest.rayon_threads));

    for entry in manifest.input_hashes.iter() {
        canonical.push_str(entry.path.as_str());
        canonical.push('=');
        canonical.push_str(entry.fnv1a64.as_str());
        canonical.push('\n');
    }

    fnv1a64_hex(canonical.as_bytes())
}

fn extract_json_string_field(doc: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\": \"", field);
    let start = doc.find(needle.as_str())?;
    let value_start = start + needle.len();
    let tail = &doc[value_start..];
    let value_end = tail.find('"')?;
    Some(tail[..value_end].to_string())
}

fn verify_replay_manifest(path: &str, expected_fingerprint: &str) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read replay manifest '{}': {}", path, err))?;

    let schema = extract_json_string_field(text.as_str(), "schema_version")
        .ok_or_else(|| format!("replay manifest '{}' is missing schema_version", path))?;
    if schema != PROVENANCE_SCHEMA_VERSION {
        return Err(format!(
            "replay manifest schema mismatch: expected '{}', got '{}'",
            PROVENANCE_SCHEMA_VERSION, schema
        ));
    }

    let fingerprint = extract_json_string_field(text.as_str(), "replay_fingerprint")
        .ok_or_else(|| format!("replay manifest '{}' is missing replay_fingerprint", path))?;

    if fingerprint != expected_fingerprint {
        return Err(format!(
            "replay manifest mismatch for '{}': expected {}, got {}",
            path, expected_fingerprint, fingerprint
        ));
    }

    Ok(())
}

fn render_provenance_json(manifest: &RunProvenanceManifest) -> String {
    let mut features_json = String::from("[");
    for (idx, feature) in manifest.build_features.iter().enumerate() {
        if idx > 0 {
            features_json.push_str(", ");
        }
        features_json.push('"');
        features_json.push_str(escape_json_string(feature).as_str());
        features_json.push('"');
    }
    features_json.push(']');

    let mut input_hashes_json = String::new();
    for (idx, entry) in manifest.input_hashes.iter().enumerate() {
        if idx > 0 {
            input_hashes_json.push_str(",\n");
        }
        input_hashes_json.push_str(&format!(
            "    {{ \"path\": \"{}\", \"fnv1a64\": \"{}\" }}",
            escape_json_string(entry.path.as_str()),
            entry.fnv1a64
        ));
    }

    format!(
        "{{\n  \"schema_version\": \"{}\",\n  \"generated_utc\": \"{}\",\n  \"binary\": {{\n    \"name\": \"pw\",\n    \"version\": \"{}\",\n    \"git_commit\": \"{}\",\n    \"build_features\": {},\n    \"fft_backend\": \"{}\"\n  }},\n  \"runtime\": {{\n    \"mpi_ranks\": {},\n    \"rayon_threads\": {},\n    \"rayon_env\": \"{}\"\n  }},\n  \"initialization\": {{\n    \"spin_scheme\": \"{}\",\n    \"kpts_scheme\": \"{}\",\n    \"k_mesh\": [{}, {}, {}],\n    \"random_seed\": {}\n  }},\n  \"input_hashes\": [\n{}\n  ],\n  \"replay_fingerprint\": \"{}\"\n}}\n",
        PROVENANCE_SCHEMA_VERSION,
        escape_json_string(manifest.generated_utc.as_str()),
        env!("CARGO_PKG_VERSION"),
        escape_json_string(manifest.git_commit.as_str()),
        features_json,
        escape_json_string(manifest.fft_backend.as_str()),
        manifest.mpi_ranks,
        manifest.rayon_threads,
        escape_json_string(manifest.rayon_env.as_str()),
        escape_json_string(manifest.spin_scheme.as_str()),
        escape_json_string(manifest.kpts_scheme.as_str()),
        manifest.k_mesh[0],
        manifest.k_mesh[1],
        manifest.k_mesh[2],
        manifest
            .random_seed
            .map(|seed| seed.to_string())
            .unwrap_or_else(|| "null".to_string()),
        input_hashes_json,
        manifest.replay_fingerprint
    )
}

fn emit_run_provenance_manifest(control: &Control, kpts: &dyn KPTS) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let generated_utc = format_unix_seconds_as_utc_iso(now);

    let git_commit = option_env!("DWWORKS_GIT_COMMIT")
        .unwrap_or("unknown")
        .trim()
        .to_string();
    let build_features = split_csv_list(option_env!("DWWORKS_BUILD_FEATURES").unwrap_or(""));
    let fft_backend = dwfft3d::backend_name().to_string();
    let mpi_ranks = dwmpi::get_comm_world_size();
    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let spin_scheme = control.get_spin_scheme().to_string();
    let kpts_scheme = control.get_kpts_scheme().to_string();
    let k_mesh = kpts.get_k_mesh();
    let random_seed = control.get_random_seed();
    let input_hashes = collect_input_hashes(control)?;

    let mut manifest = RunProvenanceManifest {
        generated_utc,
        git_commit,
        build_features,
        fft_backend,
        mpi_ranks,
        rayon_threads,
        rayon_env,
        spin_scheme,
        kpts_scheme,
        k_mesh,
        random_seed,
        input_hashes,
        replay_fingerprint: String::new(),
    };
    manifest.replay_fingerprint = compute_replay_fingerprint(&manifest);

    if control.get_provenance_check() {
        verify_replay_manifest(
            control.get_provenance_manifest(),
            manifest.replay_fingerprint.as_str(),
        )?;
    }

    let content = render_provenance_json(&manifest);
    fs::write(control.get_provenance_manifest(), content).map_err(|err| {
        format!(
            "failed to write provenance manifest '{}': {}",
            control.get_provenance_manifest(),
            err
        )
    })?;

    Ok(())
}

fn display_grid_information(fftgrid: &FFTGrid, pwden: &PWDensity) {
    const OUT_WIDTH1: usize = 28;

    println!("   {:-^88}", " grid information ");
    println!();
    println!("   FFTGrid : {}", fftgrid);
    println!("   npw_rho = {}", pwden.get_n_plane_waves());
    println!(
        "   {:<width1$} = {}",
        "nfft",
        fftgrid.get_ntot(),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {}",
        "rho_gshells",
        pwden.get_n_gshell(),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:16.8} 1/bohr ({:16.8} 1/A)",
        "gmax_rho",
        pwden.get_gmax(),
        pwden.get_gmax() * ANG_TO_BOHR,
        width1 = OUT_WIDTH1
    );
}

fn display_symmetry_equivalent_atoms(crystal: &Crystal, symdrv: &dyn SymmetryDriver) {
    let sym_atom = symdrv.get_sym_atom();
    let natoms = sym_atom.len();
    let n_sym = symdrv.get_n_sym_ops();

    println!();
    println!("   {:-^88}", " symmetry-equivalent atoms ");
    println!("   mapping convention: atom(i) --sym_op--> atom(j), 1-based atom index");
    println!("   n_atoms = {}, n_sym_ops = {}", natoms, n_sym);

    let atom_species = crystal.get_atom_species();
    for (iat, mapping_row) in sym_atom.iter().enumerate() {
        let mapped_one_based: Vec<usize> = mapping_row
            .iter()
            .map(|&jat| jat.saturating_add(1))
            .collect();
        let species = atom_species.get(iat).map(|s| s.as_str()).unwrap_or("?");
        println!(
            "   atom {:>4} {:<6} -> {:?}",
            iat + 1,
            format!("({})", species),
            mapped_one_based
        );
    }

    let mut visited = vec![false; natoms];
    let mut classes: Vec<Vec<usize>> = Vec::new();

    for iat in 0..natoms {
        if visited[iat] {
            continue;
        }

        let mut class = vec![iat];
        if let Some(row) = sym_atom.get(iat) {
            for &jat in row.iter() {
                if jat < natoms {
                    class.push(jat);
                }
            }
        }
        class.sort_unstable();
        class.dedup();

        for &jat in class.iter() {
            visited[jat] = true;
        }
        classes.push(class);
    }

    println!("   equivalence classes ({} total)", classes.len());
    for (iclass, class) in classes.iter().enumerate() {
        let one_based: Vec<usize> = class.iter().map(|&iat| iat + 1).collect();
        println!("   class {:>4} -> {:?}", iclass + 1, one_based);
    }
}

struct GeometryStepContext {
    fftgrid: FFTGrid,
    rgtrans: rgtransform::RGTransform,
    gvec: GVector,
    pwden: PWDensity,
    blatt: lattice::Lattice,
}

impl GeometryStepContext {
    fn new(crystal: &Crystal, ecutrho: f64) -> Self {
        let fftgrid = FFTGrid::new(crystal.get_latt(), ecutrho);
        let [n1, n2, n3] = fftgrid.get_size();
        let rgtrans = rgtransform::RGTransform::new(n1, n2, n3);
        let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);
        let pwden = PWDensity::new(ecutrho, &gvec);
        let blatt = crystal.get_latt().reciprocal();

        Self {
            fftgrid,
            rgtrans,
            gvec,
            pwden,
            blatt,
        }
    }

    fn fft_shape(&self) -> [usize; 3] {
        self.fftgrid.get_size()
    }
}

struct OrchestrationWorkspace {
    rhocoreg: Vec<c64>,
    rhocore_3d: Option<Array3<c64>>,
    atom_positions: Vec<[f64; 3]>,
}

impl OrchestrationWorkspace {
    fn new() -> Self {
        Self {
            rhocoreg: Vec::new(),
            rhocore_3d: None,
            atom_positions: Vec::new(),
        }
    }

    fn ensure_shape(&mut self, npw_rho: usize, fft_shape: [usize; 3]) {
        if self.rhocoreg.len() != npw_rho {
            self.rhocoreg.resize(npw_rho, c64::zero());
        } else {
            self.rhocoreg.fill(c64::zero());
        }

        let needs_rhocore_resize = self
            .rhocore_3d
            .as_ref()
            .map(|rho| rho.shape() != fft_shape)
            .unwrap_or(true);
        if needs_rhocore_resize {
            self.rhocore_3d = Some(Array3::<c64>::new(fft_shape));
        }

        self.rhocore_3d
            .as_mut()
            .expect("orchestration workspace rhocore_3d should be initialized")
            .set_value(c64::zero());
    }

    fn update_atom_positions(&mut self, crystal: &Crystal) {
        let natom = crystal.get_n_atoms();
        if self.atom_positions.len() != natom {
            self.atom_positions = vec![[0.0; 3]; natom];
        }

        let vatoms = crystal.get_atom_positions();
        for (dst, src) in self.atom_positions.iter_mut().zip(vatoms.iter()) {
            dst[0] = src.x;
            dst[1] = src.y;
            dst[2] = src.z;
        }
    }

    fn core_charge_buffers_mut(&mut self) -> (&mut [c64], &mut Array3<c64>) {
        let rhocoreg = self.rhocoreg.as_mut_slice();
        let rhocore_3d = self
            .rhocore_3d
            .as_mut()
            .expect("orchestration workspace rhocore_3d should be initialized");
        (rhocoreg, rhocore_3d)
    }

    fn rhocore_3d(&self) -> &Array3<c64> {
        self.rhocore_3d
            .as_ref()
            .expect("orchestration workspace rhocore_3d should be initialized")
    }

    fn atom_positions(&self) -> &[[f64; 3]] {
        self.atom_positions.as_slice()
    }
}

struct RuntimeContext<'a> {
    control: &'a Control,
    crystal: &'a Crystal,
    pots: &'a PSPot,
    kpts: &'a dyn KPTS,
    spin_scheme: SpinScheme,
}

impl<'a> RuntimeContext<'a> {
    fn new(
        control: &'a Control,
        crystal: &'a Crystal,
        pots: &'a PSPot,
        kpts: &'a dyn KPTS,
        spin_scheme: SpinScheme,
    ) -> Self {
        Self {
            control,
            crystal,
            pots,
            kpts,
            spin_scheme,
        }
    }

    fn kpoint_domain(&self) -> KPointDomain {
        let nkpt = self.kpts.get_n_kpts();
        let nrank = dwmpi::get_comm_world_size() as usize;
        KPointDomain::for_current_rank(nkpt, nrank)
    }

    fn checkpoint_meta(&self) -> CheckpointMeta {
        checkpoint_meta_for_run(self.control, self.spin_scheme, self.kpts.get_k_mesh())
    }
}

struct ElectronicStepContext {
    k_domain: KPointDomain,
    vpwwfc: Vec<PWBasis>,
    vvnl: Vec<VNL>,
}

impl ElectronicStepContext {
    fn build(runtime: &RuntimeContext, blatt: &lattice::Lattice, gvec: &GVector) -> Self {
        let k_domain = runtime.kpoint_domain();

        let mut vpwwfc = Vec::<PWBasis>::with_capacity(k_domain.len());
        for slot in k_domain.iter() {
            let ik = slot.global_index;
            let k_frac = runtime.kpts.get_k_frac(ik);
            let k_cart = runtime.kpts.frac_to_cart(&k_frac, blatt);
            let pwwfc = PWBasis::new(k_cart, ik, runtime.control.get_ecut(), gvec);
            vpwwfc.push(pwwfc);
        }

        let mut vvnl = Vec::<VNL>::with_capacity(k_domain.len());
        for slot in k_domain.iter() {
            let ik = slot.global_index;
            let ilocal = slot.local_slot;
            let vnl = VNL::new(ik, runtime.pots, &vpwwfc[ilocal], runtime.crystal);
            vvnl.push(vnl);
        }

        Self {
            k_domain,
            vpwwfc,
            vvnl,
        }
    }

    fn global_k_first_or_zero(&self) -> usize {
        self.k_domain.global_first_or_zero()
    }

    fn kpoint_domain(&self) -> &KPointDomain {
        &self.k_domain
    }

    fn local_nkpt(&self) -> usize {
        self.k_domain.len()
    }

    fn build_scf_state<'a>(
        &'a self,
        runtime: &RuntimeContext<'a>,
        gvec: &'a GVector,
        pwden: &'a PWDensity,
        fft_shape: [usize; 3],
    ) -> (VKSCF<'a>, VKEigenValue, VKEigenVector) {
        let nband = runtime.control.get_nband();
        let my_nkpt = self.local_nkpt();

        let vkscf = match runtime.spin_scheme {
            SpinScheme::NonSpin => {
                let channel = build_kscf_channel(runtime, self, gvec, pwden, fft_shape, 0);
                VKSCF::NonSpin(channel)
            }
            SpinScheme::Spin => {
                let channel_up = build_kscf_channel(runtime, self, gvec, pwden, fft_shape, 0);
                let channel_dn = build_kscf_channel(runtime, self, gvec, pwden, fft_shape, 1);
                VKSCF::Spin(channel_up, channel_dn)
            }
            SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in KSCF setup"),
        };

        let vkevals = allocate_eigenvalues(runtime.spin_scheme, nband, my_nkpt);
        let vkevecs = allocate_eigenvectors(runtime.spin_scheme, nband, &self.vpwwfc);

        (vkscf, vkevals, vkevecs)
    }
}

fn build_kscf_channel<'a>(
    runtime: &RuntimeContext<'a>,
    electronic_ctx: &'a ElectronicStepContext,
    gvec: &'a GVector,
    pwden: &'a PWDensity,
    fft_shape: [usize; 3],
    random_stream_id: u64,
) -> Vec<KSCF<'a>> {
    let mut channel = Vec::<KSCF>::with_capacity(electronic_ctx.k_domain.len());
    let blatt = runtime.crystal.get_latt().reciprocal();

    for slot in electronic_ctx.k_domain.iter() {
        let ik = slot.global_index;
        let ilocal = slot.local_slot;
        let k_frac = runtime.kpts.get_k_frac(ik);
        let k_cart = runtime.kpts.frac_to_cart(&k_frac, &blatt);
        let k_weight = runtime.kpts.get_k_weight(ik);

        let kscf = KSCF::new(
            runtime.control,
            gvec,
            pwden,
            runtime.crystal,
            runtime.pots,
            &electronic_ctx.vpwwfc[ilocal],
            &electronic_ctx.vvnl[ilocal],
            fft_shape,
            ik,
            k_cart,
            k_weight,
            random_stream_id,
        );
        channel.push(kscf);
    }

    channel
}

fn allocate_eigenvalues(spin_scheme: SpinScheme, nband: usize, nk_local: usize) -> VKEigenValue {
    match spin_scheme {
        SpinScheme::NonSpin => VKEigenValue::NonSpin(vec![vec![0.0; nband]; nk_local]),
        SpinScheme::Spin => {
            VKEigenValue::Spin(vec![vec![0.0; nband]; nk_local], vec![vec![0.0; nband]; nk_local])
        }
        SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in eigen setup"),
    }
}

fn allocate_eigenvector_channel(vpwwfc: &[PWBasis], nband: usize) -> Vec<Matrix<c64>> {
    let mut channel = Vec::with_capacity(vpwwfc.len());
    for pwwfc in vpwwfc.iter() {
        channel.push(Matrix::new(pwwfc.get_n_plane_waves(), nband));
    }
    channel
}

fn allocate_eigenvectors(
    spin_scheme: SpinScheme,
    nband: usize,
    vpwwfc: &[PWBasis],
) -> VKEigenVector {
    match spin_scheme {
        SpinScheme::NonSpin => VKEigenVector::NonSpin(allocate_eigenvector_channel(vpwwfc, nband)),
        SpinScheme::Spin => VKEigenVector::Spin(
            allocate_eigenvector_channel(vpwwfc, nband),
            allocate_eigenvector_channel(vpwwfc, nband),
        ),
        SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in eigen setup"),
    }
}

fn main() {
    // Top-level program flow:
    // 1) initialize MPI/runtime and read all inputs
    // 2) build per-geometry numerical context (FFT/G-vectors/k-points)
    // 3) run SCF to obtain energies, forces, and stress
    // 4) optionally perform geometry optimization step
    // 5) export requested artifacts (charge/wfc/eig files)
    // first statement
    dwmpi::init();

    // start the timer-main

    let stopwatch_main = std::time::Instant::now();

    // read in control parameters

    let control = match Control::from_file("in.ctrl") {
        Ok(control) => control,
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("failed to load control file: {}", err);
            }
            dwmpi::barrier(MPI_COMM_WORLD);
            dwmpi::finalize();
            std::process::exit(1);
        }
    };

    // dwfft3d
    dwfft3d::init_backend();
    let fft_planning_mode = match control.get_fft_planner_enum() {
        FftPlannerScheme::Estimate => dwfft3d::FftPlanningMode::Estimate,
        FftPlannerScheme::Measure => dwfft3d::FftPlanningMode::Measure,
    };
    let fft_wisdom_file = control.get_fft_wisdom_file().trim();
    dwfft3d::configure_runtime(dwfft3d::BackendOptions {
        threads: control.get_fft_threads(),
        planning_mode: fft_planning_mode,
        wisdom_file: if fft_wisdom_file.is_empty() {
            None
        } else {
            Some(fft_wisdom_file.to_string())
        },
    });

    if dwmpi::is_root() {
        display_program_header();
        display_system_information();
        control.display();
    }
    // read in crystal

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let latt0 = crystal.get_latt().clone();
    let geom_optim_mask_cell = crystal.get_cell_mask().clone();

    // read in pots

    let pots = PSPot::new(control.get_pot_scheme_enum());

    if dwmpi::is_root() {
        pots.display();
    }

    // zions

    let zions = crystal.get_zions(&pots);

    // println!("zions = {:?}", zions);

    // read in kpts

    let kpts = match kpts::try_new(
        control.get_kpts_scheme_enum(),
        &crystal,
        control.get_symmetry(),
    ) {
        Ok(kpts) => kpts,
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("failed to initialize k-points: {}", err);
            }
            dwmpi::barrier(MPI_COMM_WORLD);
            dwmpi::finalize();
            std::process::exit(1);
        }
    };

    if dwmpi::is_root() {
        kpts.display();
    }

    let mut provenance_status: i32 = 0;
    if dwmpi::is_root() {
        if let Err(err) = emit_run_provenance_manifest(&control, kpts.as_ref()) {
            eprintln!("failed to write/verify provenance manifest: {}", err);
            provenance_status = 1;
        } else {
            println!(
                "   provenance_manifest = {}",
                control.get_provenance_manifest()
            );
        }
    }
    dwmpi::bcast_scalar(&mut provenance_status, MPI_COMM_WORLD);
    if provenance_status != 0 {
        dwmpi::barrier(MPI_COMM_WORLD);
        dwmpi::finalize();
        std::process::exit(1);
    }

    //

    let mut stress_total = Matrix::<f64>::new(3, 3);
    let mut force_total = vec![Vector3f64::zeros(); crystal.get_n_atoms()];

    // Geometry optimizer state (BFGS/DIIS depending on in.ctrl).

    let mut geom_iter = 1;

    let mut geom_driver = geom::new(
        control.get_geom_optim_scheme(),
        control.get_geom_optim_alpha(),
        control.get_geom_optim_history_steps(),
    );

    // Self-consistent field drivers selected by spin scheme.

    let spin_scheme = control.get_spin_scheme_enum();

    let scf_driver = scf::new(spin_scheme);

    let density_driver = density::new(spin_scheme);
    let mut orchestration_workspace = OrchestrationWorkspace::new();

    // crystal.display();

    loop {
        // Rebuild reciprocal/FFT objects each geometry step because lattice may
        // change during cell optimization.
        let geom_ctx = GeometryStepContext::new(&crystal, control.get_ecutrho());
        let fftgrid = &geom_ctx.fftgrid;
        let rgtrans = &geom_ctx.rgtrans;
        let gvec = &geom_ctx.gvec;
        let pwden = &geom_ctx.pwden;
        let blatt = &geom_ctx.blatt;
        let [n1, n2, n3] = geom_ctx.fft_shape();
        let npw_rho = pwden.get_n_plane_waves();
        let runtime_ctx = RuntimeContext::new(&control, &crystal, &pots, kpts.as_ref(), spin_scheme);

        if dwmpi::is_root() {
            display_grid_information(fftgrid, pwden);
        }

        //loop {

        if dwmpi::is_root() {
            println!();
            crystal.display();
        }

        // Ewald

        let ewald = ewald::Ewald::new(&crystal, &zions, gvec, pwden);

        let mut rhog = match spin_scheme {
            SpinScheme::NonSpin => RHOG::NonSpin(vec![c64::zero(); npw_rho]),
            SpinScheme::Spin => RHOG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho]),
            SpinScheme::Ncl => {
                panic!("spin_scheme='ncl' is not implemented yet in pw initialization")
            }
        };

        let mut rho_3d = match spin_scheme {
            SpinScheme::NonSpin => RHOR::NonSpin(Array3::<c64>::new([n1, n2, n3])),
            SpinScheme::Spin => RHOR::Spin(
                Array3::<c64>::new([n1, n2, n3]),
                Array3::<c64>::new([n1, n2, n3]),
            ),
            SpinScheme::Ncl => {
                panic!("spin_scheme='ncl' is not implemented yet in pw initialization")
            }
        };

        // Initialize density either from previous converged file (restart) or
        // from atomic superposition.

        let expected_checkpoint_meta = runtime_ctx.checkpoint_meta();

        if dwmpi::is_root() {
            let mut loaded_from_restart = false;
            if geom_iter == 1 && control.get_restart() {
                match try_load_density_checkpoint(
                    spin_scheme,
                    &expected_checkpoint_meta,
                    blatt,
                    rgtrans,
                    gvec,
                    pwden,
                    &mut rhog,
                    &mut rho_3d,
                ) {
                    Ok(message) => {
                        println!("   {}", message);
                        loaded_from_restart = true;
                    }
                    Err(err) => {
                        panic!("{}", err);
                    }
                }
            }

            if !loaded_from_restart {
                density_driver.from_atomic_super_position(
                    &pots,
                    &crystal,
                    rgtrans,
                    gvec,
                    pwden,
                    &mut rhog,
                    &mut rho_3d,
                );

                if geom_iter == 1 && !control.get_restart() && restart_density_files_exist(spin_scheme)
                {
                    println!(
                        "   restart=false: ignore existing checkpoint files and build atomic initial density"
                    );
                } else {
                    println!();
                    println!("   construct charge density from constituent atoms");
                }
            }
        }

        match spin_scheme {
            SpinScheme::Spin => {
                let (rhog_up, rhog_dn) = rhog.as_spin_mut().unwrap();

                dwmpi::bcast_slice(rhog_up.as_mut_slice(), MPI_COMM_WORLD);
                dwmpi::bcast_slice(rhog_dn.as_mut_slice(), MPI_COMM_WORLD);

                let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin_mut().unwrap();

                dwmpi::bcast_slice(rho_3d_up.as_mut_slice(), MPI_COMM_WORLD);
                dwmpi::bcast_slice(rho_3d_dn.as_mut_slice(), MPI_COMM_WORLD);
            }
            SpinScheme::NonSpin => {
                dwmpi::bcast_slice(
                    rhog.as_non_spin_mut().unwrap().as_mut_slice(),
                    MPI_COMM_WORLD,
                );

                dwmpi::bcast_slice(
                    rho_3d.as_non_spin_mut().unwrap().as_mut_slice(),
                    MPI_COMM_WORLD,
                );
            }
            SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in pw broadcast"),
        }

        let mut total_rho = 0.0;

        if let RHOR::NonSpin(ref rho_3d) = &rho_3d {
            total_rho = rho_3d.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();
        } else if let RHOR::Spin(ref rho_3d_up, ref rho_3d_dn) = rho_3d {
            let total_rho_up =
                rho_3d_up.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();
            let total_rho_dn =
                rho_3d_dn.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();

            total_rho = total_rho_up + total_rho_dn;
        }

        if dwmpi::is_root() {
            println!("   initial_charge = {}", total_rho);
        }

        // Core charge used by NLCC terms.
        orchestration_workspace.ensure_shape(npw_rho, [n1, n2, n3]);
        let (rhocoreg, rhocore_3d_workspace) = orchestration_workspace.core_charge_buffers_mut();

        nlcc::from_atomic_super_position(
            &pots,
            &crystal,
            rgtrans,
            gvec,
            pwden,
            rhocoreg,
            rhocore_3d_workspace,
        );

        // Symmetry helper used by force/stress post-processing.
        orchestration_workspace.update_atom_positions(&crystal);

        let symdrv = symmetry::new(
            &crystal.get_latt().as_2d_array_row_major(),
            orchestration_workspace.atom_positions(),
            &crystal.get_atom_types(),
            EPS6,
        );

        if control.get_symmetry() && dwmpi::is_root() {
            println!();
            println!("   {:-^88}", " symmetry analysis ");
            symdrv.display();

            let n_sym = symdrv.get_n_sym_ops();
            let fft_comm_ops =
                symdrv.get_fft_commensurate_ops(fftgrid.get_size(), kpts.get_k_mesh(), EPS6);
            println!(
                "   commensurate_ops (fft+kmesh) = {} / {}",
                fft_comm_ops.len(),
                n_sym
            );

            let kmesh = kpts.get_k_mesh();
            if kmesh[0] > 0 && kmesh[1] > 0 && kmesh[2] > 0 {
                let nk_full = kmesh[0] as usize * kmesh[1] as usize * kmesh[2] as usize;
                println!(
                    "   ir_kpoints (symmetry-reduced) = {} / {}",
                    kpts.get_n_kpts(),
                    nk_full
                );
            }

            println!(
                "   sym_atom mapping dimensions   = {} atoms x {} ops",
                symdrv.get_sym_atom().len(),
                n_sym
            );

            display_symmetry_equivalent_atoms(&crystal, symdrv.as_ref());
        }

        //
        let electronic_ctx = ElectronicStepContext::build(&runtime_ctx, blatt, gvec);
        let (mut vkscf, mut vkevals, mut vkevecs) =
            electronic_ctx.build_scf_state(&runtime_ctx, gvec, pwden, fftgrid.get_size());

        if geom_iter == 1 && control.get_restart() {
            match try_load_wavefunction_checkpoint(
                spin_scheme,
                electronic_ctx.kpoint_domain(),
                blatt,
                &expected_checkpoint_meta,
                &electronic_ctx.vpwwfc,
                &mut vkevecs,
            ) {
                Ok(message) => {
                    if dwmpi::is_root() {
                        println!("   {}", message);
                    }
                }
                Err(err) => {
                    if dwmpi::is_root() {
                        println!("   NOTE: {}", err);
                        println!("   NOTE: continue with default wavefunction initialization");
                    }
                }
            }
        }

        // ions optimization

        if dwmpi::is_root() {
            println!("\n   #step: geom-{}\n", geom_iter);
        }

        let rhocore_3d = orchestration_workspace.rhocore_3d();

        //loop {
        scf_driver.run(
            geom_iter,
            &control,
            &crystal,
            gvec,
            pwden,
            &pots,
            rgtrans,
            kpts.as_ref(),
            &ewald,
            &electronic_ctx.vpwwfc,
            &mut vkscf,
            &mut rhog,
            &mut rho_3d,
            rhocore_3d,
            &mut vkevals,
            &mut vkevecs,
            symdrv.as_ref(),
            &mut stress_total,
            &mut force_total,
        );

        // save rho

        if dwmpi::is_root() {
            if control.get_save_rho() {
                rho_3d.save_hdf5_with_meta(blatt, Some(&expected_checkpoint_meta));
            }

            crystal.output();
        }

        // save wavefunction
        if control.get_save_wfc() || control.get_wannier90_export() {
            vkevecs.save_hdf5_with_meta(
                electronic_ctx.global_k_first_or_zero(),
                &electronic_ctx.vpwwfc,
                blatt,
                Some(&expected_checkpoint_meta),
            );
        }

        // if converged, then exit

        let force_max = force::get_max_force(&force_total);

        let stress_max = stress::get_max_stress(&stress_total);

        let mut should_exit = false;

        if control.get_geom_optim_cell() {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA
                && stress_max < control.get_geom_optim_stress_tolerance() * STRESS_KB_TO_HA
            {
                if dwmpi::is_root() {
                    println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                }
                should_exit = true;
            }
        } else {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA {
                if dwmpi::is_root() {
                    println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                }
                should_exit = true;
            }
        }

        // if not converged, but reach the max geometry optimization steps, then exit

        if !should_exit && geom_iter >= control.get_geom_optim_max_steps() {
            if dwmpi::is_root() {
                println!("\n   {} : {:<5}", "geom_exit_max_steps_reached", geom_iter);
            }
            should_exit = true;
        }

        if should_exit {
            if control.get_wannier90_export() {
                match wannier90::write_eig_inputs(
                    &control,
                    &vkevals,
                    electronic_ctx.global_k_first_or_zero(),
                ) {
                    Ok(summary) => {
                        if dwmpi::is_root() {
                            println!();
                            println!("   {:-^88}", " wannier90 eig export ");
                            for file in summary.written_files.iter() {
                                println!("   wrote {}", file);
                            }
                            println!("   run `w90-win` and `w90-amn` to generate remaining Wannier90 inputs");
                        }
                    }
                    Err(err) => {
                        if dwmpi::is_root() {
                            eprintln!("   wannier90 eig export failed: {}", err);
                        }
                    }
                }
            }

            post_processing(&control, &vkevals, &vkevecs, &vkscf);

            break;
        }

        // if not converged, get the atom positions and (lattice vectors if cell is also relaxed) for the next optim iteration
        let geom_optim_mask_ions = vec![
            Vector3f64 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            };
            crystal.get_n_atoms()
        ];

        let mut bcell_move = control.get_geom_optim_cell();

        geom_driver.compute_next_configuration(
            &mut crystal,
            &force_total,
            &stress_total,
            &geom_optim_mask_ions,
            &geom_optim_mask_cell,
            &latt0,
            bcell_move,
        );

        // end: geom relaxation

        geom_iter += 1;
    }

    // computing time statistics
    let elapsed_main_seconds = stopwatch_main.elapsed().as_secs_f64();

    if dwmpi::is_root() {
        println!();
        println!("   {:-^88}", " statistics ");
        println!();

        println!(
            "   {:16}{:5}{:16.2} seconds {:16.2} hours",
            "Total",
            ":",
            elapsed_main_seconds,
            elapsed_main_seconds / 3600.0
        );
    }

    // last statement

    dwmpi::barrier(MPI_COMM_WORLD);

    dwmpi::finalize();
}

fn restart_density_files_exist(spin_scheme: SpinScheme) -> bool {
    match spin_scheme {
        SpinScheme::NonSpin => Path::new("out.scf.rho.hdf5").exists(),
        SpinScheme::Spin => {
            Path::new("out.scf.rho.up.hdf5").exists() && Path::new("out.scf.rho.dn.hdf5").exists()
        }
        SpinScheme::Ncl => false,
    }
}

fn try_load_density_checkpoint(
    spin_scheme: SpinScheme,
    expected_meta: &CheckpointMeta,
    expected_blatt: &lattice::Lattice,
    rgtrans: &rgtransform::RGTransform,
    gvec: &GVector,
    pwden: &PWDensity,
    rhog: &mut RHOG,
    rho_3d: &mut RHOR,
) -> Result<String, String> {
    if !restart_density_files_exist(spin_scheme) {
        return Err(match spin_scheme {
            SpinScheme::NonSpin => "restart requested but 'out.scf.rho.hdf5' is missing".to_string(),
            SpinScheme::Spin => {
                "restart requested but spin density checkpoints ('out.scf.rho.up.hdf5' and/or 'out.scf.rho.dn.hdf5') are missing".to_string()
            }
            SpinScheme::Ncl => "restart requested for unsupported spin_scheme='ncl'".to_string(),
        });
    }

    let (checkpoint_blatt, loaded_rho, checkpoint_meta_opt) =
        RHOR::try_load_hdf5(matches!(spin_scheme, SpinScheme::Spin))?;

    let checkpoint_meta = checkpoint_meta_opt.ok_or_else(|| {
        "checkpoint metadata is missing; regenerate checkpoints with the current schema".to_string()
    })?;

    checkpoint_meta.validate_against(expected_meta, RESTART_META_TOL)?;

    validate_checkpoint_lattice(
        expected_blatt,
        &checkpoint_blatt,
        RESTART_LATTICE_TOL,
        "density",
    )?;

    *rho_3d = loaded_rho;

    match (spin_scheme, rhog, rho_3d) {
        (SpinScheme::NonSpin, RHOG::NonSpin(rhog_ns), RHOR::NonSpin(rho_ns)) => {
            rgtrans.r3d_to_g1d(gvec, pwden, rho_ns.as_slice(), rhog_ns);
            Ok("loaded restart density from out.scf.rho.hdf5".to_string())
        }
        (SpinScheme::Spin, RHOG::Spin(rhog_up, rhog_dn), RHOR::Spin(rho_up, rho_dn)) => {
            rgtrans.r3d_to_g1d(gvec, pwden, rho_up.as_slice(), rhog_up);
            rgtrans.r3d_to_g1d(gvec, pwden, rho_dn.as_slice(), rhog_dn);
            Ok("loaded restart density from out.scf.rho.up.hdf5/out.scf.rho.dn.hdf5".to_string())
        }
        _ => Err("restart density spin-scheme mismatch while loading checkpoint".to_string()),
    }
}

fn try_load_wavefunction_checkpoint(
    spin_scheme: SpinScheme,
    k_domain: &KPointDomain,
    expected_blatt: &lattice::Lattice,
    expected_meta: &CheckpointMeta,
    vpwwfc: &[PWBasis],
    vkevecs: &mut VKEigenVector,
) -> Result<String, String> {
    if k_domain.is_empty() || vpwwfc.is_empty() {
        return Err("skip wavefunction restart: no local k-points on this rank".to_string());
    }

    let local_nk = k_domain.len();
    if local_nk != vpwwfc.len() {
        return Err(format!(
            "wavefunction restart mismatch: local_nk={} but local basis count={}",
            local_nk,
            vpwwfc.len()
        ));
    }

    for slot in k_domain.iter() {
        for filename in checkpoint_wavefunction_filenames(spin_scheme, slot.global_index)? {
            if !Path::new(&filename).exists() {
                return Err(format!(
                    "wavefunction restart files are incomplete for local slot {} (global k-index {}): missing '{}'",
                    slot.local_slot,
                    slot.global_index,
                    filename
                ));
            }
        }
    }

    validate_wavefunction_checkpoint_metadata(spin_scheme, k_domain, expected_meta)?;

    let (ik_first, ik_last) = k_domain
        .global_range()
        .ok_or_else(|| "skip wavefunction restart: no local k-points on this rank".to_string())?;

    let (loaded_pwbasis, checkpoint_blatt, loaded_evecs) =
        VKEigenVector::try_load_hdf5(matches!(spin_scheme, SpinScheme::Spin), ik_first, ik_last)?;

    validate_checkpoint_lattice(
        expected_blatt,
        &checkpoint_blatt,
        RESTART_LATTICE_TOL,
        "wavefunction",
    )?;

    if loaded_pwbasis.len() != vpwwfc.len() {
        return Err(format!(
            "wavefunction restart mismatch: loaded basis count={} but expected {}",
            loaded_pwbasis.len(),
            vpwwfc.len()
        ));
    }

    for (slot, (loaded, expected)) in k_domain
        .iter()
        .zip(loaded_pwbasis.iter().zip(vpwwfc.iter()))
    {
        if loaded.get_k_index() != slot.global_index {
            return Err(format!(
                "wavefunction restart mismatch at local slot {}: loaded k_index={} expected={}",
                slot.local_slot,
                loaded.get_k_index(),
                slot.global_index
            ));
        }
        if expected.get_k_index() != slot.global_index {
            return Err(format!(
                "wavefunction setup mismatch at local slot {}: expected basis k_index={} but domain expects {}",
                slot.local_slot,
                expected.get_k_index(),
                slot.global_index
            ));
        }
        if loaded.get_n_plane_waves() != expected.get_n_plane_waves() {
            return Err(format!(
                "wavefunction restart mismatch at k_index {}: loaded npw={} expected={}",
                expected.get_k_index(),
                loaded.get_n_plane_waves(),
                expected.get_n_plane_waves()
            ));
        }
    }

    validate_loaded_wavefunction_shapes(&loaded_evecs, vkevecs, k_domain)?;

    *vkevecs = loaded_evecs;

    Ok(format!(
        "loaded restart wavefunctions for local k-range [{}..={}] (local_nk={})",
        ik_first, ik_last, local_nk
    ))
}

fn checkpoint_wavefunction_filenames(
    spin_scheme: SpinScheme,
    ik_global: usize,
) -> Result<Vec<String>, String> {
    match spin_scheme {
        SpinScheme::NonSpin => Ok(vec![format!("out.wfc.k.{}.hdf5", ik_global)]),
        SpinScheme::Spin => Ok(vec![
            format!("out.wfc.up.k.{}.hdf5", ik_global),
            format!("out.wfc.dn.k.{}.hdf5", ik_global),
        ]),
        SpinScheme::Ncl => Err("restart requested for unsupported spin_scheme='ncl'".to_string()),
    }
}

fn validate_wavefunction_checkpoint_metadata(
    spin_scheme: SpinScheme,
    k_domain: &KPointDomain,
    expected_meta: &CheckpointMeta,
) -> Result<(), String> {
    for slot in k_domain.iter() {
        for filename in checkpoint_wavefunction_filenames(spin_scheme, slot.global_index)? {
            let checkpoint_meta = read_checkpoint_meta_required(&filename)?;
            checkpoint_meta.validate_against(expected_meta, RESTART_META_TOL)?;
        }
    }

    Ok(())
}

fn read_checkpoint_meta_required(filename: &str) -> Result<CheckpointMeta, String> {
    let checkpoint_meta = CheckpointMeta::read_from_path_optional(filename)?;
    checkpoint_meta.ok_or_else(|| {
        format!(
            "checkpoint metadata is missing in '{}'; regenerate checkpoints with the current schema",
            filename
        )
    })
}

fn validate_loaded_wavefunction_shapes(
    loaded: &VKEigenVector,
    expected: &VKEigenVector,
    k_domain: &KPointDomain,
) -> Result<(), String> {
    match (loaded, expected) {
        (VKEigenVector::NonSpin(loaded_ns), VKEigenVector::NonSpin(expected_ns)) => {
            if loaded_ns.len() != expected_ns.len() {
                return Err(format!(
                    "wavefunction restart mismatch: loaded {} k-point blocks, expected {}",
                    loaded_ns.len(),
                    expected_ns.len()
                ));
            }
            for (local_slot, (loaded_mat, expected_mat)) in
                loaded_ns.iter().zip(expected_ns.iter()).enumerate()
            {
                if loaded_mat.nrow() != expected_mat.nrow() || loaded_mat.ncol() != expected_mat.ncol()
                {
                    let k_index = k_domain
                        .global_index(local_slot)
                        .unwrap_or(k_domain.global_first_or_zero() + local_slot);
                    return Err(format!(
                        "wavefunction restart shape mismatch at k_index {}: loaded {}x{}, expected {}x{}",
                        k_index,
                        loaded_mat.nrow(),
                        loaded_mat.ncol(),
                        expected_mat.nrow(),
                        expected_mat.ncol()
                    ));
                }
            }
        }
        (VKEigenVector::Spin(loaded_up, loaded_dn), VKEigenVector::Spin(expected_up, expected_dn)) => {
            if loaded_up.len() != expected_up.len() || loaded_dn.len() != expected_dn.len() {
                return Err(format!(
                    "wavefunction restart mismatch: loaded blocks (up={}, dn={}), expected (up={}, dn={})",
                    loaded_up.len(),
                    loaded_dn.len(),
                    expected_up.len(),
                    expected_dn.len()
                ));
            }
            for (local_slot, (loaded_mat, expected_mat)) in
                loaded_up.iter().zip(expected_up.iter()).enumerate()
            {
                if loaded_mat.nrow() != expected_mat.nrow() || loaded_mat.ncol() != expected_mat.ncol()
                {
                    let k_index = k_domain
                        .global_index(local_slot)
                        .unwrap_or(k_domain.global_first_or_zero() + local_slot);
                    return Err(format!(
                        "wavefunction restart shape mismatch (up) at k_index {}: loaded {}x{}, expected {}x{}",
                        k_index,
                        loaded_mat.nrow(),
                        loaded_mat.ncol(),
                        expected_mat.nrow(),
                        expected_mat.ncol()
                    ));
                }
            }
            for (local_slot, (loaded_mat, expected_mat)) in
                loaded_dn.iter().zip(expected_dn.iter()).enumerate()
            {
                if loaded_mat.nrow() != expected_mat.nrow() || loaded_mat.ncol() != expected_mat.ncol()
                {
                    let k_index = k_domain
                        .global_index(local_slot)
                        .unwrap_or(k_domain.global_first_or_zero() + local_slot);
                    return Err(format!(
                        "wavefunction restart shape mismatch (dn) at k_index {}: loaded {}x{}, expected {}x{}",
                        k_index,
                        loaded_mat.nrow(),
                        loaded_mat.ncol(),
                        expected_mat.nrow(),
                        expected_mat.ncol()
                    ));
                }
            }
        }
        _ => {
            return Err(
                "wavefunction restart spin-scheme mismatch between loaded and expected storage"
                    .to_string(),
            )
        }
    }

    Ok(())
}

fn validate_checkpoint_lattice(
    expected: &lattice::Lattice,
    checkpoint: &lattice::Lattice,
    tol: f64,
    checkpoint_kind: &str,
) -> Result<(), String> {
    let max_diff = lattice_max_abs_diff(expected, checkpoint);
    if max_diff > tol {
        Err(format!(
            "{} checkpoint lattice mismatch: max |delta(b)|={:.3e} exceeds tolerance {:.3e}",
            checkpoint_kind, max_diff, tol
        ))
    } else {
        Ok(())
    }
}

fn lattice_max_abs_diff(lhs: &lattice::Lattice, rhs: &lattice::Lattice) -> f64 {
    let lhs_arr = lhs.as_2d_array_row_major();
    let rhs_arr = rhs.as_2d_array_row_major();
    let mut max_diff = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (lhs_arr[i][j] - rhs_arr[i][j]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    max_diff
}

fn checkpoint_meta_for_run(
    control: &Control,
    spin_scheme: SpinScheme,
    k_mesh: [i32; 3],
) -> CheckpointMeta {
    let spin_channels = match spin_scheme {
        SpinScheme::NonSpin => 1,
        SpinScheme::Spin => 2,
        SpinScheme::Ncl => 4,
    };

    let nband_usize = control.get_nband();
    if nband_usize > u32::MAX as usize {
        panic!("nband={} exceeds checkpoint metadata limit {}", nband_usize, u32::MAX);
    }
    let nband = nband_usize as u32;

    CheckpointMeta::new(
        spin_channels,
        nband,
        control.get_ecut(),
        control.get_ecutrho(),
        k_mesh,
    )
}

fn post_processing(
    control: &Control,
    vkevals: &VKEigenValue,
    vkevecs: &VKEigenVector,
    vkscf: &VKSCF,
) {
    // total density of states

    // println!("   compute total density of states");

    //dos::compute_total_density_of_states(control, vkevals, vkevecs, vkscf);
}

fn matrix3x3_to_vector3(mat: &Matrix<f64>) -> Vec<Vector3f64> {
    let mut v = vec![Vector3f64::zeros(); 3];

    for i in 0..3 {
        v[i].x = mat[[0, i]];
        v[i].y = mat[[1, i]];
        v[i].z = mat[[2, i]];
    }

    v
}
