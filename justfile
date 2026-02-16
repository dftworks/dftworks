default:
  just --list --unsorted

docker-build:
  docker build -t rust-dev .

docker-run:
  docker run -it --rm -v $(pwd):/usr/src/app rust-dev

# ---- Silicon Wannier90 staged test (sp3 projectors) ----

si_w90_case := "/tmp/si-w90-staged"

si-w90-test-prepare:
  rm -rf {{si_w90_case}}
  mkdir -p {{si_w90_case}}
  cp {{justfile_directory()}}/test_example/si-oncv/wannier90-projected/in.* {{si_w90_case}}/
  cp -R {{justfile_directory()}}/test_example/si-oncv/wannier90-projected/pot {{si_w90_case}}/

si-w90-test-build:
  cargo build --release -p pw -p wannier90

si-w90-test-scf:
  docker run --rm \
    -v {{justfile_directory()}}:/usr/src/app \
    -v {{si_w90_case}}:/work \
    -w /work \
    rust-dev \
    bash -lc 'source $HOME/.cargo/env && CARGO_TARGET_DIR=/usr/src/app/target-docker-linux cargo run --manifest-path /usr/src/app/Cargo.toml -p pw > out.pw.log 2>&1'

si-w90-test-win:
  docker run --rm \
    -v {{justfile_directory()}}:/usr/src/app \
    -v {{si_w90_case}}:/work \
    -w /work \
    rust-dev \
    bash -lc 'source $HOME/.cargo/env && CARGO_TARGET_DIR=/usr/src/app/target-docker-linux cargo run --manifest-path /usr/src/app/Cargo.toml -p wannier90 --bin w90-win'

si-w90-test-set-sp3:
  perl -0777 -i -pe 's/begin projections.*?end projections/begin projections\nSi1:sp3\nend projections/s' {{si_w90_case}}/si.win

si-w90-test-amn:
  docker run --rm \
    -v {{justfile_directory()}}:/usr/src/app \
    -v {{si_w90_case}}:/work \
    -w /work \
    rust-dev \
    bash -lc 'source $HOME/.cargo/env && CARGO_TARGET_DIR=/usr/src/app/target-docker-linux cargo run --manifest-path /usr/src/app/Cargo.toml -p wannier90 --bin w90-amn'

si-w90-test-wannier:
  docker run --rm \
    -v {{si_w90_case}}:/work \
    -w /work \
    rust-dev \
    bash -lc 'wannier90.x si > out.w90.log 2>&1'

si-w90-test-summary:
  ls -l {{si_w90_case}}/si.win {{si_w90_case}}/si.eig {{si_w90_case}}/si.nnkp {{si_w90_case}}/si.mmn {{si_w90_case}}/si.amn {{si_w90_case}}/si.wout {{si_w90_case}}/si.chk
  tail -n 40 {{si_w90_case}}/si.wout

si-w90-test-all: si-w90-test-prepare si-w90-test-build si-w90-test-scf si-w90-test-win si-w90-test-set-sp3 si-w90-test-amn si-w90-test-wannier si-w90-test-summary
