defaut:
  just --list --unsorted

docker-build:
  docker build -t rust-dev .

docker-run:
  docker run -it --rm -v $(pwd):/usr/src/app rust-dev
