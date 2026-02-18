# Use the official Rust image as a base
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive 

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    unzip \
    gnupg \
    libcurl4-openssl-dev \
    libssl-dev \
    libfftw3-dev liblapack-dev libblas-dev libhdf5-dev \
    less curl vim emacs mpich pkg-config \
    wannier90 \
    quantum-espresso

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN wget https://cmake.org/files/v3.25/cmake-3.25.0.tar.gz &&\
    tar xf cmake-3.25.0.tar.gz &&\
    cd cmake-3.25.0 &&\
    ./bootstrap --prefix=/usr/local &&\
    make -j8 && make install && make clean

WORKDIR /usr/src/app
# Command to run your application
CMD ["bash"]
