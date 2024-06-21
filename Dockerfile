# Use the official Rust image as a base
FROM ubuntu:20.04

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
    less curl vim emacs mpich pkg-config

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN wget https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz &&\
    tar xf cmake-3.20.6.tar.gz &&\
    cd cmake-3.20.6 &&\
    ./bootstrap --prefix=/usr/local &&\
    make -j8 && make install && make clean


# Install spglib from source
RUN mkdir -p /opt/spglib && \
    cd /opt/spglib && \
    git clone https://github.com/spglib/spglib.git && \
    cd spglib && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/spglib && \
    make && \
    make install

ENV LD_LIBRARY_PATH=/opt/spglib/lib:$LD_LIBRARY_PATH

WORKDIR /usr/src/app
# Command to run your application
CMD ["bash"]

