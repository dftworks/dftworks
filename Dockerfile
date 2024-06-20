# Use the official Rust image as a base
FROM rust:1.79

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    python3 \
    python3-pip \
    wget \
    unzip \
    gnupg \
    libcurl4-openssl-dev \
    libssl-dev \
    libfftw3-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    less

RUN apt-get install -y mpich

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

# Create a directory for your Rust project
WORKDIR /usr/src/app

RUN arch=$(uname -m) && \
	if [ "$arch" = "x86_64" ]; then \
		rustup install stable-x86_64-unknown-linux-gnu; rustup default stable-x86_64-unknown-linux-gnu; \
    	elif [ "$arch" = "aarch64" ]; then \
		rustup install stable-aarch64-unknown-linux-gnu; rustup default stable-aarch64-unknown-linux-gnu; \
	fi

RUN apt-get install -y vim emacs

ENV LD_LIBRARY_PATH=/opt/spglib/lib:$LD_LIBRARY_PATH

# Command to run your application
CMD ["bash"]

