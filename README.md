**The goal of this project is to employ Rust as the programming language to implement a plane-wave pseudopotential density functional theory simulation package.**

## Code structure

* Main program: pw
* Testing: test_example
* Library: all others

## Install Rust
If you are running macOS, Linux, or another Unix-like Operating Systems, to set up the Rust working environment, please run the following command in your terminal and then follow the on-screen instructions.

<code>
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
</code><br/>
<br/>

All Rust development tools will be installed to the ~/.cargo/bin directory which needs to be added to the definition of the environmental variable PATH. This can be done by adding the following line to ~/.bash_profile.

<code>
    export PATH=~/.cargo/bin:$PATH
</code>

<br/>
<br/>

Running <code>source ~/.bash_profile</code> will update PATH.

## Download the code

<code>
    git clone https://github.com/dftworks/dftworks.git
</code>

## Intel MKL library

The [Intel MKL library](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html) is used for diagalization and FFT. It should be installed into the directory <code>/opt/intel/oneapi/mkl/2021.1.1/lib</code>. If the library is installed in other directories, the library location specified in matrix/build.rs and dwfft/build.rs should be updated.

## Symmetry analysis library

[Spglib](http://spglib.github.io/spglib/
) is used for finding and handling crystal symmetries. It should be installed into the directory <code>/opt/spglib/lib</code>. If the library is installed in other directories, the library location specified in symmetry/build.rs should be updated.

## Build the code

In the directory dftworks, run the following command.

<code>cargo build --release</code>

This will download the dependency modules and compile the code to generate the executable **pw** in the directory **target/release**.
