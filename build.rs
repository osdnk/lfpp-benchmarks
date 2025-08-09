fn main() {
    println!("cargo:rustc-link-lib=dylib=hexl_wrapper"); // Link to the shared library
    println!("cargo:rustc-link-search=native=."); // Current directory to find libhexl_wrapper.so
    println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib"); // Path to hexl library
}
