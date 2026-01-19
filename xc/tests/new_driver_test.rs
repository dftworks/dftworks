
use xc::driver::XCDriver;
use xc::kernels::cpu_scalar::LdaPzScalar;

use dfttypes::*;
use dwgrid::Array3;
use types::c64;

#[test]
fn test_new_xc_driver_ldapz() {
    // 1. Setup a small density grid (2x2x2)
    // 1. Setup a small density grid (2x2x2)
    let shape_arr = [2, 2, 2];
    let mut rho_arr = Array3::<c64>::new(shape_arr);
    
    // Fill with some dummy density
    // Array3 implements IndexMut<[usize; 3]>
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                rho_arr[[x, y, z]] = c64::new(1.0, 0.0);
            }
        }
    }
    
    // 2. Wrap in RHOR
    let rho_t = RHOR::NonSpin(rho_arr);
    
    let mut vxc = VXCR::NonSpin(Array3::<c64>::new(shape_arr));
    let mut exc = Array3::<c64>::new(shape_arr);
    
    // 3. Create Driver with Scalar Kernel
    let kernel = Box::new(LdaPzScalar);
    let driver = XCDriver::new(kernel);
    
    // 4. Compute
    driver.compute(&rho_t, None, &mut vxc, &mut exc);
    
    // 5. Verify results
    // We need to extract the array from the enum to check
    let vxc_arr = vxc.as_non_spin().expect("Should be NonSpin");
    let v_val = vxc_arr[[0,0,0]].re;
    assert!(v_val < -0.5);
    assert!(v_val > -2.0);
    
    println!("Computed VXC at rho=1.0: {}", v_val);
}
