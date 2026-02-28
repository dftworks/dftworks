use control::Control;
use crystal::Crystal;
use dfttypes::*;
use ewald::Ewald;
use gvector::GVector;
use kpts::KPTS;
use matrix::Matrix;
use ndarray::Array3;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use symmetry::SymmetryDriver;
use types::c64;
use vector3::Vector3f64;

pub(crate) struct ScfExecutionContext<'a, 'ks> {
    pub geom_iter: usize,
    pub control: &'a Control,
    pub crystal: &'a Crystal,
    pub gvec: &'a GVector,
    pub pwden: &'a PWDensity,
    pub pots: &'a PSPot,
    pub rgtrans: &'a rgtransform::RGTransform,
    pub kpts: &'a dyn KPTS,
    pub ewald: &'a Ewald,
    pub vpwwfc: &'a [PWBasis],
    pub vkscf: &'a mut VKSCF<'ks>,
    pub rhog: &'a mut RHOG,
    pub rho_3d: &'a mut RHOR,
    pub rhocore_3d: &'a Array3<c64>,
    pub vkevals: &'a mut VKEigenValue,
    pub vkevecs: &'a mut VKEigenVector,
    pub symdrv: &'a dyn SymmetryDriver,
    pub stress_total: &'a mut Matrix<f64>,
    pub force_total: &'a mut Vec<Vector3f64>,
    pub scf_driver: &'a dyn scf::SCF,
}

pub(crate) fn run_scf_phase<'a, 'ks>(ctx: ScfExecutionContext<'a, 'ks>) {
    ctx.scf_driver.run(
        ctx.geom_iter,
        ctx.control,
        ctx.crystal,
        ctx.gvec,
        ctx.pwden,
        ctx.pots,
        ctx.rgtrans,
        ctx.kpts,
        ctx.ewald,
        ctx.vpwwfc,
        ctx.vkscf,
        ctx.rhog,
        ctx.rho_3d,
        ctx.rhocore_3d,
        ctx.vkevals,
        ctx.vkevecs,
        ctx.symdrv,
        ctx.stress_total,
        ctx.force_total,
    );
}
