use std::marker::ConstParamTy;

#[cfg(feature = "hexl")]
use crate::hexl::bindings::{ntt_forward_in_place, ntt_inverse_in_place, eltwise_mult_mod};
#[cfg(feature = "tfhe")]
use tfhe_ntt::*;
use once_cell::sync::OnceCell;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::Arc;


pub trait RingOps<const MOD_Q: u64, const N: usize> {
    fn fwd(data: &mut [u64]);
    fn inv(data: &mut [u64]);
    fn multiply(result: &mut [u64], left: &mut [u64], right: &mut [u64]);
}

#[derive(Clone, Debug, Eq, PartialEq, ConstParamTy)]
pub struct NTT<const MOD_Q: u64, const N: usize>;

#[cfg(feature = "hexl")]
impl<const MOD_Q: u64, const N: usize> RingOps<MOD_Q, N> for NTT<MOD_Q, N> { 
    fn fwd(data: &mut [u64]) {
        unsafe {
            ntt_forward_in_place(data.as_mut_ptr(), N, MOD_Q)
        }
    }
    fn inv(data: &mut [u64]) {
        unsafe {
            ntt_inverse_in_place(data.as_mut_ptr(), N, MOD_Q)
        }
    }

    fn multiply(result: &mut [u64], left: &mut [u64], right: &mut [u64]) {
        unsafe {
            eltwise_mult_mod(
                result.as_mut_ptr(),
                left.as_ptr(),
                right.as_ptr(),
                N as u64,
                MOD_Q
            )
        }
    }
}

#[cfg(feature = "tfhe")]
static PLAN_CACHE: OnceCell<Mutex<HashMap<(usize, u64), Arc<prime64::Plan>>>> = OnceCell::new();

#[cfg(feature = "tfhe")]
impl<const MOD_Q: u64, const N: usize> RingOps<MOD_Q, N> for NTT<MOD_Q, N> { 
    fn fwd(data: &mut [u64]) {
        let plan = get_plan(N, MOD_Q);
        plan.fwd(data);
    }
    fn inv(data: &mut [u64]) {
        let plan = get_plan(N, MOD_Q);
        plan.inv(data);
    }

    fn multiply(result: &mut [u64], left: &mut [u64], right: &mut [u64]) {
        let plan = get_plan(N, MOD_Q);
        plan.mul_assign_normalize(left, &right);
        result.copy_from_slice(left);
    }
}

#[cfg(feature = "tfhe")]
fn get_plan(n: usize, q: u64) -> Arc<prime64::Plan> {
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();

    map.entry((n, q)).or_insert_with(|| {
        Arc::new(prime64::Plan::try_new(n, q).expect("Plan creation failed"))
    }).clone()
}
