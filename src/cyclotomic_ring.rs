use crate::hexl::bindings::{eltwise_add_mod, eltwise_mult_mod, ntt_forward_in_place, ntt_inverse_in_place};
use crate::ringops::{NTT, RingOps};
use memoize::memoize;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Representation {
    Coefficient,
    NTT,
    IncompleteNTT,
}

#[derive(Clone, Debug)]
pub struct CyclotomicRing<const MOD_Q: u64, const N: usize>
where
    [(); N / 2]:,
{
    pub data: [u64; N],
    pub representation: Representation,
}



impl<const MOD_Q: u64, const N: usize> CyclotomicRing<MOD_Q, N>
where
    [(); N / 2]:,
{
    pub fn new() -> Self {
        Self { data: [0u64; N], representation: Representation::Coefficient}
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N];
        for i in 0..N {
            data[i] = rng.random_range(0..MOD_Q);
        }
        Self { data, representation: Representation::Coefficient}
    }

    pub fn forward_partial_ntt(&mut self) {
        if self.representation == Representation::IncompleteNTT {
            return; // already in NTT form
        }
        if self.representation == Representation::NTT {
            self.to_coeff_representation();
        }

        let mut even = [0u64; N / 2];
        let mut odd = [0u64; N / 2];
        for i in 0..N / 2 {
            even[i] = self.data[i * 2];
            odd[i] = self.data[i * 2 + 1];
        }

        unsafe {
            ntt_forward_in_place(even.as_mut_ptr(), even.len(), MOD_Q);
            ntt_forward_in_place(odd.as_mut_ptr(), odd.len(), MOD_Q);
        }

        for i in 0..N / 2 {
            self.data[i] = even[i];
            self.data[i + N / 2] = odd[i];
        }
        self.representation = Representation::IncompleteNTT;
    }

    pub fn to_ntt_representation(&mut self) {
        if self.representation == Representation::NTT {
            return; // already in NTT form
        }

        if self.representation == Representation::IncompleteNTT {
            self.to_coeff_representation();
        }

        NTT::<MOD_Q, N>::fwd(&mut self.data);

        self.representation = Representation::NTT;
    }

    pub fn to_coeff_representation(&mut self) {
        if self.representation == Representation::Coefficient {
            return; // already in coefficient form
        }

        if self.representation == Representation::IncompleteNTT {
            let mut even = [0u64; N / 2];
            let mut odd = [0u64; N / 2];
            for i in 0..N / 2 {
                even[i] = self.data[i];
                odd[i] = self.data[i + N / 2];
            }

            NTT::<MOD_Q, N>::inv(&mut even);
            NTT::<MOD_Q, N>::inv(&mut odd);

            for i in 0..N / 2 {
                self.data[i * 2] = even[i];
                self.data[i * 2 + 1] = odd[i];
            }
        }

        if self.representation == Representation::NTT {

            NTT::<MOD_Q, N>::inv(&mut self.data);
        }

        self.representation = Representation::Coefficient;
    }

    pub fn incomplete_ntt_multiplication(operand1: &mut CyclotomicRing<MOD_Q, N>, 
        operand2: &mut CyclotomicRing<MOD_Q, N>,
    ) -> CyclotomicRing<MOD_Q, N>
    {
        // Static local cache (thread-safe, mutable via Mutex)
        static SHIFT_FACTORS_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> = OnceLock::new();

        // Initialize or get the cache
        let cache = SHIFT_FACTORS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache_guard = cache.lock().unwrap();

        // Get or compute shift factors
        let shift_factors = cache_guard
            .entry(N)
            .or_insert_with(|| get_shift_factors::<MOD_Q, N>());


        let mut result_even_2 = [0u64; N / 2];
        let mut result_odd_2 = [0u64; N / 2];
        let mut operand1_even = [0u64; N / 2];
        let mut operand1_odd = [0u64; N / 2];
        let mut operand2_even = [0u64; N / 2];
        let mut operand2_odd = [0u64; N / 2];
        let mut result = CyclotomicRing::<MOD_Q, N>::new();

        operand1.forward_partial_ntt();
        operand2.forward_partial_ntt();

        operand1_even.copy_from_slice(&operand1.data[..N / 2]);
        operand1_odd.copy_from_slice(&operand1.data[N / 2..]);
        operand2_even.copy_from_slice(&operand2.data[..N / 2]);
        operand2_odd.copy_from_slice(&operand2.data[N / 2..]);



        unsafe {
            eltwise_mult_mod(
                result_even_2.as_mut_ptr(),
                operand1_even.as_ptr(),
                operand2_even.as_ptr(),
                result_even_2.len() as u64,
                MOD_Q,
            );
            eltwise_mult_mod(
                result_odd_2.as_mut_ptr(),
                operand1_odd.as_ptr(),
                operand2_even.as_ptr(),
                result_even_2.len() as u64,
                MOD_Q,
            );
            eltwise_mult_mod(
                operand1_odd.as_mut_ptr(),
                operand1_odd.as_ptr(),
                operand2_odd.as_ptr(),
                operand1_odd.len() as u64,
                MOD_Q,
            );
            eltwise_mult_mod(
                operand1_odd.as_mut_ptr(),
                operand1_odd.as_ptr(),
                shift_factors.as_ptr(),
                operand1_odd.len() as u64,
                MOD_Q,
            );
            eltwise_mult_mod(
                operand1_even.as_mut_ptr(),
                operand1_even.as_ptr(),
                operand2_odd.as_ptr(),
                operand1_even.len() as u64,
                MOD_Q,
            );
            eltwise_add_mod(
                result_even_2.as_mut_ptr(),
                result_even_2.as_ptr(),
                operand1_odd.as_ptr(),
                result_even_2.len() as u64,
                MOD_Q,
            );
            // use FMA!
            eltwise_add_mod(
                result_odd_2.as_mut_ptr(),
                result_odd_2.as_ptr(),
                operand1_even.as_ptr(),
                result_odd_2.len() as u64,
                MOD_Q,
            );
        }

        result.representation = Representation::IncompleteNTT;
        result.data[..N / 2].copy_from_slice(&result_even_2);
        result.data[N / 2..].copy_from_slice(&result_odd_2);
        result

    }

    pub fn fully_splitting_ntt_multiplication(operand1: &mut CyclotomicRing<MOD_Q, N>, 
        operand2: &mut CyclotomicRing<MOD_Q, N>,
    ) -> CyclotomicRing<MOD_Q, N>
    {
        operand1.to_ntt_representation();
        operand2.to_ntt_representation();

        let mut result = CyclotomicRing::<MOD_Q, N>::new();


        NTT::<MOD_Q, N>::multiply(&mut result.data, &mut operand1.data, &mut operand2.data);
        result.representation = Representation::NTT;
        result
    }
}



fn get_shift_factors<const MOD_Q: u64, const N: usize>() -> Vec<u64>
    where
        [(); N / 2]:,
    {
        let mut factors = vec![0u64; N / 2];
        factors[1] = 1;
        unsafe { ntt_forward_in_place(factors.as_mut_ptr(), factors.len(), MOD_Q) };
        factors
    }

pub fn naive_multiply<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
) -> CyclotomicRing<MOD_Q, N>
where
    [(); N / 2]:,
{
    operand1.to_coeff_representation();
    operand2.to_coeff_representation();
    let mut result = CyclotomicRing::<MOD_Q, N>::new();
    for i in 0..N {
        for j in 0..N {
            if i + j < N {
                result.data[i + j] =
                    (result.data[i + j] + 
                        ((operand1.data[i] as u128 * operand2.data[j] as u128) % MOD_Q as u128) as u64) % MOD_Q;
            } else {
                result.data[i + j - N] = 
                    (result.data[i + j - N] + MOD_Q -
                        ((operand1.data[i] as u128 * operand2.data[j] as u128) % MOD_Q as u128) as u64 ) % MOD_Q;
            }
        }
    }
    result
}
