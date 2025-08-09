use crate::hexl::bindings::{eltwise_add_mod, eltwise_mult_mod, eltwise_reduce_mod, eltwise_sub_mod, multiply_mod, ntt_forward_in_place, ntt_inverse_in_place, power_mod};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, RwLock};
use std::ops::{Add,Mul,Sub};
use std::iter::Sum;


#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum Representation {
    Coefficient,
    NTT,
    IncompleteNTT,
}


#[derive(Clone, Debug, Copy, PartialEq)]
pub struct CyclotomicRing<const MOD_Q: u64, const N: usize> {
    pub data: [u64; N],
    pub representation: Representation,
}

impl <const MOD_Q: u64, const N: usize> Add<&CyclotomicRing<MOD_Q, N>> for &mut CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn add(self, other: &CyclotomicRing<MOD_Q, N>) -> Self::Output {
        self.adjust_representation(other.representation);
        let mut result = CyclotomicRing::<MOD_Q, N>::new();
        unsafe { 
            eltwise_add_mod(
                result.data.as_mut_ptr(),
                self.data.as_ptr(),
                other.data.as_ptr(),
                N as u64,
                MOD_Q
            )
        } 
        result.representation = self.representation.clone();
        result
    }   
}


impl <const MOD_Q: u64, const N: usize,> Add for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn add(mut self, other: Self) -> Self::Output {
        &mut self + &other 
    }   
}


impl <const MOD_Q: u64, const N: usize> Sub<&CyclotomicRing<MOD_Q, N>> for &mut CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn sub(self, other: &CyclotomicRing<MOD_Q, N>) -> Self::Output {
        self.adjust_representation(other.representation);
        let mut result = CyclotomicRing::<MOD_Q, N>::new();
        unsafe { 
            eltwise_sub_mod(
                result.data.as_mut_ptr(),
                self.data.as_ptr(),
                other.data.as_ptr(),
                N as u64,
                MOD_Q
            )
        } 
        result.representation = self.representation.clone();
        result
    }   
}


impl <const MOD_Q: u64, const N: usize,> Sub for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn sub(mut self, other: Self) -> Self::Output {
        &mut self - &other 
    }   
}


// impl<const MOD_Q: u64, const N: usize> Sub for CyclotomicRing<MOD_Q, N> {
//     type Output = CyclotomicRing<MOD_Q, N>;
//     fn sub(mut self, other: Self) -> Self::Output {
//         self.adjust_representation(other.representation);
//         let mut result = CyclotomicRing::<MOD_Q, N>::new();
//         for i in 0..N {
//             result.data[i] = (self.data[i] + MOD_Q - other.data[i]) % MOD_Q;
//         }
//         result.representation = self.representation.clone();
//         result
//     }
// }

impl <const MOD_Q: u64, const N: usize> Mul for &mut CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn mul(self, other: Self) -> Self::Output {
        incomplete_ntt_multiplication::<MOD_Q, N>(self, other, true)
    }
}

impl <const MOD_Q: u64, const N: usize> Mul for CyclotomicRing<MOD_Q, N> {
    type Output = CyclotomicRing<MOD_Q, N>;

    fn mul(mut self, mut other: Self) -> Self::Output {
        &mut self * &mut other 
    }   
}


#[test]
fn test_addition_same_representation() {
    const MOD_Q: u64 = 17;
    const N: usize = 4;
    let mut a = CyclotomicRing::<MOD_Q, N>::new();
    let mut b = CyclotomicRing::<MOD_Q, N>::new();
    a.data = [1, 2, 3, 4];
    b.data = [4, 3, 2, 1];

    let c = &mut a + &b;

    assert_eq!(c.data, [5, 5, 5, 5]);
}

#[test]
fn test_subtraction_same_representation() {
    const MOD_Q: u64 = 17;
    const N: usize = 4;
    let mut a = CyclotomicRing::<MOD_Q, N>::new();
    let mut b = CyclotomicRing::<MOD_Q, N>::new();
    a.data = [5, 6, 7, 8];
    b.data = [4, 3, 2, 1];

    let c = &mut a - &b;

    assert_eq!(c.data, [1, 3, 5, 7]);
}

#[test]
fn test_multiplication_same_representation() {
    const MOD_Q: u64 = 17;
    const N: usize = 4;
    let mut a = CyclotomicRing::<MOD_Q, N>::new();
    let mut b = CyclotomicRing::<MOD_Q, N>::new();
    a.data = [1, 2, 1, 0];
    b.data = [1, 1, 1, 0];

    let mut c = &mut a * &mut b;
    c.to_coeff_representation();

    assert_eq!(c.data, [0, 3, 4, 3]);
}

static NORMALIZE_INCOMPLETE_NTT_FACTORS_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> = OnceLock::new();
static NORMALIZE_INCOMPLETE_NTT_FACTORS_INVERSE_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> = OnceLock::new();

impl<const MOD_Q: u64, const N: usize> CyclotomicRing<MOD_Q, N> {
    pub fn new() -> Self {
        Self { data: [0u64; N], representation: Representation::Coefficient }
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N];
        for i in 0..N {
            data[i] = rng.random_range(0..MOD_Q);
        }
        let mut  t = Self { data, representation: Representation::Coefficient };
        t.to_coeff_representation();
        t
    }

    pub fn random_real() -> Self {
        let t = CyclotomicRing::random();
        let mut res = t + t.conjugate();
        // assert_eq!(res, res.conjugate());
        res
    }

    pub fn random_bounded(bound: u64) -> Self {
        let mut rng = rand::rng();
        let mut data = [0u64; N];
        for i in 0..N {
            data[i] = rng.random_range(0..bound);
            if rng.random_bool(0.5) {
                data[i] = MOD_Q - data[i]; // Randomly negate the value
            }
        }
        unsafe {
            eltwise_reduce_mod(data.as_mut_ptr(), data.as_mut_ptr(), data.len() as u64, MOD_Q);
        }

        // TODO 
        let mut t = Self { data, representation: Representation::Coefficient };
        t.to_coeff_representation();
        t
        // t + t.conjugate()
    }

    pub fn constant(value: u64) -> Self {
        let mut data = [0u64; N];
        data[0] = value;
        Self { data, representation: Representation::Coefficient }
    }

    pub fn one() -> Self {
        let mut data = [0u64; N];
        data[0] = 1;
        Self { data, representation: Representation::Coefficient }
    }

    fn forward_ntt(&mut self) {
        unsafe { ntt_forward_in_place(self.data.as_mut_ptr(), self.data.len(), MOD_Q) };
    }

    fn inverse_ntt(&mut self) {
        unsafe { ntt_inverse_in_place(self.data.as_mut_ptr(), self.data.len(), MOD_Q) };
    }

    pub fn conjugate(&self) -> Self {
        let mut conjugated = self.clone();
        conjugated.to_coeff_representation();
        let conjugated_clone = conjugated.clone();

        for i in 1..N {
            if conjugated_clone.data[N - i] == 0 {
                conjugated.data[i] = 0;
                continue;
            }
            conjugated.data[i] = MOD_Q - conjugated_clone.data[N - i];
        }
        conjugated.adjust_representation(self.representation);
        conjugated
    }

    fn adjust_representation(&mut self, new_representation: Representation) {
        if self.representation == new_representation {
            return; // already in the desired representation
        }

        match new_representation {
            Representation::Coefficient => self.to_coeff_representation(),
            Representation::NTT => self.to_ntt_representation(),
            Representation::IncompleteNTT => self.to_incomplete_ntt_representation()
        }
    }

    pub fn to_incomplete_ntt_representation(&mut self) {
        if self.representation == Representation::IncompleteNTT {
            return; // already in NTT form
        }
        if self.representation == Representation::NTT {
            self.to_coeff_representation();
        }

        // Allocate a single array of size N for both even and odd parts
        let mut even_odd = [0u64; N];
        // Fill even and odd parts
        for i in 0..N / 2 {
            even_odd[i] = self.data[i * 2];           // even indices
            even_odd[i + N / 2] = self.data[i * 2 + 1]; // odd indices
        }

        // Perform NTT on both halves in-place
        unsafe {
            ntt_forward_in_place(even_odd.as_mut_ptr(), N / 2, MOD_Q);           // even part
            ntt_forward_in_place(even_odd.as_mut_ptr().add(N / 2), N / 2, MOD_Q); // odd part
        }

        // Assign back to even and odd for further use
        let even = &even_odd[..N / 2];
        let odd = &even_odd[N / 2..];

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

        unsafe { ntt_forward_in_place(self.data.as_mut_ptr(), self.data.len(), MOD_Q) };
        self.representation = Representation::NTT;
    }


    pub fn to_coeff_representation(&mut self) {
        if self.representation == Representation::Coefficient {
            return; // already in coefficient form
        }

        if self.representation == Representation::IncompleteNTT {
            // Use a single array to hold both even and odd parts
            let mut even_odd = [0u64; N];
            // Copy even and odd parts from self.data
            for i in 0..N / 2 {
                even_odd[i] = self.data[i];           // even part
                even_odd[i + N / 2] = self.data[i + N / 2]; // odd part
            }

            unsafe {
                ntt_inverse_in_place(even_odd.as_mut_ptr(), N / 2, MOD_Q);           // inverse NTT on even part
                ntt_inverse_in_place(even_odd.as_mut_ptr().add(N / 2), N / 2, MOD_Q); // inverse NTT on odd part
            }

            // Interleave even and odd back into self.data
            for i in 0..N / 2 {
                self.data[i * 2] = even_odd[i];
                self.data[i * 2 + 1] = even_odd[i + N / 2];
            }

        } 

        if self.representation == Representation::NTT {
            unsafe { ntt_inverse_in_place(self.data.as_mut_ptr(), self.data.len(), MOD_Q) };
        }

        self.representation = Representation::Coefficient;
    }
    
}

fn get_shift_factors<const MOD_Q: u64, const N: usize>() -> Vec<u64> {
        let mut factors = vec![0u64; N / 2];
        factors[1] = 1;
        unsafe { ntt_forward_in_place(factors.as_mut_ptr(), factors.len(), MOD_Q) };
        factors
    }


    



static SHIFT_FACTORS_CACHE: OnceLock<Vec<u64>> = OnceLock::new();


fn get_shift_factors_cached<const MOD_Q: u64, const N: usize>() -> Vec<u64> {
    // Ensure the cache is initialized
    // Safe to access without locking since OnceLock + HashMap is read-only after init
    if cfg!(test) {
        return  get_shift_factors::<MOD_Q, N>();
    }
    SHIFT_FACTORS_CACHE
        .get_or_init(get_shift_factors::<MOD_Q, N>)
        .clone()
}

pub fn incomplete_ntt_multiplication<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
    use_shift_factors: bool,
) -> CyclotomicRing<MOD_Q, N> {
    // Static local cache (thread-safe, mutable via Mutex)

    // Initialize or get the cache
    // let cache = SHIFT_FACTORS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // let mut cache_guard = cache.lock().unwrap();

    // Get or compute shift factors
    // init_shift_factors_cached::<MOD_Q, N>();
    let shift_factors = get_shift_factors_cached::<MOD_Q, N>();

    // if !use_shift_factors { 
    //     for i in 1..N / 2 {
    //         shift_factors[i] = shift_factors[0]; // Use identity for multiplication
    //     }
    // }

    // Use 3 arrays: result, op1, op2 (each N elements, even in first N/2, odd in second N/2)
    let mut result_parts = [0u64; N];
    let mut op1_parts = [0u64; N];
    let mut op2_parts = [0u64; N];
    let mut result = CyclotomicRing::<MOD_Q, N>::new();

    operand1.to_incomplete_ntt_representation();
    operand2.to_incomplete_ntt_representation();

    op1_parts.copy_from_slice(&operand1.data);
    op2_parts.copy_from_slice(&operand2.data);

    // result_even = op1_even * op2_even
    unsafe {
        eltwise_mult_mod(
            result_parts.as_mut_ptr(),
            op1_parts.as_ptr(),
            op2_parts.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );
        // result_odd = op1_odd * op2_even
        eltwise_mult_mod(
            result_parts.as_mut_ptr().add(N / 2),
            op1_parts.as_ptr().add(N / 2),
            op2_parts.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );
        // tmp1 = op1_odd * op2_odd
        eltwise_mult_mod(
            op1_parts.as_mut_ptr().add(N / 2),
            op1_parts.as_ptr().add(N / 2),
            op2_parts.as_ptr().add(N / 2),
            (N / 2) as u64,
            MOD_Q,
        );
        // tmp1 = tmp1 * shift_factors
        if use_shift_factors {
            eltwise_mult_mod(
                op1_parts.as_mut_ptr().add(N / 2),
                op1_parts.as_ptr().add(N / 2),
                shift_factors.as_ptr(),
                (N / 2) as u64,
                MOD_Q
            ) 
        } else {
            // If not using shift factors, just copy the odd part to tmp1
            eltwise_mult_mod(
                op1_parts.as_mut_ptr().add(N / 2),
                op1_parts.as_ptr().add(N / 2),
                vec![shift_factors[0]; N/2].as_ptr(),
                (N / 2) as u64,
                MOD_Q
            ) 
        }
        // tmp2 = op1_even * op2_odd
        eltwise_mult_mod(
            op1_parts.as_mut_ptr(),
            op1_parts.as_ptr(),
            op2_parts.as_ptr().add(N / 2),
            (N / 2) as u64,
            MOD_Q,
        );
        // result_even += tmp1
        eltwise_add_mod(
            result_parts.as_mut_ptr(),
            result_parts.as_ptr(),
            op1_parts.as_ptr().add(N / 2),
            (N / 2) as u64,
            MOD_Q,
        );
        // result_odd += tmp2
        eltwise_add_mod(
            result_parts.as_mut_ptr().add(N / 2),
            result_parts.as_ptr().add(N / 2),
            op1_parts.as_ptr(),
            (N / 2) as u64,
            MOD_Q,
        );
    }

    result.representation = Representation::IncompleteNTT;
    result.data.copy_from_slice(&result_parts);
    result
}

pub fn fully_splitting_ntt_multiplication<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
) -> CyclotomicRing<MOD_Q, N> {
    operand1.to_ntt_representation();
    operand2.to_ntt_representation();

    let mut result = CyclotomicRing::<MOD_Q, N>::new();

    unsafe {
        eltwise_mult_mod(
            result.data.as_mut_ptr(),
            operand1.data.as_ptr(),
            operand2.data.as_ptr(),
            result.data.len() as u64,
            MOD_Q,
        ) 
    };

    result.representation = Representation::NTT;
    result
}

pub fn naive_multiply<const MOD_Q: u64, const N: usize>(
    operand1: &mut CyclotomicRing<MOD_Q, N>,
    operand2: &mut CyclotomicRing<MOD_Q, N>,
) -> CyclotomicRing<MOD_Q, N> {
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
