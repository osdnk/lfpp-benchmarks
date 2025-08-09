
#![feature(generic_const_exprs)]
use ring_arith::cyclotomic_ring::*;
use rand::Rng;
use tfhe_ntt::{prime::largest_prime_in_arithmetic_progression64, *};

fn main() {
    const N: usize = 256;
    //const MOD_Q: u64 = 4546383823830515713; // Example modulus
    const MOD_Q: u64 = 1125899904679969;
    // const MOD_Q: u64 = 257; // Example modulus
    let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
    let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();

    let expected_result = naive_multiply(&mut operand1, &mut operand2);
    let mut rng = rand::rng();
    let mut lhs_poly = vec![0; N];
    let mut rhs_poly = vec![0; N];
    for i in 0..N {
        lhs_poly[i] = rng.random_range(0..MOD_Q);
        rhs_poly[i] = rng.random_range(0..MOD_Q);
    }

    let plan = native64::Plan32::try_new(N).unwrap();
    let mut product_poly = vec![0; N];

    plan.negacyclic_polymul(&mut product_poly, &lhs_poly, &rhs_poly);

    //let mut result = incomplete_ntt_multiplication::<MOD_Q, N>(&mut operand1, &mut operand2);
    //let mut result_old = fully_splitting_ntt_multiplication::<MOD_Q, N>(&mut operand1, &mut operand2);

    //result.to_coeff_representation();
    //result_old.to_coeff_representation();
    assert_eq!(product_poly, expected_result.data);
    //assert_eq!(result_old.data, expected_result.data);
}
