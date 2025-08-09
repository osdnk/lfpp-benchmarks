#[link(name = "hexl_wrapper")]
extern "C" {
    pub fn multiply_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn power_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64;

    pub fn eltwise_mult_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64);

    pub fn get_roots(n: u64, modulus: u64) -> *const u64;
    pub fn get_inv_roots(n: u64, modulus: u64) -> *const u64;

    pub fn eltwise_add_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64);

    pub fn eltwise_sub_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64);

    pub fn multiply_poly(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64);

    pub fn eltwise_reduce_mod(result: *mut u64, operand: *const u64, n: u64, modulus: u64);

    pub fn polynomial_multiply_cyclotomic_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, phi: u64, mod_q: u64);
    pub fn ntt_forward_in_place(
        operand: *mut u64,
        n: usize,
        modulus: u64,
    );

    pub fn ntt_inverse_in_place(
        operand: *mut u64,
        n: usize,
        modulus: u64,
    );
}


pub fn cpp_multiply_mod(a: u64, b: u64, modulus: u64) -> u64 {
    unsafe { multiply_mod(a, b, modulus) }
}

pub fn cpp_eltwise_mult_mod(result: &mut [u64], a: &[u64], b: &[u64], modulus: u64) {
    assert_eq!(result.len(), a.len());
    assert_eq!(a.len(), b.len());
    unsafe {
        eltwise_mult_mod(result.as_mut_ptr(), a.as_ptr(), b.as_ptr(), a.len() as u64, modulus);
    }
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_mod() {
        let a: u64 = 12345;
        let b: u64 = 67890;
        let modulus: u64 = 1000000007;
        let result = cpp_multiply_mod(a, b, modulus);
        assert_eq!(result, 838102050); // This is the expected result for (12345 * 67890) % 1000000007
    }

    #[test]
    fn test_eltwise_mult_mod() {
        let a: Vec<u64> = vec![1, 2, 3, 4, 5];
        let b: Vec<u64> = vec![6, 7, 8, 9, 10];
        let modulus: u64 = 100;
        let mut result: Vec<u64> = vec![0; a.len()];

        cpp_eltwise_mult_mod(&mut result, &a, &b, modulus);

        assert_eq!(result, vec![6, 14, 24, 36, 50]); // Expected results for element-wise multiplication
    }
    #[test]
    fn test_multiply_poly() {
        let n = 8;
        let modulus = 65537; // Example modulus
        let operand1: Vec<u64> = vec![1, 2, 3, 1, 0, 0, 0, 0];
        let operand2: Vec<u64> = vec![8, 7, 6, 1, 0, 0, 0, 0];
        let mut result = vec![0u64; n];

        // The expected result must be computed using the same polynomial multiplication logic.
        // For simplicity, we use a direct multiplication method here assuming the input is small.
        let mut expected_result = vec![0u64; n];
        for i in 0..n / 2 {
            for j in 0..n / 2 {
                expected_result[i + j] = (expected_result[i + j] + operand1[i] * operand2[j]) % modulus;
            }
        }

        unsafe {
            multiply_poly(result.as_mut_ptr(), operand1.as_ptr(), operand2.as_ptr(), n as u64, modulus);
        }

        assert_eq!(result, expected_result);
    }
}
