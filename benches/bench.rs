use std::hint::black_box;
use ring_arith::{cyclotomic_ring::*, hexl::bindings::eltwise_add_mod};
use criterion::{criterion_group, criterion_main, Criterion};

const N: usize = 64;
// const MOD_Q: u64 = 4546383823830515713; // Example modulus
const MOD_Q: u64 = 1125899904679937; // Example modulus IFMA
// 1125899904679937
const K: usize = 2; 
const WIT_DIM: usize = 1048576; // 2^20
const LOG_B:usize = 10;


fn add_avx512(data: [u64; N], other: [u64; N]) -> [u64; N] {
    use std::arch::x86_64::*;
    #[cfg(target_feature = "avx512f")]
    unsafe {
        let mut result = [0u64; N];
        let chunks = N / 8;
        for i in 0..chunks {
            let a = _mm512_loadu_si512(data.as_ptr().add(i * 8) as *const _);
            let b = _mm512_loadu_si512(other.as_ptr().add(i * 8) as *const _);
            let sum = _mm512_add_epi64(a, b);
            _mm512_storeu_si512(result.as_mut_ptr().add(i * 8) as *mut _, sum);
        }
        return result;
    }
    panic!("AVX512 is not supported on this architecture");
}



fn bench_lfpp(c: &mut Criterion) {
    // 3.2501
    c.bench_function("lfp compute double commitment", |b| {
        b.iter_with_setup(
            || {
                let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
                let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();
                (operand1, operand2)
            },
            |(mut operand1, mut operand2)| {
                unsafe {
                    for _ in 0..WIT_DIM * K * N {
                        eltwise_add_mod(
                            black_box(operand1.data).as_mut_ptr(),
                            black_box(operand1.data).as_ptr(),
                            black_box(operand2.data).as_ptr(),
                            N as u64,
                            MOD_Q,
                        );
                    }
                }
            }, 
        )
    });

    // 1.3090 s
    c.bench_function("lfp compute double commitment no mod", |b| {
        b.iter_with_setup(
            || {
                let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
                let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();
                (operand1, operand2)
            },
            |(mut operand1, mut operand2)| {
                unsafe {
                    for _ in 0..WIT_DIM * K * N {
                        add_avx512(
                            black_box(operand1.data),
                            black_box(operand2.data),
                        );
                    }
                }
            }, 
        )
    });

    // 291.54 ms
    c.bench_function("lfpp compute extension commitment", |b| {
        b.iter_with_setup(
            || {
                let mut operand1 = CyclotomicRing::<MOD_Q, N>::random();
                let mut operand2 = CyclotomicRing::<MOD_Q, N>::random();
                (operand1, operand2)
            },
            |(mut operand1, mut operand2)| {
                for _ in 0..WIT_DIM * LOG_B {
                    fully_splitting_ntt_multiplication(&mut operand1, &mut operand2);
                }

            }, 
        )
    });
}




fn configure_criterion() -> Criterion {
    Criterion::default().sample_size(10)
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_lfpp
}
criterion_main!(benches);
