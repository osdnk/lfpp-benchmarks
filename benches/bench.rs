#![feature(generic_const_exprs)]

use std::hint::black_box;
use ring_arith::cyclotomic_ring::{*};
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use tfhe_ntt::{prime::largest_prime_in_arithmetic_progression64, *};

macro_rules! bench_polynomial_multiplication {
    ( $( ($p:expr, $n:expr) ),* ) => {
        pub fn polynomial_multiplication(c: &mut Criterion) {
            $(
                let backend = if cfg!(feature = "hexl") {
                    "hexl"
                } else if cfg!(feature = "tfhe") {
                    "zama"
                } else {
                    "unknown"
                };
                let bench_name = format!("NTT::fwd backend={} (p={}, n={})", backend, $p.1, $n);
                c.bench_function(&bench_name, |b| {
                    b.iter_batched(
                        || {
                            let mut ring = CyclotomicRing::< { $p.0 }, { $n }>::random();
                            ring.to_ntt_representation();
                            ring
                        },
                        |mut ring| {
                            ring.to_ntt_representation();
                        },
                        BatchSize::LargeInput,
                    );
                });

                let bench_name = format!("NTT::inv backend={} (p={}, n={})", backend, $p.1, $n);
                c.bench_function(&bench_name, |b| {

                    b.iter_batched(
                        || {
                            let mut ring = CyclotomicRing::< { $p.0 }, { $n }>::random();
                            ring.to_ntt_representation();
                            ring
                        },
                        |mut ring| {
                            ring.to_coeff_representation();
                        },
                        BatchSize::LargeInput,
                    );
                });

                let bench_name = format!("NTT::mul backend={} (p={}, n={})", backend, $p.1, $n);
                c.bench_function(&bench_name, |b| {

                    b.iter_batched(
                        || {
                            let mut left = CyclotomicRing::< { $p.0 }, { $n }>::random();
                            let mut right = CyclotomicRing::< { $p.0 }, { $n }>::random();
                            right.to_ntt_representation();
                            left.to_ntt_representation();
                            (right, left)
                        },
                        |(mut right, mut left)| {
                            let mut result = CyclotomicRing::fully_splitting_ntt_multiplication(&mut left, &mut right);
                        },
                        BatchSize::LargeInput
                    )
                });


                let bench_name = format!("NTT::full backend={} (p={}, n={})", backend, $p.1, $n);
                c.bench_function(&bench_name, |b| {
                    b.iter_batched(
                        || {
                            let mut left = CyclotomicRing::< { $p.0 }, { $n }>::random();
                            let mut right = CyclotomicRing::< { $p.0 }, { $n }>::random();
                            (right, left)
                        },
                        |(mut right, mut left)| {
                            let mut result = CyclotomicRing::fully_splitting_ntt_multiplication(&mut left, &mut right);
                            result.to_coeff_representation();
                        },
                        BatchSize::LargeInput
                    )
                });

                // Benchmark 4: Incomplete NTT
                /*let bench_name = format!("partial NTT multiplication (q={}, n={})", $p, $n);
                c.bench_function(&bench_name, |b| {
                    let mut operand1 = ring1.clone();
                    let mut operand2 = ring2.clone();
                    b.iter(|| incomplete_ntt_multiplication(black_box(&mut operand1), black_box(&mut operand2)))
                });
                */
            )*
        }
    };
}


type PrimeTest = (u64, &'static str);

const P1: PrimeTest = (largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 50, 1 << 51).unwrap(), "51 bits");
const P2: PrimeTest = (largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63).unwrap(), "63 bits");
const P3: PrimeTest = (prime64::Solinas::P, "64 bits Solinas");
const P4: PrimeTest = (largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 63, u64::MAX).unwrap(), "64 bits");


bench_polynomial_multiplication!(
    (P1, 256),
    (P1, 512),
    (P1, 1024),
    (P1, 2048),
    (P1, 4096),
    (P2, 256),
    (P2, 512),
    (P2, 1024),
    (P2, 2048),
    (P2, 4096),
    (P3, 256),
    (P3, 512),
    (P3, 1024),
    (P3, 2048),
    (P3, 4096),
    (P4, 256),
    (P4, 512),
    (P4, 1024),
    (P4, 2048),
    (P4, 4096)
);

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = polynomial_multiplication
}
criterion_main!(benches);
