#![allow(warnings)]
use dwmpi;
use std::slice::Chunks;

// Helper functions for direct computation without allocations
#[inline]
fn compute_k_first(nkpt: usize, nrank: usize, rank: usize) -> usize {
    let base_size = nkpt / nrank;
    let remainder = nkpt % nrank;

    if rank < remainder {
        rank * (base_size + 1)
    } else {
        rank * base_size + remainder
    }
}

#[inline]
fn compute_k_total(nkpt: usize, nrank: usize, rank: usize) -> usize {
    let base_size = nkpt / nrank;
    let remainder = nkpt % nrank;

    if rank < remainder {
        base_size + 1
    } else {
        base_size
    }
}

#[inline]
fn compute_k_last(nkpt: usize, nrank: usize, rank: usize) -> usize {
    compute_k_first(nkpt, nrank, rank) + compute_k_total(nkpt, nrank, rank) - 1
}

// Keep original function for compatibility if needed elsewhere
fn get_chunks(nkpt: usize, nrank: usize) -> Vec<Vec<usize>> {
    assert!(nkpt >= nrank);

    // Use more efficient computation of chunk sizes
    let base_size = nkpt / nrank;
    let remainder = nkpt % nrank;

    let mut vchunks = Vec::with_capacity(nrank);
    let mut start = 0;

    for rank in 0..nrank {
        let chunk_size = if rank < remainder {
            base_size + 1
        } else {
            base_size
        };
        let chunk: Vec<usize> = (start..start + chunk_size).collect();
        vchunks.push(chunk);
        start += chunk_size;
    }

    vchunks
}

pub fn get_k_first(nkpt: usize, nrank: usize, rank: usize) -> usize {
    assert!(nkpt >= nrank);
    compute_k_first(nkpt, nrank, rank)
}

pub fn get_k_last(nkpt: usize, nrank: usize, rank: usize) -> usize {
    assert!(nkpt >= nrank);
    compute_k_last(nkpt, nrank, rank)
}

pub fn get_my_k_first(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank() as usize;
    get_k_first(nkpt, nrank, rank)
}

pub fn get_my_k_last(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank() as usize;
    get_k_last(nkpt, nrank, rank)
}

pub fn get_my_k_total(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank() as usize;
    assert!(nkpt >= nrank);
    compute_k_total(nkpt, nrank, rank)
}

#[test]
fn test_kpts_distribution() {
    let nrank = 5;
    let nkpt = 31;

    for rank in 0..nrank {
        println!(
            "k1, k2 for rank {} = {}, {}",
            rank,
            get_k_first(nkpt, nrank, rank),
            get_k_last(nkpt, nrank, rank)
        );
    }
}

#[test]
fn test_optimization_correctness() {
    // Test against original implementation to ensure correctness
    let test_cases = [(10, 3), (31, 5), (100, 7), (1000, 16)];

    for (nkpt, nrank) in test_cases {
        for rank in 0..nrank {
            // Compare optimized functions with original chunk-based approach
            let chunks = get_chunks(nkpt, nrank);
            let original_first = chunks[rank].first().unwrap().clone();
            let original_last = chunks[rank].last().unwrap().clone();
            let original_total = chunks[rank].len();

            assert_eq!(get_k_first(nkpt, nrank, rank), original_first);
            assert_eq!(get_k_last(nkpt, nrank, rank), original_last);
            assert_eq!(compute_k_total(nkpt, nrank, rank), original_total);
        }
    }
}
