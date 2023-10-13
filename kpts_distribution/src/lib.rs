#![allow(warnings)]
use dwmpi;
use std::slice::Chunks;

fn get_chunks(nkpt: usize, nrank: usize) -> Vec<Vec<usize>> {
    assert!(nkpt >= nrank);

    let v: Vec<usize> = (0..nkpt).collect();
    let chunks = v.chunks(nrank);

    let vchunks = chunks.map(|chunk| chunk.to_owned()).collect();

    vchunks
}

pub fn get_k_first(nkpt: usize, nrank: usize, rank: usize) -> usize {
    let chunks = get_chunks(nkpt, nrank);

    chunks[rank].first().unwrap().clone()
}

pub fn get_k_last(nkpt: usize, nrank: usize, rank: usize) -> usize {
    let chunks = get_chunks(nkpt, nrank);

    chunks[rank].last().unwrap().clone()
}

pub fn get_my_k_first(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank();

    let chunks = get_chunks(nkpt, nrank);

    chunks[rank as usize].first().unwrap().clone()
}

pub fn get_my_k_last(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank();

    let chunks = get_chunks(nkpt, nrank);

    chunks[rank as usize].last().unwrap().clone()
}

pub fn get_my_k_total(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank();

    let chunks = get_chunks(nkpt, nrank);

    chunks[rank as usize].len()
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
