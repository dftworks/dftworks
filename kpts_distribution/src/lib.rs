#![allow(warnings)]
use dwmpi;

// K-point distribution helpers for MPI ranks.
//
// Strategy:
// - contiguous block distribution
// - first `remainder` ranks take one extra point
// - O(1) computations (no temporary chunk vectors needed)
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
    let first = compute_k_first(nkpt, nrank, rank);
    let total = compute_k_total(nkpt, nrank, rank);

    if total == 0 {
        first.saturating_sub(1)
    } else {
        first + total - 1
    }
}

// Reference chunk implementation retained for tests/compatibility.
fn get_chunks(nkpt: usize, nrank: usize) -> Vec<Vec<usize>> {
    assert!(nrank > 0);

    // Same rule as O(1) helpers, materialized explicitly.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KPointSlot {
    pub local_slot: usize,
    pub global_index: usize,
}

impl KPointSlot {
    #[inline]
    pub fn local_index(&self) -> usize {
        self.local_slot
    }
}

#[derive(Clone, Debug)]
pub struct KPointDomain {
    first: usize,
    total: usize,
}

impl KPointDomain {
    pub fn new(nkpt: usize, nrank: usize, rank: usize) -> Self {
        assert!(nrank > 0);
        assert!(rank < nrank);

        let first = compute_k_first(nkpt, nrank, rank);
        let total = compute_k_total(nkpt, nrank, rank);

        Self { first, total }
    }

    pub fn for_current_rank(nkpt: usize, nrank: usize) -> Self {
        let rank = dwmpi::get_comm_world_rank() as usize;
        Self::new(nkpt, nrank, rank)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    #[inline]
    pub fn global_range(&self) -> Option<(usize, usize)> {
        if self.total == 0 {
            None
        } else {
            Some((self.first, self.first + self.total - 1))
        }
    }

    #[inline]
    pub fn global_first_or_zero(&self) -> usize {
        self.global_range().map(|(first, _)| first).unwrap_or(0)
    }

    #[inline]
    pub fn global_last_or_first_minus_one(&self) -> usize {
        self.global_range()
            .map(|(_, last)| last)
            .unwrap_or_else(|| self.first.saturating_sub(1))
    }

    #[inline]
    pub fn contains_global(&self, global_index: usize) -> bool {
        self.local_index(global_index).is_some()
    }

    #[inline]
    pub fn slot(&self, local_slot: usize) -> Option<KPointSlot> {
        self.global_index(local_slot).map(|global_index| KPointSlot {
            local_slot,
            global_index,
        })
    }

    #[inline]
    pub fn slot_from_global(&self, global_index: usize) -> Option<KPointSlot> {
        self.local_index(global_index).map(|local_slot| KPointSlot {
            local_slot,
            global_index,
        })
    }

    #[inline]
    pub fn global_index(&self, local_index: usize) -> Option<usize> {
        if local_index < self.total {
            Some(self.first + local_index)
        } else {
            None
        }
    }

    #[inline]
    pub fn local_index(&self, global_index: usize) -> Option<usize> {
        if global_index < self.first || global_index >= self.first + self.total {
            None
        } else {
            Some(global_index - self.first)
        }
    }

    pub fn iter(&self) -> KPointDomainIter {
        KPointDomainIter {
            first: self.first,
            total: self.total,
            next_local: 0,
        }
    }
}

pub struct KPointDomainIter {
    first: usize,
    total: usize,
    next_local: usize,
}

impl Iterator for KPointDomainIter {
    type Item = KPointSlot;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_local >= self.total {
            return None;
        }

        let local_slot = self.next_local;
        self.next_local += 1;

        Some(KPointSlot {
            local_slot,
            global_index: self.first + local_slot,
        })
    }
}

pub fn get_k_first(nkpt: usize, nrank: usize, rank: usize) -> usize {
    assert!(nrank > 0);
    compute_k_first(nkpt, nrank, rank)
}

pub fn get_k_last(nkpt: usize, nrank: usize, rank: usize) -> usize {
    assert!(nrank > 0);
    compute_k_last(nkpt, nrank, rank)
}

pub fn get_k_range(nkpt: usize, nrank: usize, rank: usize) -> Option<(usize, usize)> {
    assert!(nrank > 0);
    let ntot = compute_k_total(nkpt, nrank, rank);
    if ntot == 0 {
        None
    } else {
        Some((
            get_k_first(nkpt, nrank, rank),
            get_k_last(nkpt, nrank, rank),
        ))
    }
}

pub fn get_my_k_first(nkpt: usize, nrank: usize) -> usize {
    // Convenience wrappers bound to current MPI rank.
    let rank = dwmpi::get_comm_world_rank() as usize;
    get_k_first(nkpt, nrank, rank)
}

pub fn get_my_k_last(nkpt: usize, nrank: usize) -> usize {
    let rank = dwmpi::get_comm_world_rank() as usize;
    get_k_last(nkpt, nrank, rank)
}

pub fn get_my_k_total(nkpt: usize, nrank: usize) -> usize {
    KPointDomain::for_current_rank(nkpt, nrank).len()
}

pub fn get_my_k_range(nkpt: usize, nrank: usize) -> Option<(usize, usize)> {
    KPointDomain::for_current_rank(nkpt, nrank).global_range()
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
    // Compare O(1) helpers against reference chunk-based implementation.
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

#[test]
fn test_oversubscribed_ranks() {
    // Edge case: more ranks than k-points.
    let nkpt = 3;
    let nrank = 5;

    let expected_total = [1, 1, 1, 0, 0];

    for rank in 0..nrank {
        let ntot = compute_k_total(nkpt, nrank, rank);
        assert_eq!(ntot, expected_total[rank]);

        let range = get_k_range(nkpt, nrank, rank);
        if ntot == 0 {
            assert!(range.is_none());
        } else {
            let (k1, k2) = range.unwrap();
            assert_eq!(k2 - k1 + 1, ntot);
        }
    }
}

#[test]
fn test_kpoint_domain_iteration_and_mapping() {
    let domain = KPointDomain::new(10, 3, 1);
    let slots = domain.iter().collect::<Vec<KPointSlot>>();

    assert_eq!(domain.global_range(), Some((4, 6)));
    assert_eq!(domain.len(), 3);
    assert_eq!(
        slots,
        vec![
            KPointSlot {
                local_slot: 0,
                global_index: 4
            },
            KPointSlot {
                local_slot: 1,
                global_index: 5
            },
            KPointSlot {
                local_slot: 2,
                global_index: 6
            },
        ]
    );
    assert_eq!(domain.slot(1), Some(KPointSlot { local_slot: 1, global_index: 5 }));
    assert_eq!(
        domain.slot_from_global(6),
        Some(KPointSlot {
            local_slot: 2,
            global_index: 6
        })
    );
    assert_eq!(domain.global_index(0), Some(4));
    assert_eq!(domain.global_index(2), Some(6));
    assert_eq!(domain.global_index(3), None);
    assert_eq!(domain.local_index(4), Some(0));
    assert_eq!(domain.local_index(6), Some(2));
    assert_eq!(domain.local_index(3), None);
    assert_eq!(domain.local_index(7), None);
    assert!(domain.contains_global(5));
    assert!(!domain.contains_global(7));
}

#[test]
fn test_empty_local_domain_invariants() {
    let domain = KPointDomain::new(3, 5, 4);
    assert!(domain.is_empty());
    assert_eq!(domain.len(), 0);
    assert_eq!(domain.global_range(), None);
    assert_eq!(domain.global_first_or_zero(), 0);
    assert_eq!(domain.slot(0), None);
    assert_eq!(domain.slot_from_global(0), None);
    assert_eq!(domain.global_index(0), None);
    assert_eq!(domain.local_index(0), None);
    assert_eq!(domain.iter().count(), 0);
}

#[test]
fn test_partition_invariants_uneven_and_oversubscribed() {
    let test_cases = [(17usize, 6usize), (3usize, 8usize)];

    for (nkpt, nrank) in test_cases {
        let mut seen = vec![false; nkpt];
        let mut seen_count = 0usize;

        for rank in 0..nrank {
            let domain = KPointDomain::new(nkpt, nrank, rank);

            // Range length and iterator length always agree.
            let expected_len = domain
                .global_range()
                .map(|(first, last)| last - first + 1)
                .unwrap_or(0);
            assert_eq!(domain.len(), expected_len);

            for slot in domain.iter() {
                // local->global and global->local transforms must be reversible.
                assert_eq!(domain.global_index(slot.local_slot), Some(slot.global_index));
                assert_eq!(domain.local_index(slot.global_index), Some(slot.local_slot));
                assert_eq!(domain.slot(slot.local_slot), Some(slot));
                assert_eq!(domain.slot_from_global(slot.global_index), Some(slot));

                // Each global k-point belongs to exactly one rank.
                assert!(!seen[slot.global_index]);
                seen[slot.global_index] = true;
                seen_count += 1;
            }
        }

        assert_eq!(seen_count, nkpt);
        assert!(seen.into_iter().all(|v| v));
    }
}
