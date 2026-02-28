#![allow(warnings)]
use dwmpi;
use std::cmp::Ordering;
use std::collections::HashMap;

// K-point distribution helpers for MPI ranks.
//
// Contiguous strategy:
// - contiguous block distribution
// - first `remainder` ranks take one extra point
// - O(1) first/last/total computations
//
// Cost-aware strategy:
// - greedy largest-processing-time (LPT) assignment based on user-provided
//   per-kpoint costs
// - deterministic tie-breaking for reproducibility

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
pub enum KPointScheduleMode {
    Contiguous,
    CostAware,
    Dynamic,
}

impl KPointScheduleMode {
    pub fn as_str(self) -> &'static str {
        match self {
            KPointScheduleMode::Contiguous => "contiguous",
            KPointScheduleMode::CostAware => "cost_aware",
            KPointScheduleMode::Dynamic => "dynamic",
        }
    }
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
    global_indices: Vec<usize>,
    global_to_local: HashMap<usize, usize>,
    empty_fallback_first: usize,
}

impl KPointDomain {
    fn from_indices_with_fallback(global_indices: Vec<usize>, empty_fallback_first: usize) -> Self {
        let mut global_to_local = HashMap::with_capacity(global_indices.len());
        for (local, &global) in global_indices.iter().enumerate() {
            global_to_local.insert(global, local);
        }

        Self {
            global_indices,
            global_to_local,
            empty_fallback_first,
        }
    }

    pub fn from_indices(global_indices: Vec<usize>) -> Self {
        Self::from_indices_with_fallback(global_indices, 0)
    }

    pub fn new(nkpt: usize, nrank: usize, rank: usize) -> Self {
        assert!(nrank > 0);
        assert!(rank < nrank);

        let first = compute_k_first(nkpt, nrank, rank);
        let total = compute_k_total(nkpt, nrank, rank);
        let global_indices = (first..first + total).collect::<Vec<usize>>();

        Self::from_indices_with_fallback(global_indices, first)
    }

    pub fn for_current_rank(nkpt: usize, nrank: usize) -> Self {
        let rank = dwmpi::get_comm_world_rank() as usize;
        Self::new(nkpt, nrank, rank)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.global_indices.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.global_indices.is_empty()
    }

    #[inline]
    pub fn global_indices(&self) -> &[usize] {
        self.global_indices.as_slice()
    }

    #[inline]
    pub fn global_range(&self) -> Option<(usize, usize)> {
        if self.global_indices.is_empty() {
            None
        } else {
            let mut min_idx = usize::MAX;
            let mut max_idx = 0usize;
            for &ik in self.global_indices.iter() {
                if ik < min_idx {
                    min_idx = ik;
                }
                if ik > max_idx {
                    max_idx = ik;
                }
            }
            Some((min_idx, max_idx))
        }
    }

    #[inline]
    pub fn global_first_or_zero(&self) -> usize {
        self.global_indices.first().copied().unwrap_or(0)
    }

    #[inline]
    pub fn global_last_or_first_minus_one(&self) -> usize {
        self.global_indices
            .last()
            .copied()
            .unwrap_or_else(|| self.empty_fallback_first.saturating_sub(1))
    }

    #[inline]
    pub fn contains_global(&self, global_index: usize) -> bool {
        self.global_to_local.contains_key(&global_index)
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
        self.global_indices.get(local_index).copied()
    }

    #[inline]
    pub fn local_index(&self, global_index: usize) -> Option<usize> {
        self.global_to_local.get(&global_index).copied()
    }

    pub fn iter(&self) -> KPointDomainIter<'_> {
        KPointDomainIter {
            domain: self,
            next_local: 0,
        }
    }
}

pub struct KPointDomainIter<'a> {
    domain: &'a KPointDomain,
    next_local: usize,
}

impl Iterator for KPointDomainIter<'_> {
    type Item = KPointSlot;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_local >= self.domain.len() {
            return None;
        }

        let local_slot = self.next_local;
        self.next_local += 1;

        Some(KPointSlot {
            local_slot,
            global_index: self.domain.global_indices[local_slot],
        })
    }
}

#[derive(Clone, Debug)]
pub struct KPointSchedulePlan {
    mode: KPointScheduleMode,
    rank_indices: Vec<Vec<usize>>,
    rank_loads: Vec<u64>,
    total_cost: u64,
}

impl KPointSchedulePlan {
    pub fn new_from_costs(costs: &[u64], nrank: usize, mode: KPointScheduleMode) -> Self {
        assert!(nrank > 0);
        let nkpt = costs.len();
        let rank_indices = match mode {
            KPointScheduleMode::Contiguous => build_contiguous_assignments(nkpt, nrank),
            KPointScheduleMode::CostAware => build_cost_aware_assignments(costs, nrank, false),
            KPointScheduleMode::Dynamic => build_cost_aware_assignments(costs, nrank, true),
        };

        let mut rank_loads = vec![0u64; nrank];
        for rank in 0..nrank {
            rank_loads[rank] = rank_indices[rank]
                .iter()
                .map(|&ik| costs.get(ik).copied().unwrap_or(0))
                .sum();
        }
        let total_cost = costs.iter().sum();

        Self {
            mode,
            rank_indices,
            rank_loads,
            total_cost,
        }
    }

    pub fn mode(&self) -> KPointScheduleMode {
        self.mode
    }

    pub fn rank_count(&self) -> usize {
        self.rank_indices.len()
    }

    pub fn total_cost(&self) -> u64 {
        self.total_cost
    }

    pub fn rank_loads(&self) -> &[u64] {
        self.rank_loads.as_slice()
    }

    pub fn max_rank_load(&self) -> u64 {
        self.rank_loads.iter().copied().max().unwrap_or(0)
    }

    pub fn min_rank_load(&self) -> u64 {
        self.rank_loads.iter().copied().min().unwrap_or(0)
    }

    pub fn mean_rank_load(&self) -> f64 {
        if self.rank_loads.is_empty() {
            return 0.0;
        }
        self.total_cost as f64 / self.rank_loads.len() as f64
    }

    pub fn imbalance_ratio(&self) -> f64 {
        let mean = self.mean_rank_load();
        if mean <= 0.0 {
            0.0
        } else {
            self.max_rank_load() as f64 / mean
        }
    }

    pub fn domain_for_rank(&self, rank: usize) -> KPointDomain {
        assert!(rank < self.rank_indices.len());
        KPointDomain::from_indices_with_fallback(self.rank_indices[rank].clone(), 0)
    }

    pub fn domain_for_current_rank(&self) -> KPointDomain {
        let rank = dwmpi::get_comm_world_rank() as usize;
        self.domain_for_rank(rank)
    }
}

fn build_contiguous_assignments(nkpt: usize, nrank: usize) -> Vec<Vec<usize>> {
    let mut assignments = Vec::with_capacity(nrank);
    for rank in 0..nrank {
        let first = compute_k_first(nkpt, nrank, rank);
        let total = compute_k_total(nkpt, nrank, rank);
        assignments.push((first..first + total).collect::<Vec<usize>>());
    }
    assignments
}

fn build_cost_aware_assignments(costs: &[u64], nrank: usize, dynamic_local_order: bool) -> Vec<Vec<usize>> {
    let mut assignments = vec![Vec::<usize>::new(); nrank];
    if costs.is_empty() {
        return assignments;
    }

    // Largest cost first with deterministic tie-breaks by global index.
    let mut order = (0..costs.len()).collect::<Vec<usize>>();
    order.sort_by(|&lhs, &rhs| {
        costs[rhs]
            .cmp(&costs[lhs])
            .then_with(|| lhs.cmp(&rhs))
    });

    let mut rank_load = vec![0u64; nrank];
    for ik in order {
        let best_rank = rank_load
            .iter()
            .enumerate()
            .min_by(|(ra, la), (rb, lb)| {
                la.cmp(lb).then_with(|| ra.cmp(rb))
            })
            .map(|(rank, _)| rank)
            .unwrap_or(0);

        assignments[best_rank].push(ik);
        rank_load[best_rank] += costs[ik];
    }

    if dynamic_local_order {
        for indices in assignments.iter_mut() {
            indices.sort_by(|&lhs, &rhs| {
                costs[rhs]
                    .cmp(&costs[lhs])
                    .then_with(|| lhs.cmp(&rhs))
            });
        }
    } else {
        for indices in assignments.iter_mut() {
            indices.sort_unstable();
        }
    }

    assignments
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
    let test_cases = [(10, 3), (31, 5), (100, 7), (1000, 16)];

    for (nkpt, nrank) in test_cases {
        for rank in 0..nrank {
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

            for slot in domain.iter() {
                assert_eq!(domain.global_index(slot.local_slot), Some(slot.global_index));
                assert_eq!(domain.local_index(slot.global_index), Some(slot.local_slot));
                assert_eq!(domain.slot(slot.local_slot), Some(slot));
                assert_eq!(domain.slot_from_global(slot.global_index), Some(slot));

                assert!(!seen[slot.global_index]);
                seen[slot.global_index] = true;
                seen_count += 1;
            }
        }

        assert_eq!(seen_count, nkpt);
        assert!(seen.into_iter().all(|v| v));
    }
}

#[test]
fn test_cost_aware_plan_partition_invariants() {
    let costs = vec![9u64, 1, 4, 7, 8, 2, 3, 6, 5];
    let plan = KPointSchedulePlan::new_from_costs(&costs, 4, KPointScheduleMode::CostAware);

    let mut seen = vec![false; costs.len()];
    let mut seen_count = 0usize;
    for rank in 0..4 {
        let domain = plan.domain_for_rank(rank);
        for &ik in domain.global_indices() {
            assert!(ik < costs.len());
            assert!(!seen[ik]);
            seen[ik] = true;
            seen_count += 1;
        }
    }

    assert_eq!(seen_count, costs.len());
    assert!(seen.into_iter().all(|v| v));
}

#[test]
fn test_cost_aware_plan_reduces_imbalance_for_skewed_costs() {
    let costs = vec![100u64, 90, 80, 1, 1, 1, 1, 1];
    let nrank = 3usize;

    let contiguous = KPointSchedulePlan::new_from_costs(&costs, nrank, KPointScheduleMode::Contiguous);
    let balanced = KPointSchedulePlan::new_from_costs(&costs, nrank, KPointScheduleMode::CostAware);

    assert!(balanced.max_rank_load() <= contiguous.max_rank_load());
}

#[test]
fn test_dynamic_mode_orders_local_tasks_by_descending_cost() {
    let costs = vec![10u64, 2, 9, 3, 8, 4, 7, 5, 6, 1];
    let plan = KPointSchedulePlan::new_from_costs(&costs, 3, KPointScheduleMode::Dynamic);

    for rank in 0..3 {
        let domain = plan.domain_for_rank(rank);
        let mut prev = u64::MAX;
        for &ik in domain.global_indices() {
            let c = costs[ik];
            assert!(c <= prev);
            prev = c;
        }
    }
}
