/// The last element is the one added last
use std::collections::VecDeque;

#[derive(Debug)]
pub struct FIFO<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> FIFO<T> {
    pub fn new(nelem: usize) -> Self {
        FIFO {
            data: VecDeque::with_capacity(nelem),
            capacity: nelem,
        }
    }

    pub fn push(&mut self, elem: T) {
        if self.data.len() < self.capacity {
            self.data.push_back(elem);
        } else {
            self.data.pop_front();
            self.data.push_back(elem);
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn last(&self) -> &T {
        &self.data.back().unwrap()
    }
}

impl<T> std::ops::Index<usize> for FIFO<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

#[test]
fn test_fifo() {
    let mut fq = FIFO::<f64>::new(5);

    assert!(fq.is_empty());
    assert_eq!(fq.len(), 0);

    fq.push(1.0);
    fq.push(2.0);
    fq.push(3.0);
    fq.push(4.0);
    fq.push(5.0);
    fq.push(6.0);

    assert_eq!(fq.len(), 5);
    assert_eq!(fq[0], 2.0);
    assert_eq!(fq[1], 3.0);
    assert_eq!(fq[2], 4.0);
    assert_eq!(fq[3], 5.0);
    assert_eq!(fq[4], 6.0);
    assert_eq!(*fq.last(), 6.0);
}

#[test]
fn test_fifo_rollover_preserves_order() {
    let mut fq = FIFO::<i32>::new(3);

    for v in 0..6 {
        fq.push(v);
    }

    assert_eq!(fq.len(), 3);
    assert_eq!(fq[0], 3);
    assert_eq!(fq[1], 4);
    assert_eq!(fq[2], 5);
    assert_eq!(*fq.last(), 5);
}

#[test]
#[should_panic]
fn test_fifo_last_panics_when_empty() {
    let fq = FIFO::<i32>::new(2);
    let _ = fq.last();
}
