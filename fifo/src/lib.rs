/// The last element is the one added last

#[derive(Debug)]
pub struct FIFO<T> {
    data: Vec<T>,
    capacity: usize,
}

impl<T> FIFO<T> {
    pub fn new(nelem: usize) -> Self {
        FIFO {
            data: Vec::with_capacity(nelem),
            capacity: nelem,
        }
    }

    pub fn push(&mut self, elem: T) {
        if self.data.len() < self.capacity {
            self.data.push(elem);
        } else {
            self.data.remove(0usize);
            self.data.push(elem);
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    pub fn last(&self) -> &T {
        self.data.last().unwrap()
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

    fq.push(1.0);
    fq.push(2.0);
    fq.push(3.0);
    fq.push(4.0);
    fq.push(5.0);
    fq.push(6.0);

    println!("{:?}", fq.data);
    println!("capacity = {}", fq.data.capacity());

    let ntot = fq.len();
    for i in 0..ntot {
        println!("{}, {}", i, fq[i]);
    }
    assert_eq!(fq[4], 6.0);
}
