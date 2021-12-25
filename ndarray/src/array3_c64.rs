use crate::Array3;

use itertools::multizip;

use types::*;

impl Array3<c64> {
    pub fn scale(&mut self, f: f64) {
        self.data.iter_mut().for_each(|x| *x *= f);
    }

    pub fn norm2(&self) -> f64 {
        self.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
    }

    pub fn scaled_assign(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = *s * factor;
        }
    }

    pub fn diff_norm_sum(&self, rhs: &Array3<c64>) -> f64 {
        let prhs = rhs.as_slice();
        let plhs = self.as_slice();

        assert_eq!(prhs.len(), plhs.len());

        let mut sum = 0.0;

        for (&l, &r) in multizip((plhs.iter(), prhs.iter())) {
            sum += (l - r).norm_sqr();
        }

        sum.sqrt()
    }

    pub fn sqr_add(&mut self, rhs: &Array3<c64>) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d += s.norm_sqr();
        }
    }

    pub fn scaled_sqr_add(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d += s.norm_sqr() * factor;
        }
    }

    pub fn scaled_add(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d += (*s) * factor;
        }
    }

    pub fn mix(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = (*d) * factor + (*s) * (1.0 - factor);
        }
    }
}

#[test]
fn test_array3_c64() {
    let mut m: Array3<c64> = Array3::<c64>::new([4, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                m[[k, j, i]] = c64 {
                    re: (i * j * k) as f64,
                    im: 0.0,
                };
            }
        }
    }

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                // check if memory is contiguous and value is correct

                println!("{:p} {} {}", &m[[k, j, i]], m[[k, j, i]], i * j * k);
            }
        }
    }

    println!("sum of m = {}", m.sum());

    m.save("a3.dat");

    let mut m_reload = Array3::<c64>::new([4, 3, 2]);
    m_reload.load("a3.dat");
    println!("{:?}", m_reload);

    m_reload.add_from(&m);

    println!("sum of m_reload = {}", m_reload.sum());
}
