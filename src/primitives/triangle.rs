use primitives::point::Point;
use primitives::vector::Vector;

/// This structure represents a triangle in 3D space.
#[derive(Debug, Hash)]
pub struct Triangle {
    points : Vec<Point>,
    normal : Option<Vector>
}

impl PartialEq for Triangle {
    fn eq(&self, other: &Triangle) -> bool {
        (self.points[0] == other.points[0]) & (self.points[1] == other.points[1]) & (self.points[2] == other.points[2])
    }
}

impl Eq for Triangle {}

impl Triangle {
    /// This method creates `Triangle` from a `Vec` of points and calculates a normal using `points`.
    /// # Arguments
    ///
    /// * `points` - A `Vec<Point>` to create the triangle.
    pub fn new(points : Vec<Point>) -> Triangle {
        let mut t  = Triangle {
            points : points,
            normal : None
        };
        let v1 = t.get_ref(0) - t.get_ref(1);
        let v2 = t.get_ref(1) - t.get_ref(2);
        let n = v1.cross_product(&v2);
        t.normal = Some(n);
        return t;
    }

    /// This method creates `Triangle` from a `Vec` of points.
    /// # Arguments
    ///
    /// * `points` - A `Vec<Point>' to create the triangle.
    /// * `normal` - A normal vector.
    pub fn new_with_normal(points : Vec<Point>, normal : Vector) -> Triangle {
        let t  = Triangle {
            points : points,
            normal : Some(normal)
        };
        return t;
    }

    /// This method returns the reference to the `Vec<Point>`, containing the triangle points.
    pub fn get_points_ref(&self) -> &Vec<Point> {
        return &self.points;
    }

    /// This method returns the reference to the `Vec<Point>`, containing the triangle points.
    pub fn get_points(self) -> Vec<Point> {
        return self.points;
    }

    /// This method returns the copy of the point, specified by `index`.
    /// # Arguments
    ///
    /// * `index` - An index of the point. It has to be less than 3.
    pub fn get(&self, index : usize) -> Point {
        self.points[index].clone()
    }

    /// This method returns the reference to the point, specified by `index`.
    /// # Arguments
    ///
    /// * `index` - An index of the point. It has to be less than 3.
    pub fn get_ref(&self, index : usize) -> &Point {
        &self.points[index]
    }

    /// This method returns a copy of the normal `Vector`.
    pub fn get_normal(&self) -> Vector {
        if self.normal.is_some() {
            return self.normal.clone().unwrap();
        } else {
            panic!("Something goes wrong!");
        }
    }

    /// This method checks the triangle and returns:
    /// 0 - if it's common triangle (there are zero coincident points).
    /// 1 - if it's a segment (there are two coincident points).
    /// 2 - is it's a point (there are three coincident points).
    pub fn degradation_level(&self) -> u64 {
        if self.points[0] == self.points[1] && self.points[1] == self.points[2] {
            return 2;
        }

        let cp = (&self.points[0] - &self.points[1]).cross_product(&(&self.points[2] - &self.points[1]));
        if cp.is_zero() {
            return 1;
        }

        return 0;
    }
}


