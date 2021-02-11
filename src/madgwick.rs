extern crate nalgebra;
use std::fmt;
use std::ops::Index;
use self::nalgebra::{Vector4, Vector6, Matrix6x4, Vector3, Vector2};

pub struct Quaternion {
    q: Vector4<f64>,
}

impl Quaternion {
    fn conj(&self) -> Quaternion {
      return Quaternion {q: Vector4::new(self.q[0], -self.q[1], -self.q[2], -self.q[3])};
    }
    fn new(w: f64, x: f64, y: f64, z:f64) -> Quaternion {
        Quaternion {q: Vector4::new(w, x, y, z)}
    }
    fn q_new(q: Vector4<f64>) -> Quaternion {
        return Quaternion {q};
    }
    fn mul(&self, other: Quaternion) -> Quaternion  {
        let w = self.q[0]*other.q[0] - self.q[1]*other.q[1] - self.q[2]*other.q[2] - self.q[3]*other.q[3];
        let x = self.q[0]*other.q[1] + self.q[1]*other.q[0] + self.q[2]*other.q[3] - self.q[3]*other.q[2];
        let y = self.q[0]*other.q[2] - self.q[1]*other.q[3] + self.q[2]*other.q[0] + self.q[3]*other.q[1];
        let z = self.q[0]*other.q[3] + self.q[1]*other.q[2] - self.q[2]*other.q[1] + self.q[3]*other.q[0];
        return Quaternion::new(w, x, y, z);
    }
    fn scalar_mul(&self, other: f64) -> Quaternion {
        return Quaternion::q_new(&self.q * other);
    }
    fn add(&self, other: Quaternion) -> Quaternion {
        return Quaternion::q_new(&self.q + &other.q);
    }
//    fn scalar_add(&self, other: f64) -> Quaternion {
//        return Quaternion::q_new(self.q.add_scalar(other));
//    }
    pub fn to_euler_angles(&self) -> (f32, f32, f32, f32) {
//        let q = &self.q;
//        let factor = q[1] * q[2] + q[3] * q[0];
//        let pitch = (2.0 * q[1] * q[2] + 2.0 * q[0] * q[3]).asin();
//        let roll: f64;
//        let yaw: f64;
//        if (factor - 0.5).abs() < 1e-2 {
//            roll = 0.0;
//            yaw = 2.0 * q[1].atan2(q[0]);
//        } else if (factor + 0.5).abs() < 1e-2 {
//            roll = -2.0 * q[1].atan2(q[0]);
//            yaw = 0.0;
//        } else {
//            roll = (2.0 * q[0] * q[1] - 2.0 * q[2] * q[3]).atan2(1.0 - 2.0 * q[1] * q[1]  - 2.0 * q[3] * q[3]);
//            yaw = (2.0 * q[0] * q[2] - 2.0 * q[1] * q[3]).atan2(1.0 - 2.0 * q[2] * q[2] - 2.0 * q[3] * q[3]);
//        }
        let q = &self.q;
        let w = q[0];
        let x = q[1];
        let y = q[2];
        let z = q[3];
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);
        let sinp = 2.0 * (w * y - z * x);
        let pitch: f64;
        if sinp.abs() >= 1.0 {
            pitch = sinp.signum() * std::f64::consts::PI / 2.0;
        } else {
            pitch = sinp.asin();
        }
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        return (roll as f32, pitch as f32, yaw as f32, q[3] as f32);
    }
}

impl Index<usize> for Quaternion {
    type Output = f64;

    fn index(&self, ind: usize) -> &f64 {
        return &self.q[ind];
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Quaternion({}, {}, {}, {})", self.q[0], self.q[1], self.q[2], self.q[3])
    }
}

pub struct Madgwick {
    pub q: Quaternion,
    beta: f64,
}

impl Madgwick {
    pub fn new(beta: f64) -> Madgwick {
        Madgwick {q: Quaternion::new(1.0, 0.0, 0.0, 0.0), beta}
    }
    pub fn update(&mut self, gyroscope: &[f64], accelerometer: &[f64], magnetometer: &[f64], delta_t: f64) -> () {
        let mut q = Quaternion::q_new(self.q.q);
//        let gyroscope = Vector3::new(gyroscope[0], gyroscope[1], gyroscope[2]);
//        let mut accelerometer = Vector3::new(accelerometer[0], accelerometer[1], accelerometer[2]);
//        let mut magnetometer = Vector3::new(magnetometer[0], magnetometer[1], magnetometer[2]);
        let gyroscope = Vector3::new(gyroscope[2], gyroscope[1], gyroscope[0]);
        let mut accelerometer = Vector3::new(accelerometer[2], accelerometer[1], accelerometer[0]);
        let mut magnetometer = Vector3::new(magnetometer[2], magnetometer[1], magnetometer[0]);
        accelerometer /= accelerometer.norm();
        magnetometer /= magnetometer.norm();
        let h = q.mul(Quaternion::new(0.0, magnetometer[0], magnetometer[1], magnetometer[2])).mul(q.conj());
        let b = Vector4::new(0.0, Vector2::new(h.q[1], h.q[2]).norm(), 0.0, h.q[3]);
        let f = Vector6::new(
            2.0*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2.0*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2.0*(0.5 - q[1].powi(2) - q[2].powi(2)) - accelerometer[2],
            2.0*b[1]*(0.5 - q[2].powi(2) - q[3].powi(2)) + 2.0*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2.0*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2.0*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2.0*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2.0*b[3]*(0.5 - q[1].powi(2) - q[2].powi(2)) - magnetometer[2]
        );
        let j = Matrix6x4::new(
            -2.0*q[2],                  2.0*q[3],                  -2.0*q[0],                  2.0*q[1],
            2.0*q[1],                   2.0*q[0],                  2.0*q[3],                   2.0*q[2],
            0.0,                        -4.0*q[1],                 -4.0*q[2],                  0.0,
            -2.0*b[3]*q[2],             2.0*b[3]*q[3],             -4.0*b[1]*q[2]-2.0*b[3]*q[0], -4.0*b[1]*q[3]+2.0*b[3]*q[1],
            -2.0*b[1]*q[3]+2.0*b[3]*q[1], 2.0*b[1]*q[2]+2.0*b[3]*q[0], 2.*b[1]*q[1]+2.0*b[3]*q[3],  -2.0*b[1]*q[0]+2.0*b[3]*q[2],
            2.0*b[1]*q[2],              2.0*b[1]*q[3]-4.0*b[3]*q[1], 2.0*b[1]*q[0]-4.0*b[3]*q[2],  2.0*b[1]*q[1]
            );
//        let f = Vector3::new(
//            2.0 * (q[1] * q[3] - q[0] * q[2]) - accelerometer[0],
//            2.0*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
//            2.0*(0.5 - q[1].powi(2) - q[2].powi(2)) - accelerometer[2],
//        );
//        let j = Matrix3x4::new(
//            -2.0*q[2],                  2.0*q[3],                  -2.0*q[0],                  2.0*q[1],
//            2.0*q[1],                   2.0*q[0],                  2.0*q[3],                   2.0*q[2],
//            0.0,                        -4.0*q[1],                 -4.0*q[2],                  0.0
//        );
        let mut step = j.transpose() * f;
        step /= step.norm();
        let q_dot = (q.mul(Quaternion::new(0.0, gyroscope[0], gyroscope[1], gyroscope[2]))).scalar_mul(0.5).add(
            Quaternion::q_new(step * (-self.beta))
        ).scalar_mul(delta_t);

        q = q.add(q_dot);
        self.q = Quaternion::q_new(q.q / q.q.norm());

    }
}
