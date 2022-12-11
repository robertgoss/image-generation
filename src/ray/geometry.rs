// Basic raytracable geometry
//
// Will be raytraced with in a entity with a coordinate system and material
//
// This just concerns itselfwith the distance along the ray we make contact

use std::collections::HashMap;

use cgmath::prelude::*;
use cgmath::{vec3, dot, Point3, Vector3, Matrix4};
use json::JsonValue;

// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

// An expanded possilby open set of ranges
// Ray we can trace along
#[derive(Debug)]
pub struct Ray {
    pub start : Point3<f64>,
    pub direction : UnitVector3
}

impl Ray {
    pub fn at(&self, t : f64) -> Point3<f64> {
        self.start + (self.direction * t)
    }

    pub fn transform(&self, transform : &Matrix4<f64>) -> Ray {
        Ray {
            start : transform.transform_point(self.start),
            direction : transform.transform_vector(self.direction),
        }
    }
}

// Axis aligned box at the origin with given half dimensions
struct OriginBox {
    half_dimensions : Vector3<f64>
}

// Sphere at the origin of a given radius
struct Sphere {
    radius : f64
}
  
// Z plane at the origin
struct Plane;


pub trait TraceGeometry {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)>;
}
  
impl TraceGeometry for Sphere {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)> {
        // Basic ray intersection
        let diff = -1.0*ray.start.to_vec();
        // Get nearest point on ray to centre
        let t_nearest = dot(ray.direction, diff);
        // Hit behind
        if t_nearest < 0.0 {
            return None;
        }
        let projection_nearest = diff - (ray.direction * t_nearest);
        let dist_nearest_sq = projection_nearest.magnitude2();
        // Ray does not contact sphere
        let radius_sq = self.radius * self.radius;
        if dist_nearest_sq > radius_sq {
            return None
        }
        // Amount before nearest point on ray that we contact
        let t_before = (radius_sq - dist_nearest_sq).sqrt();
        let t_contact = t_nearest - t_before;
        // Inside the sphere
        if t_contact < 0.0 {
            return None;
        }
        let pos = ray.start + ray.direction *t_contact;
        Some(
            (t_contact, pos.to_vec().normalize())
        )
    }
}

impl TraceGeometry for Plane {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)> {
        if ray.start.z <= 0.0 || ray.direction.z >=0.0 {
            return None
        }  
        let t_contact = -ray.start.z / ray.direction.z;
        Some(
            (t_contact, vec3(0.0,0.0,1.0))
        )
    }
}

fn sign(x: f64) -> f64 {
    if x > 0.0 { 
        1.0
    } else {
        -1.0
    }
}

enum Axis {
    X,
    Y,
    Z
}

impl TraceGeometry for OriginBox {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)> {
        let mut range = None;
        let mut axis = None;
        if let Some(t_contact_x) = self.solve(ray, Axis::X) {
            axis = Some(Axis::X);
            range = Some(t_contact_x)
        } else {
            if ray.start.x.abs() > self.half_dimensions.x {
                return None;
            } 
        }
        if let Some((min_y, max_y)) = self.solve(ray, Axis::Y) {
            if let Some((min_r, max_r)) = range.as_mut() {
                if *min_r < min_y { 
                    *min_r = min_y;
                    axis = Some(Axis::Y);
                }
                if *max_r > max_y {
                    *max_r = max_y;
                }
            } else {
                axis = Some(Axis::Y);
                range = Some((min_y, max_y))
            }
        } else {
            if ray.start.y.abs() > self.half_dimensions.y {
                return None;
            } 
        }
        if let Some((min_z, max_z)) = self.solve(ray, Axis::Z) {
            if let Some((min_r, max_r)) = range.as_mut() {
                if *min_r < min_z { 
                    *min_r = min_z;
                    axis = Some(Axis::Z);
                }
                if *max_r > max_z {
                    *max_r = max_z;
                }
            } else {
                axis = Some(Axis::Z);
                range = Some((min_z, max_z))
            }
        } else {
            if ray.start.z.abs() > self.half_dimensions.z {
                return None;
            } 
        }
        if let Some((min_r, max_r)) = range {
            if min_r < 0.0 || min_r > max_r {
                return None;
            }
            return match axis {
                Some(Axis::X) => Some((min_r, vec3(-sign(ray.direction.x), 0.0, 0.0))),
                Some(Axis::Y) => Some((min_r, vec3(0.0, -sign(ray.direction.y), 0.0))),
                Some(Axis::Z) => Some((min_r, vec3(0.0, 0.0, -sign(ray.direction.z)))),
                None => None
            }
        } ;
        return None;
    }
}

impl Sphere {
    fn from_json(input : &JsonValue) -> Option<Box<dyn TraceGeometry>> {
        let r = input["radius"].as_f64()?;
        Some(
            Box::new(
                Sphere {
                    radius : r
                }
            )
        )
    }
}

impl OriginBox {
    fn from_json(input : &JsonValue) -> Option<Box<dyn TraceGeometry>> {
        let x = input["x"].as_f64()?;
        let y = input["y"].as_f64()?;
        let z = input["z"].as_f64()?;
        Some(
            Box::new(
                OriginBox {
                    half_dimensions : vec3(x/2.0, y/2.0, z/2.0)
                }
            )
        )
    }

    fn solve(&self, ray : &Ray, axis : Axis) -> Option<(f64,f64)> {
        let (dir, dim, start) = match axis {
            Axis::X => (ray.direction.x, self.half_dimensions.x, ray.start.x),
            Axis::Y => (ray.direction.y, self.half_dimensions.y, ray.start.y),
            Axis::Z => (ray.direction.z, self.half_dimensions.z, ray.start.z)
        };
        if dir != 0.0 {
            let sign_dim = sign(dir) * dim;
            let diff = (-sign_dim) - start;
            let t_contact = diff/dir;
            let extra = 2.0 * sign_dim / dir;
            Some((t_contact, t_contact + extra))
        } else {
            None
        }
    }
}

impl Plane {
    fn from_json() -> Option<Box<dyn TraceGeometry>> {
        Some(
            Box::new(
                Plane
            )
        )
    }
}

pub struct Geometries {
    names : HashMap<String, Box<dyn TraceGeometry>>,
}

fn geometry_from_json(input : &JsonValue) -> Option<Box<dyn TraceGeometry>> {
    match input["type"].as_str() {
        Some("sphere") => Sphere::from_json(input),
        Some("plane") => Plane::from_json(),
        Some("box") => OriginBox::from_json(input),
        _ => None
    }
}

impl Geometries {
    pub fn from_json(input : &JsonValue) -> Geometries {
        let geometries : HashMap<String, Box<dyn TraceGeometry>> = input["geometries"].entries().filter_map(
            |(name, value)| 
              geometry_from_json(value).map(|geometry| (name.to_string(), geometry))
        ).collect();
        println!("{} geometries loaded", geometries.len());
        Geometries {
            names : geometries
        }
    }

    pub fn lookup<'a>(self : &'a Self, name : &str) -> Option<&'a dyn TraceGeometry> {
        self.names.get(name).map(
            |boxed| boxed.as_ref()
        )
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{point3, assert_abs_diff_eq};

    use super::*;

    #[test]
    fn test_trace_plane_down() {
        let plane = Plane {};
        // Fire directy down
        let direct_down = plane.trace(&
            Ray {start : point3(0.0,0.0,1.0), direction : vec3(0.0,0.0,-1.0)}
        );
        assert!(direct_down.is_some());
        assert_abs_diff_eq!(direct_down.unwrap().0, 1.0);
        assert_abs_diff_eq!(direct_down.unwrap().1, vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_trace_plane_slanted() {
        let plane = Plane {};
        // Fire directy down
        let l :f64 = 1.0 / 2.0_f64.sqrt();
        let slanted = plane.trace(&
            Ray {start : point3(0.0,0.0,1.0), direction : vec3(0.0,l,-l)}
        );
        assert!(slanted.is_some());
        assert_abs_diff_eq!(slanted.unwrap().0, 2.0_f64.sqrt());
        assert_abs_diff_eq!(slanted.unwrap().1, vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_trace_plane_up() {
        let plane = Plane {};
        // Fire directy up
        let up = plane.trace(&
            Ray {start : point3(0.0,0.0,1.0), direction : vec3(0.0,0.0,1.0)}
        );
        // Miss
        assert!(up.is_none());
    }

    #[test]
    fn test_trace_plane_horizontal() {
        let plane = Plane {};
        // Fire directy up
        let hori = plane.trace(&
            Ray {start : point3(0.0,0.0,1.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(hori.is_none());
    }

    #[test]
    fn test_trace_plane_under() {
        let plane = Plane {};
        // Fire directy up ( under)
        let up = plane.trace(&
            Ray {start : point3(0.0,0.0,-1.0), direction : vec3(0.0,0.0,1.0)}
        );
        // Miss
        assert!(up.is_none());
    }

    #[test]
    fn test_trace_box_axis_hit() {
        let cube = OriginBox { half_dimensions : vec3(1.0, 1.0, 1.0)};
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray {start : point3(-10.0,0.0,0.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, 9.0);
        assert_abs_diff_eq!(trace.unwrap().1, vec3(-1.0, 0.0, 0.0));
    }

    #[test]
    fn test_trace_box_slanted_down_hit() {
        let cube = OriginBox { half_dimensions : vec3(1.0, 1.0, 1.0)};
        // Fire along axis and hit
        let dir = vec3(1.0, 0.0, -1e-3).normalize();
        let trace = cube.trace(&
            Ray {start : point3(-10.0,0.0,0.0), direction : dir}
        );
        // Miss
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().1, vec3(-1.0, 0.0, 0.0));
    }

    #[test]
    fn test_trace_box_axis_miss() {
        let cube = OriginBox { half_dimensions : vec3(1.0, 1.0, 1.0)};
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray {start : point3(-10.0,4.0,0.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(trace.is_none());
    }
    #[test]
    fn test_trace_box_axis_inside() {
        let cube = OriginBox { half_dimensions : vec3(1.0, 1.0, 1.0)};
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray {start : point3(0.0,0.0,0.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(trace.is_none());
    }
}