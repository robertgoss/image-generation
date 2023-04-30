// Basic raytracable geometry
//
// Will be raytraced with in a entity with a coordinate system and material
//
// This just concerns itselfwith the distance along the ray we make contact

use std::collections::HashMap;

use cgmath::{prelude::*, point3};
use cgmath::{point2, vec2, vec3, dot, Point3, Vector3, Matrix4, EuclideanSpace};
use json::JsonValue;

use crate::ray::geometry2::{Ray2, Circle, OriginSquare, TraceGeometry2};

// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

// Ray we can trace along
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

    fn xy(&self) -> Ray2 {
        Ray2 {
            start : point2(self.start.x, self.start.y),
            direction : vec2(self.direction.x, self.direction.y)
        }
    }
}

struct Prism<Base> {
    base : Base,
    half_length : f64
}

struct Cylinder {
    prism : Prism<Circle>
}

// Axis aligned box at the origin with given half dimensions
struct OriginBox {
    prism : Prism<OriginSquare>
}

// A axis aligned box to bound geomenty
#[derive(Clone)]
pub struct AABox {
    pub min : Point3<f64>,
    pub max : Point3<f64>
}

// Sphere at the origin of a given radius
struct Sphere {
    radius : f64
}
  
// Z plane at the origin
struct Plane;


pub trait TraceGeometry {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)>;

    fn bound(&self, mat : &Matrix4<f64>) -> Option<AABox>; 
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

    fn bound(&self, mat : &Matrix4<f64>) -> Option<AABox> {
        let origin = mat.transform_point(point3(0.0, 0.0, 0.0));
        let diff = vec3(self.radius, self.radius, self.radius);
        Some(AABox { min: origin - diff, max: origin + diff })
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

    fn bound(&self, _ : &Matrix4<f64>) -> Option<AABox> {
        None
    }
}

impl TraceGeometry for OriginBox {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)> {
        self.prism.trace(ray)
    }

    fn bound(&self, mat : &Matrix4<f64>) -> Option<AABox> {
        AABox::from_points(self.vertices().into_iter().map(
            |pt| mat.transform_point(pt)
        ))
    }
}

impl TraceGeometry for Cylinder {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)> {
        self.prism.trace(ray)
    }

    fn bound(&self, mat : &Matrix4<f64>) -> Option<AABox> {
        let half_length = self.prism.half_length;
        let radius = self.prism.base.radius;
        let top_centre = mat.transform_point(point3(0.0,0.0,half_length));
        let base_centre = mat.transform_point(point3(0.0,0.0,-half_length));
        let diff = vec3(radius, radius, radius);
        let top_box = AABox {
            min : top_centre - diff,
            max : top_centre + diff
        };
        let base_box = AABox {
            min : base_centre - diff,
            max : base_centre + diff
        };
        Some(top_box.merge(&base_box))
    }
}

impl<Base> Prism<Base> 
  where Base : TraceGeometry2 {
    fn trace(&self, ray : &Ray) -> Option<(f64, UnitVector3)> {
        if ray.start.z > self.half_length {
            if ray.direction.z >= 0.0 {
                return None;
            }
            // Get intersection with top plane 
            let t_contact_top = (self.half_length - ray.start.z) / ray.direction.z;
            let pos_top = ray.at(t_contact_top);
            if self.base.inside(&point2(pos_top.x, pos_top.y)) {
                return Some((t_contact_top, vec3(0.0, 0.0, 1.0)));
            }
        } 
        if ray.start.z < -self.half_length {
            if ray.direction.z <= 0.0 {
                return None;
            }
            // Get intersection with top plane 
            let t_contact_bot = (-self.half_length - ray.start.z) / ray.direction.z;
            let pos_bot = ray.at(t_contact_bot);
            if self.base.inside(&point2(pos_bot.x, pos_bot.y)) {
                return Some((t_contact_bot, vec3(0.0, 0.0, -1.0)));
            }
        } 
        // If we can do a 2D check aginst the side
        let mut ray2 = ray.xy();
        let ray_length_sq = ray2.direction.magnitude2();
        if ray_length_sq < 1e-9 {
            return None;
        }
        // Fiddle the length
        let ray_length = ray_length_sq.sqrt();
        ray2.direction = ray2.direction / ray_length;
        if let Some((t_contact_2d, norm)) = self.base.trace(&ray2) {
            // Check the z pos of hit
            let t_contact = t_contact_2d / ray_length;
            let z = ray.at(t_contact).z;
            if z.abs() < self.half_length {
                Some((t_contact, vec3(norm.x, norm.y, 0.0)))
            } else {
                None
            }
        } else {
            None
        }
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
                OriginBox::new(&vec3(x,y,z))
            )
        )
    }

    fn new(half_lengths : &Vector3<f64>) -> OriginBox {
        OriginBox {
            prism : Prism {
                half_length : half_lengths.z / 2.0,
                base : OriginSquare {
                    x_half : half_lengths.x / 2.0,
                    y_half : half_lengths.y / 2.0
                }
            }
        }
    }

    fn vertices(&self) -> [Point3<f64>; 8] {
        let x = self.prism.base.x_half;
        let y = self.prism.base.y_half;
        let z = self.prism.half_length;
        [
            point3(x,y,z),
            point3(x,y,-z),
            point3(x,-y,z),
            point3(x,-y,-z),
            point3(-x,y,z),
            point3(-x,y,-z),
            point3(-x,-y,z),
            point3(-x,-y,-z)
        ]
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

impl Cylinder {
    fn from_json(input : &JsonValue) -> Option<Box<dyn TraceGeometry>> {
        let r = input["radius"].as_f64()?;
        let l = input["length"].as_f64()?;
        Some(
            Box::new(
                Cylinder::new(r, l / 2.0)
            )
        )
    }

    fn new(radius : f64, half_length : f64) -> Cylinder {
        Cylinder {
            prism : Prism { 
                base: Circle { radius: radius }, 
                half_length: half_length
            }
        }
    }
}

fn min_d(a : f64, b: f64) -> f64 {
    if a < b {a} else {b}
}

fn max_d(a : f64, b: f64) -> f64 {
    if a < b {b} else {a}
}

impl AABox {
    pub fn from_points<I>(mut iter : I) -> Option<AABox>
        where I : Iterator<Item = Point3<f64>>
    {
        if let Some(first) = iter.next() {
            let base = AABox{ min: first, max : first};
            Some (
                iter.fold(base, |aabox, point| aabox.merge_pt(point))
            )
        } else {
            None
        }
    }

    fn merge_pt(&self, pt : Point3<f64>) -> AABox {
        AABox { 
            min: point3(
                min_d(self.min.x, pt.x), 
                min_d(self.min.y, pt.y), 
                min_d(self.min.z, pt.z)
            ), 
            max: point3(
                max_d(self.max.x, pt.x), 
                max_d(self.max.y, pt.y), 
                max_d(self.max.z, pt.z)
            ), 
        }
    }

    pub fn merge(&self, other : &AABox) -> AABox {
        AABox { 
            min: point3(
                min_d(self.min.x, other.min.x), 
                min_d(self.min.y, other.min.y), 
                min_d(self.min.z, other.min.z)
            ), 
            max: point3(
                max_d(self.max.x, other.max.x), 
                max_d(self.max.y, other.max.y), 
                max_d(self.max.z, other.max.z)
            ), 
        }
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
        Some("cylinder") => Cylinder::from_json(input),
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
        let cube = OriginBox::new(&vec3(1.0, 1.0, 1.0));
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray {start : point3(-10.0,0.0,0.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, 9.5);
        assert_abs_diff_eq!(trace.unwrap().1, vec3(-1.0, 0.0, 0.0));
    }

    #[test]
    fn test_trace_box_slanted_down_hit() {
        let cube = OriginBox::new(&vec3(1.0, 1.0, 1.0));
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
        let cube = OriginBox::new(&vec3(1.0, 1.0, 1.0));
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray {start : point3(-10.0,4.0,0.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(trace.is_none());
    }
    #[test]
    fn test_trace_box_axis_inside() {
        let cube = OriginBox::new(&vec3(1.0, 1.0, 1.0));
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray {start : point3(0.0,0.0,0.0), direction : vec3(1.0,0.0,0.0)}
        );
        // Miss
        assert!(trace.is_none());
    }

    #[test]
    fn test_trace_cylinder_under_hit() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(0.0,0.0,-1.5), direction : vec3(0.0,0.0,1.0)}
        );
        // Hit
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, 1.0);
        assert_abs_diff_eq!(trace.unwrap().1, vec3(0.0, 0.0, -1.0));
    }

    #[test]
    fn test_trace_cylinder_under_miss_dir() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(0.0,0.0,-1.0), direction : vec3(0.0,0.0,-1.0)}
        );
        // Miss
        assert!(trace.is_none());
    }

    #[test]
    fn test_trace_cylinder_under_miss_pos() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(2.0,0.0,-1.0), direction : vec3(0.0,0.0,1.0)}
        );
        // Miss
        assert!(trace.is_none());
    }

    #[test]
    fn test_trace_cylinder_over_hit() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(0.0,0.0,1.5), direction : vec3(0.0,0.0,-1.0)}
        );
        // Hit
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, 1.0);
        assert_abs_diff_eq!(trace.unwrap().1, vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_trace_cylinder_over_miss_dir() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(0.0,0.0,2.0), direction : vec3(0.0,0.0,1.0)}
        );
        // Miss
        assert!(trace.is_none());
    }

    #[test]
    fn test_trace_cylinder_over_miss_pos() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(2.0,0.0,2.0), direction : vec3(0.0,0.0,-1.0)}
        );
        // Miss
        assert!(trace.is_none());
    }

    #[test]
    fn test_trace_cylinder_horizontal_hit() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(2.0,0.0,0.0), direction : vec3(-1.0,0.0,0.0)}
        );
        // Hit
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, 1.5);
        assert_abs_diff_eq!(trace.unwrap().1, vec3(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_trace_cylinder_horizontal_miss() {
        let cylinder = Cylinder::new(0.5, 0.5);
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(2.0,0.0,1.1), direction : vec3(-1.0,0.0,0.0)}
        );
        // Hit
        assert!(trace.is_none());
    }

    #[test]
    fn test_trace_cylinder_slant_hit() {
        let cylinder = Cylinder::new(0.5, 0.5);
        let l :f64 = 1.0 / 2.0_f64.sqrt();
        // Fire directy up ( under)
        let trace = cylinder.trace(&
            Ray {start : point3(-1.0,0.0,0.25), direction : vec3(l,0.0,-l)}
        );
        // Hit
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, l);
        assert_abs_diff_eq!(trace.unwrap().1, vec3(-1.0, 0.0, 0.0));
    }

    #[test]
    fn test_trace_cylinder_slant_miss() {
        let cylinder = Cylinder::new(0.5, 0.5);
        let l :f64 = 1.0 / 2.0_f64.sqrt();
        // Aim over the top but from the middle
        let trace = cylinder.trace(&
            Ray {start : point3(-10.0,0.0,0.25), direction : vec3(l,0.0,-l)}
        );
        // Hit
        assert!(trace.is_none());
    }


}