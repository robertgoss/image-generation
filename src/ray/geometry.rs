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