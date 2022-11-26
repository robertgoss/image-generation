// Make a raytraced image
//
// Current quadratic and only supports spheres and planes

use cgmath::prelude::*;
use cgmath::{point3, vec3, dot, Point3, Vector3};

use json::JsonValue;
use image::{Rgb, RgbImage};
use std::io::{Error, ErrorKind};


// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

struct Sphere {
  centre : Point3<f64>,
  radius : f64,
  colour : Rgb<u8>
}

struct Plane {
    centre : Point3<f64>,
    normal : UnitVector3,
    colour : Rgb<u8>
}

struct Ray {
    start : Point3<f64>,
    direction : UnitVector3
}

#[allow(dead_code)]
struct Contact {
    pos : Point3<f64>,
    normal : UnitVector3,
    distance : f64,
    colour : Rgb<u8>
}

struct Camera {
    fov : f64,
    pos  : Point3<f64>
}

trait Entity {
    fn intersect(self : &Self, ray : &Ray) -> Option<Contact>;
}

struct Scene {
    camera : Camera,
    spheres : Vec<Box<dyn Entity>>,
    background : Rgb<u8>,
    resolution : (usize, usize)
}

impl Sphere {
    fn from_json(input : &JsonValue) -> Option<Box<dyn Entity>> {
        let x = input["x"].as_f64()?;
        let y = input["y"].as_f64()?;
        let z = input["z"].as_f64()?;
        let r = input["r"].as_f64()?;
        Some(
            Box::new(
                Sphere{
                    centre : point3(x, y, z),
                    radius : r,
                    colour : Rgb([200, 50, 50])
                }
            )
        )
    }
}

impl Entity for Sphere {
    fn intersect(self : &Self, ray : &Ray) -> Option<Contact> {
        // Basic ray intersection
        let diff = self.centre - ray.start;
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
        let contact_pos = ray.start + t_contact * ray.direction;
        let contact_normal = (contact_pos - self.centre) / self.radius;
        Some(
            Contact { 
                pos: contact_pos, 
                normal : contact_normal,
                distance : t_contact,
                colour : self.colour
            }
        )
    } 
}

impl Plane {
    fn from_json(input : &JsonValue) -> Option<Box<dyn Entity>> {
        let z = input["z"].as_f64()?;
        Some(
            Box::new(
                Plane{
                    centre : point3(0.0, 0.0, z),
                    normal : vec3(0.0, 0.0, 1.0),
                    colour : Rgb([50, 50, 200])
                }
            )
        )
    }
}

impl Entity for Plane {
    fn intersect(self : &Self, ray : &Ray) -> Option<Contact> {
        // Start below
        if ray.start.z < self.centre.z {
            return None
        }
        // Not going to hit
        if ray.direction.z >= 0.0 {
            return None
        }
        // Find contact
        let t_contact = (self.centre.z - ray.start.z) / ray.direction.z;
        let contact_pos = ray.start + t_contact * ray.direction;
        Some(
            Contact { 
                pos: contact_pos, 
                normal : self.normal,
                distance : t_contact,
                colour : self.colour
            }
        )
    } 
}

impl Camera {
    fn ray(&self, x : f64, y : f64) -> Ray {
        let diff = vec3(self.fov*(x - 0.5), 1.0, self.fov*(y - 0.5));
        Ray {
            start : self.pos,
            direction : diff / diff.magnitude()
        }
    }
}

fn entity_from_json(input : &JsonValue) -> Option<Box<dyn Entity>> {
    let entity_type = input["type"].as_str();
    match entity_type {
        Some("sphere") => Sphere::from_json(input),
        Some("plane") => Plane::from_json(input),
        _ => None
    }
}

impl Scene {
    fn trace_ray(self : &Self, ray : &Ray, _depth : u8) -> Rgb<u8> {
        self.find_best_contact(ray).map(
            |contact| contact.colour
        ).unwrap_or(self.background)
    }

    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact> {
        self.spheres.iter().filter_map(
            |entity| entity.intersect(ray)
        ).min_by(
            |contact1, contact2| contact1.distance.partial_cmp(&contact2.distance).unwrap()
        )
    }

    fn from_json(input : &JsonValue) -> std::io::Result<Scene> {
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let entities : Vec<Box<dyn Entity>> = input["entities"].members().filter_map(
            |value| entity_from_json(value) 
        ).collect();
        println!("Scene has {} entities", entities.len());
        Ok(Scene {
            camera : Camera{fov : 1.0,  pos : point3(0.0,0.0,1.0)},
            resolution : (res_x, res_y),
            background : Rgb([200,200,200]),
            spheres : entities
        })
    }


    fn make_image(&self) -> RgbImage {
        let mut img = RgbImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        for i in 0..self.resolution.0 {
            for j in 0..self.resolution.1 {
                let x = 1.0 - (i as f64 / self.resolution.0 as f64);
                let y = 1.0 - (j as f64 / self.resolution.1 as f64);
                let ray = self.camera.ray(x, y);
                let colour = self.trace_ray(&ray, 3);
                img.put_pixel(i as u32, j as u32, colour);
            }
        }

        img
    }
}

pub fn generate(input : &JsonValue) -> std::io::Result<()> {
    println!("Generating ray trace scene");
    let scene = Scene::from_json(input)?;
    let image = scene.make_image();
    image.save("output.png").map_err(
        |_| Error::new(ErrorKind::InvalidData, "Couldn't write image")
    )
}