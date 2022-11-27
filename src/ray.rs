// Make a raytraced image
//
// Current quadratic and only supports spheres and planes

use cgmath::prelude::*;
use cgmath::{point3, vec3, dot, Point3, Vector3};

use json::JsonValue;
use image::{Rgb, RgbImage};
use std::collections::HashMap;
use std::io::{Error, ErrorKind};


// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

#[derive(Clone, Copy)]
struct Material {
    colour : Rgb<u8>,
    reflectivity : u8
}

struct Materials {
    names : HashMap<String, Material>,
    default : Material
}

struct Sphere {
  centre : Point3<f64>,
  radius : f64,
  material : Material
}

struct Plane {
    centre : Point3<f64>,
    normal : UnitVector3,
    material : Material
}

struct Ray {
    start : Point3<f64>,
    direction : UnitVector3
}

#[allow(dead_code)]
struct Contact<'a> {
    pos : Point3<f64>,
    normal : UnitVector3,
    distance : f64,
    material : &'a Material
}

struct Camera {
    fov : f64,
    pos  : Point3<f64>
}

trait Entity {
    fn intersect<'a>(self : &'a Self, ray : &Ray) -> Option<Contact<'a>>;
}

struct Scene {
    camera : Camera,
    spheres : Vec<Box<dyn Entity>>,
    background : Rgb<u8>,
    resolution : (usize, usize),
    max_depth : u8
}

impl Sphere {
    fn from_json(input : &JsonValue, material : &Material) -> Option<Box<dyn Entity>> {
        let x = input["x"].as_f64()?;
        let y = input["y"].as_f64()?;
        let z = input["z"].as_f64()?;
        let r = input["r"].as_f64()?;
        Some(
            Box::new(
                Sphere{
                    centre : point3(x, y, z),
                    radius : r,
                    material : *material
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
                material : &self.material
            }
        )
    } 
}

impl Plane {
    fn from_json(input : &JsonValue, material : &Material) -> Option<Box<dyn Entity>> {
        let z = input["z"].as_f64()?;
        Some(
            Box::new(
                Plane{
                    centre : point3(0.0, 0.0, z),
                    normal : vec3(0.0, 0.0, 1.0),
                    material : *material
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
                material : &self.material
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


impl Materials {
    fn from_json(input : &JsonValue) -> Materials {
        let materials : HashMap<String, Material> = input["materials"].entries().filter_map(
            |(name, value)| 
              Material::from_json(value).map(|material| (name.to_string(), material))
        ).collect();
        println!("{} materials loaded", materials.len());
        Materials {
            names : materials,
            default : Material { colour: Rgb([145,145,145]), reflectivity: 0 }
        }
    }

    fn lookup<'a>(self : &'a Self, name : &str) -> &'a Material {
        self.names.get(name).unwrap_or(&self.default)
    }
}

impl Material {
    fn from_json(input : &JsonValue) -> Option<Material> {
        let r = input["r"].as_u8()?;
        let g = input["g"].as_u8()?;
        let b = input["b"].as_u8()?;
        let refl = input["reflect"].as_f64()?;
        let refl_u8 = (refl * 255.0) as u8;
        Some(
            Material { colour: Rgb([r,g,b]), reflectivity: refl_u8 }
        )
    }
}


fn entity_from_json(input : &JsonValue, materials : &Materials) -> Option<Box<dyn Entity>> {
    let entity_type = input["type"].as_str();
    let material = materials.lookup(input["mat"].as_str().unwrap_or("none"));
    match entity_type {
        Some("sphere") => Sphere::from_json(input, material),
        Some("plane") => Plane::from_json(input, material),
        _ => None
    }
}

impl<'a> Contact<'a> {
    fn reflection_ray(self : &Self, ray_in : &Ray) -> Ray {
        let align = 2.0*ray_in.direction.dot(self.normal);
        let dir = ray_in.direction - align * self.normal;
        Ray{start: self.pos, direction : dir}
    }
}

fn merge_colour(c1 : Rgb<u8>, c2 : Rgb<u8>, par : u8 ) -> Rgb<u8> {
    let inv_par = 255 - par;
    let r1 = par as u16 * c1.0[0] as u16;
    let g1 = par as u16 * c1.0[1] as u16;
    let b1 = par as u16 * c1.0[2] as u16;
    let r2 = inv_par as u16 * c2.0[0] as u16;
    let g2 = inv_par as u16 * c2.0[1] as u16;
    let b2 = inv_par as u16 * c2.0[2] as u16;
    let r = (r1 + r2) / 255;
    let g = (g1 + g2) / 255;
    let b = (b1 + b2) / 255;
    Rgb([r as u8, g as u8, b as u8])
}

impl Scene {
    fn trace_ray(self : &Self, ray : &Ray, depth : u8) -> Rgb<u8> {
        self.find_best_contact(ray).map(
            |contact| self.trace_contact(ray, &contact, depth)
        ).unwrap_or(self.background)
    }

    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact> {
        self.spheres.iter().filter_map(
            |entity| entity.intersect(ray)
        ).min_by(
            |contact1, contact2| contact1.distance.partial_cmp(&contact2.distance).unwrap()
        )
    }

    fn trace_contact(self : &Self, ray : &Ray, contact : &Contact, depth : u8) -> Rgb<u8> {
        let relfection = if depth == 0 {
            Rgb([0,0,0])
        } else {
            self.trace_ray(&contact.reflection_ray(ray), depth - 1)
        };
        merge_colour(contact.material.colour, relfection, contact.material.reflectivity)
    }

    fn from_json(input : &JsonValue, materials : &Materials) -> std::io::Result<Scene> {
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let max_depth = input["max_depth"].as_u8().unwrap_or(4);
        let entities : Vec<Box<dyn Entity>> = input["entities"].members().filter_map(
            |value| entity_from_json(value, materials) 
        ).collect();
        println!("Scene has {} entities", entities.len());
        Ok(Scene {
            camera : Camera{fov : 1.0,  pos : point3(0.0,0.0,1.0)},
            resolution : (res_x, res_y),
            background : Rgb([200,200,200]),
            spheres : entities,
            max_depth : max_depth
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
                let colour = self.trace_ray(&ray, self.max_depth);
                img.put_pixel(i as u32, j as u32, colour);
            }
        }

        img
    }
}

pub fn generate(input : &JsonValue) -> std::io::Result<()> {
    println!("Generating ray trace scene");
    let materials = Materials::from_json(input);
    let scene = Scene::from_json(input, &materials)?;
    let image = scene.make_image();
    image.save("output.png").map_err(
        |_| Error::new(ErrorKind::InvalidData, "Couldn't write image")
    )
}