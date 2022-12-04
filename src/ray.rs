// Make a raytraced image
//
// Current quadratic and only supports spheres and planes

use cgmath::{prelude::*};
use cgmath::{point3, vec3, Point3, Vector3, Matrix4, Deg};

use json::JsonValue;
use image::{Rgb, RgbImage};
use std::io::{Error, ErrorKind};

mod geometry;
mod materials;

use geometry::{TraceGeometry, Ray, Geometries};
use materials::{Material, Materials};

// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

struct Entity<'geom, 'mat> {
    geometry : &'geom dyn TraceGeometry,
    coords : Matrix4<f64>,
    inv_coords : Matrix4<f64>,
    material : &'mat Material
}

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

struct Scene<'mat, 'geom> {
    camera : Camera,
    spheres : Vec<Entity<'mat, 'geom>>,
    background : Rgb<u8>,
    resolution : (usize, usize),
    max_depth : u8
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


impl<'g, 'm> Entity<'g, 'm> {
    fn from_json<'geom, 'mat>(
        input : &JsonValue, 
        geometries : &'geom Geometries,
        materials : &'mat Materials
    ) -> Option<Entity<'geom, 'mat>> {
        let material = materials.lookup(input["mat"].as_str().unwrap_or("none"));
        let geometry = geometries.lookup(input["geom"].as_str().unwrap_or("none"))?;
        let x = input["x"].as_f64()?;
        let y = input["y"].as_f64()?;
        let z = input["z"].as_f64()?;
        let mut transform = Matrix4::from_translation(vec3(x,y,z));
        if let Some(roll) = input["roll"].as_f64() {
            let roll_matrix = Matrix4::from_angle_z(Deg::<f64>(roll));
            transform = transform * roll_matrix;
        }
        let inv = transform.invert().unwrap();
        Some(Entity { geometry: geometry, coords: transform, inv_coords: inv, material: material }) 
    }

    fn trace(&self, ray : &Ray) -> Option<Contact<'m>> {
        let transformed_ray = ray.transform(&self.inv_coords);
        let (t_contact, transformed_norm) = self.geometry.trace(&transformed_ray)?;
        let normal = self.coords.transform_vector(transformed_norm);
        Some(Contact { 
            pos: ray.at(t_contact), 
            normal: normal, 
            distance: t_contact, 
            material: &self.material
        })
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

impl<'geom, 'mat> Scene<'geom, 'mat> {
    fn trace_ray(self : &Self, ray : &Ray, depth : u8) -> Rgb<u8> {
        self.find_best_contact(ray).map(
            |contact| self.trace_contact(ray, &contact, depth)
        ).unwrap_or(self.background)
    }

    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact> {
        self.spheres.iter().filter_map(
            |entity| entity.trace(ray)
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

    fn from_json(input : &JsonValue, geometry : &'geom Geometries, materials : &'mat Materials) -> std::io::Result<Scene<'geom, 'mat>> {
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let max_depth = input["max_depth"].as_u8().unwrap_or(4);
        let entities : Vec<Entity> = input["entities"].members().filter_map(
            |value| Entity::from_json(value, geometry, materials) 
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
    let geometries = Geometries::from_json(input);
    let scene = Scene::from_json(input, &geometries, &materials)?;
    let image = scene.make_image();
    image.save("output.png").map_err(
        |_| Error::new(ErrorKind::InvalidData, "Couldn't write image")
    )
}