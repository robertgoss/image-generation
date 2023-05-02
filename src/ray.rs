// Make a raytraced image
//
// Current quadratic and only supports spheres and planes

use std::io::{Error, ErrorKind};

use cgmath::{prelude::*, Matrix3};
use cgmath::{point3, vec3, Point3};

use json::JsonValue;
use image::{Rgb, RgbImage};

use rand::{Rng, SeedableRng};

mod geometry;
mod geometry2;
mod materials;
mod trace;

use geometry::{Ray, Geometries};
use materials::{Materials, rgb_lerp, rgb_scale};
use crate::ray::materials::rgb_sum;
use crate::ray::trace::{Contact, Entity, BSPTreeScene, BoxedTraceSpace, LinearScene, BSPTree};

// Helper to remember which we want to always be pre-unitized
type Colour = Rgb<f64>;

struct Camera {
    fov : f64,
    pos  : Point3<f64>,
    mat : Matrix3<f64>,
    automatic : bool, // Fit the whole scene in camera
}

struct RenderScene<'a,Scene>  where Scene : BoxedTraceSpace<'a> {
    camera : Camera,
    materials : &'a Materials,
    scene : Scene,
    resolution : (usize, usize),
    antialiasing_samples : usize,
    max_depth : u8
}

impl Camera {
    fn ray(&self, x : f64, y : f64) -> Ray {
        let base_ray = vec3(self.fov*(x - 0.5), self.fov*(y - 0.5), 1.0);
        let diff = self.mat * base_ray;
        Ray {
            start : self.pos,
            direction : diff / diff.magnitude()
        }
    }

    fn from_json(input : &JsonValue) -> Option<Camera> {
        // Set the generic
        let x = input["x"].as_f64()?;
        let y = input["y"].as_f64()?;
        let z = input["z"].as_f64()?;
        let pos = point3(x, y, z);
        let fov = input["fov"].as_f64().unwrap_or(1.0);
        // Get the direction in different ways
        let dir_vec = if input.has_key("dir_x") {
            let dir_x = input["dir_x"].as_f64()?;
            let dir_y = input["dir_y"].as_f64()?;
            let dir_z = input["dir_z"].as_f64()?;
            vec3(dir_x, dir_y, dir_z).normalize()
        } else {
            let target_x = input["target_x"].as_f64()?;
            let target_y = input["target_y"].as_f64()?;
            let target_z = input["target_z"].as_f64()?;
            let target = point3(target_x, target_y, target_z);
            (target - pos).normalize()
        };
        // Want camera with z pointing down dir, y points (as close as possible to z)
        // and x whatever is left.
        let mat = if dir_vec.z.abs() > 1.0 - 1.0e-8 {
            // we are pointing directly up align other with y
            let x = dir_vec.cross(vec3(0.0, 1.0, 0.0));
            let y = x.cross(dir_vec);
            Matrix3{ x : x, y : y, z : dir_vec}
        } else {
            // we are pointing directly up align other with y
            let x = dir_vec.cross(vec3(0.0, 0.0, 1.0));
            let y = x.cross(dir_vec);
            Matrix3{ x : x, y : y, z : dir_vec}
        };
        let automatic = input["automatic"].as_bool().unwrap_or(false);
        Some(Camera {
            pos : pos,
            mat : mat,
            fov : fov,
            automatic : automatic
        })
    }
}

impl<'a, Scene> RenderScene<'a, Scene> where Scene : BoxedTraceSpace<'a> {

    fn background(self : &Self, ray : &Ray) -> Colour {
        let white = Rgb([1.0, 1.0, 1.0]);
        let blue = Rgb([0.5, 0.6, 1.0]);
        rgb_lerp(0.5 * (ray.direction.z + 1.0), &blue, &white)
    }

    fn trace_ray<Rng>(self : &Self, rng : &mut Rng, ray : &Ray, depth : u8) -> Colour
      where Rng : rand::Rng
    {
        self.scene.find_best_contact(ray).map(
            |contact| self.trace_contact(rng, ray, &contact, depth)
        ).unwrap_or(self.background(ray))
    }

    fn trace_contact<Rng>(self : &Self, rng: &mut Rng, ray : &Ray, contact : &Contact, depth : u8) -> Colour
      where Rng : rand::Rng
    {
        if depth == 0 {
            return Rgb([0.0, 0.0, 0.0]);
        }
        let (scatter, attenuation) = contact.material.colour(self.materials,&contact.local_pos);
        let scatter_ray_opt = contact.scatter_ray(&scatter, rng, ray);
        if let Some(scatter_ray) = scatter_ray_opt {
            let scatter_colour = self.trace_ray(rng, &scatter_ray, depth - 1);
            rgb_scale(&scatter_colour, &attenuation)
        } else {
            attenuation
        }
    }

    fn from_json<'mat, 'ent>(
        input : &JsonValue,
        materials : &'mat Materials,
        entities : &'ent Vec<Entity<'a>>
    ) -> std::io::Result<RenderScene<'a, Scene>>
    where 'mat : 'a, 'ent : 'a
    {
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let max_depth = input["max_depth"].as_u8().unwrap_or(4);
        let antialiasing_samples = input["antialiasing_samples"].as_usize().unwrap_or(100);
        let camera = Camera::from_json(&input["camera"]).ok_or(
            Error::new(ErrorKind::InvalidData, "Missing camera")
        )?;
        let scene = Scene::from_iter(
            entities.iter().map(|entity| (entity.bound(), entity))
        );
        println!("Scene has {} entities", entities.len());
        Ok(RenderScene {
            camera,
            resolution : (res_x, res_y),
            scene,
            antialiasing_samples,
            max_depth,
            materials,
        })
    }

    fn make_image(&self) -> RgbImage {
        let mut img = RgbImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        let x_res = self.resolution.0 as f64;
        let y_res = self.resolution.1 as f64;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        for i in 0..self.resolution.0 {
            println!("Line: {}", i);
            for j in 0..self.resolution.0 {
                let mut accumulated_colour = Rgb([0.0, 0.0, 0.0]);
                for _ in 0..self.antialiasing_samples {
                    let ai = i as f64 + rng.gen::<f64>() - 0.5;
                    let aj = j as f64 + rng.gen::<f64>() - 0.5;
                    let x = 1.0 - (ai / x_res);
                    let y = 1.0 - (aj / y_res);
                    let ray = self.camera.ray(x, y);
                    let colour = self.trace_ray(&mut rng, &ray, self.max_depth);
                    rgb_sum(&mut accumulated_colour, &colour);
                }
                let sample_scale = 1.0 / self.antialiasing_samples as f64;
                let r = (accumulated_colour.0[0] * 255.0 * sample_scale) as u8;
                let g = (accumulated_colour.0[1] * 255.0 * sample_scale) as u8;
                let b = (accumulated_colour.0[2] * 255.0 * sample_scale) as u8;
                img.put_pixel(i as u32, j as u32,Rgb([r,g,b]));
            }
        }

        img
    }
}

pub fn generate(input : &JsonValue) -> std::io::Result<RgbImage> {
    println!("Generating ray trace scene");
    let materials = Materials::from_json(input);
    let geometries = Geometries::from_json(input);
    let entities = input["entities"].members().filter_map(
        |value| Entity::from_json(value, &geometries, &materials)
    ).collect();
    let scene: RenderScene<BSPTreeScene> = RenderScene::from_json(input, &materials, &entities)?;
    Ok(scene.make_image())
}