// Make a raytraced image
//
// Current quadratic and only supports spheres and planes

use std::f64::consts::PI;

use cgmath::{prelude::*};
use cgmath::{point3, vec3, Point3, Vector3, Matrix4, Deg};

use json::JsonValue;
use image::{Rgb, RgbImage};

mod geometry;
mod materials;

use geometry::{TraceGeometry, Ray, Geometries};
use materials::{Material, Materials};

// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

struct Entity<'a> {
    geometry : &'a dyn TraceGeometry,
    coords : Matrix4<f64>,
    inv_coords : Matrix4<f64>,
    material : &'a dyn Material
}

struct Contact<'a> {
    pos : Point3<f64>,
    local_pos : Point3<f64>,
    normal : UnitVector3,
    distance : f64,
    material : &'a dyn Material
}

struct Camera {
    fov : f64,
    pos  : Point3<f64>
}

struct Scene<'a> {
    camera : Camera,
    materials : &'a Materials,
    spheres : Vec<Entity<'a>>,
    background : Rgb<u8>,
    background_illumination : u8,
    use_ambiant_illumination : bool,
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


impl<'a> Entity<'a> {
    fn from_json(
        input : &JsonValue, 
        geometries : &'a Geometries,
        materials : &'a Materials
    ) -> Option<Entity<'a>> {
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

    fn trace(&self, ray : &Ray) -> Option<Contact<'a>> {
        let transformed_ray = ray.transform(&self.inv_coords);
        let (t_contact, transformed_norm) = self.geometry.trace(&transformed_ray)?;
        let normal = self.coords.transform_vector(transformed_norm);
        Some(Contact { 
            pos : ray.at(t_contact),
            local_pos: transformed_ray.at(t_contact), 
            normal: normal, 
            distance: t_contact, 
            material: self.material
        })
    }
}

impl<'a> Contact<'a> {
    fn reflection_ray(self : &Self, ray_in : &Ray) -> Ray {
        let align = 2.0*ray_in.direction.dot(self.normal);
        let dir = ray_in.direction - align * self.normal;
        Ray{start: self.pos, direction : dir}
    }

    fn light_rays5(self : &Self) -> [UnitVector3;5] {
        let inv_sqrt2 = 0.5_f64.sqrt();
        let u: UnitVector3 = if self.normal.x.abs() > 1.0-1e-6 {
            (vec3(1.0,0.0,0.0) - self.normal.x * self.normal).normalize()
        } else {
            (vec3(0.0,1.0,0.0) - self.normal.x * self.normal).normalize()
        };
        let v = self.normal.cross(u);
        [
            self.normal,
            inv_sqrt2 * (u + self.normal),
            inv_sqrt2 * (-u + self.normal),
            inv_sqrt2 * (v + self.normal),
            inv_sqrt2 * (-v + self.normal)
        ]
    }

    fn light_rays9(self : &Self) -> [UnitVector3;9] {
        let inv_sqrt2 = 0.5_f64.sqrt();
        let u: UnitVector3 = if self.normal.x.abs() > 1.0-1e-6 {
            (vec3(1.0,0.0,0.0) - self.normal.x * self.normal).normalize()
        } else {
            (vec3(0.0,1.0,0.0) - self.normal.x * self.normal).normalize()
        };
        let v = self.normal.cross(u);
        let lambda = 3.0 * inv_sqrt2 / 5.0;
        let n1 = inv_sqrt2 * self.normal;
        let n2 = 4.0/5.0 * self.normal;
        [
            self.normal,
            (inv_sqrt2 * u) + n1,
            (inv_sqrt2 * -u) + n1,
            (inv_sqrt2 * v) + n1,
            (inv_sqrt2 * -v) + n1,
            (lambda * (u+v)) + n2,
            (lambda * (u-v)) + n2,
            (lambda * (-u+v)) + n2,
            (lambda * (-u-v)) + n2,
        ]
    }

    fn light_rays_fine(&self) -> Vec<UnitVector3> {
        let u: UnitVector3 = if self.normal.x.abs() > 1.0-1e-6 {
            (vec3(1.0,0.0,0.0) - self.normal.x * self.normal).normalize()
        } else {
            (vec3(0.0,1.0,0.0) - self.normal.x * self.normal).normalize()
        };
        let v = self.normal.cross(u);
        let num = 128;
        let turns = 5;
        // Spiral up - warning too much trig
        (0..num).map(
            |i| {
                let theta = (i as f64 * PI) / (2.0 * num as f64);
                let phi = (i as f64 * PI * 2.0 * turns as f64) / (num as f64);
                theta.sin() * self.normal 
                + theta.cos() * (phi.sin() * u + phi.cos() * v)
            }
        ).collect()
    }
}

impl<'a> Scene<'a> {
    fn trace_ray(self : &Self, ray : &Ray, depth : u8) -> Rgb<u8> {
        self.find_best_contact(ray).map(
            |contact| self.trace_contact(ray, &contact, depth)
        ).unwrap_or(self.background)
    }

    fn trace_illumination(self : &Self, ray : &Ray, depth : u8) -> u8 {
        self.find_best_contact(ray).map(
            |contact| self.trace_contact_illumination(&contact, false, depth)
        ).unwrap_or(self.background_illumination)
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
            None
        } else {
            Some(self.trace_ray(&contact.reflection_ray(ray), depth - 1))
        };
        let illumination = if self.use_ambiant_illumination {
            if depth == 0 {
                self.trace_contact_illumination(contact, depth == self.max_depth, 1)
            } else {
                self.trace_contact_illumination(contact, depth == self.max_depth, 2)
            }
        } else {
            255
        };
        contact.material.colour(
            self.materials,
             &contact.local_pos, 
             illumination,
             relfection
        )
    }

    fn trace_contact_illumination(self : &Self, contact : &Contact, fine : bool, depth : u8) -> u8 {
        let surface_glow = contact.material.illumination(
            self.materials,
             &contact.local_pos
        ) as u32;
        let merged_light :u32 = if fine {
            let fine_dirs = contact.light_rays_fine();
            let num = fine_dirs.len() as u32;
            fine_dirs.into_iter().map(
                |dir| self.trace_illumination(
                    &Ray { start: contact.pos, direction : dir }, 
                    1
                ) as u32
            ).sum::<u32>() / num
        } else {
            if depth == 0 {
                0
            } else {
                if depth == 1 {
                    contact.light_rays5().into_iter().map(
                        |dir| self.trace_illumination(
                            &Ray { start: contact.pos, direction : dir }, 
                            0
                        ) as u32
                    ).sum::<u32>() / 5
                } else {
                    contact.light_rays9().into_iter().map(
                        |dir| self.trace_illumination(
                            &Ray { start: contact.pos, direction : dir }, 
                            depth - 1
                        ) as u32
                    ).sum::<u32>() / 9
                }
            }
        };
        if surface_glow != 0 {
            if surface_glow + merged_light > 255 {
                255
            } else {
                (surface_glow + merged_light) as u8
            }
        } else { 
            merged_light as u8
        }
    }

    fn from_json<'geom, 'mat>(
        input : &JsonValue, 
        geometry : &'geom Geometries, 
        materials : &'mat Materials
    ) -> std::io::Result<Scene<'a>> 
    where 'geom : 'a, 'mat : 'a
    {
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let max_depth = input["max_depth"].as_u8().unwrap_or(4);
        let use_ambiant = input["ambiant_lighting"].as_bool().unwrap_or(false);
        let entities : Vec<Entity> = input["entities"].members().filter_map(
            |value| Entity::from_json(value, geometry, materials) 
        ).collect();
        println!("Scene has {} entities", entities.len());
        Ok(Scene {
            camera : Camera{fov : 1.0,  pos : point3(0.0,0.0,1.0)},
            resolution : (res_x, res_y),
            background : Rgb([200,200,200]),
            background_illumination : 240,
            spheres : entities,
            max_depth : max_depth,
            materials : materials,
            use_ambiant_illumination: use_ambiant
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

pub fn generate(input : &JsonValue) -> std::io::Result<RgbImage> {
    println!("Generating ray trace scene");
    let materials = Materials::from_json(input);
    let geometries = Geometries::from_json(input);
    let scene = Scene::from_json(input, &geometries, &materials)?;
    Ok(scene.make_image())
}