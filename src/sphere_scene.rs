use std::io::{Error, ErrorKind};
use json::*;
use image::{Rgb, RgbImage};
use rand::random;

use cgmath::{point2, point3, Point2, vec2, Vector2};
use cgmath::{InnerSpace, MetricSpace};

struct Sphere {
    id : usize,
    radius : f64,
    pos : Point2<f64>,
    colour : Rgb<u8>
}

struct SphereScene {
    number_of_spheres : usize,
    resolution : (usize, usize),
}

fn random_in_range(min : f64, max : f64) -> f64 {
    ((max-min) * random::<f64>()) + min
}

fn random_in_disc() -> Vector2<f64> {
    let base = 2.0*vec2(random::<f64>() - 0.5, random::<f64>() - 0.5);
    if base.magnitude2() < 1.0 {
        base
    } else {
        random_in_disc()
    }
}

fn random_material_type() -> String {
    if random::<u8>() % 2 == 0 {
        "shiny".to_string()
    } else {
        "dull".to_string()
    }
}

impl SphereScene {
    fn from_json(input : &JsonValue) -> Option<SphereScene> {
        let number_of_spheres = input["number"].as_usize()?;
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        Some(
            SphereScene {
                number_of_spheres,
                resolution: (res_x, res_y)
            }
        )
    }

    fn generate_sphere(&self, index : usize, spheres : &Vec<Sphere>) -> Sphere {
        let sphere = Sphere {
            id : index,
            radius : random_in_range(0.2, 0.8),
            pos : point2(0.0, 0.0) + (random_in_disc() * 10.0),
            colour : Rgb([
                random::<u8>(),
                random::<u8>(),
                random::<u8>()
            ])
        };
        // Check overlap
        let overlap = spheres.iter().any(
            |other| sphere.overlap(other)
        );
        if overlap {
            self.generate_sphere(index, spheres)
        } else {
            sphere
        }
    }

    fn generate_spheres(&self) -> Vec<Sphere> {
        let mut spheres = Vec::new();
        for i in 0..self.number_of_spheres {
            let sphere = self.generate_sphere(i, &spheres);
            spheres.push(sphere);
        }
        spheres
    }

    fn generate_scene(&self) -> JsonValue {
        let mut new_scene = object! {
            "resolution_x": self.resolution.0,
            "resolution_y": self.resolution.1,
            "depth": 5,
            "algorithm": "raytrace",
            "antialiasing_samples" : 200,
            "camera" : {
                "x" : 0, "y" : -15.0, "z" : 4.0,
                "target_x" : 0, "target_y" : 0.0, "target_z" : 0.0,
            },
            "geometries" : {
                "plane" : { "type" : "plane" },
            },
            "materials" : {
                "plane_mat" :  {"type" : "dull", "r" : 200, "g" : 200, "b" : 200}
            },
            "entities" : [
                { "geom" : "plane", "x" : 0.2, "y" : 0, "z" : 0, "mat" : "plane_mat"}
            ]
        };
        for sphere in self.generate_spheres() {
            sphere.add_to_scene(&mut new_scene);
        }

        new_scene
    }
}

impl Sphere {
    fn add_to_scene(&self, scene : &mut JsonValue) {
        self.add_geometry(scene);
        self.add_material(scene);
        self.add_entity(scene);
    }

    fn add_geometry(&self, scene : &mut JsonValue) {
        scene["geometries"].insert(
            &self.geom_id(),
            object!{
                "type" : "sphere",
                "radius" : self.radius
            }
        ).expect("Missing geometries from scene");
    }

    fn add_material(&self, scene : &mut JsonValue) {
        scene["materials"].insert(
            &self.mat_id(),
            object!{
                "type" : random_material_type(),
                "r" : self.colour.0[0],
                "g" : self.colour.0[1],
                "b" : self.colour.0[2],
            }
        ).expect("Missing materials from scene");
    }

    fn add_entity(&self, scene : &mut JsonValue) {
        scene["entities"].push(
            object!{
                "geom" : self.geom_id(),
                "mat" : self.mat_id(),
                "x" : self.pos.x,
                "y" : self.pos.y,
                "z" : self.radius,
            }
        ).expect("Missing entities from scene");
    }

    fn geom_id(&self) -> String {
        format!("geom_{}", self.id)
    }

    fn mat_id(&self) -> String {
        format!("mat_{}", self.id)
    }

    fn overlap(&self, other : &Sphere) -> bool {
        let pos = point3(self.pos.x, self.pos.y, self.radius);
        let pos_other = point3(other.pos.x, other.pos.y, other.radius);
        pos.distance(pos_other) < self.radius + other.radius
    }
}


pub fn generate(input : &JsonValue) -> std::io::Result<RgbImage> {
    let sphere_scene = SphereScene::from_json(input).ok_or(
        Error::new(ErrorKind::InvalidData, "Missing camera")
    )?;
    let new_scene = sphere_scene.generate_scene();

    // Useful output for debugging
    let mut f = std::fs::File::create("test.json")?;
    new_scene.write_pretty(&mut f, 4)?;

    crate::ray::generate(&new_scene)
}