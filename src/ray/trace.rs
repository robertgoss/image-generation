use std::cmp::min;
use cgmath::{Matrix4, Point3, vec3, vec4, Vector3};
use cgmath::{prelude::*};
use json::JsonValue;
// Scene definitions made faster to scan
use crate::ray::geometry::{Ray, TraceGeometry, Geometries, AABox};
use crate::ray::materials::{Materials, Material, ScatterRay};

// Helper to remember which we want to always be pre-unitized
type UnitVector3 = Vector3<f64>;

pub struct Entity<'a> {
    geometry : &'a dyn TraceGeometry,
    to_local : Matrix4<f64>,
    to_world : Matrix4<f64>,
    material : &'a dyn Material
}

pub struct Contact<'a> {
    pos : Point3<f64>,
    pub(crate) local_pos : Point3<f64>,
    normal : UnitVector3,
    distance : f64,
    pub(crate) material : &'a dyn Material
}

pub trait BoxedTraceSpace<'a> {
    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact>;

    fn from_iter<I>(boxed_entities : I) -> Self
        where I : Iterator<Item = (Option<AABox>, &'a Entity<'a>)>;

    fn bound(self : &Self) -> Option<AABox>;
}

// Very basic scene that stores a list of all the boxed entities
pub struct LinearScene<'a> {
    entities : Vec<(Option<AABox>, &'a Entity<'a>)>
}

#[derive(Clone, Copy)]
enum Coord {
    X, Y, Z
}

pub struct BSPTreeScene<'a> {
    bound : Option<AABox>,
    unsplit : LinearScene<'a>,
    split : Option<(Coord, Box<BSPTreeScene<'a>>, Box<BSPTreeScene<'a>>)>
}



impl<'a> BoxedTraceSpace<'a> for LinearScene<'a> {
    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact> {
        self.entities.iter().filter_map(
            |(_,entity)| entity.trace(ray)
        ).min_by(
            |contact1, contact2| contact1.distance.partial_cmp(&contact2.distance).unwrap()
        )
    }

    fn from_iter<I>(boxed_entities : I) -> Self
        where I : Iterator<Item = (Option<AABox>, &'a Entity<'a>)> {
        LinearScene{ entities : boxed_entities.collect() }
    }

    fn bound(self: &Self) -> Option<AABox> {
        let mut aabox_total_opt : Option<AABox> = None;
        for (aabox_opt,_) in self.entities.iter() {
            if let Some(aabox) = aabox_opt {
                aabox_total_opt = if let Some(aabox_total) = &mut aabox_total_opt {
                    Some(aabox_total.merge(aabox))
                } else {
                    Some(aabox.clone())
                }
            }
        }
        aabox_total_opt
    }
}

fn best_contact<'a>(contact1_opt : Option<Contact<'a>>, contact2_opt : Option<Contact<'a>>) -> Option<Contact<'a>> {
    if let Some(contact1) = &contact1_opt {
        if let Some(contact2) = &contact2_opt {
            if contact1.distance < contact2.distance {
                contact1_opt
            } else {
                contact2_opt
            }
        } else {
            contact1_opt
        }
    } else {
        contact2_opt
    }
}

impl<'a> BoxedTraceSpace<'a> for BSPTreeScene<'a> {
    fn find_best_contact(self: &Self, ray: &Ray) -> Option<Contact> {
        // TODO: Make efficient
        let unsplit_contact = self.unsplit.find_best_contact(ray);
        let split_contact = if let Some((_, left, right)) = &self.split {
            let left_contact = left.find_best_contact(ray);
            let right_contact = right.find_best_contact(ray);
            best_contact(left_contact, right_contact)
        } else {
            None
        };
        best_contact(unsplit_contact, split_contact)
    }

    fn from_iter<I>(boxed_entities: I) -> Self where I: Iterator<Item=(Option<AABox>, &'a Entity<'a>)> {
        let unsplit = LinearScene::from_iter(boxed_entities);
        let bound = unsplit.bound();
        let mut tree = BSPTreeScene { bound, unsplit, split : None};
        tree.split();
        tree
    }

    fn bound(self: &Self) -> Option<AABox> {
        self.bound.clone()
    }
}

#[derive(Clone, Copy)]
enum Split {
    Unsplit,
    Left,
    Right
}

fn split_state_dir(world_mid : f64, box_min : f64, box_max : f64) -> Split {
  if box_min > world_mid {
      Split::Right
  } else if box_max < world_mid {
      Split::Left
  } else {
      Split::Unsplit
  }
}

fn split_state(world : &AABox, dir : Coord, aabox_opt : &Option<AABox>) -> Split{
    if let Some(aabox) = aabox_opt {
        match dir {
            Coord::X => split_state_dir(0.5*(world.min.x + world.max.x), aabox.min.x, aabox.max.x),
            Coord::Y => split_state_dir(0.5*(world.min.y + world.max.y), aabox.min.y, aabox.max.y),
            Coord::Z => split_state_dir(0.5*(world.min.z + world.max.z), aabox.min.z, aabox.max.z)
        }
    } else {
        Split::Unsplit
    }
}

impl<'a> BSPTreeScene<'a> {
    fn split(self : &mut Self) {
        if let Some(bound) = &self.bound {
            let dir_opt = self.unsplit.split_dir();
            if let Some(dir) = dir_opt {
                let mut unsplit = LinearScene{ entities : Vec::new() };
                let mut left = Vec::new();
                let mut right = Vec::new();
                for (aabox, entity) in &self.unsplit.entities {
                    match split_state(bound, dir, &aabox) {
                        Split::Unsplit => unsplit.entities.push((aabox.clone(), entity)),
                        Split::Left => left.push((aabox.clone(), *entity)),
                        Split::Right => right.push((aabox.clone(), *entity)),
                    }
                }
                self.unsplit = unsplit;
                self.split = Some((
                    dir,
                    Box::new(BSPTreeScene::from_iter(left.into_iter())),
                    Box::new(BSPTreeScene::from_iter(right.into_iter()))
                ));
            }
        }
    }

}

impl <'a> LinearScene<'a> {
    fn split_dir(self : &Self) -> Option<Coord> {
        let world_opt = self.bound();
        if let Some(world) = world_opt {
            let x_stats = self.split_stats(Coord::X, &world);
            let y_stats = self.split_stats(Coord::Y, &world);
            let z_stats = self.split_stats(Coord::Z, &world);
            // Find the best and return if greater than 0
            if x_stats > y_stats && x_stats > z_stats && x_stats > 0 {
                Some(Coord::X)
            } else if y_stats > x_stats && y_stats > z_stats && y_stats > 0 {
                Some(Coord::Y)
            } else if z_stats > x_stats && z_stats > y_stats && z_stats > 0 {
                Some(Coord::Z)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn split_stats(self : &Self, dir : Coord, world : &AABox) -> usize {
        let mut left : usize = 0;
        let mut right : usize = 0;
        for (aabox, _) in self.entities.iter() {
            match split_state(world, dir, &aabox) {
                Split::Unsplit => (),
                Split::Left => left+=1,
                Split::Right => right+=1,
            }
        }
        min(left, right)
    }
}

impl<'a> Entity<'a> {
    pub(crate) fn from_json(
        input : &JsonValue,
        geometries : &'a Geometries,
        materials : &'a Materials
    ) -> Option<Entity<'a>> {
        let material = materials.lookup(input["mat"].as_str().unwrap_or("none"));
        let geometry = geometries.lookup(input["geom"].as_str().unwrap_or("none"))?;
        let x = input["x"].as_f64()?;
        let y = input["y"].as_f64()?;
        let z = input["z"].as_f64()?;
        let dir_x = input["dir_x"].as_f64().unwrap_or(0.0);
        let dir_y = input["dir_y"].as_f64().unwrap_or(0.0);
        let dir_z = input["dir_z"].as_f64().unwrap_or(1.0);
        let dir = vec3(dir_x, dir_y, dir_z).normalize();
        // Want camera with z pointing down dir, y points (as close as possible to z)
        // and x whatever is left.
        let xv = if dir.z.abs() > 1.0 - 1.0e-8 {
            dir.cross(vec3(0.0, -1.0, 0.0)).normalize()
        } else {
            dir.cross(vec3(0.0, 0.0, -1.0)).normalize()
        };
        let yv = -xv.cross(dir);
        let to_world = Matrix4{
            x : vec4(xv.x, xv.y, xv.z, 0.0),
            y : vec4(yv.x, yv.y, yv.z, 0.0),
            z : vec4(dir.x, dir.y, dir.z, 0.0),
            w : vec4(x, y, z, 1.0)
        };
        let inv = to_world.invert().unwrap();
        Some(Entity { geometry: geometry, to_local: inv, to_world: to_world, material: material })
    }

    pub fn bound(&self) -> Option<AABox> {
        self.geometry.bound(&self.to_world)
    }

    fn trace(&self, ray : &Ray) -> Option<Contact<'a>> {
        let transformed_ray = ray.transform(&self.to_local);
        let (t_contact, transformed_norm) = self.geometry.trace(&transformed_ray)?;
        let normal = self.to_world.transform_vector(transformed_norm);
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
        Ray{start: self.pos + dir * 0.0001, direction : dir}
    }

    fn diffuse_ray<Rng>(self : &Self, rng : &mut Rng) -> Ray
        where Rng : rand::Rng
    {
        // Generate random point on unit sphere
        let vec = vec3(rng.gen::<f64>()-0.5, rng.gen::<f64>()-0.5, rng.gen::<f64>()-0.5);
        let dir = (self.normal + vec.normalize()).normalize();
        Ray{start: self.pos + dir * 0.0001, direction : dir}
    }

    pub(crate) fn scatter_ray<Rng>(self : &Self, scatter : &ScatterRay, rng : &mut Rng, ray_in : &Ray) -> Option<Ray>
        where Rng : rand::Rng
    {
        match scatter {
            ScatterRay::None => None,
            ScatterRay::Diffuse => Some(self.diffuse_ray(rng)),
            ScatterRay::Reflection => Some(self.reflection_ray(ray_in))
        }
    }
}