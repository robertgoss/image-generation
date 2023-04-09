use cgmath::{Matrix4, Point3, vec3, vec4, Vector3};
use cgmath::{prelude::*};
use json::JsonValue;
// Scene definitions made faster to scan
use crate::ray::geometry::{Ray, TraceGeometry, Geometries};
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

pub trait SceneSpace<'a> {
    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact>;

    fn from_iter<I>(entity_iter : I) -> Self
        where I : Iterator<Item = Entity<'a>>;

    fn size(&self) -> usize;
}

// Very basic scene that stores a list of all the entities
pub struct LinearScene<'a> {
    entities : Vec<Entity<'a>>
}

impl<'a> SceneSpace<'a> for LinearScene<'a> {
    fn find_best_contact(self : &Self, ray : &Ray) -> Option<Contact> {
        self.entities.iter().filter_map(
            |entity| entity.trace(ray)
        ).min_by(
            |contact1, contact2| contact1.distance.partial_cmp(&contact2.distance).unwrap()
        )
    }

    fn from_iter<I>(entity_iter : I) -> Self
        where I : Iterator<Item = Entity<'a>>
    {
        LinearScene {
            entities : entity_iter.collect()
        }
    }

    fn size(&self) -> usize {
        self.entities.len()
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