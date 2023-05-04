use std::cmp::min;
use cgmath::{Matrix4, Point3, vec3, vec4, Vector3};
use cgmath::{prelude::*};
use json::JsonValue;
// Scene definitions made faster to scan
use crate::ray::geometry::{Ray, TraceGeometry, Geometries, AABox, Coord};
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
    fn find_best_contact(self : &Self, ray : &Ray, bound : &Option<f64>) -> Option<Contact>;

    fn from_iter<I>(boxed_entities : I) -> Self
        where I : Iterator<Item = (Option<AABox>, &'a Entity<'a>)>;

    fn bound(self : &Self) -> Option<AABox>;
}

// Very basic scene that stores a list of all the boxed entities
pub struct LinearScene<'a> {
    entities : Vec<(Option<AABox>, &'a Entity<'a>)>
}

pub struct BSPTreeScene<'a> {
    bounded_tree : Option<(AABox, BSPTree<'a>)>,
    unbounded : LinearScene<'a>
}

pub struct BSPTree<'a> {
    cut : f64,
    bound : AABox,
    unsplit : LinearScene<'a>,
    split : Option<(Coord, Box<BSPTree<'a>>, Box<BSPTree<'a>>)>
}


impl<'a> BoxedTraceSpace<'a> for LinearScene<'a> {
    fn find_best_contact(self : &Self, ray : &Ray, _ : &Option<f64>) -> Option<Contact> {
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
            } else {
                return None
            }
        }
        aabox_total_opt
    }
}

// Is a or b the best bound
//
// No distance compare less than a distance
fn cmp_bound(dist_a_opt : &Option<f64>, dist_b_opt : &Option<f64>) -> bool {
    if let Some(dist_a) = &dist_a_opt {
        if let Some(dist_b) = &dist_b_opt {
            dist_a < dist_b
        } else {
            true
        }
    } else {
        false
    }
}

fn best_bound<'a>(dist_a_opt : Option<f64>, dist_b_opt : Option<f64>) -> Option<f64> {
    if cmp_bound(&dist_a_opt, &dist_a_opt){
        dist_a_opt
    } else {
        dist_b_opt
    }
}

fn best_contact<'a>(contact1_opt : Option<Contact<'a>>, contact2_opt : Option<Contact<'a>>) -> Option<Contact<'a>> {
    let cmp = cmp_bound(
        &contact1_opt.as_ref().map(|c| {c.distance}),
        &contact2_opt.as_ref().map(|c| {c.distance})
    );
    if cmp { contact1_opt } else { contact2_opt }
}

impl<'a> BoxedTraceSpace<'a> for BSPTreeScene<'a> {
    fn find_best_contact(self: &Self, ray: &Ray, initial_dist_bound : &Option<f64>) -> Option<Contact> {
        let unbounded_contact = self.unbounded.find_best_contact(ray, initial_dist_bound);
        let dist_bound_opt = best_bound(
            initial_dist_bound.clone(),
            unbounded_contact.as_ref().map(|c| c.distance)
        );
        // Get to box if ray start not inside
        if let Some((bound, tree)) = &self.bounded_tree {
            if bound.contains(&ray.start) {
                let tree_contact = tree.find_best_contact(ray, &dist_bound_opt);
                return best_contact(unbounded_contact, tree_contact)
            }
            if let Some((t_box, _)) = bound.trace(ray) {
                let mut proj_dist_bound : Option<f64> = None;
                if let Some(dist_bound) = dist_bound_opt {
                    if dist_bound < t_box {
                        // Cant contact
                        return unbounded_contact
                    } else {
                        proj_dist_bound = Some(dist_bound - t_box);
                    }
                }
                let proj_ray = Ray { direction : ray.direction, start : ray.at(t_box)};
                assert!(bound.contains_eps(&proj_ray.start, 1e-6));
                let tree_contact = tree.find_best_contact(&proj_ray, &proj_dist_bound).map(
                    |contact| contact.push_back(t_box)
                );
                return best_contact(unbounded_contact, tree_contact)
            }
        }
        unbounded_contact
    }

    fn from_iter<I>(boxed_entities: I) -> Self where I: Iterator<Item=(Option<AABox>, &'a Entity<'a>)> {
        let mut unbounded = LinearScene { entities : Vec::new() };
        let mut unsplit_boxed = LinearScene { entities : Vec::new() };
        for (aabox_opt, entity) in boxed_entities {
            if aabox_opt. is_some(){
                unsplit_boxed.entities.push((aabox_opt, entity));
            } else {
                unbounded.entities.push((aabox_opt, entity));
            }
        };
        let tree_bound = unsplit_boxed.bound();
        let bounded_tree = tree_bound.map(
            |bound| {
                let mut tree = BSPTree { cut : 0.0, bound : bound.clone(), unsplit : unsplit_boxed, split : None};
                tree.split(&bound);
                (bound, tree)
            }
        );
        BSPTreeScene {
            bounded_tree,
            unbounded
        }
    }

    fn bound(self: &Self) -> Option<AABox> {
        if self.unbounded.entities.is_empty() {
            self.bounded_tree.as_ref().map(
                |(bound, _)| bound.clone()
            )
        } else {
            None
        }
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
            Coord::X => split_state_dir(0.5 * (world.min.x + world.max.x), aabox.min.x, aabox.max.x),
            Coord::Y => split_state_dir(0.5 * (world.min.y + world.max.y), aabox.min.y, aabox.max.y),
            Coord::Z => split_state_dir(0.5 * (world.min.z + world.max.z), aabox.min.z, aabox.max.z)
        }
    } else {
        Split::Unsplit
    }
}

fn vec_coord(dir : Coord, vec : &UnitVector3) -> f64 {
    match dir {
        Coord::X => vec.x,
        Coord::Y => vec.y,
        Coord::Z => vec.z
    }
}

fn point_coord(dir : Coord, pt : &Point3<f64>) -> f64 {
    match dir {
        Coord::X => pt.x,
        Coord::Y => pt.y,
        Coord::Z => pt.z
    }
}

impl<'a> BSPTree<'a> {
    fn split(self : &mut Self, bound : &AABox) {
        let dir_opt = self.unsplit.split_dir();
        if let Some(dir) = dir_opt {
            let mut unsplit = LinearScene { entities: Vec::new() };
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
            self.cut = point_coord(dir, &bound.center());
            self.split = Some((
                dir,
                Box::new(BSPTree::from_iter(bound.cut_left(dir), left.into_iter())),
                Box::new(BSPTree::from_iter(bound.cut_right(dir), right.into_iter()))
            ));
        }
    }

    fn from_iter<I>(bound : AABox, boxed_entities: I) -> Self where I: Iterator<Item=(Option<AABox>, &'a Entity<'a>)> {
        let unsplit = LinearScene::from_iter(
            boxed_entities.filter(|(aabox, _)| aabox.is_some())
        );
        let mut tree = BSPTree { cut : 0.0, bound : bound.clone(), unsplit, split : None};
        tree.split(&bound);
        tree
    }

    fn split_ray(self : &Self, ray : &Ray) -> Option<(f64, Ray)> {
        let coord = self.split.as_ref().unwrap().0;
        let ray_dir = vec_coord(coord, &ray.direction);
        let ray_pos = point_coord(coord, &ray.start);
        if ray_dir.abs() < 1e-10 {
            return None
        };
        let t = (self.cut - ray_pos) / ray_dir;
        if t < 0.0 {
            return None;
        }
        let cross = ray.at(t);
        if self.bound.contains(&cross) {
            Some((t, Ray { start : cross, direction : ray.direction }))
        } else {
            None
        }
    }

    fn find_best_contact(self: &Self, ray: &Ray, initial_dist_bound : &Option<f64>) -> Option<Contact> {
        assert!(self.bound.contains_eps(&ray.start, 1e-6));
        let unsplit_contact = self.unsplit.find_best_contact(ray, initial_dist_bound);
        let best_dist_bound = best_bound(
            initial_dist_bound.clone(),
            unsplit_contact.as_ref().map(|c| c.distance)
        );
        let split_contact = self.find_best_contact_split(ray, &best_dist_bound);
        best_contact(unsplit_contact, split_contact)
    }

    fn find_best_contact_split(self: &Self, ray: &Ray, initial_dist_bound : &Option<f64>) -> Option<Contact> {
        assert!(self.bound.contains_eps(&ray.start, 1e-6));
        if let Some((coord, left, right)) = &self.split {
            let left_right = vec_coord(*coord, &ray.direction) > 0.0;
            let ray_pos = point_coord(*coord, &ray.start);
            let initial_cont = if left_right {
                if ray_pos > self.cut {
                    return right.find_best_contact(ray, initial_dist_bound);
                }
                left.find_best_contact(ray, initial_dist_bound)
            } else {
                if ray_pos < self.cut {
                    return left.find_best_contact(ray, initial_dist_bound);
                }
                right.find_best_contact(ray, initial_dist_bound)
            };
            let second_cont = self.split_ray(ray).and_then(
                |(t, split_ray)| {
                    let mut split_dist_bound : Option<f64> = None;
                    if let Some(dist_bound) = initial_dist_bound {
                        if *dist_bound < t {
                            return None
                        }
                        split_dist_bound = Some(dist_bound - t);
                    }
                    let contact_opt= if left_right {
                        right.find_best_contact(&split_ray, &split_dist_bound)
                    } else {
                        left.find_best_contact(&split_ray, &split_dist_bound)
                    };
                    contact_opt.map(|contact| contact.push_back(t))
                }
            );
            best_contact(initial_cont, second_cont)
        } else {
            None
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
            if x_stats >= y_stats && x_stats >= z_stats && x_stats > 0 {
                Some(Coord::X)
            } else if y_stats >= x_stats && y_stats >= z_stats && y_stats > 0 {
                Some(Coord::Y)
            } else if z_stats >= x_stats && z_stats >= y_stats && z_stats > 0 {
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
            normal,
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

    fn push_back(self: &Self, t : f64) -> Contact<'a> {
        Contact {
            pos : self.pos,
            local_pos : self.local_pos,
            normal : self.normal,
            distance : t + self.distance,
            material : self.material
        }
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

#[cfg(test)]
mod tests {
    use cgmath::{assert_abs_diff_eq, point3};
    use image::Rgb;
    use crate::ray::geometry;
    use crate::ray::geometry::Sphere;
    use crate::ray::materials::Dull;

    use super::*;

    #[test]
    fn test_bsp_split_3_spheres() {
        // Given a scene with 3 spheres in a line the bsp should split into 2 children with one
        // in each
        let sphere = Sphere { radius : 1.0 };
        let mat = Dull { colour: Rgb([0.0,0.0,0.0]) };
        let mut entities: Vec<Entity> = vec!(
            Entity {
                geometry: &sphere,
                to_local: Matrix4::from_translation(vec3(-10.0, 0.0, 0.0)),
                to_world: Matrix4::from_translation(vec3(10.0, 0.0, 0.0)),
                material: &mat,
            },
            Entity {
                geometry: &sphere,
                to_local: Matrix4::from_translation(vec3(0.0, 0.0, 0.0)),
                to_world: Matrix4::from_translation(vec3(0.0, 0.0, 0.0)),
                material: &mat,
            },
            Entity {
                geometry: &sphere,
                to_local: Matrix4::from_translation(vec3(10.0, 0.0, 0.0)),
                to_world: Matrix4::from_translation(vec3(-10.0, 0.0, 0.0)),
                material: &mat,
            }
        );
        let scene = BSPTreeScene::from_iter(
            entities.iter().map(|entity| (entity.bound(), entity))
        );
    }
}