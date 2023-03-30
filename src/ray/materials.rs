use std::collections::HashMap;
use image::Rgb;
use json::JsonValue;
use cgmath::Point3;


pub fn rgb_lerp(
    parameter : f64,
    colour1 : Rgb<f64>,
    colour2 : Rgb<f64>
) -> Rgb<f64> {
    let inverse = 1.0 - parameter;
    Rgb([
        colour1.0[0] * parameter + colour2.0[0] * inverse,
        colour1.0[1] * parameter + colour2.0[1] * inverse,
        colour1.0[2] * parameter + colour2.0[2] * inverse
    ])
}

fn rgb_lerp3(
    parameter : f64,
    parameter2 : f64,
    colour1 : &Rgb<f64>,
    colour2 : &Rgb<f64>,
    colour3 : &Rgb<f64>
) -> Rgb<f64> {
    let inverse = 1.0 - parameter - parameter2;
    Rgb([
        colour1.0[0] * parameter + colour2.0[0] * parameter2 + colour3.0[0] * inverse,
        colour1.0[1] * parameter + colour2.0[1] * parameter2 + colour3.0[1] * inverse,
        colour1.0[2] * parameter + colour2.0[2] * parameter2 + colour3.0[2] * inverse,
    ])
}

pub fn rgb_sum(
    colour1 : &mut Rgb<f64>,
    colour2 : &Rgb<f64>
) {
    colour1.0[0] += colour2.0[0];
    colour1.0[1] += colour2.0[1];
    colour1.0[2] += colour2.0[2];
}

pub struct Colour {
    colour : Rgb<f64>,
    reflectivity : f64,
    diffusion : f64,
}

pub struct Checker {
    step : f64,
    material_odd : usize,
    material_even : usize,
}

// Internally materials that want the reference others store a index to the materials
// vector. 
pub struct Materials {
    names : HashMap<String, usize>,
    materials : Vec<Box<dyn Material>>,
    default : Colour
}

impl Materials {
    pub fn from_json(input : &JsonValue) -> Materials {
        let mut materials = Materials {
            names : HashMap::new(),
            materials : Vec::new(),
            default : Colour { colour: Rgb([0.8,0.8,0.8]), reflectivity: 0.0, diffusion : 0.5 }
        };
        for (mat_name, mat_input) in input["materials"].entries() {
            if let Some(mat) = parse_material(mat_input, &materials) {
                let index = materials.materials.len();
                materials.materials.push(mat);
                materials.names.insert(mat_name.to_string(), index);
            }
        }
        println!("{} materials loaded", materials.names.len());
        materials
    }

    pub fn lookup(&self, name : &str) -> &dyn Material {
        self.names.get(name).and_then(
            |index| self.materials.get(*index).map(
                |boxed| boxed.as_ref()
            )
        ).unwrap_or(&self.default)
    }

    pub fn lookup_index(&self, name : &str) -> Option<usize> {
        self.names.get(name).cloned()
    }
}

fn parse_material(input : &JsonValue, materials : &Materials) -> Option<Box<dyn Material>> {
    match input["type"].as_str() {
        Some("colour") => Colour::from_json(input),
        Some("checker") => Checker::from_json(input, materials),
        _ => None
    }
}

pub trait Material {
    fn colour(
        &self, 
        mats : &Materials,
        local : &Point3<f64>,
        reflection : &Rgb<f64>,
        diffuse : &Rgb<f64>
    ) -> Rgb<f64>;
}

impl Colour {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let r = input["r"].as_u8()?;
        let r_f64 = r as f64 / 255.0;
        let g = input["g"].as_u8()?;
        let g_f64 = g as f64 / 255.0;
        let b = input["b"].as_u8()?;
        let b_f64 = b as f64 / 255.0;
        let reflectivity = input["reflect"].as_f64()?;
        let diffusion = 0.5 * (1.0 - reflectivity);
        Some(
            Box::new(
                Colour { colour: Rgb([r_f64,g_f64,b_f64]), reflectivity, diffusion }
            )
        )
    }
}

impl Checker {
    pub fn from_json(input : &JsonValue, materials : &Materials) -> Option<Box<dyn Material>> {
        let step = input["step"].as_f64().unwrap_or(1.0);
        let name_even = input["odd"].as_str()?;
        let name_odd = input["even"].as_str()?;
        let index_even = materials.lookup_index(name_even)?;
        let index_odd = materials.lookup_index(name_odd)?;
        Some(
            Box::new(
                Checker { 
                    step: step, 
                    material_even : index_even,
                    material_odd : index_odd
                }
            )
        )
    }
}

impl Material for Colour {
    fn colour(
        &self,
        _mats : &Materials,
        _local_point : &Point3<f64>,
        reflection : &Rgb<f64>,
        diffuse : &Rgb<f64>
    ) -> Rgb<f64> {
        rgb_lerp3(
            self.reflectivity,
            self.diffusion,
            reflection,
            diffuse,
            &self.colour
        )
    }

}

impl Material for Checker {
    fn colour(
        &self,
        mats : &Materials,
        local_point : &Point3<f64>,
        reflection : &Rgb<f64>,
        diffuse : &Rgb<f64>
    ) -> Rgb<f64> {
        let x = (local_point.x * self.step) as i64;
        let y = (local_point.y * self.step) as i64;
        let z = (local_point.z * self.step) as i64;
        let index = if (x+y+z) % 2 == 0 { self.material_even } else { self.material_odd };
        mats.materials[index].colour(mats, local_point, reflection, diffuse)
    }
}