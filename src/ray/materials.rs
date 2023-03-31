use std::collections::HashMap;
use image::Rgb;
use json::JsonValue;
use cgmath::Point3;


pub fn rgb_lerp(
    parameter : f64,
    colour1 : &Rgb<f64>,
    colour2 : &Rgb<f64>
) -> Rgb<f64> {
    let inverse = 1.0 - parameter;
    Rgb([
        colour1.0[0] * parameter + colour2.0[0] * inverse,
        colour1.0[1] * parameter + colour2.0[1] * inverse,
        colour1.0[2] * parameter + colour2.0[2] * inverse
    ])
}

pub fn rgb_scale(
    colour : &Rgb<f64>,
    scale : &Rgb<f64>
) -> Rgb<f64> {
    Rgb([
        colour.0[0] * scale.0[0],
        colour.0[1] * scale.0[1],
        colour.0[2] * scale.0[2]
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

pub struct Plain {
    colour : Rgb<f64>
}

pub struct Shiny {
    colour : Rgb<f64>
}

pub struct Dull {
    colour : Rgb<f64>
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
    default : Plain
}

impl Materials {
    pub fn from_json(input : &JsonValue) -> Materials {
        let mut materials = Materials {
            names : HashMap::new(),
            materials : Vec::new(),
            default : Plain { colour: Rgb([0.8,0.8,0.8]) }
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
        Some("plain") => Plain::from_json(input),
        Some("dull") => Dull::from_json(input),
        Some("shiny") => Shiny::from_json(input),
        Some("checker") => Checker::from_json(input, materials),
        _ => None
    }
}

pub enum ScatterRay {
    None,
    Diffuse,
    Reflection
}

pub trait Material {
    fn colour(
        &self, 
        mats : &Materials,
        local : &Point3<f64>
    ) -> (ScatterRay, Rgb<f64>);
}

fn rgb_from_json(input : &JsonValue) -> Option<Rgb<f64>> {
    let r = input["r"].as_u8()?;
    let r_f64 = r as f64 / 255.0;
    let g = input["g"].as_u8()?;
    let g_f64 = g as f64 / 255.0;
    let b = input["b"].as_u8()?;
    let b_f64 = b as f64 / 255.0;
    Some(Rgb([r_f64,g_f64,b_f64]))
}

impl Plain {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let colour = rgb_from_json(input)?;
        Some(
            Box::new(
                Plain { colour }
            )
        )
    }
}


impl Shiny {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let colour = rgb_from_json(input)?;
        Some(
            Box::new(
                Shiny { colour }
            )
        )
    }
}


impl Dull {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let colour = rgb_from_json(input)?;
        Some(
            Box::new(
                Dull { colour }
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
                    step,
                    material_even : index_even,
                    material_odd : index_odd
                }
            )
        )
    }
}

impl Material for Plain {
    fn colour(
        &self,
        _mats : &Materials,
        _local_point : &Point3<f64>,
    ) -> (ScatterRay, Rgb<f64>) {
        (ScatterRay::None, self.colour)
    }
}

impl Material for Shiny {
    fn colour(
        &self,
        _mats : &Materials,
        _local_point : &Point3<f64>,
    ) -> (ScatterRay, Rgb<f64>) {
        (ScatterRay::Reflection, self.colour)
    }
}

impl Material for Dull {
    fn colour(
        &self,
        _mats : &Materials,
        _local_point : &Point3<f64>,
    ) -> (ScatterRay, Rgb<f64>) {
        (ScatterRay::Diffuse, self.colour)
    }
}

impl Material for Checker {
    fn colour(
        &self,
        mats : &Materials,
        local_point : &Point3<f64>
    ) -> (ScatterRay, Rgb<f64>) {
        let x = (local_point.x * self.step) as i64;
        let y = (local_point.y * self.step) as i64;
        let z = (local_point.z * self.step) as i64;
        let index = if (x+y+z) % 2 == 0 { self.material_even } else { self.material_odd };
        mats.materials[index].colour(mats, local_point)
    }
}