use std::collections::HashMap;
use image::Rgb;
use json::JsonValue;
use cgmath::Point3;

pub struct Colour {
    colour : Rgb<u8>,
    reflectivity : u8
}

pub struct Checker {
    step : f64
}

pub struct Materials {
    names : HashMap<String, Box<dyn Material>>,
    default : Colour
}

impl Materials {
    pub fn from_json(input : &JsonValue) -> Materials {
        let materials : HashMap<String, Box<dyn Material>> = input["materials"].entries().filter_map(
            |(name, value)| 
              parse_material(value).map(|material| (name.to_string(), material))
        ).collect();
        println!("{} materials loaded", materials.len());
        Materials {
            names : materials,
            default : Colour { colour: Rgb([145,145,145]), reflectivity: 0 }
        }
    }

    pub fn lookup<'a>(self : &'a Self, name : &str) -> &'a dyn Material {
        self.names.get(name).map(
            |boxed| boxed.as_ref()
        ).unwrap_or(&self.default)
    }
}

fn parse_material(input : &JsonValue) -> Option<Box<dyn Material>> {
    match input["type"].as_str() {
        Some("colour") => Colour::from_json(input),
        Some("checker") => Checker::from_json(input),
        _ => None
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

pub trait Material {
    fn colour(&self, local : &Point3<f64>, reflection : Option<Rgb<u8>>) -> Rgb<u8>;
}

impl Colour {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let r = input["r"].as_u8()?;
        let g = input["g"].as_u8()?;
        let b = input["b"].as_u8()?;
        let refl = input["reflect"].as_f64()?;
        let refl_u8 = (refl * 255.0) as u8;
        Some(
            Box::new(
                Colour { colour: Rgb([r,g,b]), reflectivity: refl_u8 }
            )
        )
    }
}

impl Checker {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let step = input["step"].as_f64().unwrap_or(1.0);
        Some(
            Box::new(
                Checker { step: step }
            )
        )
    }
}

impl Material for Colour {
    fn colour(
        &self,
        _local_point : &Point3<f64>,
        reflection : Option<Rgb<u8>>
    ) -> Rgb<u8> {
        merge_colour(
            reflection.unwrap_or(Rgb([0,0,0])), 
            self.colour,
            self.reflectivity
        )
    }
}

impl Material for Checker {
    fn colour(
        &self,
        local_point : &Point3<f64>,
        _reflection : Option<Rgb<u8>>,
    ) -> Rgb<u8> {
        let x = (local_point.x * self.step) as i64;
        let y = (local_point.y * self.step) as i64;
        let z = (local_point.z * self.step) as i64;
        if (x+y+z) % 2 == 0 {
            Rgb([0,0,0])
        } else {
            Rgb([255,255,255])
        }
    }
}