use std::collections::HashMap;
use image::Rgb;
use json::JsonValue;

#[derive(Clone, Copy)]
pub struct Material {
    pub colour : Rgb<u8>,
    pub reflectivity : u8
}

pub struct Materials {
    names : HashMap<String, Material>,
    default : Material
}

impl Materials {
    pub fn from_json(input : &JsonValue) -> Materials {
        let materials : HashMap<String, Material> = input["materials"].entries().filter_map(
            |(name, value)| 
              Material::from_json(value).map(|material| (name.to_string(), material))
        ).collect();
        println!("{} materials loaded", materials.len());
        Materials {
            names : materials,
            default : Material { colour: Rgb([145,145,145]), reflectivity: 0 }
        }
    }

    pub fn lookup<'a>(self : &'a Self, name : &str) -> &'a Material {
        self.names.get(name).unwrap_or(&self.default)
    }
}

impl Material {
    pub fn from_json(input : &JsonValue) -> Option<Material> {
        let r = input["r"].as_u8()?;
        let g = input["g"].as_u8()?;
        let b = input["b"].as_u8()?;
        let refl = input["reflect"].as_f64()?;
        let refl_u8 = (refl * 255.0) as u8;
        Some(
            Material { colour: Rgb([r,g,b]), reflectivity: refl_u8 }
        )
    }
}