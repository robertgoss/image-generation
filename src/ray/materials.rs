use std::collections::HashMap;
use image::Rgb;
use json::JsonValue;
use cgmath::Point3;

pub struct Colour {
    colour : Rgb<u8>,
    glow : u8,
    reflectivity : u8
}

pub struct Checker {
    step : f64,
    material_odd : usize,
    material_even : usize,
}

// Internally materials that want the reference others store a inddex to the materials
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
            default : Colour { colour: Rgb([145,145,145]), glow: 0, reflectivity: 0 }
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

fn scale_colour(c : Rgb<u8>, par : u8 ) -> Rgb<u8> {
    let r1 = par as u16 * c.0[0] as u16;
    let g1 = par as u16 * c.0[1] as u16;
    let b1 = par as u16 * c.0[2] as u16;
    let r = r1 / 255;
    let g = g1 / 255;
    let b = b1 / 255;
    Rgb([r as u8, g as u8, b as u8])
}

pub trait Material {
    fn colour(
        &self, 
        mats : &Materials,
        local : &Point3<f64>, 
        illumination : u8,
        reflection : Option<Rgb<u8>>
    ) -> Rgb<u8>;

    fn illumination(
        &self,
        mats : &Materials,
        local : &Point3<f64>
    ) -> u8;
}

impl Colour {
    pub fn from_json(input : &JsonValue) -> Option<Box<dyn Material>> {
        let r = input["r"].as_u8()?;
        let g = input["g"].as_u8()?;
        let b = input["b"].as_u8()?;
        let glow = input["glow"].as_u8().unwrap_or(0);
        let refl = input["reflect"].as_f64()?;
        let refl_u8 = (refl * 255.0) as u8;
        Some(
            Box::new(
                Colour { colour: Rgb([r,g,b]), glow : glow, reflectivity: refl_u8 }
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
        illumination : u8,
        reflection : Option<Rgb<u8>>
    ) -> Rgb<u8> {
        let refl_colour = merge_colour(
            reflection.unwrap_or(Rgb([0,0,0])), 
            self.colour,
            self.reflectivity
        );
        if self.glow != 0 {
            let g_illumination = illumination as u16 + self.glow as u16;
            if g_illumination > 255 {
                refl_colour
            } else {
                scale_colour(refl_colour, g_illumination as u8)
            }
        } else {
            scale_colour(refl_colour, illumination)
        }
    }

    fn illumination(
        &self,
        _mats : &Materials,
        _local : &Point3<f64>
    ) -> u8 {
        self.glow
    }

}

impl Material for Checker {
    fn colour(
        &self,
        mats : &Materials,
        local_point : &Point3<f64>,
        illumination : u8,
        reflection : Option<Rgb<u8>>,
    ) -> Rgb<u8> {
        let x = (local_point.x * self.step) as i64;
        let y = (local_point.y * self.step) as i64;
        let z = (local_point.z * self.step) as i64;
        let index = if (x+y+z) % 2 == 0 { self.material_even } else { self.material_odd };
        mats.materials[index].colour(mats, local_point, illumination, reflection)
    }

    fn illumination(
        &self,
        mats : &Materials,
        local_point : &Point3<f64>
    ) -> u8 {
        let x = (local_point.x * self.step) as i64;
        let y = (local_point.y * self.step) as i64;
        let z = (local_point.z * self.step) as i64;
        let index = if (x+y+z) % 2 == 0 { self.material_even } else { self.material_odd };
        mats.materials[index].illumination(mats, local_point)
    }
}