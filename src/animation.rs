use json::JsonValue;
use std::io::{Error, ErrorKind};

struct Modifications {
    mods : Vec<ModificationPoly>
}

impl Modifications {
    fn from_json(input : &JsonValue) -> std::io::Result<Modifications> {
        if !input.is_array() {
            return Err(Error::new(ErrorKind::InvalidData, "Missing modifications"))
        }
        let mods = input.members().filter_map(
            |i| ModificationPoly::from_json(i)
        ).collect();
        Ok(Modifications { mods: mods })
    }
}

// A modification which sets the parameter at the given path to 
// the result of evaluating polynomial with given coefficients at the
// given time
struct ModificationPoly {
    param_path : Vec<String>,
    coefficients : Vec<f64>,
    to_int : bool
}

fn set_json_param(
    root : &mut JsonValue,
    path : &[String],
    val : JsonValue 
) {
    let name = &path[0];
    if path.len() == 1 {
        if root.is_array() {
            let id = name.parse::<usize>().unwrap_or(0);
            root[id] = val;
        } else {
            root[name] = val;
        }
    } else {
        if root.is_array() {
            let id = name.parse::<usize>().unwrap_or(0);
            set_json_param(&mut root[id], &path[1..], val);
        } else {
            set_json_param(&mut root[name], &path[1..], val);
        }
    }
}

impl ModificationPoly {
    fn from_json(input : &JsonValue) -> Option<ModificationPoly> {
        let path_s = input["param"].as_str()?;
        let path = path_s.split("/").map(
            |str| str.to_string()
        ).collect();
        if !input["coeff"].is_array() {
            return None;
        }
        let is_int = input["integral"].as_bool().unwrap_or(false);
        let coefficients : Vec<f64> = input["coeff"].members().filter_map(
            |i| i.as_f64()
        ).rev().collect();
        Some(
            ModificationPoly { param_path: path, coefficients: coefficients, to_int : is_int }
        )
    }

    fn modify(&self, input : &mut JsonValue, time : f64)  {
        set_json_param(
            input, 
            &self.param_path, 
            self.evaluate(time)
        )
    }

    fn evaluate(&self, time : f64) -> JsonValue {
        let mut acc = 0.0;
        let mut curr_pow = 1.0;
        for coeff in self.coefficients.iter() {
            acc += coeff * curr_pow;
            curr_pow *= time; 
        }
        if self.to_int {
            JsonValue::from(acc.round() as i64)
        } else {
            JsonValue::from(acc)
        }
    }
}

pub fn make_frames(input : &JsonValue) -> std::io::Result<Vec<JsonValue>> {
    // Make a number of frames from a base image and interpolated parameters
    let base = &input["base"];
    let frame_count = input["frames_count"].as_usize().unwrap_or(60);
    let per_frame = 1.0 / (frame_count as f64);
    let mods = Modifications::from_json(&input["mods"])?;
    // The per frame modifications
    let frames = (0..frame_count).map(
        |i| {
            let mut frame = base.clone();
            let time = i as f64 * per_frame; 
            for modification in mods.mods.iter() {
                modification.modify(&mut frame, time);
            }
            frame
        }
    ).collect();
    Ok(frames)
}