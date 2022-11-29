// Load a json file that specifies which image we are going to make and
// set's it's parameters

use std::io::{Error, Read, ErrorKind};
use std::fs::File;
use std::env;

mod newton_raphson;
mod ray;

fn main() -> std::io::Result<()> {
    // Get file to use else default
    let filename = env::args().nth(1).unwrap_or("input.json".to_string());
    println!("Loading input file: {}", filename);
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    println!("Parsing input file");
    let input = json::parse(&contents).map_err(
        |_| Error::new(ErrorKind::InvalidData, "Couldn't parse input")
    )?;
    let algorithm = input["algorithm"].as_str().unwrap_or("none");
    match algorithm {
        "newton-raphson" => newton_raphson::generate(&input),
        "raytrace" => ray::generate(&input),
        _ => Err(Error::new(ErrorKind::InvalidData, "Unknown algorithm"))
    }
}
