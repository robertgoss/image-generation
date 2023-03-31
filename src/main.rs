// Load a json file that specifies which image we are going to make and
// set's it's parameters

use std::io::{Error, Read, ErrorKind};
use std::fs::{File, create_dir_all};
use std::env;
use std::path::Path;

use image::RgbImage;

use json::JsonValue;

mod newton_raphson;
mod ray;
mod animation;
mod l_system;
mod sphere_scene;

fn make_directory_for_image(path_str : &str) -> std::io::Result<()> {
    let path = Path::new(path_str);
    if let Some(dir) = path.parent() {
        create_dir_all(dir)
    } else {
        Ok(()) 
    }
}

fn make_image(input : &JsonValue) -> std::io::Result<RgbImage> {
    let algorithm = input["algorithm"].as_str().unwrap_or("none");
    match algorithm {
        "newton-raphson" => newton_raphson::generate(&input),
        "raytrace" => ray::generate(&input),
        "l_system" => l_system::generate(&input),
        "sphere_scene" => sphere_scene::generate(&input),
        _ => Err(Error::new(ErrorKind::InvalidData, "Unknown algorithm"))
    }
}

fn main() -> std::io::Result<()> {
    // Get file to use else default
    let in_filename = env::args().nth(1).unwrap_or("input.json".to_string());
    let out_filename = env::args().nth(2).unwrap_or("output.png".to_string());
    println!("Loading input file: {}", in_filename);
    let mut file = File::open(in_filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    println!("Parsing input file");
    let input = json::parse(&contents).map_err(
        |_| Error::new(ErrorKind::InvalidData, "Couldn't parse input")
    )?;
    let algorithm = input["algorithm"].as_str().unwrap_or("none");
    let out_filename_base = out_filename.strip_suffix(".png").unwrap_or(&out_filename);
    if algorithm == "animation" {
        let frames = animation::make_frames(&input)?;
        for (i,frame) in frames.iter().enumerate() {
            println!("Frame {} of {}", i+1, frames.len());
            let image = make_image(frame)?;
            let out_filename_frame = format!("{}/{}.png",out_filename_base, i);
            make_directory_for_image(&out_filename_frame)?;
            image.save(out_filename_frame).map_err(
                |_| Error::new(ErrorKind::InvalidData, "Couldn't write image")
            )?;
        }

    } else {
        let image = make_image(&input)?;
        println!("Writing output to {}", out_filename);
        make_directory_for_image(&out_filename)?;
        image.save(out_filename).map_err(
            |_| Error::new(ErrorKind::InvalidData, "Couldn't write image")
        )?;
    };
    Ok(())
}
