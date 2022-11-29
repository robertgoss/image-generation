// Make an image out of the newton ralphson root finding for a complex polynomial
//
// For each pixel track the number of iterations taken to converge and the 
// root converged to use this to determine intensity and colour.

use json::JsonValue;
use num::complex::{Complex};
use image::{Rgb,RgbImage};

use std::io::{Error, ErrorKind};

struct ComplexPolynomial {
    coefficients : Vec<f64>
}

impl ComplexPolynomial {
    fn from_json(input : &JsonValue) -> std::io::Result<ComplexPolynomial> {
        if !input.is_array() {
            return Err(Error::new(ErrorKind::InvalidData, "Missing coefficients"))
        }
        let coefficients : Vec<f64> = input.members().filter_map(|i| i.as_f64()).rev().collect();
        Ok(ComplexPolynomial { coefficients: coefficients })
    }

    fn differentiate(&self) -> ComplexPolynomial {
        ComplexPolynomial { 
            coefficients: self.coefficients.iter().enumerate().skip(1).map(
                |(i, v)| (i as f64) * v
            ).collect() 
        }
    }

    fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    fn evaluate(&self, z : Complex<f64>) -> Complex<f64> {
        let mut acc = Complex::new(0.0, 0.0);
        let mut curr_pow = Complex::new(1.0, 0.0);
        for coeff in self.coefficients.iter() {
            acc += coeff * curr_pow;
            curr_pow *= z; 
        }
        acc
    }
}

fn mod2(val : f64) -> f64 {
    (val / 2.0).fract() * 2.0
}

fn make_base_rgb(hue : f64) -> Rgb<u8> {
    let x_val : f64 = 1.0 - (mod2(hue / 60.0) - 1.0).abs();
    let x = (255.0 * x_val) as u8;
    if hue < 60.0 {
        Rgb([255, x, 0])
    } else if hue < 120.0 {
        Rgb([x, 255, 0])
    }else if hue < 180.0 {
        Rgb([0, 255, x])
    }else if hue < 240.0 {
        Rgb([0, x, 255])
    }else if hue < 300.0 {
        Rgb([x, 0, 255])
    } else {
        Rgb([255, 0, x])
    }
}

// Given a root 
//    if it is in the list return it's colour.
//    if it isnt add it
fn find_root_colour(
    root_colours : &mut Vec<(Complex<f64>, Rgb<u8>)>, 
    root : &Complex<f64>, 
    root_num : usize,
    eps : f64
) -> Rgb<u8> {
    let eps_sq = eps*eps;
    let found = root_colours.iter().find(
        |&(x,_)| (root - x).norm_sqr() < eps_sq
    );
    if found.is_some() {
        return found.unwrap().1;
    }
    let index = root_colours.len();
    let hue : f64 = 360.0 * index as f64 / root_num as f64; 
    let colour = make_base_rgb(hue);
    root_colours.push((*root, colour));
    colour
}

fn scale_colour(val : u8, colour : &Rgb<u8>) -> Rgb<u8> {
    let r = (colour.0[0] as u16 * val as u16) / 255;
    let b = (colour.0[1] as u16 * val as u16) / 255;
    let g = (colour.0[2] as u16 * val as u16) / 255;
    Rgb([r as u8, b as u8, g as u8])
}

struct NewtonRaphson {
    polynomial : ComplexPolynomial,
    differential : ComplexPolynomial,
    resolution : (usize, usize),
    centre : (f64, f64),
    scale : f64,
    max_iterations : usize,
    convergence : f64,
    use_colour : bool
}

impl NewtonRaphson {
    fn from_json(input : &JsonValue) -> std::io::Result<NewtonRaphson> {
        let polynomial = ComplexPolynomial::from_json(&input["polynomial"])?;
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let centre_x = input["centre_x"].as_f64().unwrap_or(0.0);
        let centre_y = input["centre_y"].as_f64().unwrap_or(0.0);
        let size = input["size"].as_f64().unwrap_or(1.0);
        let scale = size / res_x as f64;
        let max_iterations = input["max_iterations"].as_usize().unwrap_or(255);
        let convergence = input["convergence_size"].as_f64().unwrap_or(1e-10);
        let differential = polynomial.differentiate();
        let use_colour = input["colour"].as_bool().unwrap_or(false);
        Ok(NewtonRaphson {
            polynomial : polynomial,
            differential : differential,
            resolution : (res_x, res_y),
            centre : (centre_x, centre_y),
            scale : scale,
            max_iterations : max_iterations,
            convergence : convergence,
            use_colour : use_colour
        })
    }

    fn converge(&self, initial : Complex<f64>) -> (usize, Option<Complex<f64>>) {
        let eps : f64 = self.convergence * self.convergence;
        let mut z : Complex<f64> = initial;
        for i in 0..self.max_iterations {
            let eval = self.polynomial.evaluate(z);
            if eval.norm_sqr() < eps {
                return (i, Some(z));
            }
            let diff = self.differential.evaluate(z);
            if diff.norm_sqr() < eps {
                break;
            }
            z = z - (eval / diff);
        }
        (self.max_iterations, None)
    }

    fn make_image(&self) -> RgbImage {
        let mut img = RgbImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        let start_x = self.centre.0 - (self.resolution.0 as f64 / (2.0 / self.scale));
        let start_y = self.centre.1 - (self.resolution.1 as f64 / (2.0 / self.scale));
        let root_num = self.polynomial.order();
        // Map from roots to colours
        let mut root_colours : Vec<(Complex<f64>, Rgb<u8>)> = Vec::new();
        for i in 0..self.resolution.0 {
            for j in 0..self.resolution.1 {
                let x = start_x + (i as f64 * self.scale);
                let y = start_y + (j as f64 * self.scale);
                let (n,root) = self.converge(Complex::new(x, y));
                let base_colour = match (self.use_colour, root) {
                    (true, Some(z)) => 
                      find_root_colour(
                        &mut root_colours,
                        &z, 
                        root_num,
                        3.0*self.convergence
                      ),
                    _ => Rgb([255,255,255])
                };
                let val = (n * 255) / self.max_iterations;

                let pixel = scale_colour(val as u8, &base_colour);
                img.put_pixel(i as u32, j as u32, pixel);
            }
        }
        img
    }

}

pub fn generate(input : &JsonValue) -> std::io::Result<()> {
    println!("Generating newton raphson image");
    let nr = NewtonRaphson::from_json(input)?;
    let image = nr.make_image();
    image.save("output.png").map_err(
        |_| Error::new(ErrorKind::InvalidData, "Couldn't write image")
    )
}