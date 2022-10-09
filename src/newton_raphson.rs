// Make an image out of the newton ralphson root finding for a complex polynomial
//
// For each pixel track the number of iterations taken to converge and the 
// root converged to use this to determine intensity and colour.

use json::JsonValue;
use num::complex::{Complex};
use image::GrayImage;

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

struct NewtonRaphson {
    polynomial : ComplexPolynomial,
    differential : ComplexPolynomial,
    resolution : (usize, usize),
    centre : (f64, f64),
    scale : f64,
    max_iterations : usize,
    convergence : f64
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
        Ok(NewtonRaphson {
            polynomial : polynomial,
            differential : differential,
            resolution : (res_x, res_y),
            centre : (centre_x, centre_y),
            scale : scale,
            max_iterations : max_iterations,
            convergence : convergence
        })
    }

    fn converge(&self, initial : Complex<f64>) -> usize {
        let eps : f64 = self.convergence * self.convergence;
        let mut z : Complex<f64> = initial;
        for i in 0..self.max_iterations {
            let eval = self.polynomial.evaluate(z);
            if eval.norm_sqr() < eps {
                return i;
            }
            let diff = self.differential.evaluate(z);
            if diff.norm_sqr() < eps {
                break;
            }
            z = z - (eval / diff);
        }
        self.max_iterations
    }

    fn make_image(&self) -> GrayImage {
        let mut img = GrayImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        let start_x = self.centre.0 - (self.resolution.0 as f64 / (2.0 / self.scale));
        let start_y = self.centre.1 - (self.resolution.1 as f64 / (2.0 / self.scale));
        for i in 0..self.resolution.0 {
            for j in 0..self.resolution.1 {
                let x = start_x + (i as f64 * self.scale);
                let y = start_y + (j as f64 * self.scale);
                let n = self.converge(Complex::new(x, y));
                let val = (n * 255) / self.max_iterations;
                let pixel = image::Luma::from([val as u8]);
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