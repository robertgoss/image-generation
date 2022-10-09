// Make an image out of the newton ralphson root finding for a complex polynomial
//
// For each pixel track the number of iterations taken to converge and the 
// root converged to use this to determine intensity and colour.

use json::JsonValue;

struct ComplexPolynomial {
    coefficients : Vec<f32>
}

impl ComplexPolynomial {
    fn from_json(input : &JsonValue) -> std::io::Result<ComplexPolynomial> {
        let coefficients : Vec<f32> = input.members().filter_map(|i| i.as_f32()).collect();
        Ok(ComplexPolynomial { coefficients: coefficients })
    }
}

struct NewtonRaphson {
    polynomial : ComplexPolynomial,
    resolution : (usize, usize),
    centre : (f32, f32),
    scale : f32
}

impl NewtonRaphson {
    fn from_json(input : &JsonValue) -> std::io::Result<NewtonRaphson> {
        let polynomial = ComplexPolynomial::from_json(&input["polynomial"])?;
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let centre_x = input["centre_x"].as_f32().unwrap_or(0.0);
        let centre_y = input["centre_y"].as_f32().unwrap_or(0.0);
        let scale = input["scale"].as_f32().unwrap_or(1.0 / 1024.0);
        Ok(NewtonRaphson {
            polynomial : polynomial,
            resolution : (res_x, res_y),
            centre : (centre_x, centre_y),
            scale : scale
        })
    }
}

pub fn generate(input : &JsonValue) -> std::io::Result<()> {
    let nr = NewtonRaphson::from_json(input)?;
    Ok(())
}