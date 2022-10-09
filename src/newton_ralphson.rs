// Make an image out of the newton ralphson root finding for a complex polynomial
//
// For each pixel track the number of iterations taken to converge and the 
// root converged to use this to determine intensity and colour.

use json::JsonValue;

struct ComplexPolynomial {
    coefficients : Vec<f32>
}

struct NewtonRaphson {
    polynomial : ComplexPolynomial,
    resolution : (usize, usize),
    centre : (f32, f32),
    scale : f32
}

pub fn generate(input : &JsonValue) -> std::io::Result<()> {
    Ok(())
}