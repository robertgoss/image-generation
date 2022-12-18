// Make a lindenmeyer system that can render basic shapes
//
// This has a set of symbol re-writing rules which replace a 
// symbol with a string of symbols.
//
// It takes a starting axiom and replaces symbols to a fixed depth.
//
// The final result is then drawn a a logo program

use std::{io::{Error, ErrorKind}, collections::HashMap, hash::Hash, cmp::{max_by, min_by, Ordering}, f64::consts::PI};

use cgmath::{Point2, vec2, point2};
use json::JsonValue;
use image::{RgbImage, Rgb};

use imageproc::drawing::{draw_antialiased_line_segment_mut};
use imageproc::pixelops;

// Instructions to control a drawing turtle
#[derive(Clone)]
enum TurtleRule {
    Turn(f64),
    ForwardDraw(f64),
    Forward(f64)
}

// Instructions for a turtle to draw a shape.
struct LogoProgram {
    instructions : Vec<TurtleRule>
}

impl TurtleRule {
    // Make from a symbol string with fixed forward and turn ditances
    fn from_string(f_dist : f64, turn : f64, input : &str) -> Option<TurtleRule> {
        let turn_rad = (turn * PI) / 180.0;
        match input {
            "F" => Some(TurtleRule::ForwardDraw(f_dist)),
            "f" => Some(TurtleRule::Forward(f_dist)),
            "-" => Some(TurtleRule::Turn(-turn_rad)),
            "+" => Some(TurtleRule::Turn(turn_rad)),
            _  => None
        }
    }
}

fn cmp_d(a : &f64, b : &f64) -> Ordering {
    a.partial_cmp(b).unwrap()
}

impl LogoProgram {
    fn render(&self, res_x : usize, res_y : usize) -> RgbImage {
        let lines = self.make_lines();
        let max_x = lines.iter().map(
            |line| max_by(line.0.x, line.1.x, cmp_d)
        ).max_by(cmp_d).unwrap();
        let min_x = lines.iter().map(
            |line| min_by(line.0.x, line.1.x, cmp_d)
        ).min_by(cmp_d).unwrap();
        let max_y = lines.iter().map(
            |line| max_by(line.0.y, line.1.y, cmp_d)
        ).max_by(cmp_d).unwrap();
        let min_y = lines.iter().map(
            |line| min_by(line.0.y, line.1.y, cmp_d)
        ).min_by(cmp_d).unwrap();
        let centre = point2((max_x+min_x) / 2.0, (max_y+min_y) / 2.0);
        let size = vec2((max_x-min_x) *1.06, (max_y-min_y) *1.06);
        let scale_x = res_x as f64 / size.x;
        let scale_y = res_y as f64 / size.y;
        let base = centre - (size * 0.5);
        let mut image = RgbImage::new(res_x as u32, res_y as u32);
        for (start, end) in lines {
            let rel_start_pos = start - base;
            let scale_start = point2(scale_x * rel_start_pos.x, scale_y * rel_start_pos.y);
            let rel_end_pos = end - base;
            let scale_end = point2(scale_x * rel_end_pos.x, scale_y * rel_end_pos.y);
            draw_antialiased_line_segment_mut(
                &mut image, 
                (scale_start.x as i32, scale_start.y as i32), 
                (scale_end.x as i32, scale_end.y as i32), 
                Rgb([255, 255, 255]),
                pixelops::interpolate
            )
        };
        image
        
    }

    fn make_lines(&self) -> Vec<(Point2<f64>, Point2<f64>)> {
        let mut pos = point2(0.0, 0.0);
        let mut angle = 0.0;
        let mut lines = Vec::new();
        for instruction in self.instructions.iter() {
            match instruction {
                TurtleRule::Turn(a_delta) => angle += a_delta,
                TurtleRule::Forward(dist) => {
                    pos += vec2(dist * angle.cos(), dist * angle.sin());
                },
                TurtleRule::ForwardDraw(dist) => {
                    let start = pos;
                    pos += vec2(dist * angle.cos(), dist * angle.sin());
                    lines.push((start, pos));
                },
            }
        }
        lines
    }
}

// An L system has a series of replacement rules for each symbol
//
// And rules to draw each symbol
struct LSystem<Symbols> {
    rules : HashMap<Symbols, Vec<Symbols>>,
    draw_instructions : HashMap<Symbols, Vec<TurtleRule>>
}

fn parse_symbol_rule(input : &JsonValue) -> Option<Vec<String>> {
    let string = input.as_str()?;
    let symbols = string.split(",").map(
        |s| s.to_string()
    ).collect();
    Some(symbols)
}

fn parse_draw_rule(f_dist : f64, turn : f64, input : &JsonValue) -> Option<Vec<TurtleRule>> {
    let string = input.as_str()?;
    let symbols = string.split(",").filter_map(
        |s| TurtleRule::from_string(f_dist, turn, s)
    ).collect();
    Some(symbols)
}

impl<Symbol> LSystem<Symbol>
  where Symbol : Eq, Symbol : Hash, Symbol : Clone
{
    fn simulate(&self, axiom : Vec<Symbol>, depth : usize) -> Vec<Symbol> {
        if depth == 0 {
            axiom
        } else {
            let replaced = axiom.into_iter().flat_map(
                |symbol| self.rules.get(&symbol).unwrap_or(
                    &vec!(symbol)
                ).iter().cloned().collect::<Vec<_>>()
            ).collect();
            self.simulate(replaced, depth-1)
        }
    } 

    fn draw(&self, word : &Vec<Symbol>) -> LogoProgram {
        let draw_instructions = word.into_iter().flat_map(
            |symbol| self.draw_instructions.get(symbol).unwrap_or(
                &vec!()
            ).iter().cloned().collect::<Vec<_>>()
        ).collect();
        LogoProgram { instructions: draw_instructions }
    }
}

impl LSystem<String> {
    fn from_json(input : &JsonValue) -> Option<LSystem<String>> {
        let forward_dist = input["forward"].as_f64()?;
        let turn_angle = input["turn"].as_f64()?;
        let rules = input["rules"].entries().filter_map(
            |(name, in_rule)| {
                parse_symbol_rule(in_rule).map(|rule| (name.to_string(), rule))
            }
        ).collect();
        let draw = input["draw_instructions"].entries().filter_map(
            |(name, in_rule)| {
                parse_draw_rule(forward_dist, turn_angle, in_rule).map(
                    |rule| (name.to_string(), rule)
                )
            }
        ).collect();
        Some(
            LSystem { rules: rules, draw_instructions: draw }
        )
    }
}



// A simulation of a LSystem with a general system an axion and a depth to simulate
struct LSystemSimulation {
    system : LSystem<String>,
    resolution : (usize, usize),
    depth : usize,
    axiom : Vec<String>
}

impl LSystemSimulation {
    fn from_json(input : &JsonValue) -> std::io::Result<LSystemSimulation> {
        let system = LSystem::from_json(input).ok_or(
            Error::new(ErrorKind::InvalidData, "Invalid L System")
        )?;
        let res_x = input["resolution_x"].as_usize().unwrap_or(1024);
        let res_y = input["resolution_y"].as_usize().unwrap_or(1024);
        let depth = input["depth"].as_usize().unwrap_or(6);
        let axiom = parse_symbol_rule(&input["axiom"]).ok_or(
            Error::new(ErrorKind::InvalidData, "Invalid Axiom")
        )?;
        Ok(
            LSystemSimulation {
                system: system, 
                resolution: (res_x, res_y), 
                depth: depth,
                axiom: axiom
            }
        )
    }


    fn make_image(&self) -> RgbImage {
        let symbols = self.system.simulate(self.axiom.clone(), self.depth);
        println!("Derivation has {} symbols", symbols.len());
        let logo = self.system.draw(&symbols);
        logo.render(self.resolution.0, self.resolution.1)
    }
}

pub fn generate(input : &JsonValue) -> std::io::Result<RgbImage> {
    println!("Generating L-System image");
    let ls = LSystemSimulation::from_json(input)?;
    Ok(ls.make_image())
}