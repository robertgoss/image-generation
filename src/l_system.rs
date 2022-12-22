// Make a lindenmeyer system that can render basic shapes
//
// This has a set of symbol re-writing rules which replace a 
// symbol with a string of symbols.
//
// It takes a starting axiom and replaces symbols to a fixed depth.
//
// The final result is then drawn a a logo program

use std::{io::{Error, ErrorKind}, collections::HashMap, hash::Hash, cmp::{max_by, min_by, Ordering}, f64::consts::PI, fs::File};

use cgmath::{Point2, vec2, point2, Matrix3, point3, Point3, Rad, EuclideanSpace, VectorSpace};
use json::JsonValue;
use image::{RgbImage, Rgb};

use imageproc::drawing::{draw_antialiased_line_segment_mut};
use imageproc::pixelops;

use crate::ray;

// Instructions to control a drawing turtle
#[derive(Clone)]
enum TurtleRule {
    TurnX(f64),
    TurnY(f64),
    TurnZ(f64),
    PushStack,
    PopStack,
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
            "-" => Some(TurtleRule::TurnZ(-turn_rad)),
            "+" => Some(TurtleRule::TurnZ(turn_rad)),
            "^" => Some(TurtleRule::TurnY(-turn_rad)),
            "&" => Some(TurtleRule::TurnY(turn_rad)),
            "/" => Some(TurtleRule::TurnX(-turn_rad)),
            "\\" => Some(TurtleRule::TurnX(turn_rad)),
            "|" => Some(TurtleRule::TurnZ(PI)),
            "[" => Some(TurtleRule::PushStack),
            "]" => Some(TurtleRule::PopStack),
            _  => None
        }
    }
}

fn cmp_d(a : &f64, b : &f64) -> Ordering {
    a.partial_cmp(b).unwrap()
}

impl LogoProgram {
    fn render(&self, res_x : usize, res_y : usize) -> RgbImage {
        let lines = self.make_lines_2d();
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
        let m_size = max_by(size.x, size.y, cmp_d);
        let scale_x = res_x as f64 / m_size;
        let scale_y = res_y as f64 / m_size;
        let pic_size = vec2(m_size, m_size);
        let base = centre - (pic_size * 0.5);
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

    fn make_lines_2d(&self) -> Vec<(Point2<f64>, Point2<f64>)> {
        let mut pos = point2(0.0, 0.0);
        let mut angle = -PI / 2.0;
        let mut lines = Vec::new();
        let mut stack = Vec::new();
        for instruction in self.instructions.iter() {
            match instruction {
                TurtleRule::TurnX(a_delta) => angle += a_delta,
                TurtleRule::TurnY(a_delta) => angle += a_delta,
                TurtleRule::TurnZ(a_delta) => angle += a_delta,
                TurtleRule::Forward(dist) => {
                    pos += vec2(dist * angle.cos(), dist * angle.sin());
                },
                TurtleRule::ForwardDraw(dist) => {
                    let start = pos;
                    pos += vec2(dist * angle.cos(), dist * angle.sin());
                    lines.push((start, pos));
                },
                TurtleRule::PopStack => {
                    if let Some((p, a)) = stack.pop() {
                        pos = p;
                        angle = a;
                    }
                }
                TurtleRule::PushStack => {
                    stack.push((pos, angle));
                }
            }
        }
        lines
    }

    fn lengths(&self) -> f64 {
        self.instructions.iter().filter_map(
            |ins| match ins {
                TurtleRule::ForwardDraw(l) => Some(*l),
                _ => None
            }
        ).next().unwrap()
    }

    fn render_3d(&self, res_x : usize, res_y : usize) -> Result<RgbImage, json::JsonError> {
        let lines = self.make_lines_3d();
        let lengths = self.lengths();
        // Make json document to render
        let mut new_scene = JsonValue::new_object();
        new_scene.insert("resolution_x", res_x)?;
        new_scene.insert("resolution_y", res_y)?;
        new_scene.insert("depth", 4)?;
        new_scene.insert("algorithm", "raytrace")?;
        // Add a camera
        let mut camera = JsonValue::new_object();
        camera.insert("x", 0.0)?;
        camera.insert("y", 0.0)?;
        camera.insert("z", 0.0)?;
        camera.insert("dir_x", 1.0)?;
        camera.insert("dir_y", 0.5)?;
        camera.insert("dir_z", -1.0)?;
        camera.insert("automatic", true)?;
        new_scene.insert("camera", camera)?;
        // Add the two bits of geometry
        let mut geom = JsonValue::new_object();
        let mut line = JsonValue::new_object();
        line.insert("radius", 0.01)?;
        line.insert("length", lengths)?;
        line.insert("type", "cylinder")?;
        geom.insert("line", line)?;
        let mut ball = JsonValue::new_object();
        ball.insert("radius", 0.01)?;
        ball.insert("type", "sphere")?;
        geom.insert("ball", ball)?;
        new_scene.insert("geometries", geom)?;
        // Add a red material
        let mut materials = JsonValue::new_object();
        let mut red = JsonValue::new_object();
        red.insert("type", "colour")?;
        red.insert("r", 200)?;
        red.insert("g", 50)?;
        red.insert("b", 50)?;
        red.insert("reflect", 0.3)?;
        materials.insert("red", red)?;
        new_scene.insert("materials", materials)?;
        // Add entities for each line
        let mut entities = JsonValue::new_array();
        let mut sball_entity = JsonValue::new_object();
        let pt = lines.first().unwrap().0;
        sball_entity.insert("geom", "ball")?;
        sball_entity.insert("x", pt.x)?;
        sball_entity.insert("y", pt.y)?;
        sball_entity.insert("z", pt.z)?;
        sball_entity.insert("mat", "red")?;
        entities.push(sball_entity)?;
        for (start, end) in lines {
            let mid = start.to_vec().lerp(end.to_vec(), 0.5);
            let diff = end - start;
            let mut line_entity = JsonValue::new_object();
            line_entity.insert("geom", "line")?;
            line_entity.insert("x", mid.x)?;
            line_entity.insert("y", mid.y)?;
            line_entity.insert("z", mid.z)?;
            line_entity.insert("dir_x", diff.x)?;
            line_entity.insert("dir_y", diff.y)?;
            line_entity.insert("dir_z", diff.z)?;
            line_entity.insert("mat", "red")?;
            let mut ball_entity = JsonValue::new_object();
            ball_entity.insert("geom", "ball")?;
            ball_entity.insert("x", end.x)?;
            ball_entity.insert("y", end.y)?;
            ball_entity.insert("z", end.z)?;
            ball_entity.insert("mat", "red")?;
            entities.push(line_entity)?;
            entities.push(ball_entity)?;
        }
        new_scene.insert("entities", entities)?;
        ray::generate(&new_scene).map_err(
            |_| json::JsonError::UnexpectedEndOfJson
        )
    }

    fn make_lines_3d(&self) -> Vec<(Point3<f64>, Point3<f64>)> {
        let mut coords : Matrix3<f64> = Matrix3::from_scale(1.0);
        let mut pos = point3(0.0, 0.0, 0.0);
        let mut stack : Vec<(Point3<f64>, Matrix3<f64>)> = Vec::new();
        let mut lines = Vec::new();
        for instruction in self.instructions.iter() {
            match instruction {
                TurtleRule::TurnX(x_delta) => coords = coords * Matrix3::from_angle_x(Rad(*x_delta)),
                TurtleRule::TurnY(y_delta) => coords = coords * Matrix3::from_angle_y(Rad(*y_delta)),
                TurtleRule::TurnZ(z_delta) => coords = coords * Matrix3::from_angle_z(Rad(*z_delta)),
                TurtleRule::Forward(dist) => {
                    pos += coords.x * (*dist);
                },
                TurtleRule::ForwardDraw(dist) => {
                    let start = pos;
                    pos += coords.x * (*dist);
                    lines.push((start, pos));
                }
                TurtleRule::PopStack => {
                    if let Some((p, c)) = stack.pop() {
                        pos = p;
                        coords = c;
                    }
                }
                TurtleRule::PushStack => {
                    stack.push((pos, coords));
                }
            };
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
    dim3d : bool,
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
        let render_dim = input["render_3d"].as_bool().unwrap_or(false);
        let axiom = parse_symbol_rule(&input["axiom"]).ok_or(
            Error::new(ErrorKind::InvalidData, "Invalid Axiom")
        )?;
        Ok(
            LSystemSimulation {
                system: system, 
                resolution: (res_x, res_y), 
                depth: depth,
                dim3d : render_dim,
                axiom: axiom
            }
        )
    }


    fn make_image(&self) -> RgbImage {
        let symbols = self.system.simulate(self.axiom.clone(), self.depth);
        println!("Derivation has {} symbols", symbols.len());
        let logo = self.system.draw(&symbols);
        if self.dim3d {
            logo.render_3d(self.resolution.0, self.resolution.1).unwrap()
        } else {
            logo.render(self.resolution.0, self.resolution.1)
        }
    }
}

pub fn generate(input : &JsonValue) -> std::io::Result<RgbImage> {
    println!("Generating L-System image");
    let ls = LSystemSimulation::from_json(input)?;
    Ok(ls.make_image())
}