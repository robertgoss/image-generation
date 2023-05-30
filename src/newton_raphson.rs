// Make an image out of the newton ralphson root finding for a complex polynomial
//
// For each pixel track the number of iterations taken to converge and the 
// root converged to use this to determine intensity and colour.

use json::JsonValue;
use num::complex::{Complex};
use image::{Rgb,RgbImage,GrayImage,Luma};

use std::cmp::Ordering;

use std::f64::consts::PI;

use std::{io::{Error, ErrorKind}, collections::HashMap};
use std::io::{Read, repeat};

use wgpu::util::DeviceExt;


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

struct Roots {
    tol_sqr : f64,
    roots : Vec<Complex<f64>>,
    colours : HashMap<u8, Rgb<u8>>,
    white : Rgb<u8>
}

fn root_angle(principle : &Complex<f64>, root : &Complex<f64>) -> f64 {
    let ang = root.arg() - principle.arg();
    if ang.abs() < 1.0e-8 {
        0.0
    } else {
        if ang < 0.0 {
            ang + (2.0*PI)
        } else {
            ang
        }
    }
}

fn root_compare(principle : &Complex<f64>, root1 : &Complex<f64>, root2 : &Complex<f64>) -> Ordering {
    root_angle(principle, root1).partial_cmp(
        &root_angle(principle, root2)
    ).unwrap()
} 

impl Roots {
    fn new(eps : f64) -> Roots {
        Roots { 
            tol_sqr: eps * eps, 
            roots: Vec::new(), 
            colours: HashMap::new(),
            white : Rgb([255,255,255])
         }
    }

    fn add_root(&mut self, root : &Complex<f64>) -> u8 {
        for (i, z) in self.roots.iter().enumerate() {
            if (root - z).norm_sqr() < self.tol_sqr {
                return i as u8;
            }
        };
        let index = self.roots.len();
        self.roots.push(*root);
        index as u8
    }

    fn root_colour(&self, index : u8) -> &Rgb<u8> {
        self.colours.get(&index).unwrap_or(&self.white)
    }

    fn principle_root(&self) -> Complex<f64> {
        *self.roots.iter().max_by(
            |&root1, &root2| root1.im.partial_cmp(&root2.im).unwrap()
        ).unwrap()
    }

    fn make_colours(&mut self) {
        if self.roots.len() == 0 {
            return;
        }
        // Make colours for each root when we have them all
        let mut indices = Vec::from_iter(0..self.roots.len());
        let p_root = self.principle_root();
        indices.sort_by(
            |i, j| root_compare(&p_root, &self.roots[*i], &self.roots[*j]) 
        );
        let root_num = self.roots.len();
        for (i, root_i) in indices.iter().enumerate() {
            let hue : f64 = 360.0 * i as f64 / root_num as f64;
            self.colours.insert(*root_i as u8, make_base_rgb(hue));
        }
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
    use_colour : bool,
    use_gpu : bool
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
        let use_gpu = input["gpu"].as_bool().unwrap_or(false);
        Ok(NewtonRaphson {
            polynomial,
            differential,
            resolution : (res_x, res_y),
            centre : (centre_x, centre_y),
            scale,
            max_iterations,
            convergence,
            use_colour,
            use_gpu
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
        if self.use_gpu && !self.use_colour {
            pollster::block_on(self.make_image_gpu())
        } else {
            self.make_image_cpu()
        }
    }

    async fn make_image_gpu(&self) -> RgbImage {
        // Do the calculation on the gpu
        let mut img = RgbImage::new(
            self.resolution.0 as u32,
            self.resolution.1 as u32
        );
        // Setup the gpu renderer
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: None
            },
            None, // Trace path
        ).await.unwrap();
        // Load the compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("newton_raphson.wgsl").into()),
        });
        // Setup the buffer that the shader will output to with
        //
        // Input buffer that we will work with and copy out of
        //
        // NB can we just have an output buffer?
        let input = vec![0 as u8; img.len()];
        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &input,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Specify the layout of buffers to the shader
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        // Actually bind the input
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(self.resolution.0 as u32, self.resolution.1 as u32, 1);
        encoder.copy_buffer_to_buffer(&input_buf, 0, &output_buf, 0, input.len() as u64);
        queue.submit(Some(encoder.finish()));


        img
    }

    fn make_image_cpu(&self) -> RgbImage {
        let mut img = RgbImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        let start_x = self.centre.0 - (self.resolution.0 as f64 / (2.0 / self.scale));
        let start_y = self.centre.1 - (self.resolution.1 as f64 / (2.0 / self.scale));
        // Make an image of the iteration count - and record the roots found so we can
        // colour them - this is needed to make the colourings stable
        // and the index of the colour
        let mut depth_img = GrayImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        let mut root_img = GrayImage::new(
            self.resolution.0 as u32, 
            self.resolution.1 as u32
        );
        let mut roots = Roots::new(self.convergence * 4.0);
        let mut max_iter : usize = 0;
        for i in 0..(self.resolution.0 as u32) {
            for j in 0..(self.resolution.1 as u32) {
                let x = start_x + (i as f64 * self.scale);
                let y = start_y + (j as f64 * self.scale);
                let (n,root) = self.converge(Complex::new(x, y));
                if self.use_colour {
                    if let Some(root_val) = root {
                        let root_index = roots.add_root(&root_val);
                        root_img.put_pixel(i, j, Luma([root_index]));
                    }
                }
                if n > max_iter {
                    max_iter = n;
                }
                depth_img.put_pixel(i, j, Luma([n as u8]));
            }
        }
        // Now we have the roots we can stabley colour them
        roots.make_colours();
        let white : Rgb<u8> = Rgb([255,255,255]);
        // Make the colour image
        for i in 0..(self.resolution.0 as u32) {
            for j in 0..(self.resolution.1 as u32) {
                let n = depth_img.get_pixel(i, j).0[0] as usize;
                let val = (n * 255) / max_iter;
                let root_colour = if self.use_colour {
                    roots.root_colour(root_img.get_pixel(i,j).0[0])
                } else {
                    &white
                };
                img.put_pixel(i, j, scale_colour(val as u8, root_colour));
            }
        }
        img
    }

}

pub fn generate(input : &JsonValue) -> std::io::Result<RgbImage> {
    println!("Generating newton raphson image");
    let nr = NewtonRaphson::from_json(input)?;
    Ok(nr.make_image())
}