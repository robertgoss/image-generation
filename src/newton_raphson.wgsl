struct DataBuf {
    data: array<u32>,
}

struct NewtonRaphson {
    res_x:  u32,
    res_y:  u32,
    centre_x : f32,
    centre_y : f32,
    scale : f32,
    converged_sq : f32,
    coeff : array<f32, 16>,
    coeff_det : array<f32, 16>,
    max_iterations : u32
}


fn converged(pt_diff : vec2<f32>) -> bool {
  return pt_diff.x * pt_diff.x + pt_diff.y * pt_diff.y < nr.converged_sq;
}

fn comp_mult(a : vec2<f32>, b : vec2<f32>) -> vec2<f32> {
    var real = a.x * b.x - a.y * b.y;
    var img = a.x *b.y + a.y * b.x;
    return vec2<f32>(real, img);
}

fn comp_inv(a: vec2<f32>) -> vec2<f32> {
  var form = a.x*a.x + a.y*a.y;
  return vec2<f32>(a.x / form, -a.y / form);
}

fn eval(z : vec2<f32>) -> vec2<f32> {
   var sum = vec2<f32>(0.0, 0.0);
   var pow = vec2<f32>(1.0, 0.0);
   for(var i = 0; i < 12; i++) {
      // Avoid issue if pow overflows to NaN
      if(nr.coeff[i] != 0.0) {
          sum += nr.coeff[i] * pow;
      }
      pow = comp_mult(z, pow);
   }
   return sum;
}

fn eval_det(z : vec2<f32>) -> vec2<f32> {
    var sum = vec2<f32>(0.0, 0.0);
    var pow = vec2<f32>(1.0, 0.0);
    for(var i = 0; i < 16; i++) {
       // Avoid issue if pow overflows to NaN
       if(nr.coeff_det[i] != 0.0) {
          sum += nr.coeff_det[i] * pow;
       }
       pow = comp_mult(z, pow);
    }
    return sum;
}


fn initial_point(id : vec3<u32>) -> vec2<f32> {
   let base_x = nr.centre_x - (f32(nr.res_x) / (2.0 / nr.scale));
   let base_y = nr.centre_y - (f32(nr.res_y) / (2.0 / nr.scale));
   var x = base_x + (f32(id.x) * nr.scale);
   var y = base_y + (f32(id.y) * nr.scale);
   return vec2<f32>(x, y);
}

fn grey_scale_colour(val : u32) -> u32 {
    var grey : u32 = val & 255u;
    var r : u32 = grey;
    var b : u32 = grey * 256u;
    var g : u32 = grey * (256u * 256u);
    return r + g + b;
}

@group(0)
@binding(0)
var<storage, read_write> res: DataBuf;

@group(0)
@binding(1)
var<storage, read> nr: NewtonRaphson;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pt : vec2<f32> = initial_point(global_id);
    var val : u32 = 64u;
    for(var i=0u; i < nr.max_iterations; i++) {
       var ev = eval(pt);
       if(converged(ev)) {
         val = i;
         break;
       }
       var diff = eval_det(pt);
       if(converged(diff)) {
          val = nr.max_iterations;
          break;
      }
       pt = pt - comp_mult(ev, comp_inv(diff));
    }
    val = (val * 255u) / nr.max_iterations;
    // Output the value as grey scale colours
    var index = global_id.x * nr.res_y + global_id.y;
    res.data[index] = grey_scale_colour(val);
}