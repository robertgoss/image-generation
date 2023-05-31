struct DataBuf {
    data: array<u32>,
}

struct NewtonRaphson {
    res_x:  u32,
    res_y:  u32,
    centre_x : f32,
    centre_y : f32,
    scale : f32,
    converged_sq : f32
}


fn converged(pt_diff : vec2<f32>) -> bool {
  return pt_diff.x * pt_diff.x + pt_diff.y * pt_diff.y < nr.converged_sq;
}

fn iterate(pt : vec2<f32>) -> vec2<f32> {
   return pt * 0.5;
}

fn point(id : vec3<u32>) -> vec2<f32> {
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
    var pt : vec2<f32> = point(global_id);
    var val : u32 = 0u;
    for(; val < 256u; val++) {
       var new_pt = iterate(pt);
       if (converged(pt - new_pt)) {
          break;
       }
       pt = new_pt;
    }
    // Output the value as grey scale colours
    var index = global_id.x * nr.res_y + global_id.y;
    res.data[index] = grey_scale_colour(val);
}