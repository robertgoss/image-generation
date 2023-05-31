struct DataBuf {
    data: array<u32>,
}

struct NewtonRaphson {
    res_x:  u32,
    res_y:  u32
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
    var index = global_id.x * nr.res_y + global_id.y;
    var val = global_id.x;
    // Output the value as grey scale colours
    var grey : u32 = val & 255u;
    var r : u32 = grey;
    var b : u32 = grey * 256u;
    var g : u32 = grey * (256u * 256u);
    res.data[index] = r + g + b;
}