import json
import subprocess
import time


# Make scene with progressively more spheres timing the time to calculate
def time_res(num_spheres):
    # Make the raytrace input
    data = {
        "algorithm": "sphere_scene",
        "resolution_x": 512,
        "resolution_y": 512,
        "number": num_spheres,
        "seed": 0
    }
    with open("input.json", "w") as temp_file:
        temp_file.write(json.dumps(data))
    time_start = time.time()
    subprocess.run(["target/release/image-generation", "input.json"], capture_output=True, check=True)
    time_end = time.time()
    return time_end - time_start


with open("profile_ray.txt", "w") as out_file:
    for i in range(0, 120, 5):
        t = time_res(i)
        print("{} out of {}".format(i, 120))
        out_file.write("{}: {}\n".format(i, t))
