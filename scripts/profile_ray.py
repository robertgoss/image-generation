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
    with open("input.json", "w") as file:
        file.write(json.dumps(data))
    time_start = time.time()
    subprocess.run(["target/release/image-generation", "input.json"], capture_output=True, check=True)
    time_end = time.time()
    return time_end - time_start


for i in range(1, 120, 2):
    t = time_res(i)
    print("{}: {}".format(i, t))