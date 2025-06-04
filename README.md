# MEBVH: Memory Efficient BVHs for Ray Tracing with Vectorized Unit

## Environment Setup
```bash
git clone git@github.com:nycu-caslab/MEBVH.git
git checkout MEBVH
mkdir build
cd build
cmake ..
make 
```

## Usage
Prepare scenes
```bash
git clone https://github.com/mmp/pbrt-v4-scenes.git
```
In `build` Run (Modify parameter if necessary)
```bash
# render with 6-wide BVH
./pbrt ~/pbrt-v4-scenes/house/scene-v4.pbrt --nthreads 12 --spp 8 --stats --wbvh

# render with MEBVH 
./pbrt ~/pbrt-v4-scenes/house/scene-v4.pbrt --nthreads 12 --spp 8 --stats --mebvh
```

