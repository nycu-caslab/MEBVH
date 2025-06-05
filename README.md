# MEBVH: Memory Efficient BVHs for Ray Tracing with Vectorized Unit
Ray tracing is widely used in the film and game industries to produce highly realistic images through physically based light simulation. However, identifying ray–object intersections remains one of its most computationally intensive tasks. Bounding Volume Hierarchies (BVHs) are commonly employed to accelerate this process by culling unnecessary intersection checks. Despite their effectiveness, BVHs introduce additional memory overhead and are frequently bottlenecked by memory latency.

We propose MEBVH, a memory-efficient BVH scheme that exploits wide BVHs and vector units. MEBVH leverages the spatial locality of wide nodes to compress child coordinates into a more compact representation, reducing memory consumption by 67%. This smaller node structure eases bandwidth requirements and fits more nodes into cache. In addition, we propose a cache- and vector-friendly memory layout that improves memory access efficiency and facilitates fast vectorized operations.

Experimental results show that ray tracing benefits significantly from this compact node structure. MEBVH lowers memory access by 33% and achieves a 1.38X geometric mean speedup over the baseline.


## Environment Setup
```bash
git clone git@github.com:nycu-caslab/MEBVH.git
cd MEBVH
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

