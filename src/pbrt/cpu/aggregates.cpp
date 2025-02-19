// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/aggregates.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <tuple>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/BVH", treeBytes);
STAT_RATIO("BVH/Primitives per leaf node", totalPrimitives, totalLeafNodes);
STAT_COUNTER("BVH/Interior nodes", interiorNodes);
STAT_COUNTER("BVH/Leaf nodes", leafNodes);
STAT_PIXEL_COUNTER("BVH/Nodes visited", bvhNodesVisited);

// MortonPrimitive Definition
struct MortonPrimitive {
    int primitiveIndex;
    uint32_t mortonCode;
};

// LBVHTreelet Definition
struct LBVHTreelet {
    size_t startIndex, nPrimitives;
    BVHBuildNode *buildNodes;
};

// BVHAggregate Utility Functions
static void RadixSort(std::vector<MortonPrimitive> *v) {
    std::vector<MortonPrimitive> tempVector(v->size());
    constexpr int bitsPerPass = 6;
    constexpr int nBits = 30;
    static_assert((nBits % bitsPerPass) == 0,
                  "Radix sort bitsPerPass must evenly divide nBits");
    constexpr int nPasses = nBits / bitsPerPass;
    for (int pass = 0; pass < nPasses; ++pass) {
        // Perform one pass of radix sort, sorting _bitsPerPass_ bits
        int lowBit = pass * bitsPerPass;
        // Set in and out vector references for radix sort pass
        std::vector<MortonPrimitive> &in = (pass & 1) ? tempVector : *v;
        std::vector<MortonPrimitive> &out = (pass & 1) ? *v : tempVector;

        // Count number of zero bits in array for current radix sort bit
        constexpr int nBuckets = 1 << bitsPerPass;
        int bucketCount[nBuckets] = {0};
        constexpr int bitMask = (1 << bitsPerPass) - 1;
        for (const MortonPrimitive &mp : in) {
            int bucket = (mp.mortonCode >> lowBit) & bitMask;
            CHECK_GE(bucket, 0);
            CHECK_LT(bucket, nBuckets);
            ++bucketCount[bucket];
        }

        // Compute starting index in output array for each bucket
        int outIndex[nBuckets];
        outIndex[0] = 0;
        for (int i = 1; i < nBuckets; ++i)
            outIndex[i] = outIndex[i - 1] + bucketCount[i - 1];

        // Store sorted values in output array
        for (const MortonPrimitive &mp : in) {
            int bucket = (mp.mortonCode >> lowBit) & bitMask;
            out[outIndex[bucket]++] = mp;
        }
    }
    // Copy final result from _tempVector_, if needed
    if (nPasses & 1)
        std::swap(*v, tempVector);
}

// BVHSplitBucket Definition
struct BVHSplitBucket {
    int count = 0;
    Bounds3f bounds;
};

// BVHPrimitive Definition
struct BVHPrimitive {
    BVHPrimitive() {}
    BVHPrimitive(size_t primitiveIndex, const Bounds3f &bounds)
        : primitiveIndex(primitiveIndex), bounds(bounds) {}
    size_t primitiveIndex;
    Bounds3f bounds;
    // BVHPrimitive Public Methods
    Point3f Centroid() const { return .5f * bounds.pMin + .5f * bounds.pMax; }
};

// BVHBuildNode Definition
struct BVHBuildNode {
    // BVHBuildNode Public Methods
    void InitLeaf(int first, int n, const Bounds3f &b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = nullptr;
        ++leafNodes;
        ++totalLeafNodes;
        totalPrimitives += n;
    }

    void InitInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nPrimitives = 0;
        ++interiorNodes;
    }

    Bounds3f bounds;
    BVHBuildNode *children[2];
    int splitAxis, firstPrimOffset, nPrimitives;
};

// LinearBVHNode Definition
struct alignas(32) LinearBVHNode {
    Bounds3f bounds;
    union {
        int primitivesOffset;   // leaf
        int secondChildOffset;  // interior
    };
    uint16_t nPrimitives;  // 0 -> interior node
    uint8_t axis;          // interior node: xyz
};

// BVHAggregate Method Definitions
BVHAggregate::BVHAggregate(std::vector<Primitive> prims, int maxPrimsInNode,
                           SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)),
      primitives(std::move(prims)),
      splitMethod(splitMethod) {
    CHECK(!primitives.empty());
    // Build BVH from _primitives_
    // Initialize _bvhPrimitives_ array for primitives
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());

    // Build BVH for primitives using _bvhPrimitives_
    // Declare _Allocator_s used for BVH construction
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });

    std::vector<Primitive> orderedPrims(primitives.size());
    BVHBuildNode *root;
    // Build BVH according to selected _splitMethod_
    std::atomic<int> totalNodes{0};
    if (splitMethod == SplitMethod::HLBVH) {
        root = buildHLBVH(alloc, bvhPrimitives, &totalNodes, orderedPrims);
    } else {
        std::atomic<int> orderedPrimsOffset{0};
        root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                              &totalNodes, &orderedPrimsOffset, orderedPrims);
        CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());
    }
    primitives.swap(orderedPrims);

    // Convert BVH into compact representation in _nodes_ array
    bvhPrimitives.resize(0);
    bvhPrimitives.shrink_to_fit();
    LOG_VERBOSE("BVH created with %d nodes for %d primitives (%.2f MB)",
                totalNodes.load(), (int)primitives.size(),
                float(totalNodes.load() * sizeof(LinearBVHNode)) / (1024.f * 1024.f));
    treeBytes += totalNodes * sizeof(LinearBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVH(root, &offset);
    CHECK_EQ(totalNodes.load(), offset);
}

BVHBuildNode *BVHAggregate::buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                           pstd::span<BVHPrimitive> bvhPrimitives,
                                           std::atomic<int> *totalNodes,
                                           std::atomic<int> *orderedPrimsOffset,
                                           std::vector<Primitive> &orderedPrims
                                           ) {
    DCHECK_NE(bvhPrimitives.size(), 0);
    Allocator alloc = threadAllocators.Get();
    BVHBuildNode *node = alloc.new_object<BVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    ++*totalNodes;
    // Compute bounds of all primitives in BVH node
    // printf("depth: %d\n", depth);
    // fflush(stdout);

    Bounds3f bounds;
    for (const auto &prim : bvhPrimitives)
        bounds = Union(bounds, prim.bounds);

    if (bounds.SurfaceArea() == 0 || bvhPrimitives.size() == 1) {
        // Create leaf _BVHBuildNode_
        int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
            int index = bvhPrimitives[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[index];
        }
        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
        return node;

    } else {
        // Compute bound of primitive centroids and choose split dimension _dim_
        Bounds3f centroidBounds;
        for (const auto &prim : bvhPrimitives)
            centroidBounds = Union(centroidBounds, prim.Centroid());
        int dim = centroidBounds.MaxDimension();

        // Partition primitives into two sets and build children
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            // Create leaf _BVHBuildNode_
            int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
            for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                int index = bvhPrimitives[i].primitiveIndex;
                orderedPrims[firstPrimOffset + i] = primitives[index];
            }
            node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
            return node;

        } else {
            int mid = bvhPrimitives.size() / 2;
            // Partition primitives based on _splitMethod_
            switch (splitMethod) {
            case SplitMethod::Middle: {
                // Partition primitives through node's midpoint
                Float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
                auto midIter = std::partition(bvhPrimitives.begin(), bvhPrimitives.end(),
                                              [dim, pmid](const BVHPrimitive &pi) {
                                                  return pi.Centroid()[dim] < pmid;
                                              });
                mid = midIter - bvhPrimitives.begin();
                // For lots of prims with large overlapping bounding boxes, this
                // may fail to partition; in that case do not break and fall through
                // to EqualCounts.
                if (midIter != bvhPrimitives.begin() && midIter != bvhPrimitives.end())
                    break;
            }
            case SplitMethod::EqualCounts: {
                // Partition primitives into equally sized subsets
                mid = bvhPrimitives.size() / 2;
                std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + mid,
                                 bvhPrimitives.end(),
                                 [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                     return a.Centroid()[dim] < b.Centroid()[dim];
                                 });

                break;
            }
            case SplitMethod::SAH:
            default: {
                // Partition primitives using approximate SAH
                if (bvhPrimitives.size() <= 2) {
                    // Partition primitives into equally sized subsets
                    mid = bvhPrimitives.size() / 2;
                    std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + mid,
                                     bvhPrimitives.end(),
                                     [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                         return a.Centroid()[dim] < b.Centroid()[dim];
                                     });

                } else {
                    // Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvhPrimitives) {
                        int b = nBuckets * centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }

                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    Float costs[nSplits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        costs[i] += countBelow * boundBelow.SurfaceArea();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        costs[i - 1] += countAbove * boundAbove.SurfaceArea();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        if (costs[i] < minCost) {
                            minCost = costs[i];
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = bvhPrimitives.size();
                    minCost = 1.f / 2.f + minCost / bounds.SurfaceArea();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (bvhPrimitives.size() > maxPrimsInNode || minCost < leafCost) {
                        auto midIter = std::partition(
                            bvhPrimitives.begin(), bvhPrimitives.end(),
                            [=](const BVHPrimitive &bp) {
                                int b =
                                    nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                                if (b == nBuckets)
                                    b = nBuckets - 1;
                                return b <= minCostSplitBucket;
                            });
                        mid = midIter - bvhPrimitives.begin();
                    } else {
                        // Create leaf _BVHBuildNode_
                        int firstPrimOffset =
                            orderedPrimsOffset->fetch_add(bvhPrimitives.size());
                        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                            int index = bvhPrimitives[i].primitiveIndex;
                            orderedPrims[firstPrimOffset + i] = primitives[index];
                        }
                        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
                        return node;
                    }
                }

                break;
            }
            }

            BVHBuildNode *children[2];
            // Recursively build BVHs for _children_
            if (bvhPrimitives.size() > 128 * 1024) {
                // Recursively build child BVHs in parallel
                ParallelFor(0, 2, [&](int i) {
                    if (i == 0)
                        children[0] = buildRecursive(
                            threadAllocators, bvhPrimitives.subspan(0, mid), totalNodes,
                            orderedPrimsOffset, orderedPrims);
                    else
                        children[1] =
                            buildRecursive(threadAllocators, bvhPrimitives.subspan(mid),
                                           totalNodes, orderedPrimsOffset, orderedPrims);
                });

            } else {
                // Recursively build child BVHs sequentially
                children[0] =
                    buildRecursive(threadAllocators, bvhPrimitives.subspan(0, mid),
                                   totalNodes, orderedPrimsOffset, orderedPrims);
                children[1] =
                    buildRecursive(threadAllocators, bvhPrimitives.subspan(mid),
                                   totalNodes, orderedPrimsOffset, orderedPrims);
            }

            node->InitInterior(dim, children[0], children[1]);
        }
    }

    return node;
}

BVHBuildNode *BVHAggregate::buildHLBVH(Allocator alloc,
                                       const std::vector<BVHPrimitive> &bvhPrimitives,
                                       std::atomic<int> *totalNodes,
                                       std::vector<Primitive> &orderedPrims) {
    // Compute bounding box of all primitive centroids
    Bounds3f bounds;
    for (const BVHPrimitive &prim : bvhPrimitives)
        bounds = Union(bounds, prim.Centroid());

    // Compute Morton indices of primitives
    std::vector<MortonPrimitive> mortonPrims(bvhPrimitives.size());
    ParallelFor(0, bvhPrimitives.size(), [&](int64_t i) {
        // Initialize _mortonPrims[i]_ for _i_th primitive
        constexpr int mortonBits = 10;
        constexpr int mortonScale = 1 << mortonBits;
        mortonPrims[i].primitiveIndex = bvhPrimitives[i].primitiveIndex;
        Vector3f centroidOffset = bounds.Offset(bvhPrimitives[i].Centroid());
        Vector3f offset = centroidOffset * mortonScale;
        mortonPrims[i].mortonCode = EncodeMorton3(offset.x, offset.y, offset.z);
    });

    // Radix sort primitive Morton indices
    RadixSort(&mortonPrims);

    // Create LBVH treelets at bottom of BVH
    // Find intervals of primitives for each treelet
    std::vector<LBVHTreelet> treeletsToBuild;
    for (size_t start = 0, end = 1; end <= mortonPrims.size(); ++end) {
        uint32_t mask = 0b00111111111111000000000000000000;
        if (end == (int)mortonPrims.size() || ((mortonPrims[start].mortonCode & mask) !=
                                               (mortonPrims[end].mortonCode & mask))) {
            // Add entry to _treeletsToBuild_ for this treelet
            size_t nPrimitives = end - start;
            int maxBVHNodes = 2 * nPrimitives - 1;
            BVHBuildNode *nodes = alloc.allocate_object<BVHBuildNode>(maxBVHNodes);
            treeletsToBuild.push_back({start, nPrimitives, nodes});

            start = end;
        }
    }

    // Create LBVHs for treelets in parallel
    std::atomic<int> orderedPrimsOffset(0);
    ParallelFor(0, treeletsToBuild.size(), [&](int i) {
        // Generate _i_th LBVH treelet
        int nodesCreated = 0;
        const int firstBitIndex = 29 - 12;
        LBVHTreelet &tr = treeletsToBuild[i];
        tr.buildNodes = emitLBVH(
            tr.buildNodes, bvhPrimitives, &mortonPrims[tr.startIndex], tr.nPrimitives,
            &nodesCreated, orderedPrims, &orderedPrimsOffset, firstBitIndex);
        *totalNodes += nodesCreated;
    });

    // Create and return SAH BVH from LBVH treelets
    std::vector<BVHBuildNode *> finishedTreelets;
    finishedTreelets.reserve(treeletsToBuild.size());
    for (LBVHTreelet &treelet : treeletsToBuild)
        finishedTreelets.push_back(treelet.buildNodes);
    return buildUpperSAH(alloc, finishedTreelets, 0, finishedTreelets.size(), totalNodes);
}

BVHBuildNode *BVHAggregate::emitLBVH(BVHBuildNode *&buildNodes,
                                     const std::vector<BVHPrimitive> &bvhPrimitives,
                                     MortonPrimitive *mortonPrims, int nPrimitives,
                                     int *totalNodes,
                                     std::vector<Primitive> &orderedPrims,
                                     std::atomic<int> *orderedPrimsOffset, int bitIndex) {
    CHECK_GT(nPrimitives, 0);
    if (bitIndex == -1 || nPrimitives < maxPrimsInNode) {
        // Create and return leaf node of LBVH treelet
        ++*totalNodes;
        BVHBuildNode *node = buildNodes++;
        Bounds3f bounds;
        int firstPrimOffset = orderedPrimsOffset->fetch_add(nPrimitives);
        for (int i = 0; i < nPrimitives; ++i) {
            int primitiveIndex = mortonPrims[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[primitiveIndex];
            bounds = Union(bounds, bvhPrimitives[primitiveIndex].bounds);
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
        return node;

    } else {
        int mask = 1 << bitIndex;
        // Advance to next subtree level if there is no LBVH split for this bit
        if ((mortonPrims[0].mortonCode & mask) ==
            (mortonPrims[nPrimitives - 1].mortonCode & mask))
            return emitLBVH(buildNodes, bvhPrimitives, mortonPrims, nPrimitives,
                            totalNodes, orderedPrims, orderedPrimsOffset, bitIndex - 1);

        // Find LBVH split point for this dimension
        int splitOffset = FindInterval(nPrimitives, [&](int index) {
            return ((mortonPrims[0].mortonCode & mask) ==
                    (mortonPrims[index].mortonCode & mask));
        });
        ++splitOffset;
        CHECK_LE(splitOffset, nPrimitives - 1);
        CHECK_NE(mortonPrims[splitOffset - 1].mortonCode & mask,
                 mortonPrims[splitOffset].mortonCode & mask);

        // Create and return interior LBVH node
        (*totalNodes)++;
        BVHBuildNode *node = buildNodes++;
        BVHBuildNode *lbvh[2] = {
            emitLBVH(buildNodes, bvhPrimitives, mortonPrims, splitOffset, totalNodes,
                     orderedPrims, orderedPrimsOffset, bitIndex - 1),
            emitLBVH(buildNodes, bvhPrimitives, &mortonPrims[splitOffset],
                     nPrimitives - splitOffset, totalNodes, orderedPrims,
                     orderedPrimsOffset, bitIndex - 1)};
        int axis = bitIndex % 3;
        node->InitInterior(axis, lbvh[0], lbvh[1]);
        return node;
    }
}

int BVHAggregate::flattenBVH(BVHBuildNode *node, int *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->bounds = node->bounds;
    int nodeOffset = (*offset)++;
    if (node->nPrimitives > 0) {
        CHECK(!node->children[0] && !node->children[1]);
        CHECK_LT(node->nPrimitives, 65536);
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    } else {
        // Create interior flattened BVH node
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVH(node->children[0], offset);
        linearNode->secondChildOffset = flattenBVH(node->children[1], offset);
    }
    return nodeOffset;
}

Bounds3f BVHAggregate::Bounds() const {
    CHECK(nodes);
    return nodes[0].bounds;
}

pstd::optional<ShapeIntersection> BVHAggregate::Intersect(const Ray &ray,
                                                          Float tMax) const {

    if (!nodes)
        return {};

    pstd::optional<ShapeIntersection> si = {};
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    int nodesVisited = 0;
    while (true) {
        ++nodesVisited;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        // Check ray against BVH node
        // __builtin_prefetch(&nodes[currentNodeIndex + 1]);
        // __builtin_prefetch(&nodes[node->secondChildOffset]);
        if (node->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                for (int i = 0; i < node->nPrimitives; ++i) {
                    // Check for intersection with primitive in BVH node
                    pstd::optional<ShapeIntersection> primSi =
                        primitives[node->primitivesOffset + i].Intersect(ray, tMax);
                    if (primSi) {
                        si = primSi;
                        tMax = si->tHit;
                    }
                }
                if (toVisitOffset == 0)
                    break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                // Put far BVH node on _nodesToVisit_ stack, advance to near node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0)
                break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    bvhNodesVisited += nodesVisited;
    return si;
}

bool BVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    if (!nodes)
        return false;
    Vector3f invDir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    int dirIsNeg[3] = {static_cast<int>(invDir.x < 0), static_cast<int>(invDir.y < 0),
                       static_cast<int>(invDir.z < 0)};
    int nodesToVisit[64];
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesVisited = 0;

    while (true) {
        ++nodesVisited;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        if (node->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
            // Process BVH node _node_ for traversal
            if (node->nPrimitives > 0) {
                for (int i = 0; i < node->nPrimitives; ++i) {
                    if (primitives[node->primitivesOffset + i].IntersectP(ray, tMax)) {
                        bvhNodesVisited += nodesVisited;
                        return true;
                    }
                }
                if (toVisitOffset == 0)
                    break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node->axis] != 0) {
                    /// second child first
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0)
                break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    bvhNodesVisited += nodesVisited;
    return false;
}

BVHBuildNode *BVHAggregate::buildUpperSAH(Allocator alloc,
                                          std::vector<BVHBuildNode *> &treeletRoots,
                                          int start, int end,
                                          std::atomic<int> *totalNodes) const {
    CHECK_LT(start, end);
    int nNodes = end - start;
    if (nNodes == 1)
        return treeletRoots[start];
    (*totalNodes)++;
    BVHBuildNode *node = alloc.new_object<BVHBuildNode>();

    // Compute bounds of all nodes under this HLBVH node
    Bounds3f bounds;
    for (int i = start; i < end; ++i)
        bounds = Union(bounds, treeletRoots[i]->bounds);

    // Compute bound of HLBVH node centroids, choose split dimension _dim_
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        Point3f centroid =
            (treeletRoots[i]->bounds.pMin + treeletRoots[i]->bounds.pMax) * 0.5f;
        centroidBounds = Union(centroidBounds, centroid);
    }
    int dim = centroidBounds.MaxDimension();
    // FIXME: if this hits, what do we need to do?
    // Make sure the SAH split below does something... ?
    CHECK_NE(centroidBounds.pMax[dim], centroidBounds.pMin[dim]);

    // Allocate _BVHSplitBucket_ for SAH partition buckets
    constexpr int nBuckets = 12;
    struct BVHSplitBucket {
        int count = 0;
        Bounds3f bounds;
    };
    BVHSplitBucket buckets[nBuckets];

    // Initialize _BVHSplitBucket_ for HLBVH SAH partition buckets
    for (int i = start; i < end; ++i) {
        Float centroid =
            (treeletRoots[i]->bounds.pMin[dim] + treeletRoots[i]->bounds.pMax[dim]) *
            0.5f;
        int b = nBuckets * ((centroid - centroidBounds.pMin[dim]) /
                            (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
        if (b == nBuckets)
            b = nBuckets - 1;
        CHECK_GE(b, 0);
        CHECK_LT(b, nBuckets);
        buckets[b].count++;
        buckets[b].bounds = Union(buckets[b].bounds, treeletRoots[i]->bounds);
    }

    // Compute costs for splitting after each bucket
    Float cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; ++i) {
        Bounds3f b0, b1;
        int count0 = 0, count1 = 0;
        for (int j = 0; j <= i; ++j) {
            b0 = Union(b0, buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for (int j = i + 1; j < nBuckets; ++j) {
            b1 = Union(b1, buckets[j].bounds);
            count1 += buckets[j].count;
        }
        cost[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) /
                              bounds.SurfaceArea();
    }

    // Find bucket to split at that minimizes SAH metric
    Float minCost = cost[0];
    int minCostSplitBucket = 0;
    for (int i = 1; i < nBuckets - 1; ++i) {
        if (cost[i] < minCost) {
            minCost = cost[i];
            minCostSplitBucket = i;
        }
    }

    // Split nodes and create interior HLBVH SAH node
    BVHBuildNode **pmid = std::partition(
        &treeletRoots[start], &treeletRoots[end - 1] + 1, [=](const BVHBuildNode *node) {
            Float centroid = (node->bounds.pMin[dim] + node->bounds.pMax[dim]) * 0.5f;
            int b = nBuckets * ((centroid - centroidBounds.pMin[dim]) /
                                (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
            if (b == nBuckets)
                b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            return b <= minCostSplitBucket;
        });
    int mid = pmid - &treeletRoots[0];
    CHECK_GT(mid, start);
    CHECK_LT(mid, end);
    node->InitInterior(dim,
                       this->buildUpperSAH(alloc, treeletRoots, start, mid, totalNodes),
                       this->buildUpperSAH(alloc, treeletRoots, mid, end, totalNodes));
    return node;
}

BVHAggregate *BVHAggregate::Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters) {
    std::string splitMethodName = parameters.GetOneString("splitmethod", "sah");
    BVHAggregate::SplitMethod splitMethod;
    if (splitMethodName == "sah")
        splitMethod = BVHAggregate::SplitMethod::SAH;
    else if (splitMethodName == "hlbvh")
        splitMethod = BVHAggregate::SplitMethod::HLBVH;
    else if (splitMethodName == "middle")
        splitMethod = BVHAggregate::SplitMethod::Middle;
    else if (splitMethodName == "equal")
        splitMethod = BVHAggregate::SplitMethod::EqualCounts;
    else {
        Warning(R"(BVH split method "%s" unknown.  Using "sah".)", splitMethodName);
        splitMethod = BVHAggregate::SplitMethod::SAH;
    }

    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 4);
    return new BVHAggregate(std::move(prims), maxPrimsInNode, splitMethod);
}

// WBVHBuildNode Definition
struct WBVHBuildNode {
    struct SubNode{
        Bounds3f bounds = Bounds3f();
        Bounds3f centroidBounds = Bounds3f();
        WBVHBuildNode* child = nullptr;
        uint32_t nPrimitives = 0;
        uint32_t firstPrimOffset = 0;
        uint32_t start = 0;
        uint32_t end = 0;
        enum {INTERNAL, LEAF, EMPTY} type = EMPTY;
        uint32_t size() { return end - start; }
        uint32_t mid() { return (start + end) / 2; }
    } subNodes[WBVHAggregate::WIDTH];

    static bool octantCompare(const WBVHBuildNode::SubNode &a, const WBVHBuildNode::SubNode &b){
        auto a_centriod = (a.bounds.pMin + a.bounds.pMax) * 0.5f;
        auto b_centriod = (b.bounds.pMin + b.bounds.pMax) * 0.5f;
        return a_centriod.x < b_centriod.x;
    }

    void sort(){
        std::sort(subNodes, subNodes + WBVHAggregate::WIDTH, octantCompare);
    }

    std::string ToString(){
        std::string str = "";
        for(int i=0; i<WBVHAggregate::WIDTH; ++i){
            str += pbrt::StringPrintf("[%d] nPrimitives: %d, firstPrimOffset: %d, se %d -> %d, bound center: %s\n",
                i,
                subNodes[i].nPrimitives,
                subNodes[i].firstPrimOffset,
                subNodes[i].start,
                subNodes[i].end,
                ((subNodes[i].bounds.pMin + subNodes[i].bounds.pMax) * .5f).ToString()
            );
        }
        return str;
    }

};


#include <immintrin.h>

struct alignas(32) LinearWBVHNode {

    __m256 AABB[3][2];

    int      offsets[WBVHAggregate::WIDTH];
    uint32_t nPrimitives[WBVHAggregate::WIDTH];
    inline bool isInternal(const int &i) const {
        return nPrimitives[i] == 0 && offsets[i] != 0;
    }
    inline bool isLeaf(const int &i) const {
        return nPrimitives[i] > 0;
    }
    inline bool isEmpty(const int &i) const {
        return nPrimitives[i] == 0 && offsets[i] == 0;
    }

    inline void initAABB(const int &i, const Bounds3f &b){
        AABB[0][0][i] = b.pMin.x;
        AABB[0][1][i] = b.pMax.x;
        AABB[1][0][i] = b.pMin.y;
        AABB[1][1][i] = b.pMax.y;
        AABB[2][0][i] = b.pMin.z;
        AABB[2][1][i] = b.pMax.z;
    }

    std::array<bool, WBVHAggregate::WIDTH> Intersect(const Point3f &ray_o, const Vector3f &ray_d, const Float &raytMax, Vector3f invDir, const int dirIsNeg[3]) const {
        std::array<bool, WBVHAggregate::WIDTH> hit;
        for(int i=0; i<WBVHAggregate::WIDTH; ++i){
            Bounds3f bounds(
                Point3f(AABB[0][0][i], AABB[1][0][i], AABB[2][0][i]),
                Point3f(AABB[0][1][i], AABB[1][1][i], AABB[2][1][i])
            );
            hit[i] = bounds.IntersectP(ray_o, ray_d, raytMax, invDir, dirIsNeg);
        }
        return hit;
    }

    std::array<bool, WBVHAggregate::WIDTH> IntersectSIMD(const Point3f &ray_o, const Vector3f &ray_d, const Float &raytMax, Vector3f invDir, const int dirIsNeg[3]) const {

        __m256 ray_o_xyz  = _mm256_set1_ps(ray_o.x);
        __m256 invDir_xyz = _mm256_set1_ps(invDir.x);
        __m256 tMin_xyz = _mm256_mul_ps(_mm256_sub_ps(AABB[0][    dirIsNeg[0]], ray_o_xyz), invDir_xyz);
        __m256 tMax_xyz = _mm256_mul_ps(_mm256_sub_ps(AABB[0][1 - dirIsNeg[0]], ray_o_xyz), invDir_xyz);

        ray_o_xyz  = _mm256_set1_ps(ray_o.y);
        invDir_xyz = _mm256_set1_ps(invDir.y);
        tMin_xyz = _mm256_max_ps(
            tMin_xyz,
            _mm256_mul_ps(
                _mm256_sub_ps(AABB[1][dirIsNeg[1]], ray_o_xyz),
                invDir_xyz
            )
        );
        tMax_xyz = _mm256_min_ps(
            tMax_xyz,
            _mm256_mul_ps(
                _mm256_sub_ps(AABB[1][1 - dirIsNeg[1]], ray_o_xyz),
                invDir_xyz
            )
        );

        ray_o_xyz = _mm256_set1_ps(ray_o.z);
        invDir_xyz = _mm256_set1_ps(invDir.z);
        tMin_xyz = _mm256_max_ps(
            tMin_xyz,
            _mm256_mul_ps(
                _mm256_sub_ps(AABB[2][dirIsNeg[2]], ray_o_xyz),
                invDir_xyz
            )
        );
        tMax_xyz = _mm256_min_ps(
            tMax_xyz,
            _mm256_mul_ps(
                _mm256_sub_ps(AABB[2][1 - dirIsNeg[2]], ray_o_xyz),
                invDir_xyz
            )
        );
    
        __m256 t_cmp = _mm256_cmp_ps(tMin_xyz, tMax_xyz, _CMP_LE_OQ);        
        t_cmp = _mm256_and_ps(t_cmp, _mm256_cmp_ps(tMax_xyz, _mm256_setzero_ps(), _CMP_GT_OQ));
        t_cmp = _mm256_and_ps(t_cmp, _mm256_cmp_ps(tMin_xyz, _mm256_set1_ps(raytMax), _CMP_LT_OQ));

        std::array<bool, WBVHAggregate::WIDTH> hit;
        for(int i=0; i<WBVHAggregate::WIDTH; ++i)
            hit[i] = std::isnan(t_cmp[i]);
        
        return hit;
    }

};

// WBVH Method Definitions
WBVHAggregate::WBVHAggregate(std::vector<Primitive> prims, int maxPrimsInNode)
    : maxPrimsInNode(std::min(127, maxPrimsInNode)),
      primitives(std::move(prims)) {

    CHECK(!primitives.empty());
    // Build BVH from _primitives_
    // Initialize _bvhPrimitives_ array for primitives
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());

    // Build BVH for primitives using _bvhPrimitives_
    // Declare _Allocator_s used for BVH construction
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });

    std::vector<Primitive> orderedPrims(primitives.size());
    WBVHBuildNode *root;
    // Build BVH according to selected _splitMethod_
    std::atomic<int> totalNodes{0};

    std::atomic<int> orderedPrimsOffset{0};
    // LOG_CONCISE("Starting WBVH build with %d primitives", (int)primitives.size());
    root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                            &totalNodes, &orderedPrimsOffset, orderedPrims);
    CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());

    primitives.swap(orderedPrims);

    // construct gbounds
    gbounds = Bounds3f();
    for (const auto &subNode: root->subNodes)
        gbounds = Union(gbounds, subNode.bounds);

    LOG_CONCISE("%d-Wide BVH total_nodes: %d empty_nodes: %d one_nodes: %d", WBVHAggregate::WIDTH, total_nodes.load(), empty_nodes.load(), one_nodes.load());

    // Convert BVH into compact representation in _nodes_ array
    bvhPrimitives.resize(0);
    bvhPrimitives.shrink_to_fit();
    LOG_CONCISE("WBVH created with %d nodes for %d primitives (%.2f MB)",
                totalNodes.load(), (int)primitives.size(),
                float(totalNodes.load() * sizeof(LinearWBVHNode)) / (1024.f * 1024.f));
    treeBytes += totalNodes * sizeof(LinearWBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = new LinearWBVHNode[totalNodes];
    int offset = flattenWBVH(root, 0, 1);

    CHECK_EQ(totalNodes.load(), offset);
}


WBVHBuildNode *WBVHAggregate::buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                           pstd::span<BVHPrimitive> bvhPrimitives,
                                           std::atomic<int> *totalNodes,
                                           std::atomic<int> *orderedPrimsOffset,
                                           std::vector<Primitive> &orderedPrims,
                                           int depth) {
    DCHECK_NE(bvhPrimitives.size(), 0);
    CHECK_LE(depth, 36);
    
    Allocator alloc = threadAllocators.Get();
    WBVHBuildNode *node = alloc.new_object<WBVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    ++*totalNodes;
    

    // Initialize the first child
    node->subNodes[0].start = 0;
    node->subNodes[0].end = bvhPrimitives.size();
    node->subNodes[0].type = WBVHBuildNode::SubNode::INTERNAL;

    for (const auto &prim : bvhPrimitives){
        node->subNodes[0].bounds = Union(node->subNodes[0].bounds, prim.bounds);
        node->subNodes[0].centroidBounds = Union(node->subNodes[0].centroidBounds, prim.Centroid());
    }
    int builtChildCnt = 1;

    // split until reach the width or not splitable
    while(builtChildCnt < WBVHAggregate::WIDTH) {
        // find the child with the largest bounds
        int largestChild = -1;
        Float largestArea = 0;
        for (int i = 0; i < builtChildCnt; i++) {
            Float area = node->subNodes[i].bounds.SurfaceArea();
            if(area <= 0)
                // prevent internal node with zero area, which leads to infinite recursion
                node->subNodes[i].type = WBVHBuildNode::SubNode::LEAF;
            else if(node->subNodes[i].size() == 0)
                node->subNodes[i].type = WBVHBuildNode::SubNode::EMPTY;
            else if (node->subNodes[i].type == WBVHBuildNode::SubNode::INTERNAL && area > largestArea){
                largestArea = node->subNodes[i].bounds.SurfaceArea();
                largestChild = i;
            }
        }
        if(largestChild == -1)
            break;

        WBVHBuildNode::SubNode &target = node->subNodes[largestChild];
        CHECK(target.size() >= 1);
        
        if(target.bounds.SurfaceArea() == 0 || target.size() == 1){
            // Create leaf _BVHBuildNode_
            target.type = WBVHBuildNode::SubNode::LEAF;
            continue;
        } else {
            int dim = target.centroidBounds.MaxDimension();
            // Partition primitives into two sets and build children
            if(target.centroidBounds.pMax[dim] == target.centroidBounds.pMin[dim]){
                // Create leaf _BVHBuildNode_
                target.type = WBVHBuildNode::SubNode::LEAF;
                continue;
            } else {
                int mid = target.mid();
                
                // Partition primitives using approximate SAH
                if (target.size() <= 2) {
                    // Partition primitives into equally sized subsets
                    mid = target.mid();
                    std::nth_element(
                        bvhPrimitives.begin() + target.start,
                        bvhPrimitives.begin() + mid,
                        bvhPrimitives.begin() + target.end,
                        [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                            return a.Centroid()[dim] < b.Centroid()[dim];
                        }
                    );
                    
                    // create a new child
                    WBVHBuildNode::SubNode &newChild = node->subNodes[builtChildCnt++];
                    newChild.start = target.mid();
                    newChild.end = target.end;
                    newChild.bounds = Bounds3f();
                    newChild.centroidBounds = Bounds3f();
                    newChild.type = WBVHBuildNode::SubNode::INTERNAL;
                    for (const auto &prim : bvhPrimitives.subspan(newChild.start, newChild.size())) {
                        newChild.bounds = Union(newChild.bounds, prim.bounds);
                        newChild.centroidBounds = Union(newChild.centroidBounds, prim.Centroid());
                    }

                    // reduce the original child
                    target.end = target.mid();
                    target.bounds = Bounds3f();
                    target.centroidBounds = Bounds3f();
                    target.type = WBVHBuildNode::SubNode::INTERNAL;
                    for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                        target.bounds = Union(target.bounds, prim.bounds);
                        target.centroidBounds = Union(target.centroidBounds, prim.Centroid());
                    }

                } else {
                    // Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                        int b = nBuckets * target.centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }
                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    Float costs[nSplits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        costs[i] += countBelow * boundBelow.SurfaceArea();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        costs[i - 1] += countAbove * boundAbove.SurfaceArea();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        if (costs[i] < minCost) {
                            minCost = costs[i];
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = target.size();
                    minCost = 1.f / 2.f + minCost / target.bounds.SurfaceArea();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (target.size() > maxPrimsInNode || minCost < leafCost) {
                        auto midIter = std::partition(
                            bvhPrimitives.begin() + target.start, bvhPrimitives.begin() + target.end,
                            [=](const BVHPrimitive &bp) {
                                int b =
                                    nBuckets * target.centroidBounds.Offset(bp.Centroid())[dim];
                                if (b == nBuckets)
                                    b = nBuckets - 1;
                                return b <= minCostSplitBucket;
                            });
                        mid = midIter - bvhPrimitives.begin();
                        
                        // create a new child
                        WBVHBuildNode::SubNode &newChild = node->subNodes[builtChildCnt++];
                        newChild.start = mid;
                        newChild.end = target.end;
                        newChild.bounds = Bounds3f();
                        newChild.centroidBounds = Bounds3f();
                        newChild.type = WBVHBuildNode::SubNode::INTERNAL;
                        for (const auto &prim : bvhPrimitives.subspan(newChild.start, newChild.size())) {
                            newChild.bounds = Union(newChild.bounds, prim.bounds);
                            newChild.centroidBounds = Union(newChild.centroidBounds, prim.Centroid());
                        }

                        // reduce the original child
                        target.start = target.start;
                        target.end = mid;
                        target.bounds = Bounds3f();
                        target.centroidBounds = Bounds3f();
                        target.type = WBVHBuildNode::SubNode::INTERNAL;
                        for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                            target.bounds = Union(target.bounds, prim.bounds);
                            target.centroidBounds = Union(target.centroidBounds, prim.Centroid());
                        }
                    } 
                    else {
                        // create leaf _BVHBuildNode_
                        target.type = WBVHBuildNode::SubNode::LEAF;
                        continue;
                    }
                }
            }
        }           
    }

    // octant sort
    node->sort();

    // unsplitable internal node check
    /*
    for(int i=0; i<WBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type != WBVHBuildNode::SubNode::INTERNAL)
            continue;

        auto &target = node->subNodes[i];
        if(target.bounds.SurfaceArea() == 0 || target.size() == 1)
            target.type = WBVHBuildNode::SubNode::LEAF;
        else {
            int dim = target.centroidBounds.MaxDimension();
            if(target.centroidBounds.pMax[dim] == target.centroidBounds.pMin[dim])
                target.type = WBVHBuildNode::SubNode::LEAF;
            else {
                int mid = target.mid();
                if (target.size() <= 2)
                    target.type = WBVHBuildNode::SubNode::LEAF;
                else {
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                        int b = nBuckets * target.centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }
                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    Float costs[nSplits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        costs[i] += countBelow * boundBelow.SurfaceArea();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        costs[i - 1] += countAbove * boundAbove.SurfaceArea();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if necessary
                        if (costs[i] < minCost) {
                            minCost = costs[i];
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = target.size();
                    minCost = 1.f / 2.f + minCost / target.bounds.SurfaceArea();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (!(target.size() > maxPrimsInNode || minCost < leafCost)){
                        target.type = WBVHBuildNode::SubNode::LEAF;
                        continue;
                    }
                }
            }
        }
    }
    */

    // statistics
    int empty_cnt = 0;
    for(int i=0; i<WBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type == WBVHBuildNode::SubNode::EMPTY)
            empty_cnt++;
    }
    this->empty_nodes.fetch_add(empty_cnt);
    this->total_nodes.fetch_add(1);
    this->one_nodes.fetch_add(empty_cnt == WBVHAggregate::WIDTH - 1);


    if(bvhPrimitives.size() > 128 * 1024){
        ParallelFor(0, WBVHAggregate::WIDTH, [&](int i){
            if(node->subNodes[i].type == WBVHBuildNode::SubNode::EMPTY)
                return;            
            else if(node->subNodes[i].type == WBVHBuildNode::SubNode::INTERNAL){
                node->subNodes[i].child = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(node->subNodes[i].start, node->subNodes[i].size()),
                    totalNodes,
                    orderedPrimsOffset,
                    orderedPrims,
                    depth + 1
                );
                node->subNodes[i].nPrimitives = 0;
                node->subNodes[i].firstPrimOffset = 0;
            }
            else if(node->subNodes[i].type == WBVHBuildNode::SubNode::LEAF){
                node->subNodes[i].child = nullptr;
                node->subNodes[i].nPrimitives = node->subNodes[i].size();
                int firstPrimOffset = orderedPrimsOffset->fetch_add(node->subNodes[i].size());
                node->subNodes[i].firstPrimOffset = firstPrimOffset;
                for(int prim = 0; prim < node->subNodes[i].size(); prim++){
                    int index = bvhPrimitives[node->subNodes[i].start + prim].primitiveIndex;
                    orderedPrims[firstPrimOffset + prim] = primitives[index];
                }
            }
        });    
    }
    else{
        for(int i = 0; i < WBVHAggregate::WIDTH; i++){
            if(node->subNodes[i].type == WBVHBuildNode::SubNode::EMPTY){
                continue;
            }
            else if(node->subNodes[i].type == WBVHBuildNode::SubNode::INTERNAL){
                node->subNodes[i].child = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(node->subNodes[i].start, node->subNodes[i].size()),
                    totalNodes,
                    orderedPrimsOffset,
                    orderedPrims,
                    depth + 1
                );
                node->subNodes[i].nPrimitives = 0;
                node->subNodes[i].firstPrimOffset = 0;
            }
            else if(node->subNodes[i].type == WBVHBuildNode::SubNode::LEAF){
                node->subNodes[i].child = nullptr;
                node->subNodes[i].nPrimitives = node->subNodes[i].size();
                int firstPrimOffset = orderedPrimsOffset->fetch_add(node->subNodes[i].size());
                node->subNodes[i].firstPrimOffset = firstPrimOffset;
                for(int prim = 0; prim < node->subNodes[i].size(); prim++){
                    int index = bvhPrimitives[node->subNodes[i].start + prim].primitiveIndex;
                    orderedPrims[firstPrimOffset + prim] = primitives[index];
                }
            }
        }

    }
    return node;
}

int WBVHAggregate::flattenWBVH(WBVHBuildNode *node, int locate, int offset) {
    LinearWBVHNode *linearNode = &nodes[locate];

    int internalCnt = 0;
    for(int i = 0; i < WBVHAggregate::WIDTH; i++)
        internalCnt += node->subNodes[i].type == WBVHBuildNode::SubNode::INTERNAL ? 1 : 0;
    int next_offset = offset + internalCnt;
    internalCnt = 0;

    for(int i = 0; i < WBVHAggregate::WIDTH; i++){
        if(node->subNodes[i].type == WBVHBuildNode::SubNode::EMPTY){
            linearNode->initAABB(i, Bounds3f());
            // linearNode->bounds[i] = Bounds3f();
            linearNode->offsets[i] = 0;
            linearNode->nPrimitives[i] = 0;
            continue;
        }
        else if(node->subNodes[i].type == WBVHBuildNode::SubNode::INTERNAL){
            linearNode->initAABB(i, node->subNodes[i].bounds);
            // linearNode->bounds[i] = node->subNodes[i].bounds;
            linearNode->nPrimitives[i] = 0;
            linearNode->offsets[i] = offset + (internalCnt++);
            next_offset = flattenWBVH(node->subNodes[i].child, linearNode->offsets[i], next_offset);
        }
        else if(node->subNodes[i].type == WBVHBuildNode::SubNode::LEAF){
            linearNode->initAABB(i, node->subNodes[i].bounds);
            // linearNode->bounds[i] = node->subNodes[i].bounds;
            linearNode->nPrimitives[i] = node->subNodes[i].nPrimitives;
            linearNode->offsets[i] = node->subNodes[i].firstPrimOffset;
        }
        else {
            CHECK(false);
        }
    }
    return next_offset;
}

pstd::optional<ShapeIntersection> WBVHAggregate::Intersect(const Ray &ray, Float tMax) const {
    if (!nodes)
        return {};
    auto start = std::chrono::high_resolution_clock::now();


    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    uint8_t rayOctant = (uint8_t(invDir.x < 0) << 2) | (uint8_t(invDir.y < 0)  << 1) | (uint8_t(invDir.z < 0));
    int toVisitOffset = 1;
    int nodesToVisit[64];
    int nodesVisited = nodesToVisit[0] = 0;

    while (toVisitOffset > 0) {
        ++nodesVisited;
        const LinearWBVHNode *node = &nodes[nodesToVisit[--toVisitOffset]];
        auto hit = node->Intersect(ray.o, ray.d, tMax, invDir, dirIsNeg);
        
        for(int octant = 7; octant >= 0; --octant){
            int child = octant ^ rayOctant;
            if(child >= WBVHAggregate::WIDTH)
                continue;
            else if(hit[child]){
                if(node->isLeaf(child)){
                    for(int i=0; i < node->nPrimitives[child]; ++i){
                        pstd::optional<ShapeIntersection> primSi =
                            primitives[node->offsets[child] + i].Intersect(ray, tMax);
                        if(primSi){
                            si = primSi;
                            tMax = si->tHit;
                        }
                    }
                }
                else if(node->isInternal(child)){
                    nodesToVisit[toVisitOffset++] = node->offsets[child];
                }
            }
        }
    }
    bvhNodesVisited += nodesVisited;
    Options->intersect_time += std::chrono::high_resolution_clock::now() - start;
    return si;
}

bool WBVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    if (!nodes)
        return {};

    auto start = std::chrono::high_resolution_clock::now();

    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    // Follow ray through BVH nodes to find primitive intersections

    int toVisitOffset = 1;
    int nodesToVisit[64];
    int nodesVisited = 0;
    nodesToVisit[0] = 0;

    while (toVisitOffset > 0) {
        ++nodesVisited;
        const LinearWBVHNode *node = &nodes[nodesToVisit[--toVisitOffset]];
        auto hit = node->Intersect(ray.o, ray.d, tMax, invDir, dirIsNeg);

        for(int child = 0; child < WBVHAggregate::WIDTH; ++child){
            if(child >= WBVHAggregate::WIDTH)
                continue;
            else if(node->isEmpty(child))
                continue;
            else if(hit[child]){
                if(node->isLeaf(child)){
                    for(int i=0; i< node->nPrimitives[child]; ++i){
                        if(primitives[node->offsets[child] + i].IntersectP(ray, tMax)){
                            bvhNodesVisited += nodesVisited;
                            Options->intersect_time += std::chrono::high_resolution_clock::now() - start;
                            return true;
                        }
                    }
                }
                else if(node->isInternal(child)){
                    nodesToVisit[toVisitOffset++] = node->offsets[child];
                }
            }
        }
    }
    bvhNodesVisited += nodesVisited;
    Options->intersect_time += std::chrono::high_resolution_clock::now() - start;
    return false;
}

WBVHAggregate *WBVHAggregate::Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters) {
    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 64);
    // LOG_CONCISE("[WBVH Create] maxPrimsInNode: %d", maxPrimsInNode);
    return new WBVHAggregate(std::move(prims), maxPrimsInNode);
}

Bounds3f WBVHAggregate::Bounds() const {
    return gbounds;
}

// MEBVH Method Definitions



// WBVHBuildNode Definition
struct MEBVHBuildNode {
    struct SubNode{
        Bounds3f bounds = Bounds3f();
        Bounds3f centroidBounds = Bounds3f();
        MEBVHBuildNode* child = nullptr;
        uint32_t nPrimitives = 0;
        uint32_t firstPrimOffset = 0;
        uint32_t start = 0;
        uint32_t end = 0;
        enum {INTERNAL, LEAF, EMPTY} type = EMPTY;
        uint32_t size() { return end - start; }
        uint32_t mid() { return (start + end) / 2; }
    } subNodes[MEBVHAggregate::WIDTH];

    static bool octantCompare(const MEBVHBuildNode::SubNode &a, const MEBVHBuildNode::SubNode &b){
        auto a_centriod = (a.bounds.pMin + a.bounds.pMax) * 0.5f;
        auto b_centriod = (b.bounds.pMin + b.bounds.pMax) * 0.5f;
        return a_centriod.x < b_centriod.x;
    }

    void sort(){
        std::sort(subNodes, subNodes + MEBVHAggregate::WIDTH, octantCompare);
    }

    std::string ToString(){
        std::string str = "";
        for(int i=0; i<WBVHAggregate::WIDTH; ++i){
            str += pbrt::StringPrintf("[%d] nPrimitives: %d, firstPrimOffset: %d, se %d -> %d, bound center: %s\n",
                i,
                subNodes[i].nPrimitives,
                subNodes[i].firstPrimOffset,
                subNodes[i].start,
                subNodes[i].end,
                ((subNodes[i].bounds.pMin + subNodes[i].bounds.pMax) * .5f).ToString()
            );
        }
        return str;
    }

};

struct alignas(64) LinearMEBVHNode{   
    Float p[3];
    uint32_t primitive_offset = UINT32_MAX;
    uint32_t internal_offset : 24;
    int8_t e[3];
    uint8_t meta[MEBVHAggregate::WIDTH];
    uint8_t q[3][2][MEBVHAggregate::WIDTH]; // dim / minmax / width

    void init(MEBVHBuildNode* const node, const Bounds3f &bounds, const int &offset, const int &primitive_offset){
        this->internal_offset = offset;
        this->primitive_offset = primitive_offset;
        for(int dim = 0; dim < 3; ++dim){
            this->p[dim] = bounds.pMin[dim];
            auto e_i = [](const Float &min, const Float &max){
                return pstd::ceil(Log2((max - min + 0.00000001f) / 255.0));
            };
            int32_t tE = e_i(bounds.pMin[dim], bounds.pMax[dim]);
            this->e[dim] = std::max(INT8_MIN, std::min(INT8_MAX, tE));
        }

        int internal_cnt = 0;
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
            if(node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL){
                for(int dim = 0; dim < 3; ++dim){
                    int32_t tQ_lo = pstd::floor((node->subNodes[i].bounds.pMin[dim] - p[dim]) / std::exp2(e[dim]));
                    int32_t tQ_hi = pstd::ceil((node->subNodes[i].bounds.pMax[dim] - p[dim]) / std::exp2(e[dim]));
                    // tQ_lo = std::max(0, tQ_lo-1);
                    // tQ_hi = std::min(255, tQ_hi+1);
                    CHECK_GE(tQ_lo, 0); CHECK_LE(tQ_lo, 255);
                    CHECK_GE(tQ_hi, 0); CHECK_LE(tQ_hi, 255);
                    this->q[dim][0][i] = tQ_lo;
                    this->q[dim][1][i] = tQ_hi;
                }
                CHECK_LT(internal_cnt, 8);
                meta[i] = 0b11111000 | (internal_cnt++);
            }
            else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF){
                for(int dim = 0; dim < 3; ++dim){
                    int32_t tQ_lo = pstd::floor((node->subNodes[i].bounds.pMin[dim] - p[dim]) / std::exp2(e[dim]));
                    int32_t tQ_hi = pstd::ceil((node->subNodes[i].bounds.pMax[dim] - p[dim]) / std::exp2(e[dim]));
                    CHECK_GE(tQ_lo, 0); CHECK_LE(tQ_lo, 255);
                    CHECK_GE(tQ_hi, 0); CHECK_LE(tQ_hi, 255);
                    this->q[dim][0][i] = tQ_lo;
                    this->q[dim][1][i] = tQ_hi;
                }
                CHECK_LT(node->subNodes[i].nPrimitives, 248);
                meta[i] = node->subNodes[i].nPrimitives;
            }
            else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::EMPTY){
                meta[i] = 0;
            }
        }
    }
    Bounds3f bounds(const int &i) const {
        Bounds3f b;
        for(int dim = 0; dim < 3; ++dim){
            b.pMin[dim] = p[dim] + q[dim][0][i] * std::exp2(e[dim]);
            b.pMax[dim] = p[dim] + q[dim][1][i] * std::exp2(e[dim]);
        }
        return b;
    }
    uint32_t offset(const int &i) const {
        CHECK(isInternal(i));
        return internal_offset + (meta[i] & 0b00000111);
    }
    uint32_t nPrimitives(const int &i) const {
        CHECK(isLeaf(i));        
        return meta[i];
    }
    inline bool isInternal(const int &i) const {
        return (meta[i] & 0b11111000) == 0b11111000;
    }
    inline bool isLeaf(const int &i) const {
        return meta[i] < 0b11111000 && meta[i] != 0;
    }
    inline bool isEmpty(const int &i) const {
        return meta[i] == 0;
    }

    std::array<bool, MEBVHAggregate::WIDTH> Intersect(const Point3f &ray_o, const Vector3f &ray_d, const Float &raytMax, const Vector3f &invDir, const int dirIsNeg[3]) const {
        std::array<bool, MEBVHAggregate::WIDTH> hit;
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
            if(isEmpty(i))
                hit[i] = false;
            else
                hit[i] = bounds(i).IntersectP(ray_o, ray_d, raytMax, invDir, dirIsNeg);
        }
        return hit;
    }

    inline float fast_exp2(int8_t e) const {
        union {
            uint32_t u;
            float f;
        } converter;
        converter.u = ((e + 127) << 23);
        return converter.f;
    }

    #define _mm256_submul_ps(a, b, c) _mm256_mul_ps(_mm256_sub_ps(a, b), c)

    inline __m256 dequant(int d, int m) const {
        return _mm256_fmadd_ps(_mm256_set_ps(0, 0, q[d][m][5], q[d][m][4], q[d][m][3], q[d][m][2], q[d][m][1], q[d][m][0]), _mm256_set1_ps(fast_exp2(e[d])), _mm256_set1_ps(p[d])); 
    }

    std::array<bool, MEBVHAggregate::WIDTH> IntersectSIMD(const Point3f &ray_o, const Vector3f &ray_d, const Float &raytMax, const Vector3f &invDir, const int dirIsNeg[3]) const {
        __m256 ray_o_xyz = _mm256_set1_ps(ray_o.x);
        __m256 invDir_xyz = _mm256_set1_ps(invDir.x);                                                                     
        __m256 tMin_xyz = _mm256_submul_ps(dequant(0,  dirIsNeg[0]), ray_o_xyz, invDir_xyz);
        __m256 tMax_xyz = _mm256_submul_ps(dequant(0, !dirIsNeg[0]), ray_o_xyz, invDir_xyz);

        ray_o_xyz  = _mm256_set1_ps(ray_o.y);
        invDir_xyz = _mm256_set1_ps(invDir.y);
        tMin_xyz = _mm256_max_ps(
            tMin_xyz,
            _mm256_submul_ps(dequant(1, dirIsNeg[1]), ray_o_xyz, invDir_xyz)
        );
        tMax_xyz = _mm256_min_ps(
            tMax_xyz,
            _mm256_submul_ps(dequant(1, !dirIsNeg[1]), ray_o_xyz, invDir_xyz)
        );

        ray_o_xyz = _mm256_set1_ps(ray_o.z);
        invDir_xyz = _mm256_set1_ps(invDir.z);
        tMin_xyz = _mm256_max_ps(
            tMin_xyz,
            _mm256_submul_ps(dequant(2, dirIsNeg[2]), ray_o_xyz, invDir_xyz)
        );
        tMax_xyz = _mm256_min_ps(
            tMax_xyz,
            _mm256_submul_ps(dequant(2, !dirIsNeg[2]), ray_o_xyz, invDir_xyz)
        );

        __m256 t_cmp = _mm256_cmp_ps(tMin_xyz, tMax_xyz, _CMP_LE_OQ);        
        t_cmp = _mm256_and_ps(t_cmp, _mm256_cmp_ps(tMax_xyz, _mm256_setzero_ps(), _CMP_GT_OQ));
        t_cmp = _mm256_and_ps(t_cmp, _mm256_cmp_ps(tMin_xyz, _mm256_set1_ps(raytMax), _CMP_LT_OQ));
        std::array<bool, MEBVHAggregate::WIDTH> hit;
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i)
            hit[i] = std::isnan(t_cmp[i]);   
        return hit;
    }


    std::string ToString(){
        std::string str = "";
        str += pbrt::StringPrintf("p: 0x%08x 0x%08x 0x%08x\n", *((uint32_t *)(&p[0])), *((uint32_t *)(&p[1])), *((uint32_t *)(&p[2])));
        str += pbrt::StringPrintf("e: %hhx %hhx %hhx\n", *((uint8_t *)(&e[0])), *((uint8_t *)(&e[1])), *((uint8_t *)(&e[2])));
        uint32_t tmp_intermal_offset = internal_offset;
        str += pbrt::StringPrintf("internal_offset: 0x%08x\n", tmp_intermal_offset);
        str += pbrt::StringPrintf("primitive_offset: 0x%08x\n", primitive_offset);
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
            str += pbrt::StringPrintf("meta[%d]: %hhx\n", i, meta[i]);
            for(int dim=0; dim<3; ++dim){
                str += pbrt::StringPrintf("q[%d][0][%d]: %hhx\n", dim, i, q[dim][0][i]);
                str += pbrt::StringPrintf("q[%d][1][%d]: %hhx\n", dim, i, q[dim][1][i]);
            }
        }
        return str;
    }
};



static const __m256i m256_0xFF = _mm256_set1_epi32(0xFF);

struct alignas(64) LinearOptimizedMEBVHNode{
    alignas(64) union {
        __m256i  m256i;
        uint32_t u32[8];
        float    f32[8];
        int8_t   i8[32];
        uint8_t  u8[32];
        // b256[0]
        // 24B | qmin[0].x qmin[0].y qmin[0].z meta[0] | ... | qmax[5].x qmax[5].y qmax[5].z meta[5] |
        // 8B  | p[0] | p[1] |  

        // b256[1]
        // 12B | qmax[0].x qmax[0].y qmax[0].z e[0] | ... | qmax[2].x qmax[2].y qmax[2].z e[2] |
        // 12B | qmax[3].x qmax[3].y qmax[3].z internal_offset | ... | qmax[5].x qmax[5].y qmax[5].z internal_offset | 
        // 4B  | p[2] |
        // 4B  | primitive_offset |
    } b256[2];
    
    void init(MEBVHBuildNode* const node, const Bounds3f &bounds, const int &offset, const int &primitive_offset){
        LinearMEBVHNode unlignedNode;
        unlignedNode.init(node, bounds, offset, primitive_offset);
        b256[0].m256i = _mm256_setzero_si256();
        b256[1].m256i = _mm256_setzero_si256();
        for(int i=0; i<6; ++i)
            b256[0].u32[i] = (unlignedNode.q[0][0][i] << 24) | (unlignedNode.q[1][0][i] << 16) | (unlignedNode.q[2][0][i] << 8) | unlignedNode.meta[i];
        b256[0].f32[6] = unlignedNode.p[0];
        b256[0].f32[7] = unlignedNode.p[1];

        alignas(4) union {
            uint32_t u32;
            uint8_t  u8[4];
        } internal;
        internal.u32 = unlignedNode.internal_offset;
        CHECK(internal.u8[3] == 0);

        for(int i=0; i<6; ++i)
            b256[1].u32[i] = (unlignedNode.q[0][1][i] << 24) | (unlignedNode.q[1][1][i] << 16) | (unlignedNode.q[2][1][i] << 8);
        for(int i=0; i<3; ++i){
            b256[1].i8[i * 4] = unlignedNode.e[i];
            b256[1].u8[12 + i * 4] = internal.u8[i];
        }        
        b256[1].f32[6] = unlignedNode.p[2];
        b256[1].u32[7] = unlignedNode.primitive_offset;
        
        // TODO:: remove after success
        // check(unlignedNode);
    }

    void check(const LinearMEBVHNode &unlignedNode) const {
        // check p
        CHECK(unlignedNode.p[0] == p(0));
        CHECK(unlignedNode.p[1] == p(1));
        CHECK(unlignedNode.p[2] == p(2));

        // check q, e
        for(int d = 0; d < 3; ++d){
            __m256 gt_qmin = _mm256_set_ps(0, 0, unlignedNode.q[d][0][5], unlignedNode.q[d][0][4], unlignedNode.q[d][0][3], unlignedNode.q[d][0][2], unlignedNode.q[d][0][1], unlignedNode.q[d][0][0]);
            __m256 gt_qmax = _mm256_set_ps(0, 0, unlignedNode.q[d][1][5], unlignedNode.q[d][1][4], unlignedNode.q[d][1][3], unlignedNode.q[d][1][2], unlignedNode.q[d][1][1], unlignedNode.q[d][1][0]);
            __m256 gt_e = _mm256_set1_ps(unlignedNode.fast_exp2(unlignedNode.e[d]));
            __m256 tt_qmin = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(b256[0].m256i, 24 - 8 * d), _mm256_set1_epi32(0xFF)));
            __m256 tt_qmax = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(b256[1].m256i, 24 - 8 * d), _mm256_set1_epi32(0xFF)));
            __m256 tt_e = _mm256_set1_ps(unlignedNode.fast_exp2(e(d)));

            for(int i=0; i<6; ++i){
                CHECK_EQ(gt_qmin[i], tt_qmin[i]);
                CHECK_EQ(gt_qmax[i], tt_qmax[i]);
                CHECK_EQ(gt_e[i], tt_e[i]);
            }

            // check dequant
            // for(int m = 0; m < 2; ++m){
            //     __m256 gt_dequant = unlignedNode.dequant(d, m);
            //     __m256 tt_dequant = dequant(d, m);
            //     for(int i=0; i<6; ++i)
            //         CHECK_EQ(gt_dequant[i], tt_dequant[i]);
            // }
        }
        // check meta
        for(int i=0; i<6; ++i)
            CHECK_EQ(unlignedNode.meta[i], meta(i));
        // check internal_offset
        CHECK_EQ(unlignedNode.internal_offset, internal_offset());
        // check primitive_offset
        CHECK_EQ(unlignedNode.primitive_offset, this->primitive_offset());

        // check leaf int attributes
        for(int i=0; i<6; ++i){
            if(unlignedNode.isInternal(i))
                CHECK_EQ(unlignedNode.offset(i), offset(i));
            else if(unlignedNode.isLeaf(i))
                CHECK_EQ(unlignedNode.nPrimitives(i), nPrimitives(i));
        }
    }

    Bounds3f bounds(const int &i) const {
        Bounds3f b;
        for(int dim = 0; dim < 3; ++dim){
            b.pMin[dim] = p(dim) + (b256[0].u32[i] >> (24 - 8 * dim)) * fast_exp2(e(dim));
            b.pMax[dim] = p(dim) + (b256[1].u32[i] >> (24 - 8 * dim)) * fast_exp2(e(dim));
        }
        return b;
    }

    inline uint32_t internal_offset() const {
        return (uint32_t)b256[1].u8[12] << 0 | (uint32_t)b256[1].u8[16] << 8 | (uint32_t)b256[1].u8[20] << 16;
    }
    inline uint32_t primitive_offset() const {
        return b256[1].u32[7];
    }
    inline int32_t offset(const int &i) const {
        // CHECK(isInternal(i));
        return internal_offset() + (meta(i) & 0b00000111);
    }
    inline int32_t nPrimitives(const int &i) const {
        // CHECK(isLeaf(i));        
        return meta(i);
    }
    inline uint8_t meta(const int &i) const {
        return b256[0].u8[4*i];
    }
    inline bool isInternal(const int &i) const {
        return (meta(i) & 0b11111000) == 0b11111000;
    }
    inline bool isLeaf(const int &i) const {
        return meta(i) < 0b11111000 && meta(i) != 0;
    }
    inline bool isEmpty(const int &i) const {
        return meta(i) == 0;
    }
    inline float fast_exp2(int8_t e) const {
        union {
            uint32_t u;
            float f;
        } converter;
        converter.u = ((e + 127) << 23);
        return converter.f;
    }
    inline float p(const int &i) const {
        return (i == 2)? b256[1].f32[6]: b256[0].f32[6+i];
    }
    inline int8_t e(const int &i) const {
        return b256[1].i8[i * 4];
    }   
    
    #define dequant_compute_t(d, m, e256, p_o256, invDir256)                                                      \
        _mm256_mul_ps(                                                                                          \
            _mm256_fmadd_ps(                                                                                    \
                _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(b256[m].m256i, 24 - 8 * d), m256_0xFF)),  \
                e256,                                                                                           \
                p_o256                                                                                          \
            ),                                                                                                  \
            invDir256                                                                                           \
        )

    std::array<bool, MEBVHAggregate::WIDTH> IntersectSIMD(const Point3f &ray_o, const Vector3f &ray_d, const Float &raytMax, const Vector3f &invDir, const int dirIsNeg[3]) const {
        __m256 p_minus_o = _mm256_set1_ps(p(0) - ray_o.x);
        __m256 invDir_xyz = _mm256_set1_ps(invDir.x);
        __m256 e256 = _mm256_set1_ps(fast_exp2(e(0)));
        __m256 tMin_xyz = dequant_compute_t(0,  dirIsNeg[0], e256, p_minus_o, invDir_xyz);
        __m256 tMax_xyz = dequant_compute_t(0, !dirIsNeg[0], e256, p_minus_o, invDir_xyz);

        p_minus_o  = _mm256_set1_ps(p(1) - ray_o.y);
        invDir_xyz = _mm256_set1_ps(invDir.y);
        e256 = _mm256_set1_ps(fast_exp2(e(1)));
        tMin_xyz = _mm256_max_ps(
            tMin_xyz,
            dequant_compute_t(1,  dirIsNeg[1], e256, p_minus_o, invDir_xyz)
        );
        tMax_xyz = _mm256_min_ps(
            tMax_xyz,
            dequant_compute_t(1, !dirIsNeg[1], e256, p_minus_o, invDir_xyz)
        );
        p_minus_o = _mm256_set1_ps(p(2) - ray_o.z);
        invDir_xyz = _mm256_set1_ps(invDir.z);
        e256 = _mm256_set1_ps(fast_exp2(e(2)));
        tMin_xyz = _mm256_max_ps(
            tMin_xyz,
            dequant_compute_t(2,  dirIsNeg[2], e256, p_minus_o, invDir_xyz)
        );
        tMax_xyz = _mm256_min_ps(
            tMax_xyz,
            dequant_compute_t(2, !dirIsNeg[2], e256, p_minus_o, invDir_xyz)
        );

        __m256 t_cmp = _mm256_cmp_ps(tMin_xyz, tMax_xyz, _CMP_LE_OQ);        
        t_cmp = _mm256_and_ps(t_cmp, _mm256_cmp_ps(tMax_xyz, _mm256_setzero_ps(), _CMP_GT_OQ));
        t_cmp = _mm256_and_ps(t_cmp, _mm256_cmp_ps(tMin_xyz, _mm256_set1_ps(raytMax), _CMP_LT_OQ));
        std::array<bool, MEBVHAggregate::WIDTH> hit;
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i)
            hit[i] = std::isnan(t_cmp[i]);   
        return hit;
    }
    
};




MEBVHAggregate::MEBVHAggregate(std::vector<Primitive> prims, int maxPrimsInNode)
    : maxPrimsInNode(std::min(247, maxPrimsInNode)),
      primitives(std::move(prims)) {

    CHECK(!primitives.empty());
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());

    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });

    std::vector<Primitive> orderedPrims(primitives.size());
    MEBVHBuildNode *root;
    // Build BVH according to selected _splitMethod_
    std::atomic<int> totalNodes{0};
    std::atomic<int> orderedPrimsOffset{0};
    LOG_CONCISE("Starting MEBVH build with %d primitives", (int)primitives.size());
    root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                            &totalNodes, &orderedPrimsOffset, orderedPrims);
    CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());
    primitives.swap(orderedPrims);

    // construct gbounds
    gbounds = Bounds3f();
    for (const auto &subNode: root->subNodes)
        gbounds = Union(gbounds, subNode.bounds);

    // Convert BVH into compact representation in _nodes_ array
    bvhPrimitives.resize(0);
    bvhPrimitives.shrink_to_fit();
    LOG_VERBOSE("MEBVH created with %d nodes for %d primitives (%.2f MB)",
                totalNodes.load(), (int)primitives.size(),
                float(totalNodes.load() * sizeof(LinearOptimizedMEBVHNode)) / (1024.f * 1024.f));
    treeBytes += totalNodes * sizeof(LinearOptimizedMEBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = new LinearOptimizedMEBVHNode[totalNodes];

    CHECK(sizeof(LinearOptimizedMEBVHNode) == 64);

    int offset = flattenMEBVH(root, 0, 1);
    CHECK_EQ(totalNodes.load(), offset);
}

MEBVHBuildNode* MEBVHAggregate::buildRecursive(
    ThreadLocal<Allocator> &threadAllocators,
    pstd::span<BVHPrimitive> bvhPrimitives,
    std::atomic<int> *totalNodes,
    std::atomic<int> *orderedPrimsOffset,
    std::vector<Primitive> &orderedPrims,
    int depth){

    DCHECK_NE(bvhPrimitives.size(), 0);
    CHECK_LE(depth, 30);
    
    Allocator alloc = threadAllocators.Get();
    MEBVHBuildNode *node = alloc.new_object<MEBVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    ++*totalNodes;
    

    // Initialize the first child
    node->subNodes[0].start = 0;
    node->subNodes[0].end = bvhPrimitives.size();
    node->subNodes[0].type = MEBVHBuildNode::SubNode::INTERNAL;

    for (const auto &prim : bvhPrimitives){
        node->subNodes[0].bounds = Union(node->subNodes[0].bounds, prim.bounds);
        node->subNodes[0].centroidBounds = Union(node->subNodes[0].centroidBounds, prim.Centroid());
    }
    int builtChildCnt = 1;

    // split until reach the width or not splitable
    while(builtChildCnt < MEBVHAggregate::WIDTH) {
        // find the child with the largest bounds
        int largestChild = -1;
        Float largestArea = 0;
        for (int i = 0; i < builtChildCnt; i++) {
            Float area = node->subNodes[i].bounds.SurfaceArea();
            if(area <= 0)
                // prevent internal node with zero area, which leads to infinite recursion
                node->subNodes[i].type = MEBVHBuildNode::SubNode::LEAF;
            else if(node->subNodes[i].size() == 0)
                node->subNodes[i].type = MEBVHBuildNode::SubNode::EMPTY;
            else if (node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL && area > largestArea){
                largestArea = node->subNodes[i].bounds.SurfaceArea();
                largestChild = i;
            }
        }
        if(largestChild == -1)
            break;

        MEBVHBuildNode::SubNode &target = node->subNodes[largestChild];
        CHECK(target.size() >= 1);
        
        if(target.bounds.SurfaceArea() == 0 || target.size() == 1){
            // Create leaf _BVHBuildNode_
            target.type = MEBVHBuildNode::SubNode::LEAF;
            continue;
        } else {
            int dim = target.centroidBounds.MaxDimension();
            // Partition primitives into two sets and build children
            if(target.centroidBounds.pMax[dim] == target.centroidBounds.pMin[dim]){
                // Create leaf _BVHBuildNode_
                target.type = MEBVHBuildNode::SubNode::LEAF;
                continue;
            } else {
                int mid = target.mid();
                
                // Partition primitives using approximate SAH
                if (target.size() <= 2) {
                    // Partition primitives into equally sized subsets
                    mid = target.mid();
                    std::nth_element(
                        bvhPrimitives.begin() + target.start,
                        bvhPrimitives.begin() + mid,
                        bvhPrimitives.begin() + target.end,
                        [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                            return a.Centroid()[dim] < b.Centroid()[dim];
                        }
                    );
                    
                    // create a new child
                    MEBVHBuildNode::SubNode &newChild = node->subNodes[builtChildCnt++];
                    newChild.start = target.mid();
                    newChild.end = target.end;
                    newChild.bounds = Bounds3f();
                    newChild.centroidBounds = Bounds3f();
                    newChild.type = MEBVHBuildNode::SubNode::INTERNAL;
                    for (const auto &prim : bvhPrimitives.subspan(newChild.start, newChild.size())) {
                        newChild.bounds = Union(newChild.bounds, prim.bounds);
                        newChild.centroidBounds = Union(newChild.centroidBounds, prim.Centroid());
                    }

                    // reduce the original child
                    target.end = target.mid();
                    target.bounds = Bounds3f();
                    target.centroidBounds = Bounds3f();
                    target.type = MEBVHBuildNode::SubNode::INTERNAL;
                    for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                        target.bounds = Union(target.bounds, prim.bounds);
                        target.centroidBounds = Union(target.centroidBounds, prim.Centroid());
                    }

                } else {
                    // Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                        int b = nBuckets * target.centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }
                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    Float costs[nSplits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        costs[i] += countBelow * boundBelow.SurfaceArea();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        costs[i - 1] += countAbove * boundAbove.SurfaceArea();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        if (costs[i] < minCost) {
                            minCost = costs[i];
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = target.size();
                    minCost = 1.f / 2.f + minCost / target.bounds.SurfaceArea();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (target.size() > maxPrimsInNode || minCost < leafCost) {
                        auto midIter = std::partition(
                            bvhPrimitives.begin() + target.start, bvhPrimitives.begin() + target.end,
                            [=](const BVHPrimitive &bp) {
                                int b =
                                    nBuckets * target.centroidBounds.Offset(bp.Centroid())[dim];
                                if (b == nBuckets)
                                    b = nBuckets - 1;
                                return b <= minCostSplitBucket;
                            });
                        mid = midIter - bvhPrimitives.begin();
                        
                        // create a new child
                        MEBVHBuildNode::SubNode &newChild = node->subNodes[builtChildCnt++];
                        newChild.start = mid;
                        newChild.end = target.end;
                        newChild.bounds = Bounds3f();
                        newChild.centroidBounds = Bounds3f();
                        newChild.type = MEBVHBuildNode::SubNode::INTERNAL;
                        for (const auto &prim : bvhPrimitives.subspan(newChild.start, newChild.size())) {
                            newChild.bounds = Union(newChild.bounds, prim.bounds);
                            newChild.centroidBounds = Union(newChild.centroidBounds, prim.Centroid());
                        }

                        // reduce the original child
                        target.start = target.start;
                        target.end = mid;
                        target.bounds = Bounds3f();
                        target.centroidBounds = Bounds3f();
                        target.type = MEBVHBuildNode::SubNode::INTERNAL;
                        for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                            target.bounds = Union(target.bounds, prim.bounds);
                            target.centroidBounds = Union(target.centroidBounds, prim.Centroid());
                        }
                    } 
                    else {
                        // create leaf _BVHBuildNode_
                        target.type = MEBVHBuildNode::SubNode::LEAF;
                        continue;
                    }
                }
            }
        }           
    }

    // octant sort
    node->sort();

    // unsplitable internal node check
    /*
    for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type != MEBVHBuildNode::SubNode::INTERNAL)
            continue;

        auto &target = node->subNodes[i];
        if(target.bounds.SurfaceArea() == 0 || target.size() == 1)
            target.type = MEBVHBuildNode::SubNode::LEAF;
        else {
            int dim = target.centroidBounds.MaxDimension();
            if(target.centroidBounds.pMax[dim] == target.centroidBounds.pMin[dim])
                target.type = MEBVHBuildNode::SubNode::LEAF;
            else {
                int mid = target.mid();
                if (target.size() <= 2)
                    target.type = MEBVHBuildNode::SubNode::LEAF;
                else {
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    for (const auto &prim : bvhPrimitives.subspan(target.start, target.size())) {
                        int b = nBuckets * target.centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }
                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    Float costs[nSplits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        costs[i] += countBelow * boundBelow.SurfaceArea();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        costs[i - 1] += countAbove * boundAbove.SurfaceArea();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if necessary
                        if (costs[i] < minCost) {
                            minCost = costs[i];
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = target.size();
                    minCost = 1.f / 2.f + minCost / target.bounds.SurfaceArea();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (!(target.size() > maxPrimsInNode || minCost < leafCost)){
                        target.type = MEBVHBuildNode::SubNode::LEAF;
                        continue;
                    }
                }
            }
        }
    }
    */

    // arrange for continous primitives 
    int totalPrims = 0;
    for(int i = 0; i < MEBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF){
            node->subNodes[i].child = nullptr;
            node->subNodes[i].nPrimitives = node->subNodes[i].size();
            totalPrims += node->subNodes[i].size();
        }
    }
    uint32_t primitive_offset = orderedPrimsOffset->fetch_add(totalPrims);

    // we initialize the leaf node first, to ensure the countious primitive arrangement
    for(int i = 0; i < MEBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF){
            node->subNodes[i].firstPrimOffset = primitive_offset;
            for(int j = 0; j < node->subNodes[i].size(); ++j){
                int index = bvhPrimitives[node->subNodes[i].start + j].primitiveIndex;
                orderedPrims[primitive_offset++] = primitives[index];
            }
        }
    }

    if(bvhPrimitives.size() > 128 * 1024){
        ParallelFor(0, MEBVHAggregate::WIDTH, [&](int i){
            if(node->subNodes[i].type == MEBVHBuildNode::SubNode::EMPTY)
                return;            
            else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL){
                node->subNodes[i].child = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(node->subNodes[i].start, node->subNodes[i].size()),
                    totalNodes,
                    orderedPrimsOffset,
                    orderedPrims,
                    depth + 1
                );
                node->subNodes[i].nPrimitives = 0;
                node->subNodes[i].firstPrimOffset = 0;
            }
            else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF)
                return ; // already handled
        });    
    }
    else{

        for(int i = 0; i < MEBVHAggregate::WIDTH; i++){
            if(node->subNodes[i].type == MEBVHBuildNode::SubNode::EMPTY)
                continue;
            else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL){
                node->subNodes[i].child = buildRecursive(
                    threadAllocators,
                    bvhPrimitives.subspan(node->subNodes[i].start, node->subNodes[i].size()),
                    totalNodes,
                    orderedPrimsOffset,
                    orderedPrims,
                    depth + 1
                );
                node->subNodes[i].nPrimitives = 0;
                node->subNodes[i].firstPrimOffset = 0;
            }
            else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF)
                continue; // already handled
        }
    }
    return node;
}

int MEBVHAggregate::flattenMEBVH(MEBVHBuildNode *node, int locate, int offset){
    LinearOptimizedMEBVHNode *linearNode = &nodes[locate];

    // find next_offset
    // find the full bound and primitive_offset of node
    int internal_offset = 0;
    Bounds3f bounds = Bounds3f();
    int32_t first_primitive_offset = -1;
    for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL){
            bounds = Union(bounds, node->subNodes[i].bounds);
            internal_offset++;
        }
        else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF){
            bounds = Union(bounds, node->subNodes[i].bounds);
            if(first_primitive_offset == -1)
                first_primitive_offset = node->subNodes[i].firstPrimOffset;
            else
                CHECK_GT(node->subNodes[i].firstPrimOffset, first_primitive_offset);
        }
    }
    int next_offset = offset + internal_offset;
    internal_offset = 0;

    // initialize LinearMEBVHNode linearNode
    linearNode->init(node, bounds, offset, first_primitive_offset);
    // LinearOptimizedMEBVHNode optimizedNode;
    // optimizedNode.init(node, bounds, offset, first_primitive_offset);

    int primitive_offset = linearNode->primitive_offset();
    for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
        if(node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL){
            // ground truth            
            // Bounds3f &orig_bounds = node->subNodes[i].bounds;
            // Bounds3f quan_bounds = linearNode->bounds(i);
            // for(int dim = 0; dim < 3; ++dim){
            //     CHECK_GE(orig_bounds.pMin[dim], quan_bounds.pMin[dim]);
            //     CHECK_LE(orig_bounds.pMax[dim], quan_bounds.pMax[dim]);
            // }
        }
        else if (node->subNodes[i].type == MEBVHBuildNode::SubNode::LEAF){

            // Bounds3f &orig_bounds = node->subNodes[i].bounds;
            // Bounds3f quan_bounds = linearNode->bounds(i);
            // for(int dim = 0; dim < 3; ++dim){
            //     CHECK_GE(orig_bounds.pMin[dim], quan_bounds.pMin[dim]);
            //     CHECK_LE(orig_bounds.pMax[dim], quan_bounds.pMax[dim]);
            // }
            // CHECK_GT(node->subNodes[i].nPrimitives, 0);
            // CHECK_EQ(node->subNodes[i].firstPrimOffset, primitive_offset);
            primitive_offset += linearNode->nPrimitives(i);
            CHECK_EQ(node->subNodes[i].nPrimitives, linearNode->nPrimitives(i));

        }
        else if(node->subNodes[i].type == MEBVHBuildNode::SubNode::EMPTY){
            CHECK(linearNode->isEmpty(i));
        }
    }

    for(int i = 0; i < MEBVHAggregate::WIDTH; i++){
        if(node->subNodes[i].type == MEBVHBuildNode::SubNode::INTERNAL){
            next_offset = flattenMEBVH(node->subNodes[i].child, linearNode->offset(i), next_offset);
        }
    }

    return next_offset;
}

MEBVHAggregate *MEBVHAggregate::Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters) {
    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 247);
    return new MEBVHAggregate(std::move(prims), maxPrimsInNode);
}

Bounds3f MEBVHAggregate::Bounds() const {
    return gbounds;
}

pstd::optional<ShapeIntersection> MEBVHAggregate::Intersect(const Ray &ray, Float tMax) const {
    if (!nodes)
        return {};

    auto start = std::chrono::high_resolution_clock::now();

    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    uint8_t rayOctant = (uint8_t(invDir.x < 0) << 2) | (uint8_t(invDir.y < 0)  << 1) | (uint8_t(invDir.z < 0));
    int toVisitOffset = 1;
    int nodesToVisit[64];
    int nodesVisited = nodesToVisit[0] = 0;
    uint32_t primitive_offsets[MEBVHAggregate::WIDTH];

    while (toVisitOffset > 0) {
        ++nodesVisited;
        const LinearOptimizedMEBVHNode *node = &nodes[nodesToVisit[--toVisitOffset]];
        auto hit = node->IntersectSIMD(ray.o, ray.d, tMax, invDir, dirIsNeg);
        
        // decompress primitive_offsets;
        int current_offset = node->primitive_offset();
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
            if(node->isLeaf(i)){
                primitive_offsets[i] = current_offset;
                current_offset += node->nPrimitives(i);
            }
        }

        for(int octant = 7; octant >= 0; --octant){
            int child = octant ^ rayOctant;
            if(child >= MEBVHAggregate::WIDTH)
                continue;
            else if(hit[child]){
                if(node->isLeaf(child)){
                    for(int i=0; i < node->nPrimitives(child); ++i){
                        pstd::optional<ShapeIntersection> primSi =
                            primitives[primitive_offsets[child] + i].Intersect(ray, tMax);
                        if(primSi){
                            si = primSi;
                            tMax = si->tHit;
                        }
                    }
                }
                else if(node->isInternal(child)){
                    nodesToVisit[toVisitOffset++] = node->offset(child);
                }
            }
        }
    }
    bvhNodesVisited += nodesVisited;

    Options->intersect_time += std::chrono::high_resolution_clock::now() - start;
    return si;
};

bool MEBVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    
    if (!nodes)
        return {};

    auto start = std::chrono::high_resolution_clock::now();


    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    // Follow ray through BVH nodes to find primitive intersections

    int toVisitOffset = 1;
    int nodesToVisit[64];
    int nodesVisited = 0;
    nodesToVisit[0] = 0;
    uint32_t primitive_offsets[MEBVHAggregate::WIDTH];


    while (toVisitOffset > 0) {
        ++nodesVisited;
        const LinearOptimizedMEBVHNode *node = &nodes[nodesToVisit[--toVisitOffset]];
        auto hit = node->IntersectSIMD(ray.o, ray.d, tMax, invDir, dirIsNeg);

        // decompress primitive_offsets;
        int current_offset = node->primitive_offset();
        for(int i=0; i<MEBVHAggregate::WIDTH; ++i){
            if(node->isLeaf(i)){
                primitive_offsets[i] = current_offset;
                current_offset += node->nPrimitives(i);
            }
        }

        for(int child = 0; child < MEBVHAggregate::WIDTH; ++child){
            if(child >= MEBVHAggregate::WIDTH)
                continue;
            else if(node->isEmpty(child))
                continue;
            else if(hit[child]){
                if(node->isLeaf(child)){
                    for(int i=0; i< node->nPrimitives(child); ++i){
                        if(primitives[primitive_offsets[child] + i].IntersectP(ray, tMax)){
                            bvhNodesVisited += nodesVisited;
                            Options->intersect_time += std::chrono::high_resolution_clock::now() - start;
                            return true;
                        }
                    }
                }
                else if(node->isInternal(child)){
                    nodesToVisit[toVisitOffset++] = node->offset(child);
                }
            }
        }
    }
    bvhNodesVisited += nodesVisited;
    Options->intersect_time += std::chrono::high_resolution_clock::now() - start;
    return false;

};


// KdNodeToVisit Definition
struct KdNodeToVisit {
    const KdTreeNode *node;
    Float tMin, tMax;
};

// KdTreeNode Definition
struct alignas(8) KdTreeNode {
    // KdTreeNode Methods
    void InitLeaf(pstd::span<const int> primNums, std::vector<int> *primitiveIndices);

    void InitInterior(int axis, int aboveChild, Float s) {
        split = s;
        flags = axis | (aboveChild << 2);
    }

    Float SplitPos() const { return split; }
    int nPrimitives() const { return flags >> 2; }
    int SplitAxis() const { return flags & 3; }
    bool IsLeaf() const { return (flags & 3) == 3; }
    int AboveChild() const { return flags >> 2; }

    union {
        Float split;                 // Interior
        int onePrimitiveIndex;       // Leaf
        int primitiveIndicesOffset;  // Leaf
    };

  private:
    uint32_t flags;
};

// EdgeType Definition
enum class EdgeType { Start, End };

// BoundEdge Definition
struct BoundEdge {
    // BoundEdge Public Methods
    BoundEdge() {}

    BoundEdge(Float t, int primNum, bool starting) : t(t), primNum(primNum) {
        type = starting ? EdgeType::Start : EdgeType::End;
    }

    Float t;
    int primNum;
    EdgeType type;
};

STAT_PIXEL_COUNTER("Kd-Tree/Nodes visited", kdNodesVisited);

// KdTreeAggregate Method Definitions
KdTreeAggregate::KdTreeAggregate(std::vector<Primitive> p, int isectCost,
                                 int traversalCost, Float emptyBonus, int maxPrims,
                                 int maxDepth)
    : isectCost(isectCost),
      traversalCost(traversalCost),
      maxPrims(maxPrims),
      emptyBonus(emptyBonus),
      primitives(std::move(p)) {
    // Build kd-tree aggregate
    nextFreeNode = nAllocedNodes = 0;
    if (maxDepth <= 0)
        maxDepth = std::round(8 + 1.3f * Log2Int(int64_t(primitives.size())));
    // Compute bounds for kd-tree construction
    std::vector<Bounds3f> primBounds;
    primBounds.reserve(primitives.size());
    for (Primitive &prim : primitives) {
        Bounds3f b = prim.Bounds();
        bounds = Union(bounds, b);
        primBounds.push_back(b);
    }

    // Allocate working memory for kd-tree construction
    std::vector<BoundEdge> edges[3];
    for (int i = 0; i < 3; ++i)
        edges[i].resize(2 * primitives.size());

    std::vector<int> prims0(primitives.size());
    std::vector<int> prims1((maxDepth + 1) * primitives.size());

    // Initialize _primNums_ for kd-tree construction
    std::vector<int> primNums(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        primNums[i] = i;

    // Start recursive construction of kd-tree
    buildTree(0, bounds, primBounds, primNums, maxDepth, edges, pstd::span<int>(prims0),
              pstd::span<int>(prims1), 0);
}

void KdTreeNode::InitLeaf(pstd::span<const int> primNums,
                          std::vector<int> *primitiveIndices) {
    flags = 3 | (primNums.size() << 2);
    // Store primitive ids for leaf node
    if (primNums.size() == 0)
        onePrimitiveIndex = 0;
    else if (primNums.size() == 1)
        onePrimitiveIndex = primNums[0];
    else {
        primitiveIndicesOffset = primitiveIndices->size();
        for (int pn : primNums)
            primitiveIndices->push_back(pn);
    }
}

void KdTreeAggregate::buildTree(int nodeNum, const Bounds3f &nodeBounds,
                                const std::vector<Bounds3f> &allPrimBounds,
                                pstd::span<const int> primNums, int depth,
                                std::vector<BoundEdge> edges[3], pstd::span<int> prims0,
                                pstd::span<int> prims1, int badRefines) {
    CHECK_EQ(nodeNum, nextFreeNode);
    // Get next free node from _nodes_ array
    if (nextFreeNode == nAllocedNodes) {
        int nNewAllocNodes = std::max(2 * nAllocedNodes, 512);
        KdTreeNode *n = new KdTreeNode[nNewAllocNodes];
        if (nAllocedNodes > 0) {
            std::memcpy(n, nodes, nAllocedNodes * sizeof(KdTreeNode));
            delete[] nodes;
        }
        nodes = n;
        nAllocedNodes = nNewAllocNodes;
    }
    ++nextFreeNode;

    // Initialize leaf node if termination criteria met
    if (primNums.size() <= maxPrims || depth == 0) {
        nodes[nodeNum].InitLeaf(primNums, &primitiveIndices);
        return;
    }

    // Initialize interior node and continue recursion
    // Choose split axis position for interior node
    int bestAxis = -1, bestOffset = -1;
    Float bestCost = Infinity, leafCost = isectCost * primNums.size();
    Float invTotalSA = 1 / nodeBounds.SurfaceArea();
    // Choose which axis to split along
    int axis = nodeBounds.MaxDimension();

    // Choose split along axis and attempt to partition primitives
    int retries = 0;
    size_t nPrimitives = primNums.size();
retrySplit:
    // Initialize edges for _axis_
    for (size_t i = 0; i < nPrimitives; ++i) {
        int pn = primNums[i];
        const Bounds3f &bounds = allPrimBounds[pn];
        edges[axis][2 * i] = BoundEdge(bounds.pMin[axis], pn, true);
        edges[axis][2 * i + 1] = BoundEdge(bounds.pMax[axis], pn, false);
    }
    // Sort _edges_ for _axis_
    std::sort(edges[axis].begin(), edges[axis].begin() + 2 * nPrimitives,
              [](const BoundEdge &e0, const BoundEdge &e1) -> bool {
                  return std::tie(e0.t, e0.type) < std::tie(e1.t, e1.type);
              });

    // Compute cost of all splits for _axis_ to find best
    int nBelow = 0, nAbove = primNums.size();
    for (size_t i = 0; i < 2 * primNums.size(); ++i) {
        if (edges[axis][i].type == EdgeType::End)
            --nAbove;
        Float edgeT = edges[axis][i].t;
        if (edgeT > nodeBounds.pMin[axis] && edgeT < nodeBounds.pMax[axis]) {
            // Compute child surface areas for split at _edgeT_
            Vector3f d = nodeBounds.pMax - nodeBounds.pMin;
            int otherAxis0 = (axis + 1) % 3, otherAxis1 = (axis + 2) % 3;
            Float belowSA =
                2 * (d[otherAxis0] * d[otherAxis1] +
                     (edgeT - nodeBounds.pMin[axis]) * (d[otherAxis0] + d[otherAxis1]));
            Float aboveSA =
                2 * (d[otherAxis0] * d[otherAxis1] +
                     (nodeBounds.pMax[axis] - edgeT) * (d[otherAxis0] + d[otherAxis1]));

            // Compute cost for split at _i_th edge
            Float pBelow = belowSA * invTotalSA, pAbove = aboveSA * invTotalSA;
            Float eb = (nAbove == 0 || nBelow == 0) ? emptyBonus : 0;
            Float cost = traversalCost +
                         isectCost * (1 - eb) * (pBelow * nBelow + pAbove * nAbove);
            // Update best split if this is lowest cost so far
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestOffset = i;
            }
        }
        if (edges[axis][i].type == EdgeType::Start)
            ++nBelow;
    }
    CHECK(nBelow == nPrimitives && nAbove == 0);

    // Try to split along another axis if no good splits were found
    if (bestAxis == -1 && retries < 2) {
        ++retries;
        axis = (axis + 1) % 3;
        goto retrySplit;
    }

    // Create leaf if no good splits were found
    if (bestCost > leafCost)
        ++badRefines;
    if ((bestCost > 4 * leafCost && nPrimitives < 16) || bestAxis == -1 ||
        badRefines == 3) {
        nodes[nodeNum].InitLeaf(primNums, &primitiveIndices);
        return;
    }

    // Classify primitives with respect to split
    int n0 = 0, n1 = 0;
    for (int i = 0; i < bestOffset; ++i)
        if (edges[bestAxis][i].type == EdgeType::Start)
            prims0[n0++] = edges[bestAxis][i].primNum;
    for (int i = bestOffset + 1; i < 2 * nPrimitives; ++i)
        if (edges[bestAxis][i].type == EdgeType::End)
            prims1[n1++] = edges[bestAxis][i].primNum;

    // Recursively initialize kd-tree node's children
    Float tSplit = edges[bestAxis][bestOffset].t;
    Bounds3f bounds0 = nodeBounds, bounds1 = nodeBounds;
    bounds0.pMax[bestAxis] = bounds1.pMin[bestAxis] = tSplit;
    buildTree(nodeNum + 1, bounds0, allPrimBounds, prims0.subspan(0, n0), depth - 1,
              edges, prims0, prims1.subspan(n1), badRefines);
    int aboveChild = nextFreeNode;
    nodes[nodeNum].InitInterior(bestAxis, aboveChild, tSplit);
    buildTree(aboveChild, bounds1, allPrimBounds, prims1.subspan(0, n1), depth - 1, edges,
              prims0, prims1.subspan(n1), badRefines);
}

pstd::optional<ShapeIntersection> KdTreeAggregate::Intersect(const Ray &ray,
                                                             Float rayTMax) const {
    // Compute initial parametric range of ray inside kd-tree extent
    Float tMin, tMax;
    if (!bounds.IntersectP(ray.o, ray.d, rayTMax, &tMin, &tMax))
        return {};

    // Prepare to traverse kd-tree for ray
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    constexpr int maxToVisit = 64;
    KdNodeToVisit toVisit[maxToVisit];
    int toVisitIndex = 0;
    int nodesVisited = 0;

    // Traverse kd-tree nodes in order for ray
    pstd::optional<ShapeIntersection> si;
    const KdTreeNode *node = &nodes[0];
    while (node) {
        // Bail out if we found a hit closer than the current node
        if (rayTMax < tMin)
            break;

        ++nodesVisited;
        if (!node->IsLeaf()) {
            // Visit kd-tree interior node
            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            Float tSplit = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get node child pointers for ray
            const KdTreeNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] < node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            } else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next child node, possibly enqueue other child
            if (tSplit > tMax || tSplit <= 0)
                node = firstChild;
            else if (tSplit < tMin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                toVisit[toVisitIndex].node = secondChild;
                toVisit[toVisitIndex].tMin = tSplit;
                toVisit[toVisitIndex].tMax = tMax;
                ++toVisitIndex;

                node = firstChild;
                tMax = tSplit;
            }

        } else {
            // Check for intersections inside leaf node
            int nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                const Primitive &p = primitives[node->onePrimitiveIndex];
                // Check one primitive inside leaf node
                pstd::optional<ShapeIntersection> primSi = p.Intersect(ray, rayTMax);
                if (primSi) {
                    si = primSi;
                    rayTMax = si->tHit;
                }

            } else {
                for (int i = 0; i < nPrimitives; ++i) {
                    int index = primitiveIndices[node->primitiveIndicesOffset + i];
                    const Primitive &p = primitives[index];
                    // Check one primitive inside leaf node
                    pstd::optional<ShapeIntersection> primSi = p.Intersect(ray, rayTMax);
                    if (primSi) {
                        si = primSi;
                        rayTMax = si->tHit;
                    }
                }
            }

            // Grab next node to visit from todo list
            if (toVisitIndex > 0) {
                --toVisitIndex;
                node = toVisit[toVisitIndex].node;
                tMin = toVisit[toVisitIndex].tMin;
                tMax = toVisit[toVisitIndex].tMax;
            } else
                break;
        }
    }
    kdNodesVisited += nodesVisited;
    return si;
}

bool KdTreeAggregate::IntersectP(const Ray &ray, Float raytMax) const {
    // Compute initial parametric range of ray inside kd-tree extent
    Float tMin, tMax;
    if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
        return false;

    // Prepare to traverse kd-tree for ray
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    constexpr int maxTodo = 64;
    KdNodeToVisit toVisit[maxTodo];
    int toVisitIndex = 0;
    int nodesVisited = 0;
    const KdTreeNode *node = &nodes[0];
    while (node) {
        ++nodesVisited;
        if (node->IsLeaf()) {
            // Check for shadow ray intersections inside leaf node
            int nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                const Primitive &p = primitives[node->onePrimitiveIndex];
                if (p.IntersectP(ray, raytMax)) {
                    kdNodesVisited += nodesVisited;
                    return true;
                }
            } else {
                for (int i = 0; i < nPrimitives; ++i) {
                    int primitiveIndex =
                        primitiveIndices[node->primitiveIndicesOffset + i];
                    const Primitive &prim = primitives[primitiveIndex];
                    if (prim.IntersectP(ray, raytMax)) {
                        kdNodesVisited += nodesVisited;
                        return true;
                    }
                }
            }

            // Grab next node to process from todo list
            if (toVisitIndex > 0) {
                --toVisitIndex;
                node = toVisit[toVisitIndex].node;
                tMin = toVisit[toVisitIndex].tMin;
                tMax = toVisit[toVisitIndex].tMax;
            } else
                break;
        } else {
            // Process kd-tree interior node

            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            Float tSplit = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get node children pointers for ray
            const KdTreeNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] < node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst != 0) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            } else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next child node, possibly enqueue other child
            if (tSplit > tMax || tSplit <= 0)
                node = firstChild;
            else if (tSplit < tMin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                toVisit[toVisitIndex].node = secondChild;
                toVisit[toVisitIndex].tMin = tSplit;
                toVisit[toVisitIndex].tMax = tMax;
                ++toVisitIndex;
                node = firstChild;
                tMax = tSplit;
            }
        }
    }
    kdNodesVisited += nodesVisited;
    return false;
}

KdTreeAggregate *KdTreeAggregate::Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters) {
    int isectCost = parameters.GetOneInt("intersectcost", 5);
    int travCost = parameters.GetOneInt("traversalcost", 1);
    Float emptyBonus = parameters.GetOneFloat("emptybonus", 0.5f);
    int maxPrims = parameters.GetOneInt("maxprims", 1);
    int maxDepth = parameters.GetOneInt("maxdepth", -1);
    return new KdTreeAggregate(std::move(prims), isectCost, travCost, emptyBonus,
                               maxPrims, maxDepth);
}

Primitive CreateAccelerator(const std::string &name, std::vector<Primitive> prims,
                            const ParameterDictionary &parameters) {
    Primitive accel = nullptr;
    LOG_CONCISE("Start create accelerator called with name %s", name);
    if (name == "bvh")
        accel = BVHAggregate::Create(std::move(prims), parameters);
    else if (name == "kdtree")
        accel = KdTreeAggregate::Create(std::move(prims), parameters);
    else if (name == "wbvh")
        accel = WBVHAggregate::Create(std::move(prims), parameters);
    else if (name == "mebvh")
        accel = MEBVHAggregate::Create(std::move(prims), parameters);
    else
        ErrorExit("%s: accelerator type unknown.", name);
    if (!accel)
        ErrorExit("%s: unable to create accelerator.", name);

    parameters.ReportUnused();
    LOG_CONCISE("End create acceleratror");
    return accel;
}

}  // namespace pbrt
