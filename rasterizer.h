#pragma once

#include "vec.h"
#include <vector>
#include "Camera.h"

struct face;

void outputMeshToImage(
    std::string const& outputDirectory,
    std::string const& outputName,
    std::vector<float3> const& aVertexPositions,
    std::vector<uint32_t> const& aiTriangles,
    CCamera const& camera,
    uint32_t iImageWidth,
    uint32_t iImageHeight);