#pragma once

#include "vec.h"
#include <string>
#include <vector>

void writeOBJFile(
    std::vector<float3> const& aVertexPositions,
    std::vector<float3> const& aVertexNormals,
    std::vector<float2> const& aVertexUV,
    std::vector<uint32_t> const& aiPositionIndices,
    std::vector<uint32_t> const& aiNormalIndices,
    std::vector<uint32_t> const& aiUVIndices,
    std::string const& outputFilePath,
    std::string const& objectName);