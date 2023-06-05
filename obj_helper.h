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

void writeTotalClusterOBJ(
    std::string const& outputTotalClusterFilePath,
    std::string const& objectName,
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices);