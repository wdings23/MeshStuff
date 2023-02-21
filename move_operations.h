#pragma once

#include "vec.h"
#include <vector>

bool moveTriangles(
    std::vector<std::vector<float3>>& aaVertexPositions,
    std::vector<std::vector<float3>>& aaVertexNormals,
    std::vector<std::vector<float2>>& aaVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexUVIndices,
    uint32_t iSrcCluster,
    uint32_t iMaxTriangleVertexCount);

bool mergeTriangles(
    std::vector<std::vector<float3>>& aaVertexPositions,
    std::vector<std::vector<float3>>& aaVertexNormals,
    std::vector<std::vector<float2>>& aaVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexUVIndices,
    uint32_t iSrcCluster);