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

void moveVertices(
    std::vector<float3>& aDestClusterVertexPositions,
    std::vector<float3>& aDestClusterVertexNormals,
    std::vector<float2>& aDestClusterVertexUVs,
    std::vector<uint32_t>& aiDestClusterPositionIndices,
    std::vector<uint32_t>& aiDestClusterNormalIndices,
    std::vector<uint32_t>& aiDestClusterUVIndices,
    std::vector<float3> const& aSrcClusterVertexPositions,
    std::vector<float3> const& aSrcClusterVertexNormals,
    std::vector<float2> const& aSrcClusterVertexUVs,
    std::vector<uint32_t> const& aiSrcClusterPositionIndices,
    std::vector<uint32_t> const& aiSrcClusterNormalIndices,
    std::vector<uint32_t> const& aiSrcClusterUVIndices,
    uint32_t iSrcCluster,
    uint32_t iDestCluster);