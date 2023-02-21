#pragma once

#include "vec.h"
#include <vector>

bool canJoinClusters(
    std::vector<float3> const& aVertexPositions0,
    std::vector<float3> const& aVertexPositions1,
    std::vector<uint32_t> const& aiTrianglePositionIndices0,
    std::vector<uint32_t> const& aiTrianglePositionIndices1);

void joinSharedClusters(
    std::vector<float3>& aTotalVertexPositions,
    std::vector<float3>& aTotalVertexNormals,
    std::vector<float2>& aTotalVertexUVs,
    std::vector<uint32_t>& aiTotalTrianglePositionIndices,
    std::vector<uint32_t>& aiTotalTriangleNormalIndices,
    std::vector<uint32_t>& aiTotalTriangleUVIndices,
    std::vector<float3> const& aVertexPositions0,
    std::vector<float3> const& aVertexPositions1,
    std::vector<float3> const& aVertexNormals0,
    std::vector<float3> const& aVertexNormals1,
    std::vector<float2> const& aVertexUVs0,
    std::vector<float2> const& aVertexUVs1,
    std::vector<uint32_t> const& aiTrianglePositionIndices0,
    std::vector<uint32_t> const& aiTrianglePositionIndices1,
    std::vector<uint32_t> const& aiTriangleNormalIndices0,
    std::vector<uint32_t> const& aiTriangleNormalIndices1,
    std::vector<uint32_t> const& aiTriangleUVIndices0,
    std::vector<uint32_t> const& aiTriangleUVIndices1);