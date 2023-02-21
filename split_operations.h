#pragma once

#include "vec.h"
#include <mutex>
#include <vector>

void checkClusterAdjacency(
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleIndices,
    std::vector<uint32_t> const& aiClusterTriangleIndices);

void createSplitClusters(
    std::vector<std::vector<float3>>& aaSplitClusterVertexPositions,
    std::vector<std::vector<float3>>& aaSplitClusterVertexNormals,
    std::vector<std::vector<float2>>& aaSplitClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleUVIndices,
    std::vector<uint32_t>& aiDeleteClusters,
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices,
    std::vector<std::vector<uint32_t>> const& aaiSplitClusterTriangleIndices,
    uint32_t iCheckCluster);

void createSplitClusters2(
    std::vector<std::vector<float3>>& aaSplitClusterVertexPositions,
    std::vector<std::vector<float3>>& aaSplitClusterVertexNormals,
    std::vector<std::vector<float2>>& aaSplitClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleUVIndices,
    std::vector<uint32_t>& aiDeleteClusters,
    std::vector<float3> const& aClusterVertexPositions,
    std::vector<float3> const& aClusterVertexNormals,
    std::vector<float2> const& aClusterVertexUVs,
    std::vector<uint32_t> const& aiClusterTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterTriangleUVIndices,
    std::vector<std::vector<uint32_t>> const& aaiSplitClusterTriangleIndices);

bool splitDiscontigousClusters(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    std::vector<uint32_t>& aiDeleteClusters,
    std::mutex& threadMutex,
    uint32_t iCheckCluster,
    uint32_t iLODLevel);



void splitCluster(
    std::vector<std::vector<float3>>& aaVertexPositions,
    std::vector<std::vector<float3>>& aaVertexNormals,
    std::vector<std::vector<float2>>& aaVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexUVIndices,
    std::vector<float3> const& aOrigVertexPositions,
    std::vector<float3> const& aOrigVertexNormals,
    std::vector<float2> const& aOrigVertexUVs,
    std::vector<uint32_t> const& aiOrigVertexPositionIndices,
    std::vector<uint32_t> const& aiOrigVertexNormalIndices,
    std::vector<uint32_t> const& aiOrigVertexUVIndices,
    uint32_t iMaxTriangles);