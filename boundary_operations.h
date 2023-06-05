#pragma once

#include <map>
#include <vector>
#include "vec.h"


struct BoundaryEdgeInfo
{
    uint32_t            miClusterGroup;
    uint32_t            miPos0;
    uint32_t            miPos1;
    float3              mPos0;
    float3              mPos1;

    BoundaryEdgeInfo() = default;

    BoundaryEdgeInfo(
        uint32_t iClusterGroup, 
        uint32_t iPos0, 
        uint32_t iPos1, 
        float3 const& pos0, 
        float3 const& pos1)
    {
        miClusterGroup = iClusterGroup;
        miPos0 = iPos0;
        miPos1 = iPos1;
        mPos0 = pos0;
        mPos1 = pos1;
    }
};


void getBoundaryEdges(
    std::vector<BoundaryEdgeInfo>& aBoundaryEdges,
    std::vector<std::vector<float3>>const& aaClusterGroupVertexPositions,
    std::vector<std::vector<float3>>const& aaClusterGroupVertexNormals,
    std::vector<std::vector<float2>>const& aaClusterGroupVertexUVs,
    std::vector<std::vector<uint32_t>>const& aaiClusterGroupTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>const& aaiClusterGroupTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>const& aaiClusterGroupTriangleUVIndices);

void getClusterGroupBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupNonBoundaryVertices,
    std::vector<std::vector<float3>> const& aaClusterGroupVertexPositions,
    uint32_t const& iNumClusterGroups);

void getInnerEdgesAndVertices(
    std::vector<std::vector<uint32_t>>& aaiValidClusterGroupEdges,
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaValidClusterGroupEdgePairs,
    std::vector<std::map<uint32_t, uint32_t>>& aaValidVertices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupTriWithEdges,
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaClusterGroupEdges,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTriangles,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupNonBoundaryVertices,
    uint32_t const& iNumClusterGroups);

void getBoundaryAndNonBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiBoundaryVertices,
    std::vector<std::vector<uint32_t>>& aaiNonBoundaryVertices,
    std::vector<std::vector<float3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiTrianglePositionIndices);