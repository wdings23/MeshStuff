#pragma once

#include "vec.h"
#include "mat4.h"

#include <map>
#include <string>
#include <vector>

struct EdgeCollapseInfo
{
    float3      mOptimalVertexPosition;
    float3      mOptimalNormal;
    float2      mOptimalUV;
    float       mfCost;
};

void simplifyClusterGroup(
    std::map<uint32_t, mat4>& aQuadrics,
    std::vector<float3>& aClusterGroupVertexPositions,
    std::vector<float3>& aClusterGroupVertexNormals,
    std::vector<float2>& aClusterGroupVertexUVs,
    std::vector<uint32_t>& aiClusterGroupNonBoundaryVertices,
    std::vector<uint32_t>& aiClusterGroupBoundaryVertices,
    std::vector<uint32_t>& aiClusterGroupTrianglePositions,
    std::vector<uint32_t>& aiClusterGroupTriangleNormals,
    std::vector<uint32_t>& aiClusterGroupTriangleUVs,
    std::vector<std::pair<uint32_t, uint32_t>>& aValidClusterGroupEdgePairs,
    float& fTotalError,
    //std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices,
    uint32_t iMaxTriangles,
    uint32_t iClusterGroup,
    uint32_t iLODLevel,
    std::string const& meshModelName,
    std::string const& homeDirectory);