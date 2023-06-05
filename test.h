#pragma once

#include <vector>

#include "vec.h"
void checkClusterGroupBoundaryVerticesCUDA(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<vec3>> const& aaClusterGroupVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTrianglePositionIndices);

void buildClusterAdjacencyCUDA(
    std::vector<std::vector<uint32_t>>& aaiNumAdjacentVertices,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    bool bOnlyEdgeAdjacent);

void getClusterGroupBoundaryVerticesCUDA(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<vec3>> const& aaClusterGroupVertexPositions);

void computeEdgeCollapseInfoCUDA(
    std::vector<float>& afCollapseCosts,
    std::vector<vec3>& aOptimalVertexPositions,
    std::vector<vec3>& aOptimalVertexNormals,
    std::vector<vec2>& aOptimalVertexUVs,
    std::vector<std::pair<uint32_t, uint32_t>>& aEdges,
    std::vector<vec3> const& aClusterGroupVertexPositions,
    std::vector<vec3> const& aClusterGroupVertexNormals,
    std::vector<vec2> const& aClusterGroupVertexUVs,
    std::vector<std::pair<uint32_t, uint32_t>> const& aiValidClusterGroupEdgePairs,
    std::vector<uint32_t> const& aiClusterGroupNonBoundaryVertices,
    std::vector<uint32_t> const& aiClusterGroupTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangleUVIndices,
    std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices);

void getShortestVertexDistancesCUDA(
    std::vector<float>& afClosestDistances,
    std::vector<uint32_t>& aiClosestVertexPositionIndices,
    std::vector<vec3> const& aVertexPositions0,
    std::vector<vec3> const& aVertexPositions1);

void buildClusterEdgeAdjacencyCUDA(
    std::vector<std::vector<uint32_t>>& aaiAdjacentEdgeClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices);

void buildClusterEdgeAdjacencyCUDA2(
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaiAdjacentEdgeClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices);

void getSortedEdgeAdjacentClustersCUDA(
    std::vector<std::vector<uint32_t>>& aaiSortedAdjacentEdgeClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices);

void getProjectVertexDistancesCUDA(
    std::vector<vec3>& aProjectedPositions,
    std::vector<vec3> const& aTriangleVertexPositions0,
    std::vector<vec3> const& aTriangleVertexPositions1);

void getClusterGroupBoundaryVerticesCUDA2(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<vec3>> const& aaClusterGroupVertexPositions);