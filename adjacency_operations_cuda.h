#pragma once

#include "vec.h"
void buildClusterEdgeAdjacencyCUDA3(
    std::vector<std::vector<uint32_t>>& aaiNumAdjacentClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiBoundaryVertexIndices);