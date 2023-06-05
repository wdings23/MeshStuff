#pragma once

#include "vec.h"

#include <map>
#include <vector>

#include "mesh_cluster.h"


struct VertexMappingInfo
{
    float3              mPosition;
    uint32_t            miClusterVertexID;
    uint32_t            miCluster;
    uint32_t            miClusterGroup;
    uint32_t            miLODLevel;
    float               mfDistance;
    uint32_t            miMIPLevel;
};

void getVertexMappingAndMaxDistances(
    std::vector<float>& afMaxClusterDistances,
    std::map<std::pair<uint32_t, uint32_t>, VertexMappingInfo>& aVertexMapping,
    std::vector<std::pair<float3, float3>>& aMaxErrorPositions,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<std::vector<MeshCluster>> const& aaMeshClusters,
    std::vector<std::vector<MeshClusterGroup>> const& aaMeshClusterGroups,
    std::vector<MeshCluster*> const& apTotalMeshClusters,
    std::vector<MeshClusterGroup*> const& apTotalMeshClusterGroups,
    uint32_t iLODLevel,
    uint32_t iUpperLODLevel);