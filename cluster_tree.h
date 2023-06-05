#pragma once

#include "mesh_cluster.h"
#include "vec.h"
#include <string>
#include <vector>

#define MAX_CLUSTER_TREE_NODE_CHILDREN      16
#define MAX_CLUSTER_TREE_NODE_PARENTS       16

struct ClusterTreeNode
{
    uint32_t        miClusterAddress = 0;
    uint32_t        miClusterGroupAddress = 0;
    uint32_t        miLevel = 0;

    uint32_t        maiChildrenAddress[MAX_CLUSTER_TREE_NODE_CHILDREN];
    uint32_t        miNumChildren = 0;

    uint32_t        maiParentAddress[MAX_CLUSTER_TREE_NODE_PARENTS];
    uint32_t        miNumParents = 0;

    float3          mMinBounds = float3(0.0f, 0.0f, 0.0f);
    float3          mMaxBounds = float3(0.0f, 0.0f, 0.0f);

    float           mfScreenSpaceError = FLT_MAX;
    float           mfAverageDistanceFromLOD0 = 0.0f;
    
    float4          mNormalCone;
};

struct ClusterGroupTreeNode
{
    uint32_t        miClusterGroupAddress = 0;
    uint32_t        miLevel = 0;

    uint32_t        maiClusterAddress[MAX_CLUSTER_TREE_NODE_CHILDREN];
    uint32_t        miNumChildClusters = 0;

    float3          mMaxDistanceCurrClusterPosition;
    float3          mMaxDistanceLOD0ClusterPosition;


    float           mfScreenSpacePixelError = 0.0f;
};

/*
**
*/
uint32_t getMeshClusterGroupAddress(
    std::vector<uint8_t>& meshClusterGroupBuffer,
    uint32_t iLODLevel,
    uint32_t iClusterGroup);

uint32_t getMeshClusterAddress(
    std::vector<uint8_t>& meshClusterBuffer,
    uint32_t iLODLevel,
    uint32_t iCluster);

uint32_t getMeshClusterAddress(
    std::vector<uint8_t>& meshClusterBuffer,
    uint32_t iLODLevel,
    uint32_t iCluster);

void createTreeNodes(
    std::vector<ClusterTreeNode>& aNodes,
    uint32_t iNumLODLevels,
    std::vector<uint8_t>& aMeshClusterData,
    std::vector<uint8_t>& aMeshClusterGroupData,
    std::vector<std::vector<MeshClusterGroup>> const& aaMeshClusterGroups,
    std::vector<std::vector<MeshCluster>> const& aaMeshClusters);

void createTreeNodes2(
    std::vector<ClusterTreeNode>& aNodes,
    uint32_t iNumLODLevels,
    std::vector<uint8_t>& aMeshClusterData,
    std::vector<uint8_t>& aMeshClusterGroupData,
    std::vector<std::vector<MeshClusterGroup>> const& aaMeshClusterGroups,
    std::vector<std::vector<MeshCluster>> const& aaMeshClusters,
    std::vector<std::pair<float3, float3>> const& aTotalMaxClusterDistancePositionFromLOD0);

void saveClusterGroupTreeNodes(
    std::string const& outputFilePath,
    std::vector<ClusterGroupTreeNode> const& aClusterGroupNodes);

void loadClusterGroupTreeNodes(
    std::vector<ClusterGroupTreeNode>& aClusterGroupNodes,
    std::string const& filePath);

void saveClusterTreeNodes(
    std::string const& outputFilePath,
    std::vector<ClusterTreeNode> const& aClusterNodes);

void loadClusterTreeNodes(
    std::vector<ClusterTreeNode>& aClusterNodes,
    std::string const& filePath);
