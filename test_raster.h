#pragma once

#include "mesh_cluster.h"
#include "vec.h"
#include <string>
#include <vector>

#define MAX_CLUSTER_TREE_NODE_CHILDREN      8
#define MAX_CLUSTER_TREE_NODE_PARENTS       8

struct ClusterTreeNode
{
    uint32_t        miClusterAddress = 0;
    uint32_t        miClusterGroupAddress = 0;
    uint32_t        miLevel = 0;

    uint32_t        maiChildrenAddress[MAX_CLUSTER_TREE_NODE_CHILDREN];
    uint32_t        miNumChildren = 0;

    uint32_t        maiParentAddress[MAX_CLUSTER_TREE_NODE_PARENTS];
    uint32_t        miNumParents = 0;

    float3          mMaxDistanceCurrLODClusterPosition;
    float3          mMaxDistanceLOD0ClusterPosition;

    float           mfScreenSpaceError = FLT_MAX;
    float           mfAverageDistanceFromLOD0 = 0.0f;

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

void testClusterLOD(
    std::vector<uint8_t>& aMeshClusterData,
    std::vector<uint8_t>& aMeshClusterGroupData,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<uint8_t>& trianglePositionIndexBuffer,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    ClusterTreeNode const& startTreeNode,
    std::vector<uint8_t> const& clusterGroupBuffer,
    std::vector<uint8_t> const& clusterBuffer,
    float fPixelErrorThreshold);

void testClusterLOD2(
    std::vector<ClusterTreeNode>& aClusterNodes,
    std::vector<ClusterGroupTreeNode>& aClusterGroupNodes,
    std::vector<uint32_t> const& aiLevelStartGroupIndices,
    std::vector<uint32_t> const& aiNumLevelGroupNodes,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    float fPixelErrorThreshold);

void testClusterLOD3(
    std::vector<uint32_t>& aiDrawClusterAddress,
    std::vector<ClusterTreeNode>& aClusterNodes,
    std::vector<ClusterGroupTreeNode>& aClusterGroupNodes,
    std::vector<uint32_t> const& aiLevelStartGroupIndices,
    std::vector<uint32_t> const& aiNumLevelGroupNodes,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    float fPixelErrorThreshold);

void drawMeshClusterImage(
    std::vector<uint32_t> const& aiClusterAddress,
    std::vector<MeshCluster*> const& aMeshClusters,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<uint8_t>& trianglePositionIndexBuffer,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    std::string const& outputDirectory,
    std::string const& outputFileName);