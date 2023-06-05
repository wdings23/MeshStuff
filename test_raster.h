#pragma once

#include "cluster_tree.h"

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


void drawMeshClusterImage2(
    std::vector<float3>& aLightIntensityBuffer,
    std::vector<float3>& aPositionBuffer,
    std::vector<float3>& aNormalBuffer,
    std::vector<float>& afDepthBuffer,
    std::vector<float3>& aColorBuffer,
    std::vector<uint32_t> const& aiClusterAddress,
    std::vector<float3> const& aClusterColors,
    std::vector<MeshCluster*> const& aMeshClusters,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<uint8_t>& vertexNormalBuffer,
    std::vector<uint8_t>& trianglePositionIndexBuffer,
    std::vector<uint8_t>& triangleNormalIndexBuffer,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight);