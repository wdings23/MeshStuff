#pragma once

#include <stdint.h>
#include <vector>

#include "mesh_cluster.h"

void testClusterRequests(
    std::vector<uint32_t>& saiNumClusterVertices,
    std::vector<uint32_t>& saiNumClusterIndices,
    std::vector<uint64_t>& saiVertexBufferArrayOffsets,
    std::vector<uint64_t>& saiIndexBufferArrayOffsets,
    std::vector<uint32_t> const& aiDrawClusters);

void testGetClusterRequests(
    std::vector<uint8_t>& aClusterRequestInfo);


void testUploadClusterData(
    void* paClusterRequestInfo,
    std::vector<uint32_t> const& aiDrawList,
    std::vector<std::vector<ConvertedMeshVertexFormat>> const& aaVertices,
    std::vector<std::vector<uint32_t>> const& aaiIndices,
    uint32_t iRequestClusterDataSize);

void testVerifyStreamClusterData(
    void const* aClusterInfoRequestBuffer,
    std::vector<uint32_t> const& aiDrawClusters,
    std::vector<std::vector<ConvertedMeshVertexFormat>> const& aaClusterTriangleVertices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleVertexIndices);

void* testGetVertexDataBuffer();
void* testGetIndexDataBuffer();
void* testGetRequestClusterInfo();
