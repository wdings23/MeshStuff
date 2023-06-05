#pragma once

#include <string>
#include <vector>

#include "vec.h"


bool readMetisClusterFile(
    std::vector<uint32_t>& aiClusters,
    std::string const& filePath);

void outputMeshClusters2(
    std::vector<float3>& aVertexPositions,
    std::vector<float3>& aVertexNormals,
    std::vector<float2>& aVertexUVs,
    std::vector<uint32_t>& aiRetTrianglePositionIndices,
    std::vector<uint32_t>& aiRetTriangleNormalIndices,
    std::vector<uint32_t>& aiRetTriangleUVIndices,
    std::vector<uint32_t> const& aiClusterIndices,
    std::vector<float3> const& aTriangleVertexPositions,
    std::vector<float3> const& aTriangleVertexNormals,
    std::vector<float2> const& aTriangleVertexUVs,
    std::vector<uint32_t> const& aiTrianglePositionIndices,
    std::vector<uint32_t> const& aiTriangleNormalIndices,
    std::vector<uint32_t> const& aiTriangleUVIndices,
    uint32_t iLODLevel,
    uint32_t iOutputClusterIndex);

void buildMETISMeshFile2(
    std::string const& outputFilePath,
    std::vector<uint32_t> const& aiTrianglePositionIndices);

void buildMETISGraphFile(
    std::string const& outputFilePath,
    std::vector<std::vector<float3>> const& aaVertexPositions,
    bool bUseCUDA,
    bool bOnlyEdgeAdjacent);

void buildMETISMeshFile(
    std::string const& outputFilePath,
    std::vector<tinyobj::shape_t> aShapes,
    tinyobj::attrib_t const& attrib);