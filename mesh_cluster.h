#pragma once

#include "vec.h"
#include <string>
#include <vector>

#define MAX_CLUSTERS_IN_GROUP       128
#define MAX_MIP_LEVELS              2
#define MAX_ASSOCIATED_GROUPS       2
#define MAX_PARENT_CLUSTERS         128

/*
**
*/
struct MeshClusterGroup
{
    uint32_t                            maiClusters[MAX_MIP_LEVELS][MAX_CLUSTERS_IN_GROUP];
    uint32_t                            maiNumClusters[MAX_MIP_LEVELS] = { 0, 0 };
    uint32_t                            miNumMIPS = 0;
    uint32_t                            miIndex = 0;

    float                               mafMinErrors[MAX_MIP_LEVELS] = { 0.0f, 0.0f };
    float                               mafMaxErrors[MAX_MIP_LEVELS] = { 0.0f, 0.0f };

    uint32_t                            miLODLevel = 0;
    float3                              mMinBounds = float3(0.0f, 0.0f, 0.0f);
    float3                              mMaxBounds = float3(0.0f, 0.0f, 0.0f);
    float3                              mMidPosition = float3(0.0f, 0.0f, 0.0f);
    float                               mfRadius = 0.0f;
    float                               mfPctError = 0.0f;

    float3                              mNormal = float3(0.0f, 0.0f, 0.0f);

    float3                              maMaxErrorPositions[MAX_MIP_LEVELS][2];

    float                               mfMinError = 0.0f;
    float                               mfMaxError = 0.0f;

public:
    MeshClusterGroup() 
    {
        for(uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
        {
            memset(&maiClusters[i], 0xff, MAX_CLUSTERS_IN_GROUP * (sizeof(uint32_t) / sizeof(char)));
        }
    }

    MeshClusterGroup(
        std::vector<uint32_t> aiClusters,
        uint32_t iLODLevel,
        uint32_t iMIPLevel,
        uint32_t iIndex,
        uint32_t iClusterIndexOffset)
    {
        maiNumClusters[iMIPLevel] = static_cast<uint32_t>(aiClusters.size());
        miLODLevel = iLODLevel;
        miIndex = iIndex;

        miNumMIPS = (miNumMIPS < iMIPLevel + 1) ? iMIPLevel + 1 : miNumMIPS;

        for(uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
        {
            memset(&maiClusters[i], 0xff, MAX_CLUSTERS_IN_GROUP * (sizeof(uint32_t) / sizeof(char)));
        }

        for(uint32_t i = 0; i < static_cast<uint32_t>(aiClusters.size()); i++)
        {
            maiClusters[iMIPLevel][i] = aiClusters[i] + iClusterIndexOffset;
        }
    }
};

/*
**
*/
struct MeshCluster
{
    uint32_t                                    miClusterGroup = 0;
    uint32_t                                    miLODLevel = 0;
    uint32_t                                    miIndex = 0;

    uint32_t                                    maiClusterGroups[MAX_ASSOCIATED_GROUPS];
    uint32_t                                    miNumClusterGroups = 0;

    uint32_t                                    maiParentClusters[MAX_PARENT_CLUSTERS];
    uint32_t                                    miNumParentClusters = 0;

    float                                       mfError = 0.0f;
    float3                                      mNormal = float3(0.0f, 0.0f, 0.0f);

    float                                       mfAverageDistanceFromLOD0 = 0.0f;

    float3                                      mMinBounds = float3(0.0f, 0.0f, 0.0f);
    float3                                      mMaxBounds = float3(0.0f, 0.0f, 0.0f);
    float3                                      mCenter = float3(0.0f, 0.0f, 0.0f);


    float3                                      mMaxErrorPosition0 = 0.0f;
    float3                                      mMaxErrorPosition1 = 0.0f;

    uint64_t                                    miVertexPositionStartArrayAddress = 0;
    uint32_t                                    miNumVertexPositions = 0;
    uint64_t                                    miVertexNormalStartArrayAddress = 0;
    uint32_t                                    miNumVertexNormals = 0;
    uint64_t                                    miVertexUVStartArrayAddress = 0;
    uint32_t                                    miNumVertexUVs = 0;
    uint64_t                                    miTrianglePositionIndexArrayAddress = 0;
    uint32_t                                    miNumTrianglePositionIndices = 0;
    uint64_t                                    miTriangleNormalIndexArrayAddress = 0;
    uint32_t                                    miNumTriangleNormalIndices = 0;
    uint64_t                                    miTriangleUVIndexArrayAddress = 0;
    uint32_t                                    miNumTriangleUVIndices = 0;

    float4                                      mNormalCone = float4(0.0f, 0.0f, 0.0f, 0.0f);

public:
    MeshCluster()
    {
        memset(maiClusterGroups, 0xff, (sizeof(uint32_t) / sizeof(char)) * MAX_ASSOCIATED_GROUPS);
        memset(maiParentClusters, 0xff, (sizeof(uint32_t) / sizeof(char)) * MAX_PARENT_CLUSTERS);
    }

    MeshCluster(
        uint64_t iVertexPositionAddress,
        uint64_t iVertexNormalAddress,
        uint64_t iVertexUVAddress,
        uint64_t iTrianglePositionIndexArrayAddress,
        uint64_t iTriangleNormalIndexArrayAddress,
        uint64_t iTriangleUVIndexArrayAddress,
        uint32_t iNumVertexPositions,
        uint32_t iNumVertexNormals,
        uint32_t iNumVertexUVs,
        uint32_t iNumTriangleIndices,
        uint32_t iClusterGroup,
        uint32_t iLODLevel,
        uint32_t iIndex,
        uint32_t iMeshClusterGroupIndexOffset)
    {
        miVertexPositionStartArrayAddress = iVertexPositionAddress;
        miTrianglePositionIndexArrayAddress = iTrianglePositionIndexArrayAddress;

        miVertexNormalStartArrayAddress = iVertexNormalAddress;
        miTriangleNormalIndexArrayAddress = iTriangleNormalIndexArrayAddress;

        miVertexUVStartArrayAddress = iVertexUVAddress;
        miTriangleUVIndexArrayAddress = iTriangleUVIndexArrayAddress;

        miNumVertexPositions = iNumVertexPositions;
        miNumTrianglePositionIndices = iNumTriangleIndices;

        miNumVertexNormals = iNumVertexNormals;
        miNumTriangleNormalIndices = iNumTriangleIndices;

        miNumVertexUVs = iNumVertexUVs;
        miNumTriangleUVIndices = iNumTriangleIndices;

        miClusterGroup = iClusterGroup + iMeshClusterGroupIndexOffset;
        miLODLevel = iLODLevel;

        miIndex = iIndex;

        memset(maiClusterGroups, 0xff, (sizeof(uint32_t) / sizeof(char)) * MAX_ASSOCIATED_GROUPS);
        memset(maiParentClusters, 0xff, (sizeof(uint32_t) / sizeof(char)) * MAX_PARENT_CLUSTERS);

        //maiClusterGroups[miNumClusterGroups] = iClusterGroup;
        //++miNumClusterGroups;
    }
};


void saveMeshClusters(
    std::string const& outputFilePath,
    std::vector<MeshCluster*> const& apMeshClusters);

void loadMeshClusters(
    std::vector<MeshCluster>& aMeshClusters,
    std::string const& filePath);

void saveMeshClusterData(
    std::vector<uint8_t> const& aVertexPositionBuffer,
    std::vector<uint8_t> const& aVertexNormalBuffer,
    std::vector<uint8_t> const& aVertexUVBuffer,
    std::vector<uint8_t> const& aiTrianglePositionIndexBuffer,
    std::vector<uint8_t> const& aiTriangleNormalIndexBuffer,
    std::vector<uint8_t> const& aiTriangleUVIndexBuffer,
    std::vector<MeshCluster*> const& apMeshClusters,
    std::string const& outputFilePath);

struct MeshVertexFormat
{
    float3          mPosition;
    float3          mNormal;
    float2          mUV;

    MeshVertexFormat() = default;

    MeshVertexFormat(float3 const& pos, float3 const& norm, float2 const& uv)
    {
        mPosition = pos; mNormal = norm; mUV = uv;
    }
};

struct ConvertedMeshVertexFormat
{
    float4          mPosition;
    float4          mNormal;
    float4          mUV;

    ConvertedMeshVertexFormat() = default;

    ConvertedMeshVertexFormat(float3 const& pos, float3 const& norm, float2 const& uv)
    {
        mPosition = float4(pos, 1.0f); 
        mNormal = float4(norm, 1.0f); 
        mUV = float4(uv.x, uv.y, 0.0f, 0.0f);
    }

    ConvertedMeshVertexFormat(MeshVertexFormat const& v)
    {
        mPosition = float4(v.mPosition, 1.0f);
        mNormal = float4(v.mNormal, 1.0f);
        mUV = float4(v.mUV.x, v.mUV.y, 0.0f, 0.0f);
    }
};

void saveMeshClusterTriangleData(
    std::vector<uint8_t> const& aVertexPositionBuffer,
    std::vector<uint8_t> const& aVertexNormalBuffer,
    std::vector<uint8_t> const& aVertexUVBuffer,
    std::vector<uint8_t> const& aiTrianglePositionIndexBuffer,
    std::vector<uint8_t> const& aiTriangleNormalIndexBuffer,
    std::vector<uint8_t> const& aiTriangleUVIndexBuffer,
    std::vector<MeshCluster*> const& apMeshClusters,
    std::string const& outputFilePath,
    std::string const& outputVertexDataFilePath,
    std::string const& outputIndexDataFilePath);

void loadMeshClusterTriangleData(
    std::string const& filePath,
    std::vector<std::vector<ConvertedMeshVertexFormat>>& aaVertices,
    std::vector<std::vector<uint32_t>>& aaiTriangleVertexIndices);

void loadMeshClusterTriangleDataTableOfContent(
    std::vector<uint32_t>& aiNumClusterVertices,
    std::vector<uint32_t>& aiNumClusterIndices,
    std::vector<uint64_t>& aiVertexBufferArrayOffsets,
    std::vector<uint64_t>& aiIndexBufferArrayOffset,
    std::string const& vertexDataFilePath,
    std::string const& indexDataFilePath);

void loadMeshClusterTriangleDataChunk(
    std::vector<ConvertedMeshVertexFormat>& aClusterTriangleVertices,
    std::vector<uint32_t>& aiClusterTriangleVertexIndices,
    std::string const& vertexDataFilePath,
    std::string const& indexDataFilePath,
    std::vector<uint32_t> const& aiNumClusterVertices,
    std::vector<uint32_t> const& aiNumClusterIndices,
    std::vector<uint64_t> const& aiVertexBufferArrayOffsets,
    std::vector<uint64_t> const& aiIndexBufferArrayOffsets,
    uint32_t iClusterIndex);

