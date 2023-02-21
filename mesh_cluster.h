#pragma once

#include "vec.h"
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
    uint32_t                            maiNumClusters[MAX_MIP_LEVELS];
    uint32_t                            miNumMIPS = 0;
    uint32_t                            miIndex;

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

    float                               mfMinError;
    float                               mfMaxError;

public:
    MeshClusterGroup() = default;

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
    uint32_t                                    miClusterGroup;
    uint32_t                                    miLODLevel;
    uint32_t                                    miIndex;

    uint32_t                                    maiClusterGroups[MAX_ASSOCIATED_GROUPS];
    uint32_t                                    miNumClusterGroups = 0;

    uint32_t                                    maiParentClusters[MAX_PARENT_CLUSTERS];
    uint32_t                                    miNumParentClusters = 0;

    float                                       mfError = 0.0f;
    float3                                      mNormal = float3(0.0f, 0.0f, 0.0f);

    float                                       mfAverageDistanceFromLOD0 = 0.0f;

    float3                                      mMinBounds;
    float3                                      mMaxBounds;
    float3                                      mCenter;

    float                                       mfPctError = 0.0f;

    float3                                      mMaxErrorPosition0;
    float3                                      mMaxErrorPosition1;

    uint64_t                                    miVertexPositionStartAddress;
    uint32_t                                    miNumVertexPositions;
    uint64_t                                    miVertexNormalStartAddress;
    uint32_t                                    miNumVertexNormals;
    uint64_t                                    miVertexUVStartAddress;
    uint32_t                                    miNumVertexUVs;
    uint64_t                                    miTrianglePositionIndexAddress;
    uint32_t                                    miNumTrianglePositionIndices;
    uint64_t                                    miTriangleNormalIndexAddress;
    uint32_t                                    miNumTriangleNormalIndices;
    uint64_t                                    miTriangleUVIndexAddress;
    uint32_t                                    miNumTriangleUVIndices;

    float4                                      mNormalCone;

public:
    MeshCluster() = default;

    MeshCluster(
        uint64_t iVertexPositionAddress,
        uint64_t iVertexNormalAddress,
        uint64_t iVertexUVAddress,
        uint64_t iTrianglePositionIndexAddress,
        uint64_t iTriangleNormalIndexAddress,
        uint64_t iTriangleUVIndexAddress,
        uint32_t iNumVertexPositions,
        uint32_t iNumVertexNormals,
        uint32_t iNumVertexUVs,
        uint32_t iNumTriangleIndices,
        uint32_t iClusterGroup,
        uint32_t iLODLevel,
        uint32_t iIndex,
        uint32_t iMeshClusterGroupIndexOffset)
    {
        miVertexPositionStartAddress = iVertexPositionAddress;
        miTrianglePositionIndexAddress = iTrianglePositionIndexAddress;

        miVertexNormalStartAddress = iVertexNormalAddress;
        miTriangleNormalIndexAddress = iTriangleNormalIndexAddress;

        miVertexUVStartAddress = iVertexUVAddress;
        miTriangleUVIndexAddress = iTriangleUVIndexAddress;

        miNumVertexPositions = iNumVertexPositions;
        miNumTrianglePositionIndices = iNumTriangleIndices;

        miNumVertexNormals = iNumVertexNormals;
        miNumTriangleNormalIndices = iNumTriangleIndices;

        miNumVertexUVs = iNumVertexUVs;
        miNumTriangleUVIndices = iNumTriangleIndices;

        miClusterGroup = iClusterGroup + iMeshClusterGroupIndexOffset;
        miLODLevel = iLODLevel;

        miIndex = iIndex;

        //maiClusterGroups[miNumClusterGroups] = iClusterGroup;
        //++miNumClusterGroups;
    }
};