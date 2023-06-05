#include "vertex_mapping_operations.h"

#include <cassert>
#include <mutex>

#include "LogPrint.h"

/*
**
*/
static std::mutex sMutex;
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
    uint32_t iUpperLODLevel)
{

    auto start = std::chrono::high_resolution_clock::now();

    auto const& aMeshClusters = aaMeshClusters[iLODLevel];
    uint32_t iNumMeshClusters = static_cast<uint32_t>(aMeshClusters.size());
    for(uint32_t iMeshCluster = 0; iMeshCluster < iNumMeshClusters; iMeshCluster++)
    {
        float fMaxVertexPositionDistance = 0.0f;

        auto const& meshCluster = aMeshClusters[iMeshCluster];
        std::pair<float3, float3> maxErrorPositions;

        uint32_t const kiMIPLevel = 0;

        // get closest vertex positions from the upper clusters (given upper LOD)
        for(uint32_t iVertex = 0; iVertex < static_cast<uint32_t>(meshCluster.miNumVertexPositions); iVertex++)
        {
            uint32_t iBestUpperMeshCluster = UINT32_MAX;
            uint32_t iBestUpperMeshClusterGroup = UINT32_MAX;
            uint32_t iBestUpperClusterVertexID = UINT32_MAX;
            float3 bestUpperClusterVertexPosition = float3(FLT_MAX, FLT_MAX, FLT_MAX);
            float fShortestDistance = FLT_MAX;

            float3 const* aClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshCluster.miVertexPositionStartArrayAddress * sizeof(float3));
            auto const& vertexPosition = aClusterVertexPositions[iVertex];

            if(iLODLevel == 0)
            {
                fShortestDistance = 0.0f;
                bestUpperClusterVertexPosition = aClusterVertexPositions[iVertex];
                iBestUpperClusterVertexID = iMeshCluster;
                iBestUpperMeshClusterGroup = 0;
                iBestUpperMeshCluster = iMeshCluster;

                continue;
            }

            // mesh cluster group of given upper LOD
            uint32_t iNumUpperMeshClusterGroups = static_cast<uint32_t>(aaMeshClusterGroups[iUpperLODLevel].size());
            for(uint32_t iUpperMeshClusterGroup = 0; iUpperMeshClusterGroup < iNumUpperMeshClusterGroups; iUpperMeshClusterGroup++)
            {
                auto const& meshClusterGroup = aaMeshClusterGroups[iUpperLODLevel][iUpperMeshClusterGroup];

                // mesh clusters of given upper LOD associated with this mesh cluster group
                for(uint32_t iUpperMeshClusterIndex = 0; iUpperMeshClusterIndex < static_cast<uint32_t>(meshClusterGroup.maiNumClusters[kiMIPLevel]); iUpperMeshClusterIndex++)
                {
                    uint32_t iMeshClusterID = meshClusterGroup.maiClusters[kiMIPLevel][iUpperMeshClusterIndex];
                    auto const& upperMeshCluster = *apTotalMeshClusters[iMeshClusterID];

                    float3 const* aUpperMeshClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + upperMeshCluster.miVertexPositionStartArrayAddress * sizeof(float3));
                    for(uint32_t iUpperClusterVertex = 0; iUpperClusterVertex < static_cast<uint32_t>(upperMeshCluster.miNumVertexPositions); iUpperClusterVertex++)
                    {
                        auto const& upperMeshClusterVertexPosition = aUpperMeshClusterVertexPositions[iUpperClusterVertex];
                        float fDistance = lengthSquared(vertexPosition - upperMeshClusterVertexPosition);
                        if(fDistance < fShortestDistance)
                        {
                            fShortestDistance = fDistance;
                            bestUpperClusterVertexPosition = upperMeshClusterVertexPosition;
                            iBestUpperClusterVertexID = iUpperClusterVertex;
                            iBestUpperMeshClusterGroup = iUpperMeshClusterGroup;
                            iBestUpperMeshCluster = iMeshClusterID;
                        }
                    }

                }   // for cluster = 0 to num upper cluster

            }   // for cluster group = 0 to num upper cluster group

            // verify
            float3 const* aCheckClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + apTotalMeshClusters[iBestUpperMeshCluster]->miVertexPositionStartArrayAddress * sizeof(float3));
            auto const& checkVertexPosition = aCheckClusterVertexPositions[iBestUpperClusterVertexID];
            float fCheckDistance = length(checkVertexPosition - bestUpperClusterVertexPosition);
            assert(fCheckDistance <= 1.0e-5f);
            //DEBUG_PRINTF("found cluster %d of lod %d with position (%.4f, %.4f, %.4f) compared to\n cluster %d of lod %d with position (%.4f, %.4f, %.4f)\ndistance: %.4f\n",
            //    iBestUpperMeshCluster,
            //    iUpperLODLevel,
            //    checkVertexPosition.x,
            //    checkVertexPosition.y,
            //    checkVertexPosition.z,
            //    iMeshCluster,
            //    iLODLevel,
            //    vertexPosition.x,
            //    vertexPosition.y,
            //    vertexPosition.z,
            //    fShortestDistance);

            VertexMappingInfo mappingInfo;
            mappingInfo.miCluster = iBestUpperMeshCluster;
            mappingInfo.miClusterGroup = iBestUpperMeshClusterGroup;
            mappingInfo.miLODLevel = iUpperLODLevel;
            mappingInfo.mPosition = bestUpperClusterVertexPosition;
            mappingInfo.miClusterVertexID = iBestUpperClusterVertexID;
            mappingInfo.mfDistance = fShortestDistance;

            aVertexMapping[std::make_pair(iMeshCluster, iVertex)] = mappingInfo;

            if(fShortestDistance > fMaxVertexPositionDistance)
            {
                fMaxVertexPositionDistance = fShortestDistance;

                maxErrorPositions = std::make_pair(mappingInfo.mPosition, vertexPosition);
            }

        }   // for vertex = 0 to num vertices in cluster

        {
            std::lock_guard<std::mutex> lock(sMutex);
            afMaxClusterDistances.push_back(fMaxVertexPositionDistance);
            aMaxErrorPositions.push_back(maxErrorPositions);
        }

    }   // for cluster = 0 to num clusters

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    DEBUG_PRINTF("took total %lld seconds to get lod distance error between lod %d and lod %d\n", iSeconds, iLODLevel, iUpperLODLevel);
}