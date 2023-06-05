#include "split_operations.h"
#include "LogPrint.h"
#include "obj_helper.h"
#include "tiny_obj_loader.h"
#include "metis_operations.h"
#include "move_operations.h"
#include "system_command.h"
#include "obj_helper.h"

#include <cassert>
#include <filesystem>
#include <map>
#include <sstream>

void visitAdjacentTris(
    std::vector<uint32_t>& aiVisitedTris,
    std::vector<std::vector<uint32_t>> const& aaiAdjacentTri,
    uint32_t iCurrTri,
    uint32_t iAdjacentIndex,
    uint32_t iStack);

/*
**
*/
bool splitDiscontigousClusters(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    std::vector<uint32_t>& aiDeleteClusters,
    std::mutex& threadMutex,
    uint32_t iCheckCluster,
    uint32_t iLODLevel)
{
    std::vector<std::vector<float3>> aaSplitClusterVertexPositions;
    std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleIndices;
    checkClusterAdjacency(
        aaiSplitClusterTriangleIndices,
        aaiClusterTrianglePositionIndices[iCheckCluster]);

    if(aaiSplitClusterTriangleIndices.size() > 1)
    {
        DEBUG_PRINTF("!!! LOD %d cluster %d (%lld clusters) is separated into %lld parts !!!\n",
            iLODLevel,
            iCheckCluster,
            aaClusterVertexPositions.size(),
            aaiSplitClusterTriangleIndices.size());

        for(uint32_t i = 0; i < static_cast<uint32_t>(aaiSplitClusterTriangleIndices.size()); i++)
        {
            std::vector<float3> aSplitClusterVertexPositions;
            std::vector<float3> aSplitClusterVertexNormals;
            std::vector<float2> aSplitClusterVertexUVs;
            std::vector<uint32_t> aiSplitClusterVertexPositionIndices;
            std::vector<uint32_t> aiSplitClusterVertexNormalIndices;
            std::vector<uint32_t> aiSplitClusterVertexUVIndices;
            uint32_t iPositionCount = 0, iNormalCount = 0, iUVCount = 0;
            for(uint32_t j = 0; j < static_cast<uint32_t>(aaiSplitClusterTriangleIndices[i].size()); j++)
            {
                uint32_t iTriIndex0 = aaiSplitClusterTriangleIndices[i][j] * 3;
                uint32_t iTriIndex1 = aaiSplitClusterTriangleIndices[i][j] * 3 + 1;
                uint32_t iTriIndex2 = aaiSplitClusterTriangleIndices[i][j] * 3 + 2;

                uint32_t iPos0 = aaiClusterTrianglePositionIndices[iCheckCluster][iTriIndex0];
                uint32_t iPos1 = aaiClusterTrianglePositionIndices[iCheckCluster][iTriIndex1];
                uint32_t iPos2 = aaiClusterTrianglePositionIndices[iCheckCluster][iTriIndex2];
                assert(iPos0 < aaClusterVertexPositions[iCheckCluster].size());
                assert(iPos1 < aaClusterVertexPositions[iCheckCluster].size());
                assert(iPos2 < aaClusterVertexPositions[iCheckCluster].size());

                uint32_t iNorm0 = aaiClusterTriangleNormalIndices[iCheckCluster][iTriIndex0];
                uint32_t iNorm1 = aaiClusterTriangleNormalIndices[iCheckCluster][iTriIndex1];
                uint32_t iNorm2 = aaiClusterTriangleNormalIndices[iCheckCluster][iTriIndex2];
                assert(iNorm0 < aaClusterVertexNormals[iCheckCluster].size());
                assert(iNorm1 < aaClusterVertexNormals[iCheckCluster].size());
                assert(iNorm2 < aaClusterVertexNormals[iCheckCluster].size());

                uint32_t iUV0 = aaiClusterTriangleUVIndices[iCheckCluster][iTriIndex0];
                uint32_t iUV1 = aaiClusterTriangleUVIndices[iCheckCluster][iTriIndex1];
                uint32_t iUV2 = aaiClusterTriangleUVIndices[iCheckCluster][iTriIndex2];
                assert(iUV0 < aaClusterVertexUVs[iCheckCluster].size());
                assert(iUV1 < aaClusterVertexUVs[iCheckCluster].size());
                assert(iUV2 < aaClusterVertexUVs[iCheckCluster].size());

                // add position and position indices
                {
                    float3 const& pos0 = aaClusterVertexPositions[iCheckCluster][iPos0];
                    float3 const& pos1 = aaClusterVertexPositions[iCheckCluster][iPos1];
                    float3 const& pos2 = aaClusterVertexPositions[iCheckCluster][iPos2];

                    auto positionIter0 = std::find_if(
                        aSplitClusterVertexPositions.begin(),
                        aSplitClusterVertexPositions.end(),
                        [pos0](float3 const& checkPos)
                        {
                            return (length(pos0 - checkPos) < 1.0e-6f);
                        });
                    if(positionIter0 == aSplitClusterVertexPositions.end())
                    {
                        aSplitClusterVertexPositions.push_back(pos0);
                        aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                        ++iPositionCount;
                    }
                    else
                    {
                        uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter0));
                        aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                    }
                    uint32_t iRemap0 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                    auto positionIter1 = std::find_if(
                        aSplitClusterVertexPositions.begin(),
                        aSplitClusterVertexPositions.end(),
                        [pos1](float3 const& checkPos)
                        {
                            return (length(pos1 - checkPos) < 1.0e-6f);
                        });
                    if(positionIter1 == aSplitClusterVertexPositions.end())
                    {
                        aSplitClusterVertexPositions.push_back(pos1);
                        aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                        ++iPositionCount;
                    }
                    else
                    {
                        uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter1));
                        aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                    }
                    uint32_t iRemap1 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                    auto positionIter2 = std::find_if(
                        aSplitClusterVertexPositions.begin(),
                        aSplitClusterVertexPositions.end(),
                        [pos2](float3 const& checkPos)
                        {
                            return (length(pos2 - checkPos) < 1.0e-6f);
                        });
                    if(positionIter2 == aSplitClusterVertexPositions.end())
                    {
                        aSplitClusterVertexPositions.push_back(pos2);
                        aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                        ++iPositionCount;
                    }
                    else
                    {
                        uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter2));
                        aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                    }
                    uint32_t iRemap2 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                    assert(iRemap0 != iRemap1);
                    assert(iRemap0 != iRemap2);
                    assert(iRemap1 != iRemap2);

                }   // add position indices


                // add normal and normal indices
                {
                    float3 const& norm0 = aaClusterVertexNormals[iCheckCluster][iNorm0];
                    float3 const& norm1 = aaClusterVertexNormals[iCheckCluster][iNorm1];
                    float3 const& norm2 = aaClusterVertexNormals[iCheckCluster][iNorm2];
                    auto normalIter0 = std::find_if(
                        aSplitClusterVertexNormals.begin(),
                        aSplitClusterVertexNormals.end(),
                        [norm0](float3 const& checkNorm)
                        {
                            return (lengthSquared(norm0 - checkNorm) < 1.0e-6f);
                        });
                    if(normalIter0 == aSplitClusterVertexNormals.end())
                    {
                        aSplitClusterVertexNormals.push_back(norm0);
                        aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                        ++iNormalCount;
                    }
                    else
                    {
                        uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter0));
                        aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                    }

                    auto normalIter1 = std::find_if(
                        aSplitClusterVertexNormals.begin(),
                        aSplitClusterVertexNormals.end(),
                        [norm1](float3 const& checkNorm)
                        {
                            return (lengthSquared(norm1 - checkNorm) < 1.0e-6f);
                        });
                    if(normalIter1 == aSplitClusterVertexNormals.end())
                    {
                        aSplitClusterVertexNormals.push_back(norm1);
                        aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                        ++iNormalCount;
                    }
                    else
                    {
                        uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter1));
                        aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                    }

                    auto normalIter2 = std::find_if(
                        aSplitClusterVertexNormals.begin(),
                        aSplitClusterVertexNormals.end(),
                        [norm2](float3 const& checkNorm)
                        {
                            return (lengthSquared(norm2 - checkNorm) < 1.0e-6f);
                        });
                    if(normalIter2 == aSplitClusterVertexNormals.end())
                    {
                        aSplitClusterVertexNormals.push_back(norm2);
                        aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                        ++iNormalCount;
                    }
                    else
                    {
                        uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter2));
                        aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                    }
                }   // add normal indices

                // add uv and uv indices
                {
                    float2 const& uv0 = aaClusterVertexUVs[iCheckCluster][iUV0];
                    float2 const& uv1 = aaClusterVertexUVs[iCheckCluster][iUV1];
                    float2 const& uv2 = aaClusterVertexUVs[iCheckCluster][iUV2];
                    auto uvIter0 = std::find_if(
                        aSplitClusterVertexUVs.begin(),
                        aSplitClusterVertexUVs.end(),
                        [uv0](float2 const& checkUV)
                        {
                            return (lengthSquared(uv0 - checkUV) < 1.0e-6f);
                        });
                    if(uvIter0 == aSplitClusterVertexUVs.end())
                    {
                        aSplitClusterVertexUVs.push_back(uv0);
                        aiSplitClusterVertexUVIndices.push_back(iUVCount);
                        ++iUVCount;
                    }
                    else
                    {
                        uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter0));
                        aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                    }

                    auto uvIter1 = std::find_if(
                        aSplitClusterVertexUVs.begin(),
                        aSplitClusterVertexUVs.end(),
                        [uv1](float2 const& checkUV)
                        {
                            return (lengthSquared(uv1 - checkUV) < 1.0e-6f);
                        });
                    if(uvIter1 == aSplitClusterVertexUVs.end())
                    {
                        aSplitClusterVertexUVs.push_back(uv1);
                        aiSplitClusterVertexUVIndices.push_back(iUVCount);
                        ++iUVCount;
                    }
                    else
                    {
                        uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter1));
                        aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                    }

                    auto uvIter2 = std::find_if(
                        aSplitClusterVertexUVs.begin(),
                        aSplitClusterVertexUVs.end(),
                        [uv2](float2 const& checkUV)
                        {
                            return (lengthSquared(uv2 - checkUV) < 1.0e-6f);
                        });
                    if(uvIter2 == aSplitClusterVertexUVs.end())
                    {
                        aSplitClusterVertexUVs.push_back(uv2);
                        aiSplitClusterVertexUVIndices.push_back(iUVCount);
                        ++iUVCount;
                    }
                    else
                    {
                        uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter2));
                        aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                    }
                }   // add uv indices

            }

            // push new clusters from existing disparate cluster
            if(aSplitClusterVertexPositions.size() > 0)
            {
                std::lock_guard<std::mutex> lock(threadMutex);

                aaClusterVertexPositions.push_back(aSplitClusterVertexPositions);
                aaClusterVertexNormals.push_back(aSplitClusterVertexNormals);
                aaClusterVertexUVs.push_back(aSplitClusterVertexUVs);

                aaiClusterTrianglePositionIndices.push_back(aiSplitClusterVertexPositionIndices);
                aaiClusterTriangleNormalIndices.push_back(aiSplitClusterVertexNormalIndices);
                aaiClusterTriangleUVIndices.push_back(aiSplitClusterVertexUVIndices);

                aiDeleteClusters.resize(aaClusterVertexPositions.size());
            }

        }   // for i = 0 to num split inner clusters

        aiDeleteClusters[iCheckCluster] = iCheckCluster;


    }   // if split cluster triangle indices > 1

    return aaiSplitClusterTriangleIndices.size() > 1;
}

/*
**
*/
void createSplitClusters(
    std::vector<std::vector<float3>>& aaSplitClusterVertexPositions,
    std::vector<std::vector<float3>>& aaSplitClusterVertexNormals,
    std::vector<std::vector<float2>>& aaSplitClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleUVIndices,
    std::vector<uint32_t>& aiDeleteClusters,
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices,
    std::vector<std::vector<uint32_t>> const& aaiSplitClusterTriangleIndices,
    uint32_t iCheckCluster)
{
    for(uint32_t i = 0; i < static_cast<uint32_t>(aaiSplitClusterTriangleIndices.size()); i++)
    {
        std::vector<float3> aSplitClusterVertexPositions;
        std::vector<float3> aSplitClusterVertexNormals;
        std::vector<float2> aSplitClusterVertexUVs;
        std::vector<uint32_t> aiSplitClusterVertexPositionIndices;
        std::vector<uint32_t> aiSplitClusterVertexNormalIndices;
        std::vector<uint32_t> aiSplitClusterVertexUVIndices;
        uint32_t iPositionCount = 0, iNormalCount = 0, iUVCount = 0;
        for(uint32_t j = 0; j < static_cast<uint32_t>(aaiSplitClusterTriangleIndices[i].size()); j++)
        {
            uint32_t iTriIndex0 = aaiSplitClusterTriangleIndices[i][j] * 3;
            uint32_t iTriIndex1 = aaiSplitClusterTriangleIndices[i][j] * 3 + 1;
            uint32_t iTriIndex2 = aaiSplitClusterTriangleIndices[i][j] * 3 + 2;

            uint32_t iPos0 = aaiClusterTrianglePositionIndices[iCheckCluster][iTriIndex0];
            uint32_t iPos1 = aaiClusterTrianglePositionIndices[iCheckCluster][iTriIndex1];
            uint32_t iPos2 = aaiClusterTrianglePositionIndices[iCheckCluster][iTriIndex2];
            assert(iPos0 < aaClusterVertexPositions[iCheckCluster].size());
            assert(iPos1 < aaClusterVertexPositions[iCheckCluster].size());
            assert(iPos2 < aaClusterVertexPositions[iCheckCluster].size());

            uint32_t iNorm0 = aaiClusterTriangleNormalIndices[iCheckCluster][iTriIndex0];
            uint32_t iNorm1 = aaiClusterTriangleNormalIndices[iCheckCluster][iTriIndex1];
            uint32_t iNorm2 = aaiClusterTriangleNormalIndices[iCheckCluster][iTriIndex2];
            assert(iNorm0 < aaClusterVertexNormals[iCheckCluster].size());
            assert(iNorm1 < aaClusterVertexNormals[iCheckCluster].size());
            assert(iNorm2 < aaClusterVertexNormals[iCheckCluster].size());

            uint32_t iUV0 = aaiClusterTriangleUVIndices[iCheckCluster][iTriIndex0];
            uint32_t iUV1 = aaiClusterTriangleUVIndices[iCheckCluster][iTriIndex1];
            uint32_t iUV2 = aaiClusterTriangleUVIndices[iCheckCluster][iTriIndex2];
            assert(iUV0 < aaClusterVertexUVs[iCheckCluster].size());
            assert(iUV1 < aaClusterVertexUVs[iCheckCluster].size());
            assert(iUV2 < aaClusterVertexUVs[iCheckCluster].size());

            // add position and position indices
            {
                float3 const& pos0 = aaClusterVertexPositions[iCheckCluster][iPos0];
                float3 const& pos1 = aaClusterVertexPositions[iCheckCluster][iPos1];
                float3 const& pos2 = aaClusterVertexPositions[iCheckCluster][iPos2];

                auto positionIter0 = std::find_if(
                    aSplitClusterVertexPositions.begin(),
                    aSplitClusterVertexPositions.end(),
                    [pos0](float3 const& checkPos)
                    {
                        return (length(pos0 - checkPos) < 1.0e-6f);
                    });
                if(positionIter0 == aSplitClusterVertexPositions.end())
                {
                    aSplitClusterVertexPositions.push_back(pos0);
                    aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                    ++iPositionCount;
                }
                else
                {
                    uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter0));
                    aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                }
                uint32_t iRemap0 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                auto positionIter1 = std::find_if(
                    aSplitClusterVertexPositions.begin(),
                    aSplitClusterVertexPositions.end(),
                    [pos1](float3 const& checkPos)
                    {
                        return (length(pos1 - checkPos) < 1.0e-6f);
                    });
                if(positionIter1 == aSplitClusterVertexPositions.end())
                {
                    aSplitClusterVertexPositions.push_back(pos1);
                    aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                    ++iPositionCount;
                }
                else
                {
                    uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter1));
                    aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                }
                uint32_t iRemap1 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                auto positionIter2 = std::find_if(
                    aSplitClusterVertexPositions.begin(),
                    aSplitClusterVertexPositions.end(),
                    [pos2](float3 const& checkPos)
                    {
                        return (length(pos2 - checkPos) < 1.0e-6f);
                    });
                if(positionIter2 == aSplitClusterVertexPositions.end())
                {
                    aSplitClusterVertexPositions.push_back(pos2);
                    aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                    ++iPositionCount;
                }
                else
                {
                    uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter2));
                    aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                }
                uint32_t iRemap2 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                assert(iRemap0 != iRemap1);
                assert(iRemap0 != iRemap2);
                assert(iRemap1 != iRemap2);

            }   // add position indices


            // add normal and normal indices
            {
                float3 const& norm0 = aaClusterVertexNormals[iCheckCluster][iNorm0];
                float3 const& norm1 = aaClusterVertexNormals[iCheckCluster][iNorm1];
                float3 const& norm2 = aaClusterVertexNormals[iCheckCluster][iNorm2];
                auto normalIter0 = std::find_if(
                    aSplitClusterVertexNormals.begin(),
                    aSplitClusterVertexNormals.end(),
                    [norm0](float3 const& checkNorm)
                    {
                        return (lengthSquared(norm0 - checkNorm) < 1.0e-6f);
                    });
                if(normalIter0 == aSplitClusterVertexNormals.end())
                {
                    aSplitClusterVertexNormals.push_back(norm0);
                    aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                    ++iNormalCount;
                }
                else
                {
                    uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter0));
                    aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                }

                auto normalIter1 = std::find_if(
                    aSplitClusterVertexNormals.begin(),
                    aSplitClusterVertexNormals.end(),
                    [norm1](float3 const& checkNorm)
                    {
                        return (lengthSquared(norm1 - checkNorm) < 1.0e-6f);
                    });
                if(normalIter1 == aSplitClusterVertexNormals.end())
                {
                    aSplitClusterVertexNormals.push_back(norm1);
                    aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                    ++iNormalCount;
                }
                else
                {
                    uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter1));
                    aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                }

                auto normalIter2 = std::find_if(
                    aSplitClusterVertexNormals.begin(),
                    aSplitClusterVertexNormals.end(),
                    [norm2](float3 const& checkNorm)
                    {
                        return (lengthSquared(norm2 - checkNorm) < 1.0e-6f);
                    });
                if(normalIter2 == aSplitClusterVertexNormals.end())
                {
                    aSplitClusterVertexNormals.push_back(norm2);
                    aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                    ++iNormalCount;
                }
                else
                {
                    uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter2));
                    aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                }
            }   // add normal indices

            // add uv and uv indices
            {
                float2 const& uv0 = aaClusterVertexUVs[iCheckCluster][iUV0];
                float2 const& uv1 = aaClusterVertexUVs[iCheckCluster][iUV1];
                float2 const& uv2 = aaClusterVertexUVs[iCheckCluster][iUV2];
                auto uvIter0 = std::find_if(
                    aSplitClusterVertexUVs.begin(),
                    aSplitClusterVertexUVs.end(),
                    [uv0](float2 const& checkUV)
                    {
                        return (lengthSquared(uv0 - checkUV) < 1.0e-6f);
                    });
                if(uvIter0 == aSplitClusterVertexUVs.end())
                {
                    aSplitClusterVertexUVs.push_back(uv0);
                    aiSplitClusterVertexUVIndices.push_back(iUVCount);
                    ++iUVCount;
                }
                else
                {
                    uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter0));
                    aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                }

                auto uvIter1 = std::find_if(
                    aSplitClusterVertexUVs.begin(),
                    aSplitClusterVertexUVs.end(),
                    [uv1](float2 const& checkUV)
                    {
                        return (lengthSquared(uv1 - checkUV) < 1.0e-6f);
                    });
                if(uvIter1 == aSplitClusterVertexUVs.end())
                {
                    aSplitClusterVertexUVs.push_back(uv1);
                    aiSplitClusterVertexUVIndices.push_back(iUVCount);
                    ++iUVCount;
                }
                else
                {
                    uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter1));
                    aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                }

                auto uvIter2 = std::find_if(
                    aSplitClusterVertexUVs.begin(),
                    aSplitClusterVertexUVs.end(),
                    [uv2](float2 const& checkUV)
                    {
                        return (lengthSquared(uv2 - checkUV) < 1.0e-6f);
                    });
                if(uvIter2 == aSplitClusterVertexUVs.end())
                {
                    aSplitClusterVertexUVs.push_back(uv2);
                    aiSplitClusterVertexUVIndices.push_back(iUVCount);
                    ++iUVCount;
                }
                else
                {
                    uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter2));
                    aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                }
            }   // add uv indices

        }

        // push new clusters from existing disparate cluster
        if(aSplitClusterVertexPositions.size() > 0)
        {
            aaSplitClusterVertexPositions.push_back(aSplitClusterVertexPositions);
            aaSplitClusterVertexNormals.push_back(aSplitClusterVertexNormals);
            aaSplitClusterVertexUVs.push_back(aSplitClusterVertexUVs);

            aaiSplitClusterTrianglePositionIndices.push_back(aiSplitClusterVertexPositionIndices);
            aaiSplitClusterTriangleNormalIndices.push_back(aiSplitClusterVertexNormalIndices);
            aaiSplitClusterTriangleUVIndices.push_back(aiSplitClusterVertexUVIndices);

            aiDeleteClusters.resize(aaClusterVertexPositions.size());
        }

    }   // for i = 0 to num split inner clusters
}

/*
**
*/
void createSplitClusters2(
    std::vector<std::vector<float3>>& aaSplitClusterVertexPositions,
    std::vector<std::vector<float3>>& aaSplitClusterVertexNormals,
    std::vector<std::vector<float2>>& aaSplitClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleUVIndices,
    std::vector<uint32_t>& aiDeleteClusters,
    std::vector<float3> const& aClusterVertexPositions,
    std::vector<float3> const& aClusterVertexNormals,
    std::vector<float2> const& aClusterVertexUVs,
    std::vector<uint32_t> const& aiClusterTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterTriangleUVIndices,
    std::vector<std::vector<uint32_t>> const& aaiSplitClusterTriangleIndices)
{
    for(uint32_t i = 0; i < static_cast<uint32_t>(aaiSplitClusterTriangleIndices.size()); i++)
    {
        std::vector<float3> aSplitClusterVertexPositions;
        std::vector<float3> aSplitClusterVertexNormals;
        std::vector<float2> aSplitClusterVertexUVs;
        std::vector<uint32_t> aiSplitClusterVertexPositionIndices;
        std::vector<uint32_t> aiSplitClusterVertexNormalIndices;
        std::vector<uint32_t> aiSplitClusterVertexUVIndices;
        uint32_t iPositionCount = 0, iNormalCount = 0, iUVCount = 0;
        for(uint32_t j = 0; j < static_cast<uint32_t>(aaiSplitClusterTriangleIndices[i].size()); j++)
        {
            uint32_t iTriIndex0 = aaiSplitClusterTriangleIndices[i][j] * 3;
            uint32_t iTriIndex1 = aaiSplitClusterTriangleIndices[i][j] * 3 + 1;
            uint32_t iTriIndex2 = aaiSplitClusterTriangleIndices[i][j] * 3 + 2;

            uint32_t iPos0 = aiClusterTrianglePositionIndices[iTriIndex0];
            uint32_t iPos1 = aiClusterTrianglePositionIndices[iTriIndex1];
            uint32_t iPos2 = aiClusterTrianglePositionIndices[iTriIndex2];
            assert(iPos0 < aClusterVertexPositions.size());
            assert(iPos1 < aClusterVertexPositions.size());
            assert(iPos2 < aClusterVertexPositions.size());

            uint32_t iNorm0 = aiClusterTriangleNormalIndices[iTriIndex0];
            uint32_t iNorm1 = aiClusterTriangleNormalIndices[iTriIndex1];
            uint32_t iNorm2 = aiClusterTriangleNormalIndices[iTriIndex2];
            assert(iNorm0 < aClusterVertexNormals.size());
            assert(iNorm1 < aClusterVertexNormals.size());
            assert(iNorm2 < aClusterVertexNormals.size());

            uint32_t iUV0 = aiClusterTriangleUVIndices[iTriIndex0];
            uint32_t iUV1 = aiClusterTriangleUVIndices[iTriIndex1];
            uint32_t iUV2 = aiClusterTriangleUVIndices[iTriIndex2];
            assert(iUV0 < aClusterVertexUVs.size());
            assert(iUV1 < aClusterVertexUVs.size());
            assert(iUV2 < aClusterVertexUVs.size());

            // add position and position indices
            {
                float3 const& pos0 = aClusterVertexPositions[iPos0];
                float3 const& pos1 = aClusterVertexPositions[iPos1];
                float3 const& pos2 = aClusterVertexPositions[iPos2];

                auto positionIter0 = std::find_if(
                    aSplitClusterVertexPositions.begin(),
                    aSplitClusterVertexPositions.end(),
                    [pos0](float3 const& checkPos)
                    {
                        return (length(pos0 - checkPos) < 1.0e-8f);
                    });
                if(positionIter0 == aSplitClusterVertexPositions.end())
                {
                    aSplitClusterVertexPositions.push_back(pos0);
                    aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                    ++iPositionCount;
                }
                else
                {
                    uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter0));
                    aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                }
                uint32_t iRemap0 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                auto positionIter1 = std::find_if(
                    aSplitClusterVertexPositions.begin(),
                    aSplitClusterVertexPositions.end(),
                    [pos1](float3 const& checkPos)
                    {
                        return (length(pos1 - checkPos) < 1.0e-8f);
                    });
                if(positionIter1 == aSplitClusterVertexPositions.end())
                {
                    aSplitClusterVertexPositions.push_back(pos1);
                    aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                    ++iPositionCount;
                }
                else
                {
                    uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter1));
                    aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                }
                uint32_t iRemap1 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                auto positionIter2 = std::find_if(
                    aSplitClusterVertexPositions.begin(),
                    aSplitClusterVertexPositions.end(),
                    [pos2](float3 const& checkPos)
                    {
                        return (length(pos2 - checkPos) < 1.0e-8f);
                    });
                if(positionIter2 == aSplitClusterVertexPositions.end())
                {
                    aSplitClusterVertexPositions.push_back(pos2);
                    aiSplitClusterVertexPositionIndices.push_back(iPositionCount);
                    ++iPositionCount;
                }
                else
                {
                    uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexPositions.begin(), positionIter2));
                    aiSplitClusterVertexPositionIndices.push_back(iPositionIndex);
                }
                uint32_t iRemap2 = aiSplitClusterVertexPositionIndices[aiSplitClusterVertexPositionIndices.size() - 1];

                assert(iRemap0 != iRemap1);
                assert(iRemap0 != iRemap2);
                assert(iRemap1 != iRemap2);

            }   // add position indices


            // add normal and normal indices
            {
                float3 const& norm0 = aClusterVertexNormals[iNorm0];
                float3 const& norm1 = aClusterVertexNormals[iNorm1];
                float3 const& norm2 = aClusterVertexNormals[iNorm2];
                auto normalIter0 = std::find_if(
                    aSplitClusterVertexNormals.begin(),
                    aSplitClusterVertexNormals.end(),
                    [norm0](float3 const& checkNorm)
                    {
                        return (lengthSquared(norm0 - checkNorm) < 1.0e-6f);
                    });
                if(normalIter0 == aSplitClusterVertexNormals.end())
                {
                    aSplitClusterVertexNormals.push_back(norm0);
                    aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                    ++iNormalCount;
                }
                else
                {
                    uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter0));
                    aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                }

                auto normalIter1 = std::find_if(
                    aSplitClusterVertexNormals.begin(),
                    aSplitClusterVertexNormals.end(),
                    [norm1](float3 const& checkNorm)
                    {
                        return (lengthSquared(norm1 - checkNorm) < 1.0e-6f);
                    });
                if(normalIter1 == aSplitClusterVertexNormals.end())
                {
                    aSplitClusterVertexNormals.push_back(norm1);
                    aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                    ++iNormalCount;
                }
                else
                {
                    uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter1));
                    aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                }

                auto normalIter2 = std::find_if(
                    aSplitClusterVertexNormals.begin(),
                    aSplitClusterVertexNormals.end(),
                    [norm2](float3 const& checkNorm)
                    {
                        return (lengthSquared(norm2 - checkNorm) < 1.0e-6f);
                    });
                if(normalIter2 == aSplitClusterVertexNormals.end())
                {
                    aSplitClusterVertexNormals.push_back(norm2);
                    aiSplitClusterVertexNormalIndices.push_back(iNormalCount);
                    ++iNormalCount;
                }
                else
                {
                    uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexNormals.begin(), normalIter2));
                    aiSplitClusterVertexNormalIndices.push_back(iNormalIndex);
                }
            }   // add normal indices

            // add uv and uv indices
            {
                float2 const& uv0 = aClusterVertexUVs[iUV0];
                float2 const& uv1 = aClusterVertexUVs[iUV1];
                float2 const& uv2 = aClusterVertexUVs[iUV2];
                auto uvIter0 = std::find_if(
                    aSplitClusterVertexUVs.begin(),
                    aSplitClusterVertexUVs.end(),
                    [uv0](float2 const& checkUV)
                    {
                        return (lengthSquared(uv0 - checkUV) < 1.0e-6f);
                    });
                if(uvIter0 == aSplitClusterVertexUVs.end())
                {
                    aSplitClusterVertexUVs.push_back(uv0);
                    aiSplitClusterVertexUVIndices.push_back(iUVCount);
                    ++iUVCount;
                }
                else
                {
                    uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter0));
                    aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                }

                auto uvIter1 = std::find_if(
                    aSplitClusterVertexUVs.begin(),
                    aSplitClusterVertexUVs.end(),
                    [uv1](float2 const& checkUV)
                    {
                        return (lengthSquared(uv1 - checkUV) < 1.0e-6f);
                    });
                if(uvIter1 == aSplitClusterVertexUVs.end())
                {
                    aSplitClusterVertexUVs.push_back(uv1);
                    aiSplitClusterVertexUVIndices.push_back(iUVCount);
                    ++iUVCount;
                }
                else
                {
                    uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter1));
                    aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                }

                auto uvIter2 = std::find_if(
                    aSplitClusterVertexUVs.begin(),
                    aSplitClusterVertexUVs.end(),
                    [uv2](float2 const& checkUV)
                    {
                        return (lengthSquared(uv2 - checkUV) < 1.0e-6f);
                    });
                if(uvIter2 == aSplitClusterVertexUVs.end())
                {
                    aSplitClusterVertexUVs.push_back(uv2);
                    aiSplitClusterVertexUVIndices.push_back(iUVCount);
                    ++iUVCount;
                }
                else
                {
                    uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aSplitClusterVertexUVs.begin(), uvIter2));
                    aiSplitClusterVertexUVIndices.push_back(iUVIndex);
                }
            }   // add uv indices

        }

        // push new clusters from existing disparate cluster
        if(aSplitClusterVertexPositions.size() > 0)
        {
            aaSplitClusterVertexPositions.push_back(aSplitClusterVertexPositions);
            aaSplitClusterVertexNormals.push_back(aSplitClusterVertexNormals);
            aaSplitClusterVertexUVs.push_back(aSplitClusterVertexUVs);

            aaiSplitClusterTrianglePositionIndices.push_back(aiSplitClusterVertexPositionIndices);
            aaiSplitClusterTriangleNormalIndices.push_back(aiSplitClusterVertexNormalIndices);
            aaiSplitClusterTriangleUVIndices.push_back(aiSplitClusterVertexUVIndices);

            aiDeleteClusters.resize(aaSplitClusterVertexPositions.size());
        }

    }   // for i = 0 to num split inner clusters
}

/*
**
*/
void checkClusterAdjacency(
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleIndices,
    std::vector<uint32_t> const& aiClusterTriangleIndices)
{
    uint32_t iNumClusterTriangleIndices = static_cast<uint32_t>(aiClusterTriangleIndices.size());
    if(iNumClusterTriangleIndices <= 0)
    {
        return;
    }

    std::vector<std::vector<uint32_t>> aaiAdjacentTri(iNumClusterTriangleIndices / 3);
    for(uint32_t iTri = 0; iTri < iNumClusterTriangleIndices - 1; iTri += 3)
    {
        uint32_t iPos0 = aiClusterTriangleIndices[iTri];
        uint32_t iPos1 = aiClusterTriangleIndices[iTri + 1];
        uint32_t iPos2 = aiClusterTriangleIndices[iTri + 2];

        for(uint32_t iCheckTri = iTri + 3; iCheckTri < iNumClusterTriangleIndices; iCheckTri += 3)
        {
            uint32_t iCheckPos0 = aiClusterTriangleIndices[iCheckTri];
            uint32_t iCheckPos1 = aiClusterTriangleIndices[iCheckTri + 1];
            uint32_t iCheckPos2 = aiClusterTriangleIndices[iCheckTri + 2];

            uint32_t iNumSamePos = 0;
            if(iCheckPos0 == iPos0 || iCheckPos0 == iPos1 || iCheckPos0 == iPos2)
            {
                ++iNumSamePos;
            }

            if(iCheckPos1 == iPos0 || iCheckPos1 == iPos1 || iCheckPos1 == iPos2)
            {
                ++iNumSamePos;
            }

            if(iCheckPos2 == iPos0 || iCheckPos2 == iPos1 || iCheckPos2 == iPos2)
            {
                ++iNumSamePos;
            }

            if(iNumSamePos >= 2)
            {
                aaiAdjacentTri[iTri / 3].push_back(iCheckTri / 3);
                aaiAdjacentTri[iCheckTri / 3].push_back(iTri / 3);
            }
        }
    }

    std::vector<uint32_t> aiVisitedTris;
    visitAdjacentTris(
        aiVisitedTris,
        aaiAdjacentTri,
        0,
        0,
        0);

    if(aiVisitedTris.size() != aiClusterTriangleIndices.size() / 3)
    {
        aaiSplitClusterTriangleIndices.push_back(aiVisitedTris);
        for(int32_t iTri = 0; iTri < static_cast<int32_t>(aiClusterTriangleIndices.size() / 3); iTri++)
        {
            auto iter = std::find(aiVisitedTris.begin(), aiVisitedTris.end(), iTri);
            if(iter == aiVisitedTris.end())
            {
                std::vector<uint32_t> aiNewClusterTris;
                visitAdjacentTris(
                    aiNewClusterTris,
                    aaiAdjacentTri,
                    iTri,
                    0,
                    0);
                aaiSplitClusterTriangleIndices.push_back(aiNewClusterTris);
                aiVisitedTris.insert(aiVisitedTris.end(), aiNewClusterTris.begin(), aiNewClusterTris.end());

                iTri = -1;
            }
        }
    }
}

/*
**
*/
void visitAdjacentTris(
    std::vector<uint32_t>& aiVisitedTris,
    std::vector<std::vector<uint32_t>> const& aaiAdjacentTri,
    uint32_t iCurrTri,
    uint32_t iAdjacentIndex,
    uint32_t iStack)
{
    auto visitedIter = std::find(aiVisitedTris.begin(), aiVisitedTris.end(), iCurrTri);
    if(visitedIter != aiVisitedTris.end())
    {
        return;
    }

    aiVisitedTris.push_back(iCurrTri);
    for(uint32_t iAdjacentTriIndex = 0; iAdjacentTriIndex < static_cast<uint32_t>(aaiAdjacentTri[iCurrTri].size()); iAdjacentTriIndex++)
    {
        visitAdjacentTris(
            aiVisitedTris,
            aaiAdjacentTri,
            aaiAdjacentTri[iCurrTri][iAdjacentTriIndex],
            iAdjacentTriIndex,
            iStack + 1);
    }
}

/*
**
*/
template <typename T>
uint32_t getVertexIndex(
    std::vector<T> const& aVertices,
    T const& v,
    float fThreshold)
{
    uint32_t iRet = UINT32_MAX;
    auto iter = std::find_if(
        aVertices.begin(),
        aVertices.end(),
        [v, fThreshold](T const& checkV)
        {
            return lengthSquared(v - checkV) <= fThreshold;
        }
    );

    if(iter != aVertices.end())
    {
        iRet = static_cast<uint32_t>(std::distance(aVertices.begin(), iter));
    }

    return iRet;
}

/*
**
*/
void addTriangle(
    std::vector<float3>& aVertexPositions,
    std::vector<float3>& aVertexNormals,
    std::vector<float2>& aVertexUVs,
    std::vector<uint32_t>& aiVertexPositionIndices,
    std::vector<uint32_t>& aiVertexNormalIndices,
    std::vector<uint32_t>& aiVertexUVIndices,
    std::vector<float3> const& aOrigVertexPositions,
    std::vector<float3> const& aOrigVertexNormals,
    std::vector<float2> const& aOrigVertexUVs,
    std::vector<uint32_t> const& aiOrigVertexPositionIndices,
    std::vector<uint32_t> const& aiOrigVertexNormalIndices,
    std::vector<uint32_t> const& aiOrigVertexUVIndices,
    uint32_t iTri)
{
    float const kfEqualityThreshold = 1.0e-8f;

    // add triangle, inserting possible new positions into the list as well
    for(uint32_t i = 0; i < 3; i++)
    {
        // position
        {
            uint32_t iPos = aiOrigVertexPositionIndices[iTri + i];
            float3 const& origPosition = aOrigVertexPositions[iPos];
            uint32_t iIndex = getVertexIndex(aVertexPositions, origPosition, kfEqualityThreshold);
            if(iIndex == UINT32_MAX)
            {
                iIndex = static_cast<uint32_t>(aVertexPositions.size());
                aVertexPositions.push_back(origPosition);
                aiVertexPositionIndices.push_back(iIndex);
            }
            else
            {
                aiVertexPositionIndices.push_back(iIndex);
            }
        }

        // normal
        {
            uint32_t iNorm = aiOrigVertexNormalIndices[iTri + i];
            float3 const& origNormal = aOrigVertexNormals[iNorm];
            uint32_t iIndex = getVertexIndex(aVertexNormals, origNormal, kfEqualityThreshold);
            if(iIndex == UINT32_MAX)
            {
                iIndex = static_cast<uint32_t>(aVertexNormals.size());
                aVertexNormals.push_back(origNormal);
                aiVertexNormalIndices.push_back(iIndex);
            }
            else
            {
                aiVertexNormalIndices.push_back(iIndex);
            }
        }

        // uv
        {
            uint32_t iUV = aiOrigVertexUVIndices[iTri + i];
            float2 const& origUV = aOrigVertexUVs[iUV];
            uint32_t iIndex = getVertexIndex(aVertexUVs, origUV, kfEqualityThreshold);
            if(iIndex == UINT32_MAX)
            {
                iIndex = static_cast<uint32_t>(aVertexUVs.size());
                aVertexUVs.push_back(origUV);
                aiVertexUVIndices.push_back(iIndex);
            }
            else
            {
                aiVertexUVIndices.push_back(iIndex);
            }
        }

    }   // for i = 0 to 3
}

/*
**
*/
void splitCluster(
    std::vector<std::vector<float3>>& aaVertexPositions,
    std::vector<std::vector<float3>>& aaVertexNormals,
    std::vector<std::vector<float2>>& aaVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexUVIndices,
    std::vector<float3> const& aOrigVertexPositions,
    std::vector<float3> const& aOrigVertexNormals,
    std::vector<float2> const& aOrigVertexUVs,
    std::vector<uint32_t> const& aiOrigVertexPositionIndices,
    std::vector<uint32_t> const& aiOrigVertexNormalIndices,
    std::vector<uint32_t> const& aiOrigVertexUVIndices,
    uint32_t iMaxTriangles)
{
    static float const kfEqualityThreshold = 1.0e-8f;

    PrintOptions printOptions;
    printOptions.mbDisplayTime = false;
    setPrintOptions(printOptions);

    std::vector<uint32_t> aiTriangleAdded(aiOrigVertexPositionIndices.size() / 3);
    memset(aiTriangleAdded.data(), 0, aiTriangleAdded.size() * sizeof(uint32_t));

    uint32_t iCluster = 0;

    uint32_t iNumTriangles = static_cast<uint32_t>(aiOrigVertexPositionIndices.size() / 3);
    uint32_t iNumTotalTriangleAdded = 0;

    // add adjacent triangles to clusters
    for(uint32_t iCluster = 0; iCluster < 1000; iCluster++)
    {
        if(iNumTotalTriangleAdded >= iNumTriangles)
        {
            break;
        }

        aaVertexPositions.resize(aaVertexPositions.size() + 1);
        aaVertexNormals.resize(aaVertexNormals.size() + 1);
        aaVertexUVs.resize(aaVertexUVs.size() + 1);

        aaiVertexPositionIndices.resize(aaiVertexPositionIndices.size() + 1);
        aaiVertexNormalIndices.resize(aaiVertexNormalIndices.size() + 1);
        aaiVertexUVIndices.resize(aaiVertexUVIndices.size() + 1);

        // add first triangle of the cluster to determine the rest using triangle's edge adjacency
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiOrigVertexPositionIndices.size()); iTri += 3)
        {
            if(aiTriangleAdded[iTri / 3] == 1)
            {
                continue;
            }

            float3 diff0 = aOrigVertexPositions[aiOrigVertexPositionIndices[iTri]] - aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 1]];
            float3 diff1 = aOrigVertexPositions[aiOrigVertexPositionIndices[iTri]] - aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 2]];
            float3 diff2 = aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 1]] - aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 2]];

            if(lengthSquared(diff0) <= 1.0e-8f || lengthSquared(diff1) <= 1.0e-8f || lengthSquared(diff2) <= 1.0e-8f)
            {
                continue;
            }

            addTriangle(
                aaVertexPositions[iCluster],
                aaVertexNormals[iCluster],
                aaVertexUVs[iCluster],
                aaiVertexPositionIndices[iCluster],
                aaiVertexNormalIndices[iCluster],
                aaiVertexUVIndices[iCluster],
                aOrigVertexPositions,
                aOrigVertexNormals,
                aOrigVertexUVs,
                aiOrigVertexPositionIndices,
                aiOrigVertexNormalIndices,
                aiOrigVertexUVIndices,
                iTri);

            aiTriangleAdded[iTri / 3] = 1;
            ++iNumTotalTriangleAdded;
            break;
        }

        bool bRestart = false;
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size()); iTri += 3)
        {
            if(aaiVertexPositionIndices[iCluster].size() >= iMaxTriangles)
            {
                break;
            }

            if(bRestart)
            {
                iTri = 0;
            }

            float3 const& pos0 = aaVertexPositions[iCluster][aaiVertexPositionIndices[iCluster][iTri]];
            float3 const& pos1 = aaVertexPositions[iCluster][aaiVertexPositionIndices[iCluster][iTri + 1]];
            float3 const& pos2 = aaVertexPositions[iCluster][aaiVertexPositionIndices[iCluster][iTri + 2]];

            uint32_t iNumClusterTriAdded = 0;
            for(uint32_t iCheckTri = 0; iCheckTri < static_cast<uint32_t>(aiOrigVertexPositionIndices.size()); iCheckTri += 3)
            {
                // already added
                if(aiTriangleAdded[iCheckTri / 3])
                {
                    continue;
                }

                // check invalid triangle
                {
                    float3 const& checkPos0 = aOrigVertexPositions[aiOrigVertexPositionIndices[iCheckTri]];
                    float3 const& checkPos1 = aOrigVertexPositions[aiOrigVertexPositionIndices[iCheckTri + 1]];
                    float3 const& checkPos2 = aOrigVertexPositions[aiOrigVertexPositionIndices[iCheckTri + 2]];

                    float3 diff0 = checkPos1 - checkPos0;
                    float3 diff1 = checkPos2 - checkPos0;
                    float3 diff2 = checkPos2 - checkPos1;

                    if(lengthSquared(diff0) <= 1.0e-8f || lengthSquared(diff1) <= 1.0e-8f || lengthSquared(diff2) <= 1.0e-8f)
                    {
                        //DEBUG_PRINTF("!!! SKIP INVALID TRIANGLE !!!\n");
                        continue;
                    }
                }

                // check shared edge
                uint32_t iNumSamePos = 0;
                float3 aSamePos[3];
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = aaiVertexPositionIndices[iCluster][iTri + i];
                    float3 const& pos = aaVertexPositions[iCluster][iPos];
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        uint32_t iCheckPos = aiOrigVertexPositionIndices[iCheckTri + j];
                        float3 const& checkPos = aOrigVertexPositions[iCheckPos];

                        if(lengthSquared(checkPos - pos) <= kfEqualityThreshold)
                        {
                            aSamePos[iNumSamePos++] = pos;
                        }
                    }
                }

                if(iNumSamePos == 2)
                {
                    // shared an edge, add 

                    uint32_t aiNewIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
                    addTriangle(
                        aaVertexPositions[iCluster],
                        aaVertexNormals[iCluster],
                        aaVertexUVs[iCluster],
                        aaiVertexPositionIndices[iCluster],
                        aaiVertexNormalIndices[iCluster],
                        aaiVertexUVIndices[iCluster],
                        aOrigVertexPositions,
                        aOrigVertexNormals,
                        aOrigVertexUVs,
                        aiOrigVertexPositionIndices,
                        aiOrigVertexNormalIndices,
                        aiOrigVertexUVIndices,
                        iCheckTri);

                    aiTriangleAdded[iCheckTri/3] = 1;
                    ++iNumTotalTriangleAdded;
                    ++iNumClusterTriAdded;

                    DEBUG_PRINTF("\t!!! ADD triangle: %d !!!\n\tpos0 (%.4f, %.4f, %.4f) pos1 (%.4f, %.4f, %.4f)\n",
                        iCheckTri,
                        aSamePos[0].x, aSamePos[0].y, aSamePos[0].z,
                        aSamePos[1].x, aSamePos[1].y, aSamePos[1].z);
                }

            }   // for check tri = tri + 3 to num triangles

            // restart from the 1st triangle
            if(iNumClusterTriAdded > 0)
            {
                bRestart = true;
            }
            else
            {
                bRestart = false;
            }

        }   // for tri = 0 to num triangles

    }   // for cluster = 0 to 1000

    printOptions.mbDisplayTime = true;
    setPrintOptions(printOptions);
}

/*
**
*/
void splitClusterGroups(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    uint32_t& iTotalClusterIndex,
    std::vector<float3> const& aClusterGroupVertexPositions,
    std::vector<float3> const& aClusterGroupVertexNormals,
    std::vector<float2> const& aClusterGroupVertexUVs,
    std::vector<uint32_t> const& aiClusterTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterTriangleUVIndices,
    uint32_t iMaxTrianglesPerCluster,
    uint32_t iNumSplitClusters,
    uint32_t iLODLevel,
    uint32_t iClusterGroup,
    std::string const& meshModelName,
    std::string const& homeDirectory)
{

    // update cluster group vertex positions and triangles, generate metis mesh file to partition, and output the max partitions of 2 into clusters

    // triangle positions
    std::vector<float3> aClusterGroupTrianglePositions(aiClusterTrianglePositionIndices.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTrianglePositionIndices.size()); iTri += 3)
    {
        uint32_t iPos0 = aiClusterTrianglePositionIndices[iTri];
        uint32_t iPos1 = aiClusterTrianglePositionIndices[iTri + 1];
        uint32_t iPos2 = aiClusterTrianglePositionIndices[iTri + 2];

        aClusterGroupTrianglePositions[iTri] = aClusterGroupVertexPositions[iPos0];
        aClusterGroupTrianglePositions[iTri + 1] = aClusterGroupVertexPositions[iPos1];
        aClusterGroupTrianglePositions[iTri + 2] = aClusterGroupVertexPositions[iPos2];
    }

    // triangle normals
    std::vector<float3> aClusterGroupTriangleNormals(aiClusterTriangleNormalIndices.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTriangleNormalIndices.size()); iTri += 3)
    {
        uint32_t iNormal0 = aiClusterTriangleNormalIndices[iTri];
        uint32_t iNormal1 = aiClusterTriangleNormalIndices[iTri + 1];
        uint32_t iNormal2 = aiClusterTriangleNormalIndices[iTri + 2];

        aClusterGroupTriangleNormals[iTri] = aClusterGroupVertexNormals[iNormal0];
        aClusterGroupTriangleNormals[iTri + 1] = aClusterGroupVertexNormals[iNormal1];
        aClusterGroupTriangleNormals[iTri + 2] = aClusterGroupVertexNormals[iNormal2];
    }

    // triangle uvs
    std::vector<float2> aClusterGroupTriangleUVs(aiClusterTriangleUVIndices.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTriangleUVIndices.size()); iTri += 3)
    {
        uint32_t iUV0 = aiClusterTriangleUVIndices[iTri];
        uint32_t iUV1 = aiClusterTriangleUVIndices[iTri + 1];
        uint32_t iUV2 = aiClusterTriangleUVIndices[iTri + 2];

        aClusterGroupTriangleUVs[iTri] = aClusterGroupVertexUVs[iUV0];
        aClusterGroupTriangleUVs[iTri + 1] = aClusterGroupVertexUVs[iUV1];
        aClusterGroupTriangleUVs[iTri + 2] = aClusterGroupVertexUVs[iUV2];
    }

    float const kfDifferenceThreshold = 1.0e-8f;

    // re-build triangle indices
    std::vector<float3> aTrimmedTotalVertexPositions;
    std::vector<uint32_t> aiTrianglePositionIndices(aClusterGroupTrianglePositions.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTrianglePositions.size()); i++)
    {
        auto const& vertexPosition = aClusterGroupTrianglePositions[i];
        auto iter = std::find_if(
            aTrimmedTotalVertexPositions.begin(),
            aTrimmedTotalVertexPositions.end(),
            [vertexPosition,
            kfDifferenceThreshold](float3 const& checkPos)
            {
                return length(checkPos - vertexPosition) < kfDifferenceThreshold;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aTrimmedTotalVertexPositions.size());
        if(iter == aTrimmedTotalVertexPositions.end())
        {
            aTrimmedTotalVertexPositions.push_back(vertexPosition);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aTrimmedTotalVertexPositions.begin(), iter));
        }

        aiTrianglePositionIndices[i] = iVertexIndex;
    }

    // check for consistency
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTrianglePositionIndices.size()); iTri += 3)
    {
        uint32_t iPos0 = aiTrianglePositionIndices[iTri];
        uint32_t iPos1 = aiTrianglePositionIndices[iTri + 1];
        uint32_t iPos2 = aiTrianglePositionIndices[iTri + 2];

        assert(iPos0 != iPos1 && iPos0 != iPos2 && iPos1 != iPos2);
    }

    // re-build triangle normals
    std::vector<float3> aTrimmedTotalVertexNormals;
    std::vector<uint32_t> aiTriangleNormalIndices(aClusterGroupTriangleNormals.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTriangleNormals.size()); i++)
    {
        auto const& vertexNormal = aClusterGroupTriangleNormals[i];
        auto iter = std::find_if(
            aTrimmedTotalVertexNormals.begin(),
            aTrimmedTotalVertexNormals.end(),
            [vertexNormal](float3 const& checkPos)
            {
                return length(checkPos - vertexNormal) < 1.0e-5f;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aTrimmedTotalVertexNormals.size());
        if(iter == aTrimmedTotalVertexNormals.end())
        {
            aTrimmedTotalVertexNormals.push_back(vertexNormal);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aTrimmedTotalVertexNormals.begin(), iter));
        }

        aiTriangleNormalIndices[i] = iVertexIndex;
    }

    // re-build triangle uvs
    std::vector<float2> aTrimmedTotalVertexUVs;
    std::vector<uint32_t> aiTriangleUVIndices(aClusterGroupTriangleUVs.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTriangleUVs.size()); i++)
    {
        auto const& vertexUV = aClusterGroupTriangleUVs[i];
        auto iter = std::find_if(
            aTrimmedTotalVertexUVs.begin(),
            aTrimmedTotalVertexUVs.end(),
            [vertexUV](float2 const& checkUV)
            {
                return length(checkUV - vertexUV) < 1.0e-5f;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aTrimmedTotalVertexUVs.size());
        if(iter == aTrimmedTotalVertexUVs.end())
        {
            aTrimmedTotalVertexUVs.push_back(vertexUV);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aTrimmedTotalVertexUVs.begin(), iter));
        }

        aiTriangleUVIndices[i] = iVertexIndex;
    }

    uint32_t iPrevNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());

    std::vector<std::vector<float3>> aaTempClusterVertexPositions;
    std::vector<std::vector<float3>> aaTempClusterVertexNormals;
    std::vector<std::vector<float2>> aaTempClusterVertexUVs;
    std::vector<std::vector<uint32_t>> aaiTempClusterTrianglePositionIndices;
    std::vector<std::vector<uint32_t>> aaiTempClusterTriangleNormalIndices;
    std::vector<std::vector<uint32_t>> aaiTempClusterTriangleUVIndices;
    {
        std::vector<std::vector<float3>> aaTempClusterGroupVertexPositions(1);
        std::vector<std::vector<float3>> aaTempClusterGroupVertexNormals(1);
        std::vector<std::vector<float2>> aaTempClusterGroupVertexUVs(1);

        std::vector<std::vector<uint32_t>> aaiTempClusterGroupTrianglePositionIndices(1);
        std::vector<std::vector<uint32_t>> aaiTempClusterGroupTriangleNormalIndices(1);
        std::vector<std::vector<uint32_t>> aaiTempClusterGroupTriangleUVIndices(1);

        aaTempClusterGroupVertexPositions[0] = aTrimmedTotalVertexPositions;
        aaTempClusterGroupVertexNormals[0] = aTrimmedTotalVertexNormals;
        aaTempClusterGroupVertexUVs[0] = aTrimmedTotalVertexUVs;

        aaiTempClusterGroupTrianglePositionIndices[0] = aiTrianglePositionIndices;
        aaiTempClusterGroupTriangleNormalIndices[0] = aiTriangleNormalIndices;
        aaiTempClusterGroupTriangleUVIndices[0] = aiTriangleUVIndices;

        splitCluster3(
            aaTempClusterVertexPositions,
            aaTempClusterVertexNormals,
            aaTempClusterVertexUVs,
            aaiTempClusterTrianglePositionIndices,
            aaiTempClusterTriangleNormalIndices,
            aaiTempClusterTriangleUVIndices,
            aaTempClusterGroupVertexPositions,
            aaTempClusterGroupVertexNormals,
            aaTempClusterGroupVertexUVs,
            aaiTempClusterGroupTrianglePositionIndices,
            aaiTempClusterGroupTriangleNormalIndices,
            aaiTempClusterGroupTriangleUVIndices,
            0,
            iMaxTrianglesPerCluster);
    }

    // move large cluster triangles to smaller ones or merge small cluster triangles to larger clusters
    {
        bool bResetLoop = false;
        for(uint32_t iCheckCluster = 0; iCheckCluster < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); iCheckCluster++)
        {
            if(bResetLoop)
            {
                iCheckCluster = 0;
                bResetLoop = false;
            }

            if(aaiTempClusterTrianglePositionIndices[iCheckCluster].size() > 128 * 3)
            {
                //for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                //{
                //    DEBUG_PRINTF("BEFORE moveTriangle cluster %d size: %d\n",
                //        j,
                //        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                //}

                bResetLoop = moveTriangles(
                    aaTempClusterVertexPositions,
                    aaTempClusterVertexNormals,
                    aaTempClusterVertexUVs,
                    aaiTempClusterTrianglePositionIndices,
                    aaiTempClusterTriangleNormalIndices,
                    aaiTempClusterTriangleUVIndices,
                    iCheckCluster,
                    128 * 3);
                if(bResetLoop)
                {
                    iCheckCluster = 0;
                }

                //for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                //{
                //    DEBUG_PRINTF("AFTER moveTriangles cluster %d size: %d\n",
                //        j,
                //        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                //}
                //DEBUG_PRINTF("\n");
            }
            else if(aaiTempClusterTrianglePositionIndices[iCheckCluster].size() <= 12)
            {
                //DEBUG_PRINTF("\n");
                //for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                //{
                //    DEBUG_PRINTF("BEFORE mergeTriangles cluster %d size: %d\n",
                //        j,
                //        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                //}

                bResetLoop = mergeTriangles(
                    aaTempClusterVertexPositions,
                    aaTempClusterVertexNormals,
                    aaTempClusterVertexUVs,
                    aaiTempClusterTrianglePositionIndices,
                    aaiTempClusterTriangleNormalIndices,
                    aaiTempClusterTriangleUVIndices,
                    iCheckCluster);
                if(bResetLoop)
                {
                    iCheckCluster = 0;
                }

                //for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                //{
                //    DEBUG_PRINTF("AFTER mergeTriangles cluster %d size: %d\n",
                //        j,
                //        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                //}
                //DEBUG_PRINTF("\n");
            }
        }
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(aaTempClusterVertexPositions.size()); i++)
    {
        if(aaTempClusterVertexPositions[i].size() > 0)
        {
            aaClusterVertexPositions.push_back(aaTempClusterVertexPositions[i]);
            aaClusterVertexNormals.push_back(aaTempClusterVertexNormals[i]);
            aaClusterVertexUVs.push_back(aaTempClusterVertexUVs[i]);

            aaiClusterTrianglePositionIndices.push_back(aaiTempClusterTrianglePositionIndices[i]);
            aaiClusterTriangleNormalIndices.push_back(aaiTempClusterTriangleNormalIndices[i]);
            aaiClusterTriangleUVIndices.push_back(aaiTempClusterTriangleUVIndices[i]);
        }
    }

    // cluster clean up here...

    uint32_t iCurrNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
    iTotalClusterIndex += (iCurrNumClusters - iPrevNumClusters);

    assert(aaClusterVertexPositions.size() == aaClusterVertexNormals.size());
    assert(aaClusterVertexPositions.size() == aaClusterVertexUVs.size());

    assert(aaiClusterTrianglePositionIndices.size() == aaiClusterTriangleNormalIndices.size());
    assert(aaiClusterTrianglePositionIndices.size() == aaiClusterTriangleUVIndices.size());
}

/*
**
*/
void splitLargeClusters(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    uint32_t& iNumClusters,
    uint32_t& iNumClusterGroups,
    uint32_t iMaxTrianglesPerCluster,
    std::string const& meshModelName,
    std::string const& homeDirectory)
{
    uint32_t iNumTotalSplitCluster = iNumClusters;
    for(int32_t iCluster = 0; iCluster < static_cast<int32_t>(aaClusterVertexPositions.size()); iCluster++)
    {
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());

        if(aaiClusterTrianglePositionIndices[iCluster].size() >= iMaxTrianglesPerCluster)
        {
            std::ostringstream metisFileFolderPath;
            {
                metisFileFolderPath << homeDirectory << "metis\\" << meshModelName << "\\";
                std::filesystem::path metisFileFolderFileSystemPath(metisFileFolderPath.str());
                if(!std::filesystem::exists(metisFileFolderFileSystemPath))
                {
                    std::filesystem::create_directory(metisFileFolderFileSystemPath);
                }
            }

            std::ostringstream outputMetisSplitMeshFilePath;
            outputMetisSplitMeshFilePath << metisFileFolderPath.str() << "split-cluster";
            outputMetisSplitMeshFilePath << iCluster << "-lod" << 0 << ".mesh";

            buildMETISMeshFile2(
                outputMetisSplitMeshFilePath.str(),
                aaiClusterTrianglePositionIndices[iCluster]);

            uint32_t iNumSplitClusters = static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()) / (128 * 3);
            iNumSplitClusters = std::max(iNumSplitClusters, 2u);
            // exec the mpmetis to generate the initial clusters
            std::ostringstream metisCommand;
            metisCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
            metisCommand << outputMetisSplitMeshFilePath.str() << " ";
            metisCommand << "-gtype=dual ";
            metisCommand << "-ncommon=2 ";
            metisCommand << "-objtype=cut ";
            metisCommand << "-ufactor=60 ";
            metisCommand << "-contig ";
            //metisCommand << "-minconn ";
            //metisCommand << "-niter=20 ";
            metisCommand << iNumSplitClusters;
            std::string result = execCommand(metisCommand.str(), false);
            if(result.find("Metis returned with an error.") != std::string::npos)
            {
                metisCommand = std::ostringstream();
                metisCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
                metisCommand << outputMetisSplitMeshFilePath.str() << " ";
                metisCommand << "-gtype=dual ";
                metisCommand << "-ncommon=2 ";
                metisCommand << "-objtype=cut ";
                //metisCommand << "-ufactor=600 ";
                metisCommand << iNumSplitClusters;
                result = execCommand(metisCommand.str(), false);
                assert(result.find("Metis returned with an error.") == std::string::npos);
            }

            std::ostringstream outputSplitPartitionFilePath;
            outputSplitPartitionFilePath << outputMetisSplitMeshFilePath.str() << ".epart.";
            outputSplitPartitionFilePath << iNumSplitClusters;

            std::vector<uint32_t> aiClusters;
            readMetisClusterFile(aiClusters, outputSplitPartitionFilePath.str());

            // map of element index (triangle index) to cluster
            std::map<uint32_t, std::vector<uint32_t>> aSplitClusterMap;
            {
                std::vector<uint32_t> aiSplitClusters;
                readMetisClusterFile(aiSplitClusters, outputSplitPartitionFilePath.str());
                for(uint32_t i = 0; i < static_cast<uint32_t>(aiSplitClusters.size()); i++)
                {
                    uint32_t const& iSplitCluster = aiSplitClusters[i];
                    aSplitClusterMap[iSplitCluster].push_back(i);
                }
            }

            std::vector<std::vector<float3>> aaSplitClusterVertexPositions(iNumSplitClusters);
            std::vector<std::vector<float3>> aaSplitClusterVertexNormals(iNumSplitClusters);
            std::vector<std::vector<float2>> aaSplitClusterVertexUVs(iNumSplitClusters);
            std::vector<std::vector<uint32_t>> aaiSplitClusterTrianglePositionIndices(iNumSplitClusters);
            std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleNormalIndices(iNumSplitClusters);
            std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleUVIndices(iNumSplitClusters);
            for(uint32_t iSplitCluster = 0; iSplitCluster < iNumSplitClusters; iSplitCluster++)
            {
                outputMeshClusters2(
                    aaSplitClusterVertexPositions[iSplitCluster],
                    aaSplitClusterVertexNormals[iSplitCluster],
                    aaSplitClusterVertexUVs[iSplitCluster],
                    aaiSplitClusterTrianglePositionIndices[iSplitCluster],
                    aaiSplitClusterTriangleNormalIndices[iSplitCluster],
                    aaiSplitClusterTriangleUVIndices[iSplitCluster],
                    aSplitClusterMap[iSplitCluster],
                    aaClusterVertexPositions[iCluster],
                    aaClusterVertexNormals[iCluster],
                    aaClusterVertexUVs[iCluster],
                    aaiClusterTrianglePositionIndices[iCluster],
                    aaiClusterTriangleNormalIndices[iCluster],
                    aaiClusterTriangleUVIndices[iCluster],
                    0,
                    iNumTotalSplitCluster);
                ++iNumTotalSplitCluster;
            }

            bool bReset = false;
            for(uint32_t iSplitCluster = 0; iSplitCluster < iNumSplitClusters; iSplitCluster++)
            {
                if(bReset)
                {
                    iSplitCluster = 0;
                    bReset = false;
                }
                for(int32_t iTri = 0; iTri < static_cast<int32_t>(aaiSplitClusterTrianglePositionIndices[iSplitCluster].size()); iTri += 3)
                {
                    uint32_t iPos0 = aaiSplitClusterTrianglePositionIndices[iSplitCluster][iTri];
                    uint32_t iPos1 = aaiSplitClusterTrianglePositionIndices[iSplitCluster][iTri + 1];
                    uint32_t iPos2 = aaiSplitClusterTrianglePositionIndices[iSplitCluster][iTri + 2];

                    if(iPos0 == iPos1 || iPos0 == iPos2 || iPos1 == iPos2)
                    {
                        aaiSplitClusterTrianglePositionIndices[iSplitCluster].erase(
                            aaiSplitClusterTrianglePositionIndices[iSplitCluster].begin() + iTri,
                            aaiSplitClusterTrianglePositionIndices[iSplitCluster].begin() + iTri + 3);

                        aaiSplitClusterTriangleNormalIndices[iSplitCluster].erase(
                            aaiSplitClusterTriangleNormalIndices[iSplitCluster].begin() + iTri,
                            aaiSplitClusterTriangleNormalIndices[iSplitCluster].begin() + iTri + 3);

                        aaiSplitClusterTriangleUVIndices[iSplitCluster].erase(
                            aaiSplitClusterTriangleUVIndices[iSplitCluster].begin() + iTri,
                            aaiSplitClusterTriangleUVIndices[iSplitCluster].begin() + iTri + 3);

                        bReset = true;

                        DEBUG_PRINTF("!!! Remove degenerate triangle %d from split cluster %d\n", iTri, iSplitCluster);

                        break;
                    }
                }
            }

            aaClusterVertexPositions.erase(aaClusterVertexPositions.begin() + iCluster);
            aaClusterVertexNormals.erase(aaClusterVertexNormals.begin() + iCluster);
            aaClusterVertexUVs.erase(aaClusterVertexUVs.begin() + iCluster);

            aaiClusterTrianglePositionIndices.erase(aaiClusterTrianglePositionIndices.begin() + iCluster);
            aaiClusterTriangleNormalIndices.erase(aaiClusterTriangleNormalIndices.begin() + iCluster);
            aaiClusterTriangleUVIndices.erase(aaiClusterTriangleUVIndices.begin() + iCluster);

            if(iCluster < aaiClusterTrianglePositionIndices.size())
            {
                assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
                assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
            }
            else
            {
                assert(aaiClusterTrianglePositionIndices[iCluster - 1].size() == aaiClusterTriangleNormalIndices[iCluster - 1].size());
                assert(aaiClusterTrianglePositionIndices[iCluster - 1].size() == aaiClusterTriangleUVIndices[iCluster - 1].size());
            }

            for(uint32_t iSplitCluster = 0; iSplitCluster < iNumSplitClusters; iSplitCluster++)
            {
                aaClusterVertexPositions.push_back(aaSplitClusterVertexPositions[iSplitCluster]);
                aaClusterVertexNormals.push_back(aaSplitClusterVertexNormals[iSplitCluster]);
                aaClusterVertexUVs.push_back(aaSplitClusterVertexUVs[iSplitCluster]);

                aaiClusterTrianglePositionIndices.push_back(aaiSplitClusterTrianglePositionIndices[iSplitCluster]);
                aaiClusterTriangleNormalIndices.push_back(aaiSplitClusterTriangleNormalIndices[iSplitCluster]);
                aaiClusterTriangleUVIndices.push_back(aaiSplitClusterTriangleUVIndices[iSplitCluster]);
            }

            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());

            iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
            iNumClusterGroups = iNumClusters / 4;

            iCluster = -1;

        }   // if cluster size > threshold

    }   // for cluster = 0 to num clusters
}

/*
**
*/
void splitCluster3(
    std::vector<std::vector<float3>>& aaSplitClusterVertexPositions,
    std::vector<std::vector<float3>>& aaSplitClusterVertexNormals,
    std::vector<std::vector<float2>>& aaSplitClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiSplitClusterTriangleUVIndices,
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices,
    uint32_t iOrigCluster,
    uint32_t iMaxTrianglesPerCluster)
{
    struct AdjacencyInfo
    {
        uint32_t        miNumAdjacentTriangles;
        uint32_t        maiAdjacentTriangles[64];
    };

    auto const& aClusterVertexPositions = aaClusterVertexPositions[iOrigCluster];
    auto const& aClusterVertexNormals = aaClusterVertexNormals[iOrigCluster];
    auto const& aClusterVertexUVs = aaClusterVertexUVs[iOrigCluster];

    auto const& aiClusterTrianglePositionIndices = aaiClusterTrianglePositionIndices[iOrigCluster];
    auto const& aiClusterTriangleNormalIndices = aaiClusterTriangleNormalIndices[iOrigCluster];
    auto const& aiClusterTriangleUVIndices = aaiClusterTriangleUVIndices[iOrigCluster];

    //uint32_t aiNumPartitionTriangles[MAX_SPLIT_CLUSTERS];
    std::vector<uint32_t> aiNumPartitionTriangles;

    uint32_t iNumTrianglePositionIndices = static_cast<uint32_t>(aiClusterTrianglePositionIndices.size());

    // build adjacency info for all the triangles
    std::vector<AdjacencyInfo> aTriangleAdjacencyInfo(iNumTrianglePositionIndices);
    for(uint32_t iTri = 0; iTri < iNumTrianglePositionIndices; iTri += 3)
    {
        for(uint32_t iCheckTri = 0; iCheckTri < iNumTrianglePositionIndices; iCheckTri += 3)
        {
            if(iTri == iCheckTri)
            {
                continue;
            }

            uint32_t iNumSamePos = 0;
            for(uint32_t i = 0; i < 3; i++)
            {
                for(uint32_t j = 0; j < 3; j++)
                {
                    if(aiClusterTrianglePositionIndices[iTri + i] == aiClusterTrianglePositionIndices[iCheckTri + j])
                    {
                        ++iNumSamePos;
                        break;
                    }
                }
            }

            if(iNumSamePos > 0)
            {
                uint32_t iTriIndex = iTri / 3;
                uint32_t iCheckTriIndex = iCheckTri / 3;

                assert(iCheckTri < aiClusterTrianglePositionIndices.size());

                aTriangleAdjacencyInfo[iTriIndex].maiAdjacentTriangles[aTriangleAdjacencyInfo[iTriIndex].miNumAdjacentTriangles] = iCheckTriIndex;
                aTriangleAdjacencyInfo[iTriIndex].miNumAdjacentTriangles += 1;
            }
        }
    }

    //std::vector<std::vector<uint32_t>> aaiClusterTriangles(MAX_SPLIT_CLUSTERS);
    //for(uint32_t i = 0; i < aaiClusterTriangles.size(); i++)
    //{
    //    aaiClusterTriangles[i].resize(1024);
    //}
    std::vector<std::vector<uint32_t>> aaiClusterTriangles;

    uint32_t iNumTotalTriAdded = 0;
    uint32_t iSplitClusterID = 0;

    uint32_t iNumClusters = uint32_t(ceilf(float(iNumTrianglePositionIndices) / float(iMaxTrianglesPerCluster)));
    uint32_t iNumTrianglesPerCluster = uint32_t(ceilf(float(iNumTrianglePositionIndices) / float(iNumClusters))) / 3;

    std::vector<uint32_t> aiAdded(1 << 16);
    memset(aiAdded.data(), 0, sizeof(uint32_t) * (1 << 16));
    uint32_t iPartition = 0;
    for(iPartition = 0;; iPartition++)
    {
        //assert(iPartition < MAX_SPLIT_CLUSTERS);

        //aiNumPartitionTriangles[iPartition] = 0;
        aiNumPartitionTriangles.push_back(0);

        iSplitClusterID += iPartition;
        if(iNumTotalTriAdded >= iNumTrianglePositionIndices / 3)
        {
            break;
        }

        aaiClusterTriangles.resize(aaiClusterTriangles.size() + 1);

        // add starting triangle ID
        uint32_t iNumClusterTriAdded = 0;
        for(uint32_t iStartTri = 0; iStartTri < iNumTrianglePositionIndices; iStartTri += 3)
        {
            uint32_t iStartTriID = iStartTri / 3;
            assert(iStartTriID < aiAdded.size());
            if(aiAdded[iStartTriID] > 0)
            {
                continue;
            }
            //assert(iPartition < MAX_SPLIT_CLUSTERS);
            //aaiClusterTriangles[iPartition][0] = iStartTri / 3;
            aaiClusterTriangles[iPartition].push_back(iStartTri / 3);

            //assert(iStartTriID < sizeof(aiAdded) / sizeof(*aiAdded));
            assert(iStartTriID < aiAdded.size());
            aiAdded[iStartTriID] = 1;

            ++iNumClusterTriAdded;
            ++iNumTotalTriAdded;
            ++aiNumPartitionTriangles[iPartition];

            break;
        }

        assert(aaiClusterTriangles[iPartition].size() == iNumClusterTriAdded);
        for(uint32_t iTri = 0; iTri < iNumClusterTriAdded; iTri++)
        {
            // add adjacent triangle IDs
            assert(iTri < aaiClusterTriangles[iPartition].size());
            uint32_t iTriID = aaiClusterTriangles[iPartition][iTri];
            AdjacencyInfo adjacencyInfo = aTriangleAdjacencyInfo[iTriID];
            for(uint32_t iAddTri = 0; iAddTri < adjacencyInfo.miNumAdjacentTriangles; iAddTri++)
            {
                if(iNumClusterTriAdded >= iNumTrianglesPerCluster)
                {
                    break;
                }

                uint32_t iAdjacentTriID = adjacencyInfo.maiAdjacentTriangles[iAddTri];
                if(iAdjacentTriID >= aiAdded.size())
                {
                    uint32_t iPrevSize = static_cast<uint32_t>(aiAdded.size());
                    aiAdded.resize(aiAdded.size() + aiAdded.size());
                    memset(aiAdded.data() + iPrevSize, 0, aiAdded.size() * sizeof(uint32_t));
                }

                if(aiAdded[iAdjacentTriID] > 0)
                {
                    continue;
                }

                assert(iAdjacentTriID * 3 < iNumTrianglePositionIndices);
                //aaiClusterTriangles[iPartition][iNumClusterTriAdded] = iAdjacentTriID;
                aaiClusterTriangles[iPartition].push_back(iAdjacentTriID);
                aiAdded[iAdjacentTriID] = 1;

                ++iNumClusterTriAdded;
                ++iNumTotalTriAdded;
                ++aiNumPartitionTriangles[iPartition];
            }

            if(iNumClusterTriAdded >= iNumTrianglesPerCluster)
            {
                break;
            }

        }   // for tri = 0 to curr num tri added 

    }   // for partition = 0 to num partitions

    aaSplitClusterVertexPositions.resize(iPartition);
    aaSplitClusterVertexNormals.resize(iPartition);
    aaSplitClusterVertexUVs.resize(iPartition);
    aaiSplitClusterTrianglePositionIndices.resize(iPartition);
    aaiSplitClusterTriangleNormalIndices.resize(iPartition);
    aaiSplitClusterTriangleUVIndices.resize(iPartition);

    for(uint32_t iCluster = 0; iCluster < iPartition; iCluster++)
    {
        uint32_t iNumClusterTriangles = aiNumPartitionTriangles[iCluster];
        auto const& aiClusterTriangles = aaiClusterTriangles[iCluster];
        
        // partition positions, normals, and uvs
        for(uint32_t iTri = 0; iTri < iNumClusterTriangles; iTri++)
        {
            uint32_t iTriID = aiClusterTriangles[iTri];
            for(uint32_t i = 0; i < 3; i++)
            {
                uint32_t iSrcIndex = iTriID * 3 + i;
                
                // position
                float3 const& position = aClusterVertexPositions[aiClusterTrianglePositionIndices[iSrcIndex]];
                auto positionIter = std::find_if(
                    aaSplitClusterVertexPositions[iCluster].begin(),
                    aaSplitClusterVertexPositions[iCluster].end(),
                    [position](float3 const& checkPosition)
                    {
                        return (lengthSquared(position - checkPosition) <= 1.0e-8f);
                    });
                if(positionIter == aaSplitClusterVertexPositions[iCluster].end())
                {
                    aaSplitClusterVertexPositions[iCluster].push_back(position);
                }

                // normal
                float3 const& normal = aClusterVertexNormals[aiClusterTriangleNormalIndices[iSrcIndex]];
                auto normalIter = std::find_if(
                    aaSplitClusterVertexNormals[iCluster].begin(),
                    aaSplitClusterVertexNormals[iCluster].end(),
                    [normal](float3 const& checkNormal)
                    {
                        return (lengthSquared(normal - checkNormal) <= 1.0e-8f);
                    });
                if(normalIter == aaSplitClusterVertexNormals[iCluster].end())
                {
                    aaSplitClusterVertexNormals[iCluster].push_back(normal);
                }
                
                // uv
                float2 const& uv = aClusterVertexUVs[aiClusterTriangleUVIndices[iSrcIndex]];
                auto uvIter = std::find_if(
                    aaSplitClusterVertexUVs[iCluster].begin(),
                    aaSplitClusterVertexUVs[iCluster].end(),
                    [uv](float2 const& checkUV)
                    {
                        return (lengthSquared(uv - checkUV) <= 1.0e-8f);
                    });
                if(uvIter == aaSplitClusterVertexUVs[iCluster].end())
                {
                    aaSplitClusterVertexUVs[iCluster].push_back(uv);
                }

            }   // for i = 0 to 3 
        
        }   // for tri = 0 to num triangles

        // partition position indices, normal indices, and uv indices
        for(uint32_t iTri = 0; iTri < iNumClusterTriangles; iTri++)
        {
            uint32_t iTriID = aiClusterTriangles[iTri];
            for(uint32_t i = 0; i < 3; i++)
            {
                uint32_t iSrcIndex = iTriID * 3 + i;
                uint32_t iDestIndex = iTri * 3 + i;

                // position
                float3 const& position = aClusterVertexPositions[aiClusterTrianglePositionIndices[iSrcIndex]];
                auto positionIter = std::find_if(
                    aaSplitClusterVertexPositions[iCluster].begin(),
                    aaSplitClusterVertexPositions[iCluster].end(),
                    [position](float3 const& checkPosition)
                    {
                        return (lengthSquared(position - checkPosition) <= 1.0e-8f);
                    });
                assert(positionIter != aaSplitClusterVertexPositions[iCluster].end());
                uint32_t iPositionIndex = static_cast<uint32_t>(std::distance(aaSplitClusterVertexPositions[iCluster].begin(), positionIter));
                aaiSplitClusterTrianglePositionIndices[iCluster].push_back(iPositionIndex);
                assert(lengthSquared(aaSplitClusterVertexPositions[iCluster][iPositionIndex] - position) <= 1.0e-8f);

                // normal
                float3 const& normal = aClusterVertexNormals[aiClusterTriangleNormalIndices[iSrcIndex]];
                auto normalIter = std::find_if(
                    aaSplitClusterVertexNormals[iCluster].begin(),
                    aaSplitClusterVertexNormals[iCluster].end(),
                    [normal](float3 const& checkNormal)
                    {
                        return (lengthSquared(normal - checkNormal) <= 1.0e-8f);
                    });
                assert(normalIter != aaSplitClusterVertexNormals[iCluster].end());
                uint32_t iNormalIndex = static_cast<uint32_t>(std::distance(aaSplitClusterVertexNormals[iCluster].begin(), normalIter));
                aaiSplitClusterTriangleNormalIndices[iCluster].push_back(iNormalIndex);
                assert(lengthSquared(aaSplitClusterVertexNormals[iCluster][iNormalIndex] - normal) <= 1.0e-8f);

                // uv
                float2 const& uv = aClusterVertexUVs[aiClusterTriangleUVIndices[iSrcIndex]];
                auto uvIter = std::find_if(
                    aaSplitClusterVertexUVs[iCluster].begin(),
                    aaSplitClusterVertexUVs[iCluster].end(),
                    [uv](float2 const& checkUV)
                    {
                        return (lengthSquared(uv - checkUV) <= 1.0e-8f);
                    });
                assert(uvIter != aaSplitClusterVertexUVs[iCluster].end());
                uint32_t iUVIndex = static_cast<uint32_t>(std::distance(aaSplitClusterVertexUVs[iCluster].begin(), uvIter));
                aaiSplitClusterTriangleUVIndices[iCluster].push_back(iUVIndex);
                assert(lengthSquared(aaSplitClusterVertexUVs[iCluster][iUVIndex] - uv) <= 1.0e-8f);

            }   // for i = 0 to 3

        }   // for tri = 0 to num cluster triangles
        
    }   // for cluster = 0 to num clusters
}