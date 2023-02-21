#include "split_operations.h"
#include "LogPrint.h"
#include "obj_helper.h"

#include <sstream>
#include <assert.h>

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
        DEBUG_PRINTF("!!! LOD %d cluster %d (%d clusters) is separated into %d parts !!!\n",
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

    aaVertexPositions.resize(2);
    aaVertexNormals.resize(2);
    aaVertexUVs.resize(2);

    aaiVertexPositionIndices.resize(2);
    aaiVertexNormalIndices.resize(2);
    aaiVertexUVIndices.resize(2);

    std::vector<uint32_t> aiAdded(aiOrigVertexPositionIndices.size());
    memset(aiAdded.data(), 0, aiAdded.size() * sizeof(uint32_t));

    uint32_t iCluster = 0;

    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiVertexPositionIndices.size()); iTri += 3)
    {
        float3 diff0 = aOrigVertexPositions[aiOrigVertexPositionIndices[iTri]] - aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 1]];
        float3 diff1 = aOrigVertexPositions[aiOrigVertexPositionIndices[iTri]] - aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 2]];
        float3 diff2 = aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 1]] - aOrigVertexPositions[aiOrigVertexPositionIndices[iTri + 2]];

        if(lengthSquared(diff0) <= 1.0e-6f || lengthSquared(diff1) <= 1.0e-6f || lengthSquared(diff2) <= 1.0e-6f)
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
            0);

        aiAdded[iTri] = 1;
        break;
    }
        
    // add triangle to the first cluster
    for(;;)
    {
        if(aaiVertexPositionIndices[iCluster].size() >= iMaxTriangles)
        {
            break;
        }

        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size()); iTri += 3)
        {
            if(aaiVertexPositionIndices[iCluster].size() >= iMaxTriangles)
            {
                break;
            }

            float3 const& pos0 = aaVertexPositions[iCluster][aaiVertexPositionIndices[iCluster][iTri]];
            float3 const& pos1 = aaVertexPositions[iCluster][aaiVertexPositionIndices[iCluster][iTri+1]];
            float3 const& pos2 = aaVertexPositions[iCluster][aaiVertexPositionIndices[iCluster][iTri+2]];
            //DEBUG_PRINTF("CLUSTER %d TRIANGLE %d IN LIST\npos0 (%.4f, %.4f, %.4f)\npos1 (%.4f, %.4f, %.4f)\npos2 (%.4f, %.4f, %.4f)\n",
            //    iCluster,
            //    iTri,
            //    pos0.x, pos0.y, pos0.z,
            //    pos1.x, pos1.y, pos1.z,
            //    pos2.x, pos2.y, pos2.z);

            for(uint32_t iCheckTri = 0; iCheckTri < static_cast<uint32_t>(aiOrigVertexPositionIndices.size()); iCheckTri += 3)
            {
                if(aaiVertexPositionIndices[iCluster].size() >= iMaxTriangles)
                {
                    break;
                }

                float3 const& checkPos0 = aOrigVertexPositions[aiOrigVertexPositionIndices[iCheckTri]];
                float3 const& checkPos1 = aOrigVertexPositions[aiOrigVertexPositionIndices[iCheckTri + 1]];
                float3 const& checkPos2 = aOrigVertexPositions[aiOrigVertexPositionIndices[iCheckTri + 2]];

                //DEBUG_PRINTF("\tCHECK TRIANGLE %d IN LIST\n\tcheck pos0 (%.4f, %.4f, %.4f)\n\tcheck pos1 (%.4f, %.4f, %.4f)\n\tcheck pos2 (%.4f, %.4f, %.4f)\n",
                //    iCheckTri,
                //    checkPos0.x, checkPos0.y, checkPos0.z,
                //    checkPos1.x, checkPos1.y, checkPos1.z,
                //    checkPos2.x, checkPos2.y, checkPos2.z);

                float3 diff0 = checkPos1 - checkPos0;
                float3 diff1 = checkPos2 - checkPos0;
                float3 diff2 = checkPos2 - checkPos1;

                if(lengthSquared(diff0) <= 1.0e-8f || lengthSquared(diff1) <= 1.0e-8f || lengthSquared(diff2) <= 1.0e-8f)
                {
                    //DEBUG_PRINTF("!!! SKIP INVALID TRIANGLE !!!\n");
                    continue;
                }

                if(aiAdded[iCheckTri])
                {
                    continue;
                }

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

                    aiAdded[iCheckTri] = 1;

                    DEBUG_PRINTF("\t!!! ADD triangle: %d !!!\n\tpos0 (%.4f, %.4f, %.4f) pos1 (%.4f, %.4f, %.4f)\n", 
                        iCheckTri,
                        aSamePos[0].x, aSamePos[0].y, aSamePos[0].z,
                        aSamePos[1].x, aSamePos[1].y, aSamePos[1].z);
                }


            }   // for check tri = tri + 3 to num triangles

        }   // for tri = 0 to num triangles

    }   // for ;;

    // add remain triangles
    iCluster = 1;
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiAdded.size()); iTri += 3)
    {
        if(aiAdded[iTri] == 0)
        {
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
        }
    }

    //for(uint32_t iCluster = 0; iCluster < 2; iCluster++)
    //{
    //    std::ostringstream objectName;
    //    objectName << "split-cluster" << iCluster;
    //
    //    std::ostringstream outputFilePath;
    //    outputFilePath << "c:\\Users\\Dingwings\\demo-models\\debug-output\\" << objectName.str() << ".obj";
    //
    //    writeOBJFile(
    //        aaVertexPositions[iCluster],
    //        aaVertexNormals[iCluster],
    //        aaVertexUVs[iCluster],
    //        aaiVertexPositionIndices[iCluster],
    //        aaiVertexNormalIndices[iCluster],
    //        aaiVertexUVIndices[iCluster],
    //        outputFilePath.str(),
    //        objectName.str());
    //}

    printOptions.mbDisplayTime = true;
    setPrintOptions(printOptions);
}