#include "cleanup_operations.h"
#include "move_operations.h"
#include "LogPrint.h"

#include <algorithm>
#include <cassert>
#include <map>

struct DestCandidate
{
    uint32_t                                    miIndex;
    uint32_t                                    miNumSamePositions;
    uint32_t                                    miNumVertices;
    std::vector<std::pair<uint32_t, uint32_t>>  maSharedTriangles;
};

/*
**
*/
uint32_t cleanupClusters(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    uint32_t iSrcCluster)
{
    uint32_t const kiMinimumClusterTriangleVertices = 24 * 3;
    uint32_t const kiMaxTriangleVertices = 128 * 3;

    uint32_t iMergedIntoCluster = UINT32_MAX;

    uint32_t iNumTotalClusters = static_cast<uint32_t>(aaiClusterTrianglePositionIndices.size());
    //for(uint32_t iLoop = 0; iLoop < 100; iLoop++)
    {
        bool bCanExit = true;
        for(uint32_t i = 0; i < iNumTotalClusters; i++)
        {
            if(aaiClusterTrianglePositionIndices[i].size() > 0 && aaiClusterTrianglePositionIndices[i].size() < kiMinimumClusterTriangleVertices)
            {
                bCanExit = false;
                break;
            }
        }

        //if(bCanExit)
        //{
        //    break;
        //}

       
        {
            std::map<uint32_t, DestCandidate> aDestCandidates;

            auto& aSrcClusterVertexPositions = aaClusterVertexPositions[iSrcCluster];
            auto& aiSrcClusterVertexPositionIndices = aaiClusterTrianglePositionIndices[iSrcCluster];

            //if(aiSrcClusterVertexPositionIndices.size() >= kiMinimumClusterTriangleVertices ||
            //    aiSrcClusterVertexPositionIndices.size() <= 0)
            //{
            //    continue;
            //}

            for(uint32_t iSrcTri = 0; iSrcTri < static_cast<uint32_t>(aiSrcClusterVertexPositionIndices.size()); iSrcTri += 3)
            {
                uint32_t const& iSrcPos0 = aiSrcClusterVertexPositionIndices[iSrcTri];
                uint32_t const& iSrcPos1 = aiSrcClusterVertexPositionIndices[iSrcTri + 1];
                uint32_t const& iSrcPos2 = aiSrcClusterVertexPositionIndices[iSrcTri + 2];
                auto const& srcPos0 = aSrcClusterVertexPositions[iSrcPos0];
                auto const& srcPos1 = aSrcClusterVertexPositions[iSrcPos1];
                auto const& srcPos2 = aSrcClusterVertexPositions[iSrcPos2];

                for(uint32_t iDestCluster = 0; iDestCluster < iNumTotalClusters; iDestCluster++)
                {
                    if(iSrcCluster == iDestCluster)
                    {
                        continue;
                    }

                    auto const& aDestClusterVertexPositions = aaClusterVertexPositions[iDestCluster];
                    auto const& aiDestClusterVertexPositionIndices = aaiClusterTrianglePositionIndices[iDestCluster];

                    for(uint32_t iDestTri = 0; iDestTri < static_cast<uint32_t>(aiDestClusterVertexPositionIndices.size()); iDestTri += 3)
                    {
                        uint32_t iNumSamePos = 0;
                        for(uint32_t i = 0; i < 3; i++)
                        {
                            uint32_t const& iSrcPos = aiSrcClusterVertexPositionIndices[iSrcTri + i];
                            auto const& srcPos = aSrcClusterVertexPositions[iSrcPos];
                            for(uint32_t j = 0; j < 3; j++)
                            {
                                uint32_t const& iDestPos = aiDestClusterVertexPositionIndices[iDestTri + j];
                                auto const& destPos = aDestClusterVertexPositions[iDestPos];

                                float3 diff = destPos - srcPos;
                                if(lengthSquared(diff) <= 1.0e-8f)
                                {
                                    ++iNumSamePos;
                                }
                            }
                        }

                        if(iNumSamePos >= 1)
                        {
                            if(aDestCandidates.find(iDestCluster) == aDestCandidates.end())
                            {
                                DestCandidate destCandidate;
                                destCandidate.miIndex = iDestCluster;
                                destCandidate.maSharedTriangles.push_back(std::make_pair(iSrcTri, iDestTri));
                                destCandidate.miNumSamePositions = iNumSamePos;
                                destCandidate.miNumVertices = static_cast<uint32_t>(aiDestClusterVertexPositionIndices.size());
                                aDestCandidates[iDestCluster] = destCandidate;
                            }
                            else
                            {
                                aDestCandidates[iDestCluster].maSharedTriangles.push_back(std::make_pair(iSrcTri, iDestTri));
                            }
                        }

                    }   // for dest tri = 0 to num triangles

                }   // for dest cluster = 0 to num clusters

            }   // for src tri = 0 to num src tri

            if(aDestCandidates.size() > 0)
            {
                DestCandidate const* pBestCandidate = nullptr;
                for(auto const& keyValue : aDestCandidates)
                {
                    if(keyValue.second.miNumVertices + aiSrcClusterVertexPositionIndices.size() <= kiMaxTriangleVertices && keyValue.second.miNumVertices >= kiMinimumClusterTriangleVertices)
                    {
                        if(pBestCandidate == nullptr || pBestCandidate->miNumVertices < keyValue.second.miNumVertices)
                        {
                            pBestCandidate = &keyValue.second;
                        }

                    }
                }

                if(pBestCandidate == nullptr)
                {
                    for(auto const& keyValue : aDestCandidates)
                    {
                        if(keyValue.second.miNumVertices >= kiMaxTriangleVertices)
                        {
                            pBestCandidate = &keyValue.second;
                            break;
                        }
                    }
                }

                // merge
                if(pBestCandidate)
                {
                    iMergedIntoCluster = pBestCandidate->miIndex;

                    uint32_t iDestCluster = pBestCandidate->miIndex;

                    auto& aSrcClusterVertexNormals = aaClusterVertexNormals[iSrcCluster];
                    auto& aSrcClusterVertexUVs = aaClusterVertexUVs[iSrcCluster];
                    auto& aiSrcTrianglePositionIndices = aaiClusterTrianglePositionIndices[iSrcCluster];
                    auto& aiSrcTriangleNormalIndices = aaiClusterTriangleNormalIndices[iSrcCluster];
                    auto& aiSrcTriangleUVIndices = aaiClusterTriangleUVIndices[iSrcCluster];

                    auto& aDestClusterVertexPositions = aaClusterVertexPositions[iDestCluster];
                    auto& aDestClusterVertexNormals = aaClusterVertexNormals[iDestCluster];
                    auto& aDestClusterVertexUVs = aaClusterVertexUVs[iDestCluster];
                    auto& aiDestTrianglePositionIndices = aaiClusterTrianglePositionIndices[iDestCluster];
                    auto& aiDestTriangleNormalIndices = aaiClusterTriangleNormalIndices[iDestCluster];
                    auto& aiDestTriangleUVIndices = aaiClusterTriangleUVIndices[iDestCluster];

                    if(pBestCandidate->miNumVertices + aiSrcClusterVertexPositionIndices.size() <= kiMaxTriangleVertices)
                    {
                        // move entire cluster into the destination cluster

                        uint32_t const& iDestCluster = pBestCandidate->miIndex;

                        moveVertices(
                            aDestClusterVertexPositions,
                            aDestClusterVertexNormals,
                            aDestClusterVertexUVs,
                            aiDestTrianglePositionIndices,
                            aiDestTriangleNormalIndices,
                            aiDestTriangleUVIndices,
                            aSrcClusterVertexPositions,
                            aSrcClusterVertexNormals,
                            aSrcClusterVertexUVs,
                            aiSrcTrianglePositionIndices,
                            aiSrcTriangleNormalIndices,
                            aiSrcTriangleUVIndices,
                            iSrcCluster,
                            iDestCluster);

                        aSrcClusterVertexPositions.clear();
                        aSrcClusterVertexNormals.clear();
                        aSrcClusterVertexUVs.clear();
                        aiSrcTrianglePositionIndices.clear();
                        aiSrcTriangleNormalIndices.clear();
                        aiSrcTriangleUVIndices.clear();
                    }
                    else if(pBestCandidate->miNumVertices + aiSrcClusterVertexPositionIndices.size() > kiMaxTriangleVertices)
                    {
                        int iDebug = 1;
                    }
                }

            }   // if has best candidate

        }   // for src cluster = 0 to num clusters

    }   // for loop to 100

    return iMergedIntoCluster;
}

#include "test.h"

/*
**
*/
void cleanupClusters2(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    std::vector<std::vector<uint32_t>>& aaiGroupClustersIndices)
{
    std::map<uint32_t, std::vector<DestCandidate>> aaDestCandidates;

    uint32_t const kiMinClusterTriangleVertices = 12 * 3;
    uint32_t const kiMaxNumTriangleIndices = 128 * 3;
    uint32_t const kiEvenOutNumTriangles = 32;

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> aaiAdjacentEdgeClusters;
    buildClusterEdgeAdjacencyCUDA2(
        aaiAdjacentEdgeClusters,
        aaClusterVertexPositions,
        aaiClusterTrianglePositionIndices);
    for(uint32_t iCluster = 0; iCluster < aaClusterVertexPositions.size(); iCluster++)
    {
        auto const& aiClusterTrianglePositionIndices = aaiClusterTrianglePositionIndices[iCluster];
        if(aiClusterTrianglePositionIndices.size() < kiMinClusterTriangleVertices)
        {
            auto copyVector = aaiAdjacentEdgeClusters[iCluster];
            std::sort(
                copyVector.begin(),
                copyVector.end(),
                [](std::pair<uint32_t, uint32_t> const& lhs, std::pair<uint32_t, uint32_t> const& rhs)
                {
                    return lhs.second > rhs.second;
                });

            std::vector<DestCandidate> aCandidates;
            for(auto const& entry : copyVector)
            {
                DestCandidate candidate;
                candidate.miIndex = entry.first;
                candidate.miNumSamePositions = entry.second;
                aaDestCandidates[iCluster].push_back(candidate);
            }
        }
    }

    for(auto& keyValue : aaDestCandidates)
    {
        //uint32_t iSrcCluster = keyValue.first;
        //uint32_t iDestCluster = keyValue.second[0].miIndex;

        uint32_t iNumTrianglesMoved = 0;

        uint32_t iSrcCluster = keyValue.second[0].miIndex;
        uint32_t iDestCluster = keyValue.first;

        uint32_t iNumSrcClusterTriangles = static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iSrcCluster].size() / 3);
        uint32_t iNumDestClusterTriangles = static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iDestCluster].size() / 3);

        if(iNumDestClusterTriangles > iNumSrcClusterTriangles)
        {
            iSrcCluster = keyValue.first;
            iDestCluster = keyValue.second[0].miIndex;
        }

        uint32_t iNumSrcClusterVertices = static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iSrcCluster].size());
        for(uint32_t i = 0; i < keyValue.second.size(); i++)
        {
            uint32_t iCheckSrcCluster = keyValue.second[i].miIndex;
            uint32_t iNumCheckSrcVertices = static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCheckSrcCluster].size());
            if(iNumCheckSrcVertices > iNumSrcClusterVertices)
            {
                iSrcCluster = iCheckSrcCluster;
                iNumSrcClusterVertices = iNumCheckSrcVertices;
            }
        }

        uint32_t iNumTotalTriangles = static_cast<uint32_t>((aaiClusterTrianglePositionIndices[iDestCluster].size() + aaiClusterTrianglePositionIndices[iSrcCluster].size()) / 3);
        uint32_t iNumDividedTriangles = iNumTotalTriangles / 2;

        {
            // move from larger cluster to smaller cluster if the larger one has been filled

            // DEBUG_PRINTF("divide up cluster triangle %d (%d) and %d (%d), %d triangles to divide, each to have %d triangles\n", 
            //     iSrcCluster, 
            //     iNumSrcClusterTriangles,
            //     iDestCluster, 
            //     iNumDestClusterTriangles,
            //     iNumTotalTriangles, 
            //     iNumDividedTriangles);

            auto& aiSrcTrianglePositionIndices = aaiClusterTrianglePositionIndices[iSrcCluster];
            auto& aiSrcTriangleNormalIndices = aaiClusterTriangleNormalIndices[iSrcCluster];
            auto& aiSrcTriangleUVIndices = aaiClusterTriangleUVIndices[iSrcCluster];
            auto& aSrcClusterVertexPositions = aaClusterVertexPositions[iSrcCluster];
            auto& aSrcClusterVertexNormals = aaClusterVertexNormals[iSrcCluster];
            auto& aSrcClusterVertexUVs = aaClusterVertexUVs[iSrcCluster];

            auto& aiDestTrianglePositionIndices = aaiClusterTrianglePositionIndices[iDestCluster];
            auto& aiDestTriangleNormalIndices = aaiClusterTriangleNormalIndices[iDestCluster];
            auto& aiDestTriangleUVIndices = aaiClusterTriangleUVIndices[iDestCluster];
            auto& aDestClusterVertexPositions = aaClusterVertexPositions[iDestCluster];
            auto& aDestClusterVertexNormals = aaClusterVertexNormals[iDestCluster];
            auto& aDestClusterVertexUVs = aaClusterVertexUVs[iDestCluster];

            //uint32_t iNumTrianglesToMerge = kiEvenOutNumTriangles - static_cast<uint32_t>(aiDestTrianglePositionIndices.size() / 3);
            uint32_t iNumTrianglesToMerge = static_cast<uint32_t>(aiSrcTrianglePositionIndices.size() / 3) - iNumDividedTriangles;

            for(uint32_t iTri = 0; iTri < iNumTrianglesToMerge; iTri++)
            {
                for(uint32_t iDestTri = 0; iDestTri < static_cast<uint32_t>(aiDestTrianglePositionIndices.size()); iDestTri += 3)
                {
                    bool bAdded = false;
                    for(uint32_t iSrcTri = 0; iSrcTri < static_cast<uint32_t>(aiSrcTrianglePositionIndices.size()); iSrcTri += 3)
                    {
                        std::vector<float3> aSamePos;
                        for(uint32_t i = 0; i < 3; i++)
                        {
                            float3 const& destPos = aDestClusterVertexPositions[aiDestTrianglePositionIndices[iDestTri + i]];
                            for(uint32_t j = 0; j < 3; j++)
                            {
                                float3 const& srcPos = aSrcClusterVertexPositions[aiSrcTrianglePositionIndices[iSrcTri + j]];
                                float3 diff = destPos - srcPos;
                                if(length(diff) <= 1.0e-8f)
                                {
                                    aSamePos.push_back(srcPos);
                                }
                            }
                        }

                        if(aSamePos.size() >= 1)
                        {
                            for(uint32_t i = 0; i < 3; i++)
                            {
                                // position
                                auto positionIter = std::find_if(
                                    aDestClusterVertexPositions.begin(),
                                    aDestClusterVertexPositions.end(),
                                    [i, iSrcTri, aSrcClusterVertexPositions, aiSrcTrianglePositionIndices](float3 const& checkPosition)
                                    {
                                        float3 const& position = aSrcClusterVertexPositions[aiSrcTrianglePositionIndices[iSrcTri + i]];
                                        return (lengthSquared(checkPosition - position) <= 1.0e-8f);
                                    });

                                uint32_t iPosIndex = static_cast<uint32_t>(aDestClusterVertexPositions.size());
                                if(positionIter != aDestClusterVertexPositions.end())
                                {
                                    iPosIndex = static_cast<uint32_t>(std::distance(aDestClusterVertexPositions.begin(), positionIter));
                                }
                                else
                                {
                                    aDestClusterVertexPositions.push_back(aSrcClusterVertexPositions[aiSrcTrianglePositionIndices[iSrcTri + i]]);
                                }
                                aiDestTrianglePositionIndices.push_back(iPosIndex);
                                assert(lengthSquared(aDestClusterVertexPositions[iPosIndex] - aSrcClusterVertexPositions[aiSrcTrianglePositionIndices[iSrcTri + i]]) <= 1.0e-8f);

                                // normal
                                auto normalIter = std::find_if(
                                    aDestClusterVertexNormals.begin(),
                                    aDestClusterVertexNormals.end(),
                                    [i, iSrcTri, aSrcClusterVertexNormals, aiSrcTriangleNormalIndices](float3 const& checkNormal)
                                    {
                                        float3 const& normal = aSrcClusterVertexNormals[aiSrcTriangleNormalIndices[iSrcTri + i]];
                                        return (lengthSquared(checkNormal - normal) <= 1.0e-8f);
                                    });

                                uint32_t iNormIndex = static_cast<uint32_t>(aDestClusterVertexNormals.size());
                                if(normalIter != aDestClusterVertexNormals.end())
                                {
                                    iNormIndex = static_cast<uint32_t>(std::distance(aDestClusterVertexNormals.begin(), normalIter));
                                }
                                else
                                {
                                    aDestClusterVertexNormals.push_back(aSrcClusterVertexNormals[aiSrcTriangleNormalIndices[iSrcTri + i]]);
                                }
                                aiDestTriangleNormalIndices.push_back(iNormIndex);
                                assert(lengthSquared(aDestClusterVertexNormals[iNormIndex] - aSrcClusterVertexNormals[aiSrcTriangleNormalIndices[iSrcTri + i]]) <= 1.0e-8f);

                                // uv
                                auto UVIter = std::find_if(
                                    aDestClusterVertexUVs.begin(),
                                    aDestClusterVertexUVs.end(),
                                    [i, iSrcTri, aSrcClusterVertexUVs, aiSrcTriangleUVIndices](float2 const& checkUV)
                                    {
                                        float2 const& UV = aSrcClusterVertexUVs[aiSrcTriangleUVIndices[iSrcTri + i]];
                                        return (lengthSquared(checkUV - UV) <= 1.0e-8f);
                                    });

                                uint32_t iUVIndex = static_cast<uint32_t>(aDestClusterVertexUVs.size());
                                if(UVIter != aDestClusterVertexUVs.end())
                                {
                                    iUVIndex = static_cast<uint32_t>(std::distance(aDestClusterVertexUVs.begin(), UVIter));
                                }
                                else
                                {
                                    aDestClusterVertexUVs.push_back(aSrcClusterVertexUVs[aiSrcTriangleUVIndices[iSrcTri + i]]);
                                }
                                aiDestTriangleUVIndices.push_back(iUVIndex);
                                assert(lengthSquared(aDestClusterVertexUVs[iUVIndex] - aSrcClusterVertexUVs[aiSrcTriangleUVIndices[iSrcTri + i]]) <= 1.0e-8f);

                            }

                            aiSrcTrianglePositionIndices.erase(aiSrcTrianglePositionIndices.begin() + iSrcTri, aiSrcTrianglePositionIndices.begin() + iSrcTri + 3);
                            aiSrcTriangleNormalIndices.erase(aiSrcTriangleNormalIndices.begin() + iSrcTri, aiSrcTriangleNormalIndices.begin() + iSrcTri + 3);
                            aiSrcTriangleUVIndices.erase(aiSrcTriangleUVIndices.begin() + iSrcTri, aiSrcTriangleUVIndices.begin() + iSrcTri + 3);

                            bAdded = true;

                            ++iNumTrianglesMoved;

                            break;
                        }

                    }

                    if(bAdded)
                    {
                        break;
                    }
                }
            }

            //DEBUG_PRINTF("moved from %d cluster to %d cluster %d triangles\n", iSrcCluster, iDestCluster, iNumTrianglesMoved);
        }

    }

    // get rid of cluster with size <= 0
    for(;;)
    {
        bool bDone = true;
        for(uint32_t i = 0; i < aaClusterVertexPositions.size(); i++)
        {
            if(aaClusterVertexPositions[i].size() <= 0)
            {
                aaClusterVertexPositions.erase(aaClusterVertexPositions.begin() + i);

                bDone = false;
                break;
            }
        }

        for(uint32_t i = 0; i < aaClusterVertexNormals.size(); i++)
        {
            if(aaClusterVertexNormals[i].size() <= 0)
            {
                aaClusterVertexNormals.erase(aaClusterVertexNormals.begin() + i);
                bDone = false;
                break;
            }
        }

        for(uint32_t i = 0; i < aaClusterVertexUVs.size(); i++)
        {
            if(aaClusterVertexUVs[i].size() <= 0)
            {
                aaClusterVertexUVs.erase(aaClusterVertexUVs.begin() + i);
                bDone = false;
                break;
            }
        }

        for(uint32_t i = 0; i < aaiClusterTrianglePositionIndices.size(); i++)
        {
            if(aaiClusterTrianglePositionIndices[i].size() <= 0)
            {
                aaiClusterTrianglePositionIndices.erase(aaiClusterTrianglePositionIndices.begin() + i);
                bDone = false;
                break;
            }
        }

        for(uint32_t i = 0; i < aaiClusterTriangleNormalIndices.size(); i++)
        {
            if(aaiClusterTriangleNormalIndices[i].size() <= 0)
            {
                aaiClusterTriangleNormalIndices.erase(aaiClusterTriangleNormalIndices.begin() + i);
                bDone = false;
                break;
            }
        }

        for(uint32_t i = 0; i < aaiClusterTriangleUVIndices.size(); i++)
        {
            if(aaiClusterTriangleUVIndices[i].size() <= 0)
            {
                aaiClusterTriangleUVIndices.erase(aaiClusterTriangleUVIndices.begin() + i);
                bDone = false;
                break;
            }
        }

        if(bDone)
        {
            break;
        }
    }
}