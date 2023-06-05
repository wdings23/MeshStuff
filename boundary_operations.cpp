#include "boundary_operations.h"
#include "test.h"
#include "LogPrint.h"

#include <cassert>
#include <mutex>

std::mutex gMutex;

/*
**
*/
void getBoundaryEdges(
    std::vector<BoundaryEdgeInfo>& aBoundaryEdges,
    std::vector<std::vector<float3>>const& aaClusterGroupVertexPositions,
    std::vector<std::vector<float3>>const& aaClusterGroupVertexNormals,
    std::vector<std::vector<float2>>const& aaClusterGroupVertexUVs,
    std::vector<std::vector<uint32_t>>const& aaiClusterGroupTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>const& aaiClusterGroupTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>const& aaiClusterGroupTriangleUVIndices)
{
    std::vector<std::vector<uint32_t>> aaiBoundaryTriangles(aaiClusterGroupTrianglePositionIndices.size());
    for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices.size()); iClusterGroup++)
    {
        std::vector<uint32_t> aiBoundaryTriangles;
        auto const& aiClusterGroupTrianglePositionIndices = aaiClusterGroupTrianglePositionIndices[iClusterGroup];
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterGroupTrianglePositionIndices.size()); iTri += 3)
        {
            bool abHasAdjacentEdges[3] = { false, false, false };
            for(uint32_t iEdge = 0; iEdge < 3; iEdge++)
            {
                // position index of the edge
                uint32_t iPos0 = 0, iPos1 = 1;
                if(iEdge == 1)
                {
                    iPos1 = 2;
                }
                else if(iEdge == 2)
                {
                    iPos0 = 1;
                    iPos1 = 2;
                }

                // check other triangles
                for(uint32_t iCheckTri = 0; iCheckTri < static_cast<uint32_t>(aiClusterGroupTrianglePositionIndices.size()); iCheckTri += 3)
                {
                    if(iTri == iCheckTri)
                    {
                        continue;
                    }

                    uint32_t iNumSamePos = 0;
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        if(aiClusterGroupTrianglePositionIndices[iTri + iPos0] == aiClusterGroupTrianglePositionIndices[iCheckTri + j] ||
                            aiClusterGroupTrianglePositionIndices[iTri + iPos1] == aiClusterGroupTrianglePositionIndices[iCheckTri + j])
                        {
                            ++iNumSamePos;
                        }
                    }

                    // shared edge -> >= 2 shared vertex positions
                    if(iNumSamePos >= 2)
                    {
                        abHasAdjacentEdges[iEdge] = true;
                        break;
                    }

                }   // for check triangle in cluster
            }

            // add boundary edges
            {
                if(abHasAdjacentEdges[0] == false)
                {
                    BoundaryEdgeInfo edgeInfo;
                    edgeInfo.miClusterGroup = iClusterGroup;
                    edgeInfo.miPos0 = aiClusterGroupTrianglePositionIndices[iTri];
                    edgeInfo.miPos1 = aiClusterGroupTrianglePositionIndices[iTri + 1];
                    edgeInfo.mPos0 = aaClusterGroupVertexPositions[iClusterGroup][edgeInfo.miPos0];
                    edgeInfo.mPos1 = aaClusterGroupVertexPositions[iClusterGroup][edgeInfo.miPos1];
                    aBoundaryEdges.push_back(edgeInfo);
                }

                if(abHasAdjacentEdges[1] == false)
                {
                    BoundaryEdgeInfo edgeInfo;
                    edgeInfo.miClusterGroup = iClusterGroup;
                    edgeInfo.miPos0 = aiClusterGroupTrianglePositionIndices[iTri];
                    edgeInfo.miPos1 = aiClusterGroupTrianglePositionIndices[iTri + 2];
                    edgeInfo.mPos0 = aaClusterGroupVertexPositions[iClusterGroup][edgeInfo.miPos0];
                    edgeInfo.mPos1 = aaClusterGroupVertexPositions[iClusterGroup][edgeInfo.miPos1];
                    aBoundaryEdges.push_back(edgeInfo);
                }

                if(abHasAdjacentEdges[2] == false)
                {
                    BoundaryEdgeInfo edgeInfo;
                    edgeInfo.miClusterGroup = iClusterGroup;
                    edgeInfo.miPos0 = aiClusterGroupTrianglePositionIndices[iTri + 1];
                    edgeInfo.miPos1 = aiClusterGroupTrianglePositionIndices[iTri + 2];
                    edgeInfo.mPos0 = aaClusterGroupVertexPositions[iClusterGroup][edgeInfo.miPos0];
                    edgeInfo.mPos1 = aaClusterGroupVertexPositions[iClusterGroup][edgeInfo.miPos1];
                    aBoundaryEdges.push_back(edgeInfo);
                }

            }   // add boundary edges

            if(abHasAdjacentEdges[0] == false || abHasAdjacentEdges[1] == false || abHasAdjacentEdges[2] == false)
            {
                aaiBoundaryTriangles[iClusterGroup].push_back(iTri);
            }

        }   // for triangle in cluster

    }   // for cluster group

    int iDebug = 1;

}   // get boundary edges


/*
**
*/
void getClusterGroupBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupNonBoundaryVertices,
    std::vector<std::vector<float3>> const& aaClusterGroupVertexPositions,
    uint32_t const& iNumClusterGroups)
{
    aaiClusterGroupBoundaryVertices.resize(iNumClusterGroups);

#if 0
    std::mutex threadMutex;

    uint32_t const kiMaxThreads = 8;
    std::unique_ptr<std::thread> apThreads[kiMaxThreads];
    std::atomic<uint32_t> iCurrClusterGroup{ 0 };
    for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
    {
        apThreads[iThread] = std::make_unique<std::thread>(
            [&iCurrClusterGroup,
            &aaiClusterGroupBoundaryVertices,
            aaClusterGroupVertexPositions,
            iNumClusterGroups,
            iThread,
            &threadMutex]()
            {
                for(;;)
                {
                    uint32_t iThreadClusterGroup = iCurrClusterGroup.fetch_add(1);
                    if(iThreadClusterGroup >= iNumClusterGroups)
                    {
                        break;
                    }

                    auto start = std::chrono::high_resolution_clock::now();

                    auto const& aClusterGroupVertexPositions = aaClusterGroupVertexPositions[iThreadClusterGroup];
                    auto const* paClusterGroupVertexPositions = aaClusterGroupVertexPositions[iThreadClusterGroup].data();
                    uint32_t iNumClusterGroupVertexPositions = static_cast<uint32_t>(aClusterGroupVertexPositions.size());

                    for(uint32_t iCheckClusterGroup = iThreadClusterGroup + 1; iCheckClusterGroup < iNumClusterGroups; iCheckClusterGroup++)
                    {
                        auto const& aCheckClusterGroupVertexPositions = aaClusterGroupVertexPositions[iCheckClusterGroup];
                        auto const* paCheckClusterGroupVertexPositions = aaClusterGroupVertexPositions[iCheckClusterGroup].data();
                        uint32_t iNumCheckClusterGroupVertexPositions = static_cast<uint32_t>(aCheckClusterGroupVertexPositions.size());

                        for(uint32_t iVert = 0; iVert < iNumClusterGroupVertexPositions; iVert++)
                        {
                            float3 const& vertPosition = paClusterGroupVertexPositions[iVert];
                            for(uint32_t iCheckVert = 0; iCheckVert < iNumCheckClusterGroupVertexPositions; iCheckVert++)
                            {
                                float3 diff = vertPosition - paCheckClusterGroupVertexPositions[iCheckVert];
                                float fLength = lengthSquared(diff);
                                if(fLength <= 1.0e-8f)
                                {
                                    {
                                        std::lock_guard<std::mutex> lock(threadMutex);
                                        aaiClusterGroupBoundaryVertices[iThreadClusterGroup].push_back(iVert);
                                        aaiClusterGroupBoundaryVertices[iCheckClusterGroup].push_back(iCheckVert);
                                    }
                                }

                            }   // for check vert = 0 to num vertices

                        }   // for vert = 0 to num vertices

                    }   // for cluster group = 0 to num cluster groups

                    auto end = std::chrono::high_resolution_clock::now();
                    uint64_t iMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    if(iThreadClusterGroup % 100 == 0)
                    {
                        DEBUG_PRINTF("thread %d took %d milliseconds to complete getting boundary vertices for cluster %d (%d)\n", iThread, iMilliseconds, iThreadClusterGroup, iNumClusterGroups);
                    }
                }   // for ;;

            }
        );
    }

    for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
    {
        if(apThreads[iThread]->joinable())
        {
            apThreads[iThread]->join();
        }
    }
#endif // #if 0

    {
        //std::vector<std::vector<uint32_t>> aaiTestClusterGroupBoundaryVertices;
        if(aaClusterGroupVertexPositions.size() > 1)
        {
            //getClusterGroupBoundaryVerticesCUDA(
            //    aaiClusterGroupBoundaryVertices, //aaiTestClusterGroupBoundaryVertices,
            //    aaClusterGroupVertexPositions);

            //for(uint32_t iVertex = 0; iVertex < aaClusterGroupVertexPositions[1].size(); iVertex++)
            //{
            //    DEBUG_PRINTF("cluster group 333 vertex %d (%.4f, %.4f, %.4f)\n",
            //        iVertex,
            //        aaClusterGroupVertexPositions[3][iVertex].x,
            //        aaClusterGroupVertexPositions[3][iVertex].y,
            //        aaClusterGroupVertexPositions[3][iVertex].z);
            //}

            getClusterGroupBoundaryVerticesCUDA2(
                aaiClusterGroupBoundaryVertices,
                aaClusterGroupVertexPositions);

            for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaiClusterGroupBoundaryVertices.size()); iClusterGroup++)
            {
                auto const& aiClusterGroupBoundaryVertices = aaiClusterGroupBoundaryVertices[iClusterGroup];
                for(auto const& iVertex : aiClusterGroupBoundaryVertices)
                {
                    assert(iVertex < aaClusterGroupVertexPositions[iClusterGroup].size());
                }
            }

            //for(uint32_t i = 0; i < aaiTestClusterGroupBoundaryVertices[0].size(); i++)
            //{
            //    auto iter = std::find(
            //        aaiClusterGroupBoundaryVertices[0].begin(),
            //        aaiClusterGroupBoundaryVertices[0].end(),
            //        aaiTestClusterGroupBoundaryVertices[0][i]);
            //
            //    if(iter == aaiClusterGroupBoundaryVertices[0].end())
            //    {
            //        int iDebug = 1;
            //    }
            //}
        }
    }

    // get non-boundary vertices by checking non-adjacent vertices from cluster groups
    aaiClusterGroupNonBoundaryVertices.resize(iNumClusterGroups);
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        auto const& aClusterGroupVertexPositions = aaClusterGroupVertexPositions[iClusterGroup];
        for(uint32_t iVert = 0; iVert < static_cast<uint32_t>(aClusterGroupVertexPositions.size()); iVert++)
        {
            auto iter = std::find(aaiClusterGroupBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupBoundaryVertices[iClusterGroup].end(), iVert);
            if(iter == aaiClusterGroupBoundaryVertices[iClusterGroup].end())
            {
                aaiClusterGroupNonBoundaryVertices[iClusterGroup].push_back(iVert);
            }
        }
    }
}

struct TriAdjacentInfo
{
    uint32_t        miClusterGroup;
    uint32_t        miTriangle;
    uint32_t        miEdge;
    float3          mEdgePosition0;
    float3          mEdgePosition1;
};

/*
**
*/
void checkClusterGroupBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    uint32_t iClusterGroup,
    uint32_t iNumClusterGroups,
    std::vector<std::vector<float3>> const& aaClusterGroupVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTrianglePositionIndices)
{
    uint32_t iNumClusterGroupTriangleIndices = static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size());
    uint32_t const* paiClusterGroupTriangles = aaiClusterGroupTrianglePositionIndices[iClusterGroup].data();
    float3 const* paClusterGroupVertexPositions = aaClusterGroupVertexPositions[iClusterGroup].data();

    std::vector<std::vector<TriAdjacentInfo>> aaAdjacentTris(iNumClusterGroupTriangleIndices);
    for(uint32_t iTri = 0; iTri < iNumClusterGroupTriangleIndices; iTri += 3)
    {
        for(uint32_t iCheckClusterGroup = 0; iCheckClusterGroup < iNumClusterGroups; iCheckClusterGroup++)
        {
            uint32_t iNumCheckClusterGroupTriangleIndices = static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iCheckClusterGroup].size());
            uint32_t const* paiCheckClusterGroupTriangles = aaiClusterGroupTrianglePositionIndices[iCheckClusterGroup].data();
            float3 const* paCheckClusterGroupVertexPositions = aaClusterGroupVertexPositions[iCheckClusterGroup].data();
            for(uint32_t iCheckTri = 0; iCheckTri < iNumCheckClusterGroupTriangleIndices; iCheckTri += 3)
            {
                if(iClusterGroup == iCheckClusterGroup && iTri == iCheckTri)
                {
                    continue;
                }

                // check the number of same vertex positions
                uint32_t aiSamePositionIndices[3] = { 0, 0, 0 };
                uint32_t iNumSamePositions = 0;
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = paiClusterGroupTriangles[iTri + i];
                    float3 const& pos = paClusterGroupVertexPositions[iPos];
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        uint32_t iCheckPos = paiCheckClusterGroupTriangles[iCheckTri + j];
                        float3 const& checkPos = paCheckClusterGroupVertexPositions[iCheckPos];
                        float3 diff = checkPos - pos;
                        if(length(diff) < 1.0e-6f)
                        {
                            aiSamePositionIndices[i] = 1;
                            ++iNumSamePositions;
                            break;
                        }
                    }
                }

                if(iNumSamePositions >= 2)
                {
                    // edge index based on the same vertex positions
                    uint32_t iEdge = UINT32_MAX;
                    if(aiSamePositionIndices[0] == 1 && aiSamePositionIndices[1] == 1)
                    {
                        iEdge = 0;
                    }
                    else if(aiSamePositionIndices[0] == 1 && aiSamePositionIndices[2] == 1)
                    {
                        iEdge = 1;
                    }
                    else if(aiSamePositionIndices[1] == 1 && aiSamePositionIndices[2] == 1)
                    {
                        iEdge = 2;
                    }

                    TriAdjacentInfo triAdjacentInfo;
                    triAdjacentInfo.miClusterGroup = iClusterGroup;
                    triAdjacentInfo.miEdge = iEdge;
                    triAdjacentInfo.miTriangle = iCheckTri / 3;
                    if(iEdge == 0)
                    {
                        triAdjacentInfo.mEdgePosition0 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri]];
                        triAdjacentInfo.mEdgePosition1 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 1]];
                    }
                    else if(iEdge == 1)
                    {
                        triAdjacentInfo.mEdgePosition0 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri]];
                        triAdjacentInfo.mEdgePosition1 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 2]];
                    }
                    else if(iEdge == 2)
                    {
                        triAdjacentInfo.mEdgePosition0 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 1]];
                        triAdjacentInfo.mEdgePosition1 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 2]];
                    }

                    aaAdjacentTris[iTri / 3].push_back(triAdjacentInfo);
                }

            }   // for check tri = 0 to num tris in check cluster group

        }   // for check cluster group = 0 to num cluster groups

        uint32_t aiNumAdjacentEdges[3] = { 0, 0, 0 };
        for(auto const& adjacentTri : aaAdjacentTris[iTri / 3])
        {
            if(adjacentTri.miEdge != UINT32_MAX)
            {
                aiNumAdjacentEdges[adjacentTri.miEdge] += 1;
            }
        }

        uint32_t iPos0 = paiClusterGroupTriangles[iTri];
        uint32_t iPos1 = paiClusterGroupTriangles[iTri + 1];
        uint32_t iPos2 = paiClusterGroupTriangles[iTri + 2];

        if(aiNumAdjacentEdges[0] <= 0)
        {
            // mark edge 0 vertices as boundary
            {
                std::lock_guard<std::mutex> lock(gMutex);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos0);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos1);
            }

            DEBUG_PRINTF("\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                paClusterGroupVertexPositions[iPos0].x,
                paClusterGroupVertexPositions[iPos0].y,
                paClusterGroupVertexPositions[iPos0].z,
                paClusterGroupVertexPositions[iPos1].x,
                paClusterGroupVertexPositions[iPos1].y,
                paClusterGroupVertexPositions[iPos1].z);

        }
        else if(aiNumAdjacentEdges[1] <= 0)
        {
            // mark edge 1 vertices as boundary
            {
                std::lock_guard<std::mutex> lock(gMutex);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos0);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos2);
            }

            DEBUG_PRINTF("\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                paClusterGroupVertexPositions[iPos0].x,
                paClusterGroupVertexPositions[iPos0].y,
                paClusterGroupVertexPositions[iPos0].z,
                paClusterGroupVertexPositions[iPos2].x,
                paClusterGroupVertexPositions[iPos2].y,
                paClusterGroupVertexPositions[iPos2].z);
        }
        else if(aiNumAdjacentEdges[2] <= 0)
        {
            // mark edge 1 vertices as boundary
            {
                std::lock_guard<std::mutex> lock(gMutex);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos1);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos2);
            }

            DEBUG_PRINTF("\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                paClusterGroupVertexPositions[iPos1].x,
                paClusterGroupVertexPositions[iPos1].y,
                paClusterGroupVertexPositions[iPos1].z,
                paClusterGroupVertexPositions[iPos2].x,
                paClusterGroupVertexPositions[iPos2].y,
                paClusterGroupVertexPositions[iPos2].z);
        }

    }   // for tri = 0 to num triangles in cluster group
}

/*
**
*/
void getInnerEdgesAndVertices(
    std::vector<std::vector<uint32_t>>& aaiValidClusterGroupEdges,
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaValidClusterGroupEdgePairs,
    std::vector<std::map<uint32_t, uint32_t>>& aaValidVertices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupTriWithEdges,
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaClusterGroupEdges,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTriangles,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupNonBoundaryVertices,
    uint32_t const& iNumClusterGroups)
{
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterGroupTriangles[iClusterGroup].size()); iTri += 3)
        {
            uint32_t iV0 = aaiClusterGroupTriangles[iClusterGroup][iTri];
            uint32_t iV1 = aaiClusterGroupTriangles[iClusterGroup][iTri + 1];
            uint32_t iV2 = aaiClusterGroupTriangles[iClusterGroup][iTri + 2];

            //if(iV0 == iV1 || iV0 == iV2 || iV1 == iV2)
            //{
            //    continue;
            //}

            assert(iV0 != iV1);
            assert(iV0 != iV2);
            assert(iV1 != iV2);

            bool bValid0 = (std::find(aaiClusterGroupNonBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupNonBoundaryVertices[iClusterGroup].end(), iV0) != aaiClusterGroupNonBoundaryVertices[iClusterGroup].end());
            bool bValid1 = (std::find(aaiClusterGroupNonBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupNonBoundaryVertices[iClusterGroup].end(), iV1) != aaiClusterGroupNonBoundaryVertices[iClusterGroup].end());
            bool bValid2 = (std::find(aaiClusterGroupNonBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupNonBoundaryVertices[iClusterGroup].end(), iV2) != aaiClusterGroupNonBoundaryVertices[iClusterGroup].end());

            aaClusterGroupEdges[iClusterGroup].push_back(std::make_pair(iV0, iV1));
            aaClusterGroupEdges[iClusterGroup].push_back(std::make_pair(iV0, iV2));
            aaClusterGroupEdges[iClusterGroup].push_back(std::make_pair(iV1, iV2));

            if(bValid0)
            {
                auto pair0 = std::make_pair(iV0, iV1);
                auto iter0 = std::find_if(
                    aaValidClusterGroupEdgePairs[iClusterGroup].begin(),
                    aaValidClusterGroupEdgePairs[iClusterGroup].end(),
                    [pair0](std::pair<uint32_t, uint32_t> const& checkPair)
                    {
                        return (checkPair.first == pair0.first && checkPair.second == pair0.second) || (checkPair.first == pair0.second && checkPair.second == pair0.first);
                    });
                if(iter0 == aaValidClusterGroupEdgePairs[iClusterGroup].end())
                {
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV0);
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV1);

                    aaValidClusterGroupEdgePairs[iClusterGroup].push_back(pair0);

                    assert(pair0.first != pair0.second);

                    // save the triangle index containing this edge
                    aaiClusterGroupTriWithEdges[iClusterGroup].push_back(iTri);
                }

                auto pair1 = std::make_pair(iV0, iV2);
                auto iter1 = std::find_if(
                    aaValidClusterGroupEdgePairs[iClusterGroup].begin(),
                    aaValidClusterGroupEdgePairs[iClusterGroup].end(),
                    [pair1](std::pair<uint32_t, uint32_t> const& checkPair)
                    {
                        return (checkPair.first == pair1.first && checkPair.second == pair1.second) || (checkPair.first == pair1.second && checkPair.second == pair1.first);
                    });
                if(iter1 == aaValidClusterGroupEdgePairs[iClusterGroup].end())
                {
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV0);
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV2);

                    aaValidClusterGroupEdgePairs[iClusterGroup].push_back(pair1);

                    assert(pair1.first != pair1.second);

                    // save the triangle index containing this edge
                    aaiClusterGroupTriWithEdges[iClusterGroup].push_back(iTri);
                }
            }

            if(bValid1 || bValid2)
            {
                auto pair2 = std::make_pair(iV1, iV2);
                auto iter = std::find_if(
                    aaValidClusterGroupEdgePairs[iClusterGroup].begin(),
                    aaValidClusterGroupEdgePairs[iClusterGroup].end(),
                    [pair2](std::pair<uint32_t, uint32_t> const& checkPair)
                    {
                        return (checkPair.first == pair2.first && checkPair.second == pair2.second) || (checkPair.first == pair2.second && checkPair.second == pair2.first);
                    });
                if(iter == aaValidClusterGroupEdgePairs[iClusterGroup].end())
                {
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV1);
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV2);

                    aaValidClusterGroupEdgePairs[iClusterGroup].push_back(pair2);

                    assert(pair2.first != pair2.second);
                }

                // save the triangle index containing this edge
                aaiClusterGroupTriWithEdges[iClusterGroup].push_back(iTri);
            }

            aaValidVertices[iClusterGroup][iV0] = (bValid0 == true) ? 1 : 0;
            aaValidVertices[iClusterGroup][iV1] = (bValid1 == true) ? 1 : 0;
            aaValidVertices[iClusterGroup][iV2] = (bValid2 == true) ? 1 : 0;

        }
    }
}

/*
**
*/
void printDebugMeshes(
    std::vector<std::vector<uint32_t>>& aaiBoundaryVertices,
    std::vector<std::vector<uint32_t>>& aaiNonBoundaryVertices,
    std::vector<std::vector<float3>> const& aaVertexPositions,
    uint32_t iClusterGroup)
{
    PrintOptions option = { false };
    DEBUG_PRINTF_SET_OPTIONS(option);
    DEBUG_PRINTF("def draw_boundary_vertices():\n");
    for(uint32_t i = 0; i < aaiBoundaryVertices[iClusterGroup].size(); i++)
    {
        uint32_t iBoundaryVertIndex = aaiBoundaryVertices[iClusterGroup][i];
        float3 const& pos = aaVertexPositions[iClusterGroup][iBoundaryVertIndex];

        DEBUG_PRINTF("\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n", pos.x, pos.y, pos.z);
    }

    DEBUG_PRINTF("\n\n##\ndef draw_non_boundary_vertices():\n");
    for(uint32_t i = 0; i < aaiNonBoundaryVertices[iClusterGroup].size(); i++)
    {
        uint32_t iNonBoundaryVertIndex = aaiNonBoundaryVertices[iClusterGroup][i];
        float3 const& pos = aaVertexPositions[iClusterGroup][iNonBoundaryVertIndex];

        DEBUG_PRINTF("\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 255, 0)\n", pos.x, pos.y, pos.z);
    }

    option.mbDisplayTime = true;
    DEBUG_PRINTF_SET_OPTIONS(option);
}

/*
**
*/
void getBoundaryAndNonBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiBoundaryVertices,
    std::vector<std::vector<uint32_t>>& aaiNonBoundaryVertices,
    std::vector<std::vector<float3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiTrianglePositionIndices)
{
    aaiBoundaryVertices.resize(aaVertexPositions.size());
    aaiNonBoundaryVertices.resize(aaVertexPositions.size());
    for(uint32_t iPartition = 0; iPartition < static_cast<uint32_t>(aaVertexPositions.size()); iPartition++)
    {
        auto const& aiPartitionTrianglePositionIndices = aaiTrianglePositionIndices[iPartition];
        uint32_t iNumTrianglePositionIndices = static_cast<uint32_t>(aiPartitionTrianglePositionIndices.size());
        std::vector<uint32_t> aiBoundaryVertexFlags(aaVertexPositions[iPartition].size());
        for(uint32_t iTri = 0; iTri < iNumTrianglePositionIndices; iTri += 3)
        {
            uint32_t aiSameEdge[3] = { 0, 0, 0 };
            uint32_t iNumSameEdges = 0;
            for(uint32_t iCheckTri = 0; iCheckTri < iNumTrianglePositionIndices; iCheckTri += 3)
            {
                if(iCheckTri == iTri)
                {
                    continue;
                }

                // record same position
                uint32_t iNumSamePos = 0;
                uint32_t aiSamePos[3] = { 0, 0, 0 };
                for(uint32_t i = 0; i < 3; i++)
                {
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        if(aiPartitionTrianglePositionIndices[iTri + i] == aiPartitionTrianglePositionIndices[iCheckTri + j])
                        {
                            ++iNumSamePos;
                            aiSamePos[i] = 1;
                            break;
                        }
                    }
                }

                // record same edge
                if(iNumSamePos >= 2)
                {
                    if(aiSamePos[0] == 1 && aiSamePos[1] == 1)
                    {
                        aiSameEdge[0] = 1;
                        ++iNumSameEdges;
                    }

                    if(aiSamePos[0] == 1 && aiSamePos[2] == 1)
                    {
                        aiSameEdge[1] = 1;
                        ++iNumSameEdges;
                    }

                    if(aiSamePos[1] == 1 && aiSamePos[2] == 1)
                    {
                        aiSameEdge[2] = 1;
                        ++iNumSameEdges;
                    }
                }

                if(iNumSameEdges >= 3)
                {
                    break;
                }

            }       // for check tri = 0 to num triangles

            // set boundary vertex flags based on found same edges
            if(aiSameEdge[0] == 0)
            {
                aiBoundaryVertexFlags[aiPartitionTrianglePositionIndices[iTri]] = 1;
                aiBoundaryVertexFlags[aiPartitionTrianglePositionIndices[iTri + 1]] = 1;
            }

            if(aiSameEdge[1] == 0)
            {
                aiBoundaryVertexFlags[aiPartitionTrianglePositionIndices[iTri]] = 1;
                aiBoundaryVertexFlags[aiPartitionTrianglePositionIndices[iTri + 2]] = 1;
            }

            if(aiSameEdge[2] == 0)
            {
                aiBoundaryVertexFlags[aiPartitionTrianglePositionIndices[iTri + 1]] = 1;
                aiBoundaryVertexFlags[aiPartitionTrianglePositionIndices[iTri + 2]] = 1;
            }

        }   // for tri = 0 to num triangles

        // check for valid flag on boundary vertices
        for(uint32_t i = 0; i < static_cast<uint32_t>(aiBoundaryVertexFlags.size()); i++)
        {
            if(aiBoundaryVertexFlags[i] > 0)
            {
                aaiBoundaryVertices[iPartition].push_back(i);
            }
            else
            {
                aaiNonBoundaryVertices[iPartition].push_back(i);
            }
        }

        //printDebugMeshes(
        //    aaiBoundaryVertices,
        //    aaiNonBoundaryVertices,
        //    aaVertexPositions,
        //    iPartition);
    }
}