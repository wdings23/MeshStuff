#include "simplify_operations.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <sstream>

/*
**
*/
void contractEdge(
    std::vector<float3>& aClusterGroupVertexPositions,
    std::vector<float3>& aClusterGroupVertexNormals,
    std::vector<float2>& aClusterGroupVertexUVs,
    std::vector<uint32_t>& aiClusterGroupTrianglePositionIndices,
    std::vector<uint32_t>& aiClusterGroupTriangleNormalIndices,
    std::vector<uint32_t>& aiClusterGroupTriangleUVIndices,
    std::map<uint32_t, mat4>& aQuadrics,
    std::vector<std::pair<uint32_t, uint32_t>>& aValidClusterGroupEdgePairs,
    std::pair<uint32_t, uint32_t> const& edge,
    float3 const& replaceVertexPosition,
    float3 const& replaceVertexNormal,
    float2 const& replaceVertexUV,
    bool bDeleteContractedVertices)
{
    float3 edgePos0 = aClusterGroupVertexPositions[edge.first];
    float3 edgePos1 = aClusterGroupVertexPositions[edge.second];

    // delete the vertices of the edge quadrics to update them later on
    aQuadrics.erase(edge.first);
    aQuadrics.erase(edge.second);

    // replace the triangle vertex index with newly added vertex position index (at the end of the vertex position list)

    bool bDeleteTriangle = false;
    for(int32_t iTri = 0; iTri < static_cast<int32_t>(aiClusterGroupTrianglePositionIndices.size()); iTri += 3)
    {
        if(bDeleteTriangle)
        {
            iTri = 0;
            bDeleteTriangle = false;
        }

        uint32_t iPos0 = aiClusterGroupTrianglePositionIndices[iTri];
        uint32_t iPos1 = aiClusterGroupTrianglePositionIndices[iTri + 1];
        uint32_t iPos2 = aiClusterGroupTrianglePositionIndices[iTri + 2];

        uint32_t iNumMatches = 0;

        // set to end of cluster group vertex position list (vertex position to be added)
        if(iPos0 == edge.first || iPos0 == edge.second)
        {
            aiClusterGroupTrianglePositionIndices[iTri] = static_cast<uint32_t>(aClusterGroupVertexPositions.size());
            ++iNumMatches;
        }
        if(iPos1 == edge.first || iPos1 == edge.second)
        {
            aiClusterGroupTrianglePositionIndices[iTri + 1] = static_cast<uint32_t>(aClusterGroupVertexPositions.size());
            ++iNumMatches;
        }
        if(iPos2 == edge.first || iPos2 == edge.second)
        {
            aiClusterGroupTrianglePositionIndices[iTri + 2] = static_cast<uint32_t>(aClusterGroupVertexPositions.size());
            ++iNumMatches;
        }

        // is triangle containing the edge, just delete and restart loop from beginning
        if(iNumMatches >= 2)
        {
            aiClusterGroupTrianglePositionIndices.erase(
                aiClusterGroupTrianglePositionIndices.begin() + iTri,
                aiClusterGroupTrianglePositionIndices.begin() + iTri + 3);

            aiClusterGroupTriangleNormalIndices.erase(
                aiClusterGroupTriangleNormalIndices.begin() + iTri,
                aiClusterGroupTriangleNormalIndices.begin() + iTri + 3);

            aiClusterGroupTriangleUVIndices.erase(
                aiClusterGroupTriangleUVIndices.begin() + iTri,
                aiClusterGroupTriangleUVIndices.begin() + iTri + 3);

            //DEBUG_PRINTF("remove triangle %d (%d, %d, %d)\n", 
            //    iTri,
            //    iPos0,
            //    iPos1,
            //    iPos2);

            // restart
            bDeleteTriangle = true;
            iTri = -3;
        }
        else
        {
            bDeleteTriangle = false;
        }

    }   // for tri = 0 to num triangles in cluster group

    // replaced vertex position
    aClusterGroupVertexPositions.push_back(replaceVertexPosition);
    aClusterGroupVertexNormals.push_back(replaceVertexNormal);
    aClusterGroupVertexUVs.push_back(replaceVertexUV);

    // if delete contracted vertex positions
    if(bDeleteContractedVertices)
    {
        // build with just positions for updating the triangle indices
        std::vector<float3> aTriPositions(aiClusterGroupTrianglePositionIndices.size());
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterGroupTrianglePositionIndices.size()); iTri += 3)
        {
            uint32_t iPos0 = aiClusterGroupTrianglePositionIndices[iTri];
            uint32_t iPos1 = aiClusterGroupTrianglePositionIndices[iTri + 1];
            uint32_t iPos2 = aiClusterGroupTrianglePositionIndices[iTri + 2];

            aTriPositions[iTri] = aClusterGroupVertexPositions[iPos0];
            aTriPositions[iTri + 1] = aClusterGroupVertexPositions[iPos1];
            aTriPositions[iTri + 2] = aClusterGroupVertexPositions[iPos2];
        }

        // delete the edge positions
        auto iter0 = std::find_if(
            aClusterGroupVertexPositions.begin(),
            aClusterGroupVertexPositions.end(),
            [edgePos0](float3 const& checkPos)
            {
                return (length(edgePos0 - checkPos) <= 1.0e-5f);
            });
        assert(iter0 != aClusterGroupVertexPositions.end());
        uint64_t iDeleteIndex0 = std::distance(aClusterGroupVertexPositions.begin(), iter0);
        aClusterGroupVertexPositions.erase(aClusterGroupVertexPositions.begin() + iDeleteIndex0);

        aClusterGroupVertexNormals.erase(aClusterGroupVertexNormals.begin() + iDeleteIndex0);
        aClusterGroupVertexUVs.erase(aClusterGroupVertexUVs.begin() + iDeleteIndex0);

        auto iter1 = std::find_if(
            aClusterGroupVertexPositions.begin(),
            aClusterGroupVertexPositions.end(),
            [edgePos1](float3 const& checkPos)
            {
                return (length(edgePos1 - checkPos) <= 1.0e-5f);
            });
        assert(iter1 != aClusterGroupVertexPositions.end());
        uint64_t iDeleteIndex1 = std::distance(aClusterGroupVertexPositions.begin(), iter1);
        aClusterGroupVertexPositions.erase(aClusterGroupVertexPositions.begin() + iDeleteIndex1);

        aClusterGroupVertexNormals.erase(aClusterGroupVertexNormals.begin() + iDeleteIndex1);
        aClusterGroupVertexUVs.erase(aClusterGroupVertexUVs.begin() + iDeleteIndex1);

        // reset the triangle indices
        for(uint32_t iVert = 0; iVert < static_cast<uint32_t>(aTriPositions.size()); iVert++)
        {
            float3 pos = aTriPositions[iVert];
            auto iter = std::find_if(
                aClusterGroupVertexPositions.begin(),
                aClusterGroupVertexPositions.end(),
                [pos](float3 const& checkPos)
                {
                    return (length(pos - checkPos) <= 1.0e-5f);
                });

            assert(iter != aClusterGroupVertexPositions.end());
            uint32_t iIndex = static_cast<uint32_t>(std::distance(aClusterGroupVertexPositions.begin(), iter));
            aiClusterGroupTrianglePositionIndices[iVert] = iIndex;
            aiClusterGroupTriangleNormalIndices[iVert] = iIndex;
            aiClusterGroupTriangleUVIndices[iVert] = iIndex;
        }

    }   // if delete old vertex positions

    // delete the edge from the cluster group
    auto edgeIter = std::find_if(
        aValidClusterGroupEdgePairs.begin(),
        aValidClusterGroupEdgePairs.end(),
        [edge](std::pair<uint32_t, uint32_t> const& checkEdgePair)
        {
            return ((edge.first == checkEdgePair.first && edge.second == checkEdgePair.second) || (edge.first == checkEdgePair.second && edge.second == checkEdgePair.first));
        });
    assert(edgeIter != aValidClusterGroupEdgePairs.end());
    aValidClusterGroupEdgePairs.erase(edgeIter);

    // replace edges with new optimized vertex position, having the same edge vertex index 0 or edge vertex index 1
    for(;;)
    {
        auto edgeIter = std::find_if(
            aValidClusterGroupEdgePairs.begin(),
            aValidClusterGroupEdgePairs.end(),
            [edge](std::pair<uint32_t, uint32_t> const& checkEdgePair)
            {
                return (checkEdgePair.first == edge.first || checkEdgePair.first == edge.second || checkEdgePair.second == edge.first || checkEdgePair.second == edge.second);
            });

        // look for edges 
        if(edgeIter != aValidClusterGroupEdgePairs.end())
        {
            uint32_t iIndex = static_cast<uint32_t>(std::distance(aValidClusterGroupEdgePairs.begin(), edgeIter));
            if(edgeIter->first == edge.first || edgeIter->first == edge.second)
            {
                aValidClusterGroupEdgePairs[iIndex].first = static_cast<uint32_t>(aClusterGroupVertexPositions.size() - 1);
            }
            else if(edgeIter->second == edge.first || edgeIter->second == edge.second)
            {
                aValidClusterGroupEdgePairs[iIndex].second = static_cast<uint32_t>(aClusterGroupVertexPositions.size() - 1);
            }

            // replaced both of the vertices on this edge with the same vertex position, ie a point
            // delete the edge
            if(aValidClusterGroupEdgePairs[iIndex].first == aValidClusterGroupEdgePairs[iIndex].second)
            {
                aValidClusterGroupEdgePairs.erase(aValidClusterGroupEdgePairs.begin() + iIndex);
            }

        }
        else
        {
            break;
        }
    }
}

/*
**
*/
void updateClusterGroup(
    std::vector<float3>& aTrimmedClusterGroupVertexPositions,
    std::vector<uint32_t>& aiTrimmedClusterGroupTriangleIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangles,
    std::vector<float3> const& aClusterGroupTriangleVertexPositions)
{
    // trim positions
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTriangleVertexPositions.size()); i++)
    {
        float3 const& clusterGroupTriangleVertexPosition = aClusterGroupTriangleVertexPositions[i];
        auto checkIter = std::find_if(
            aTrimmedClusterGroupVertexPositions.begin(),
            aTrimmedClusterGroupVertexPositions.end(),
            [clusterGroupTriangleVertexPosition](float3 const& checkPosition)
            {
                return length(checkPosition - clusterGroupTriangleVertexPosition) <= 1.0e-5f;
            }
        );

        if(checkIter == aTrimmedClusterGroupVertexPositions.end())
        {
            aTrimmedClusterGroupVertexPositions.push_back(aClusterGroupTriangleVertexPositions[i]);
        }
    }

    // build triangle indices
    aiTrimmedClusterGroupTriangleIndices.resize(aClusterGroupTriangleVertexPositions.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTriangleVertexPositions.size()); i++)
    {
        float3 const& clusterGroupTriangleVertexPosition = aClusterGroupTriangleVertexPositions[i];
        auto checkIter = std::find_if(
            aTrimmedClusterGroupVertexPositions.begin(),
            aTrimmedClusterGroupVertexPositions.end(),
            [clusterGroupTriangleVertexPosition](float3 const& checkPosition)
            {
                return length(checkPosition - clusterGroupTriangleVertexPosition) <= 1.0e-7f;
            }
        );

        uint32_t iVertIndex = static_cast<uint32_t>(std::distance(aTrimmedClusterGroupVertexPositions.begin(), checkIter));
        aiTrimmedClusterGroupTriangleIndices[i] = iVertIndex;
    }
}

/*
**
*/
mat4 computeQuadric(
    float& fTotalNormalPlaneAngles,
    uint32_t iVertIndex,
    float3 const& vertexNormal,
    std::vector<uint32_t> const& aiClusterGroupTriangles,
    std::vector<float3> const& aVertexPositions)
{
    uint32_t iNumTriangles = static_cast<uint32_t>(aiClusterGroupTriangles.size());
    uint32_t iNumVertices = static_cast<uint32_t>(aVertexPositions.size());
    mat4 totalQuadricMatrix;
    memset(totalQuadricMatrix.mafEntries, 0, sizeof(float) * 16);

    uint32_t const* paiClusterGroupTriangles = aiClusterGroupTriangles.data();
    float3 const* paVertexPositions = aVertexPositions.data();

    float fAdjacentCount = 0.0f;
    for(uint32_t iTri = 0; iTri < iNumTriangles; iTri += 3)
    {
        uint32_t iV0 = paiClusterGroupTriangles[iTri];
        uint32_t iV1 = paiClusterGroupTriangles[iTri + 1];
        uint32_t iV2 = paiClusterGroupTriangles[iTri + 2];

        // compute plane for this shared face
        float4 plane = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if(iV0 == iVertIndex)
        {
            float3 diff0 = normalize(paVertexPositions[iV1] - paVertexPositions[iV0]);
            float3 diff1 = normalize(paVertexPositions[iV2] - paVertexPositions[iV0]);
            float3 normal = cross(diff1, diff0);
            plane = float4(
                normal.x,
                normal.y,
                normal.z,
                -dot(normal, paVertexPositions[iV0]));

            fTotalNormalPlaneAngles += fabsf(dot(vertexNormal, normal));

            mat4 matrix;
            matrix.mafEntries[0] = plane.x * plane.x;  matrix.mafEntries[1] = plane.x * plane.y; matrix.mafEntries[2] = plane.x * plane.z; matrix.mafEntries[3] = plane.x * plane.w;
            matrix.mafEntries[4] = plane.x * plane.y;  matrix.mafEntries[5] = plane.y * plane.y; matrix.mafEntries[6] = plane.y * plane.z; matrix.mafEntries[7] = plane.y * plane.w;
            matrix.mafEntries[8] = plane.x * plane.z;  matrix.mafEntries[9] = plane.y * plane.z; matrix.mafEntries[10] = plane.z * plane.z; matrix.mafEntries[11] = plane.z * plane.w;
            matrix.mafEntries[12] = plane.x * plane.w;  matrix.mafEntries[13] = plane.y * plane.w; matrix.mafEntries[14] = plane.z * plane.w; matrix.mafEntries[15] = plane.w * plane.w;

            totalQuadricMatrix += matrix;
            fAdjacentCount += 1.0f;
        }
        else if(iV1 == iVertIndex)
        {
            float3 diff0 = normalize(paVertexPositions[iV0] - paVertexPositions[iV1]);
            float3 diff1 = normalize(paVertexPositions[iV2] - paVertexPositions[iV1]);
            float3 normal = cross(diff1, diff0);
            plane = float4(
                normal.x,
                normal.y,
                normal.z,
                -dot(normal, paVertexPositions[iV0]));

            fTotalNormalPlaneAngles += fabsf(dot(vertexNormal, normal));

            mat4 matrix;
            matrix.mafEntries[0] = plane.x * plane.x;  matrix.mafEntries[1] = plane.x * plane.y; matrix.mafEntries[2] = plane.x * plane.z; matrix.mafEntries[3] = plane.x * plane.w;
            matrix.mafEntries[4] = plane.x * plane.y;  matrix.mafEntries[5] = plane.y * plane.y; matrix.mafEntries[6] = plane.y * plane.z; matrix.mafEntries[7] = plane.y * plane.w;
            matrix.mafEntries[8] = plane.x * plane.z;  matrix.mafEntries[9] = plane.y * plane.z; matrix.mafEntries[10] = plane.z * plane.z; matrix.mafEntries[11] = plane.z * plane.w;
            matrix.mafEntries[12] = plane.x * plane.w;  matrix.mafEntries[13] = plane.y * plane.w; matrix.mafEntries[14] = plane.z * plane.w; matrix.mafEntries[15] = plane.w * plane.w;

            totalQuadricMatrix += matrix;
            fAdjacentCount += 1.0f;
        }
        else if(iV2 == iVertIndex)
        {
            float3 diff0 = normalize(paVertexPositions[iV0] - paVertexPositions[iV2]);
            float3 diff1 = normalize(paVertexPositions[iV1] - paVertexPositions[iV2]);
            float3 normal = cross(diff1, diff0);
            plane = float4(
                normal.x,
                normal.y,
                normal.z,
                -dot(normal, paVertexPositions[iV0]));

            fTotalNormalPlaneAngles += fabsf(dot(vertexNormal, normal));

            mat4 matrix;
            matrix.mafEntries[0] = plane.x * plane.x;  matrix.mafEntries[1] = plane.x * plane.y; matrix.mafEntries[2] = plane.x * plane.z; matrix.mafEntries[3] = plane.x * plane.w;
            matrix.mafEntries[4] = plane.x * plane.y;  matrix.mafEntries[5] = plane.y * plane.y; matrix.mafEntries[6] = plane.y * plane.z; matrix.mafEntries[7] = plane.y * plane.w;
            matrix.mafEntries[8] = plane.x * plane.z;  matrix.mafEntries[9] = plane.y * plane.z; matrix.mafEntries[10] = plane.z * plane.z; matrix.mafEntries[11] = plane.z * plane.w;
            matrix.mafEntries[12] = plane.x * plane.w;  matrix.mafEntries[13] = plane.y * plane.w; matrix.mafEntries[14] = plane.z * plane.w; matrix.mafEntries[15] = plane.w * plane.w;

            totalQuadricMatrix += matrix;
            fAdjacentCount += 1.0f;
        }

    }   // for tri = 0 to num triangle0s in this cluster group

    fTotalNormalPlaneAngles /= fAdjacentCount;

    return totalQuadricMatrix;
}


/*
**
*/
void computeEdgeCollapseInfo(
    std::vector<std::pair<std::pair<uint32_t, uint32_t>, EdgeCollapseInfo>>& aSortedCollapseInfo,
    std::map<uint32_t, mat4>& aQuadrics,
    std::vector<float3> const& aClusterGroupVertexPositions,
    std::vector<float3> const& aClusterGroupVertexNormals,
    std::vector<float2> const& aClusterGroupVertexUVs,
    std::vector<std::pair<uint32_t, uint32_t>> const& aValidClusterGroupEdgePairs,
    std::vector<uint32_t> const& aiClusterGroupNonBoundaryVertices,
    std::vector<uint32_t> const& aiClusterGroupTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangleUVIndices,
    //std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices,
    uint32_t iClusterGroup,
    std::vector<std::pair<uint32_t, uint32_t>> const& aClusterGroupEdgePositions,
    std::vector<std::pair<uint32_t, uint32_t>> const& aClusterGroupEdgeNormals,
    std::vector<std::pair<uint32_t, uint32_t>> const& aClusterGroupEdgeUVs)
{
    std::vector<EdgeCollapseInfo> aEdgeCollapseCosts;
    std::vector<std::pair<uint32_t, uint32_t>> aEdges;

    std::pair<uint32_t, uint32_t> const* paValidClusterGroupEdgePairs = aValidClusterGroupEdgePairs.data();

    float3 const* paClusterGroupVertexPositions = aClusterGroupVertexPositions.data();
    float3 const* paClusterGroupVertexNormals = aClusterGroupVertexNormals.data();
    float2 const* paClusterGroupVertexUVs = aClusterGroupVertexUVs.data();

    uint32_t const* paiClusterGroupTrianglePositionIndices = aiClusterGroupTrianglePositionIndices.data();
    uint32_t const* paiClusterGroupTriangleNormalIndices = aiClusterGroupTriangleNormalIndices.data();
    uint32_t const* paiClusterGroupTriangleUVIndices = aiClusterGroupTriangleUVIndices.data();

    uint64_t iMicroseconds0 = 0, iMicroseconds1 = 0, iMicroseconds2 = 0, iMicroseconds3 = 0, iMicroseconds4 = 0, iMicroseconds5 = 0;
    uint64_t iMicroseconds8 = 0;

    auto start = std::chrono::high_resolution_clock::now();

    //for(auto const& edge : aValidClusterGroupEdgePairs)
    uint32_t iNumClusterGroupTrianglePositionIndices = static_cast<uint32_t>(aiClusterGroupTrianglePositionIndices.size());
    uint32_t iNumValidClusterGroupEdgePairs = static_cast<uint32_t>(aValidClusterGroupEdgePairs.size());
    for(uint32_t iEdge = 0; iEdge < iNumValidClusterGroupEdgePairs; iEdge++)
    {
        std::pair<uint32_t, uint32_t> const& edge = paValidClusterGroupEdgePairs[iEdge];
        assert(edge.first != edge.second);

        auto start0 = std::chrono::high_resolution_clock::now();

        uint32_t iNorm0 = UINT32_MAX, iNorm1 = UINT32_MAX;
        uint32_t iUV0 = UINT32_MAX, iUV1 = UINT32_MAX;
        {
            uint32_t iMatchingTri = 0;
            uint32_t aiTriIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
            for(iMatchingTri = 0; iMatchingTri < iNumClusterGroupTrianglePositionIndices; iMatchingTri += 3)
            {
                uint32_t iNumSamePosition = 0;
                aiTriIndices[0] = aiTriIndices[1] = aiTriIndices[2] = UINT32_MAX;
                for(uint32_t i = 0; i < 3; i++)
                {
                    if(paiClusterGroupTrianglePositionIndices[iMatchingTri + i] == edge.first ||
                        paiClusterGroupTrianglePositionIndices[iMatchingTri + i] == edge.second)
                    {
                        aiTriIndices[iNumSamePosition] = i;
                        ++iNumSamePosition;
                    }
                }

                if(iNumSamePosition >= 2)
                {
                    break;
                }
            }

            // didn't find a matching triangle for this edge, continue to the next edge
            if(iMatchingTri >= aiClusterGroupTrianglePositionIndices.size())
            {
                continue;
            }

            iNorm0 = paiClusterGroupTriangleNormalIndices[iMatchingTri + aiTriIndices[0]];
            iNorm1 = paiClusterGroupTriangleNormalIndices[iMatchingTri + aiTriIndices[1]];

            iUV0 = paiClusterGroupTriangleUVIndices[iMatchingTri + aiTriIndices[0]];
            iUV1 = paiClusterGroupTriangleUVIndices[iMatchingTri + aiTriIndices[1]];
        }

        auto end0 = std::chrono::high_resolution_clock::now();
        iMicroseconds0 += std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count();

        assert(iNorm0 != UINT32_MAX);
        assert(iNorm1 != UINT32_MAX);
        assert(iUV0 != UINT32_MAX);
        assert(iUV1 != UINT32_MAX);

        start0 = std::chrono::high_resolution_clock::now();

        bool bValid0 = false;
        bool bValid1 = false;
        uint32_t const* paiClusterGroupNonBoundaryVertices = aiClusterGroupNonBoundaryVertices.data();
        uint32_t iNumClusterGroupNonBoundaryVertices = static_cast<uint32_t>(aiClusterGroupNonBoundaryVertices.size());
        for(uint32_t i = 0; i < iNumClusterGroupNonBoundaryVertices; i++)
        {
            if(paiClusterGroupNonBoundaryVertices[i] == edge.first)
            {
                bValid0 = true;
            }

            if(paiClusterGroupNonBoundaryVertices[i] == edge.second)
            {
                bValid1 = true;
            }
        }

        assert(bValid0 || bValid1);

        end0 = std::chrono::high_resolution_clock::now();
        iMicroseconds1 += std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count();

        // set to boundary position if one of the vertex is on the boundary
        EdgeCollapseInfo edgeCollapseInfo;
        edgeCollapseInfo.mOptimalVertexPosition = float3(0.0f, 0.0f, 0.0f);
        edgeCollapseInfo.mOptimalNormal = float3(0.0f, 0.0f, 0.0f);
        edgeCollapseInfo.mOptimalUV = float2(0.0f, 0.0f);
        edgeCollapseInfo.mfCost = 0.0f;

        start0 = std::chrono::high_resolution_clock::now();

        // compute quadrics if not found in cache and add to cache
        float fTotalNormalPlaneAngles0 = 0.0f;
        if(aQuadrics.find(edge.first) == aQuadrics.end())
        {
            aQuadrics[edge.first] = computeQuadric(
                fTotalNormalPlaneAngles0,
                edge.first,
                aClusterGroupVertexNormals[iNorm0],
                aiClusterGroupTrianglePositionIndices,
                aClusterGroupVertexPositions);

            //DEBUG_PRINTF("compute quadric for vertex %d\n", edge.first);
        }

        float fTotalNormalPlaneAngles1 = 0.0f;
        if(aQuadrics.find(edge.second) == aQuadrics.end())
        {
            aQuadrics[edge.second] = computeQuadric(
                fTotalNormalPlaneAngles1,
                edge.second,
                aClusterGroupVertexNormals[iNorm1],
                aiClusterGroupTrianglePositionIndices,
                aClusterGroupVertexPositions);

            //DEBUG_PRINTF("compute quadric for vertex %d\n", edge.first);
        }

        end0 = std::chrono::high_resolution_clock::now();
        iMicroseconds2 += std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count();

        start0 = std::chrono::high_resolution_clock::now();

        // feature value
        float const kfFeatureMult = 1.0f;
        float fEdgeLength = lengthSquared(paClusterGroupVertexPositions[edge.second] - paClusterGroupVertexPositions[edge.first]);
        float fFeatureValue = fEdgeLength * (1.0f + 0.5f * (fTotalNormalPlaneAngles0 + fTotalNormalPlaneAngles1));

        // get the contraction position
        //mat4 edgeQuadric = quadric0 + quadric1;
        mat4 const& q0 = aQuadrics[edge.first];
        mat4 const& q1 = aQuadrics[edge.second];
        mat4 edgeQuadric = aQuadrics[edge.first] + aQuadrics[edge.second];
        edgeQuadric.mafEntries[15] += fFeatureValue;
        if(bValid0 == false)
        {
            // boundary
            edgeCollapseInfo.mOptimalVertexPosition = paClusterGroupVertexPositions[edge.first];
            edgeCollapseInfo.mOptimalNormal = paClusterGroupVertexNormals[iNorm0];
            edgeCollapseInfo.mOptimalUV = paClusterGroupVertexUVs[iUV0];
        }
        else if(bValid1 == false)
        {
            // boundary
            edgeCollapseInfo.mOptimalVertexPosition = paClusterGroupVertexPositions[edge.second];
            edgeCollapseInfo.mOptimalNormal = paClusterGroupVertexNormals[iNorm1];
            edgeCollapseInfo.mOptimalUV = paClusterGroupVertexUVs[iUV1];
        }
        else
        {
            // not on boundary, compute optimal position
            mat4 newMatrix;
            newMatrix.mafEntries[0] = edgeQuadric.mafEntries[0];
            newMatrix.mafEntries[1] = edgeQuadric.mafEntries[1];
            newMatrix.mafEntries[2] = edgeQuadric.mafEntries[2];
            newMatrix.mafEntries[3] = edgeQuadric.mafEntries[3];

            newMatrix.mafEntries[4] = edgeQuadric.mafEntries[1];
            newMatrix.mafEntries[5] = edgeQuadric.mafEntries[5];
            newMatrix.mafEntries[6] = edgeQuadric.mafEntries[6];
            newMatrix.mafEntries[7] = edgeQuadric.mafEntries[7];

            newMatrix.mafEntries[8] = edgeQuadric.mafEntries[2];
            newMatrix.mafEntries[9] = edgeQuadric.mafEntries[9];
            newMatrix.mafEntries[10] = edgeQuadric.mafEntries[10];
            newMatrix.mafEntries[11] = edgeQuadric.mafEntries[11];

            newMatrix.mafEntries[12] = 0.0f;
            newMatrix.mafEntries[13] = 0.0f;
            newMatrix.mafEntries[14] = 0.0f;
            newMatrix.mafEntries[15] = 1.0f + fFeatureValue;

            // mid point
            edgeCollapseInfo.mOptimalVertexPosition = (paClusterGroupVertexPositions[edge.first] + paClusterGroupVertexPositions[edge.second]) * 0.5f;
            edgeCollapseInfo.mOptimalNormal = (paClusterGroupVertexNormals[iNorm0] + paClusterGroupVertexNormals[iNorm1]) * 0.5f;
            edgeCollapseInfo.mOptimalUV = (paClusterGroupVertexUVs[iUV0] + paClusterGroupVertexUVs[iUV1]) * 0.5f;


#if 0
            // compute the optimal vertex if q0 + q1 quadric matrix is invertible, otherwise just set the optimal position to the mid point of the edge
            mat4 inverse = invert(newMatrix);
            if(inverse.mafEntries[0] == FLT_MAX)
            {
                // mid point
                edgeCollapseInfo.mOptimalVertexPosition = (paClusterGroupVertexPositions[edge.first] + paClusterGroupVertexPositions[edge.second]) * 0.5f;
                edgeCollapseInfo.mOptimalNormal = (paClusterGroupVertexNormals[iNorm0] + paClusterGroupVertexNormals[iNorm1]) * 0.5f;
                edgeCollapseInfo.mOptimalUV = (paClusterGroupVertexUVs[iUV0] + paClusterGroupVertexUVs[iUV1]) * 0.5f;
            }
            else
            {
                auto st = std::chrono::high_resolution_clock::now();
                edgeCollapseInfo.mOptimalVertexPosition = inverse * float4(0.0f, 0.0f, 0.0f, 1.0f);

                // get the best barycentric coordinate of the optimal vertex position
                float3 barycentricCoord = float3(FLT_MAX, FLT_MAX, FLT_MAX);
                uint32_t iBarycentricTri = UINT32_MAX;

                // TODO: verify this
                if(barycentricCoord.x != FLT_MAX)
                {
                    uint32_t iNormal0 = paiClusterGroupTriangleNormalIndices[iBarycentricTri];
                    uint32_t iNormal1 = paiClusterGroupTriangleNormalIndices[iBarycentricTri + 1];
                    uint32_t iNormal2 = paiClusterGroupTriangleNormalIndices[iBarycentricTri + 2];

                    uint32_t iUV0 = paiClusterGroupTriangleUVIndices[iBarycentricTri];
                    uint32_t iUV1 = paiClusterGroupTriangleUVIndices[iBarycentricTri + 1];
                    uint32_t iUV2 = paiClusterGroupTriangleUVIndices[iBarycentricTri + 2];

                    edgeCollapseInfo.mOptimalNormal =
                        paClusterGroupVertexNormals[iNormal0] * barycentricCoord.x +
                        paClusterGroupVertexNormals[iNormal1] * barycentricCoord.y +
                        paClusterGroupVertexNormals[iNormal2] * barycentricCoord.z;
                    edgeCollapseInfo.mOptimalUV =
                        paClusterGroupVertexUVs[iUV0] * barycentricCoord.x +
                        paClusterGroupVertexUVs[iUV1] * barycentricCoord.y +
                        paClusterGroupVertexUVs[iUV2] * barycentricCoord.z;
                }
                else
                {
                    edgeCollapseInfo.mOptimalNormal = (paClusterGroupVertexNormals[iNorm0] + paClusterGroupVertexNormals[iNorm1]) * 0.5f;
                    edgeCollapseInfo.mOptimalUV = (paClusterGroupVertexUVs[iUV0] + paClusterGroupVertexUVs[iUV1]) * 0.5f;
                }
            }
#endif // #if 0
        }

        end0 = std::chrono::high_resolution_clock::now();
        iMicroseconds3 += std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count();

        start0 = std::chrono::high_resolution_clock::now();

        // compute the cost of the contraction (transpose(v_optimal) * M * v_optimal)
        float4 ret = transpose(edgeQuadric) * float4(edgeCollapseInfo.mOptimalVertexPosition.x, edgeCollapseInfo.mOptimalVertexPosition.y, edgeCollapseInfo.mOptimalVertexPosition.z, 1.0f);
        //edgeCollapseInfo.mfCost = dot(ret, edgeCollapseInfo.mOptimalVertexPosition);
        edgeCollapseInfo.mfCost =
            edgeQuadric.mafEntries[0] * edgeCollapseInfo.mOptimalVertexPosition.x * edgeCollapseInfo.mOptimalVertexPosition.x +
            2.0f * edgeQuadric.mafEntries[1] * edgeCollapseInfo.mOptimalVertexPosition.x * edgeCollapseInfo.mOptimalVertexPosition.y +
            2.0f * edgeQuadric.mafEntries[2] * edgeCollapseInfo.mOptimalVertexPosition.x * edgeCollapseInfo.mOptimalVertexPosition.z +
            2.0f * edgeQuadric.mafEntries[3] * edgeCollapseInfo.mOptimalVertexPosition.x +

            edgeQuadric.mafEntries[5] * edgeCollapseInfo.mOptimalVertexPosition.y * edgeCollapseInfo.mOptimalVertexPosition.y +
            2.0f * edgeQuadric.mafEntries[6] * edgeCollapseInfo.mOptimalVertexPosition.y * edgeCollapseInfo.mOptimalVertexPosition.z +
            2.0f * edgeQuadric.mafEntries[7] * edgeCollapseInfo.mOptimalVertexPosition.y +

            edgeQuadric.mafEntries[10] * edgeCollapseInfo.mOptimalVertexPosition.z * edgeCollapseInfo.mOptimalVertexPosition.z +
            2.0f * edgeQuadric.mafEntries[11] * edgeCollapseInfo.mOptimalVertexPosition.z +

            edgeQuadric.mafEntries[15];

        end0 = std::chrono::high_resolution_clock::now();
        iMicroseconds4 += std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count();

        start0 = std::chrono::high_resolution_clock::now();

#if 0
        // apply large multiplier penalty if any of the vertices of the edge is on the boundary
        auto checkBoundaryVertexIter = std::find_if(
            aBoundaryVertices.begin(),
            aBoundaryVertices.end(),
            [iClusterGroup, edge](std::pair<uint32_t, uint32_t> const& checkBoundaryVertex)
            {
                return ((checkBoundaryVertex.first == iClusterGroup && checkBoundaryVertex.second == edge.first) ||
                    (checkBoundaryVertex.first == iClusterGroup && checkBoundaryVertex.second == edge.second));
            }
        );
        if(checkBoundaryVertexIter != aBoundaryVertices.end())
        {
            edgeCollapseInfo.mfCost = 1.0e+10f;
        }
#endif // #if 0

        aEdges.push_back(edge);
        aEdgeCollapseCosts.push_back(edgeCollapseInfo);

        end0 = std::chrono::high_resolution_clock::now();
        iMicroseconds5 += std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count();

    }

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t iMicroseconds6 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();

    // collapse cost list starting from smallest 
    aSortedCollapseInfo.resize(aEdgeCollapseCosts.size());
    {
        uint32_t iIndex = 0;
        for(uint32_t iIndex = 0; iIndex < static_cast<uint32_t>(aEdgeCollapseCosts.size()); iIndex++)
        {
            auto const& edge = aEdges[iIndex];
            //aSortedCollapseInfo.push_back(keyValue);
            aSortedCollapseInfo[iIndex] = std::make_pair(edge, aEdgeCollapseCosts[iIndex]);
        }

        std::sort(
            aSortedCollapseInfo.begin(),
            aSortedCollapseInfo.end(),
            [](std::pair<std::pair<uint32_t, uint32_t>, EdgeCollapseInfo>& collapseInfo0,
                std::pair<std::pair<uint32_t, uint32_t>, EdgeCollapseInfo>& collapseInfo1)
            {
                return collapseInfo0.second.mfCost < collapseInfo1.second.mfCost;
            }
        );
    }

    end = std::chrono::high_resolution_clock::now();
    uint64_t iMicroseconds7 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    int iDebug = 1;
}

/*
**
*/
void simplifyClusterGroup(
    std::map<uint32_t, mat4>& aQuadrics,
    std::vector<float3>& aClusterGroupVertexPositions,
    std::vector<float3>& aClusterGroupVertexNormals,
    std::vector<float2>& aClusterGroupVertexUVs,
    std::vector<uint32_t>& aiClusterGroupNonBoundaryVertices,
    std::vector<uint32_t>& aiClusterGroupBoundaryVertices,
    std::vector<uint32_t>& aiClusterGroupTrianglePositions,
    std::vector<uint32_t>& aiClusterGroupTriangleNormals,
    std::vector<uint32_t>& aiClusterGroupTriangleUVs,
    std::vector<std::pair<uint32_t, uint32_t>>& aValidClusterGroupEdgePairs,
    float& fTotalError,
    //std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices,
    uint32_t iMaxTriangles,
    uint32_t iClusterGroup,
    uint32_t iLODLevel,
    std::string const& meshModelName,
    std::string const& homeDirectory)
{
    uint32_t const kiNumEdgesToCollapse = 10;

    fTotalError = 0.0f;
    while(aiClusterGroupTrianglePositions.size() >= iMaxTriangles)
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<uint32_t, uint32_t>> aClusterGroupTriEdgePositions;
        std::vector<std::pair<uint32_t, uint32_t>> aClusterGroupTriEdgeNormals;
        std::vector<std::pair<uint32_t, uint32_t>> aClusterGroupTriEdgeUVs;
        if(aValidClusterGroupEdgePairs.size() <= 0)
        {
            break;
        }

        std::vector<std::pair<std::pair<uint32_t, uint32_t>, EdgeCollapseInfo>> aSortedCollapseInfo;
        computeEdgeCollapseInfo(
            aSortedCollapseInfo,
            aQuadrics,
            aClusterGroupVertexPositions,
            aClusterGroupVertexNormals,
            aClusterGroupVertexUVs,
            aValidClusterGroupEdgePairs,
            aiClusterGroupNonBoundaryVertices,
            aiClusterGroupTrianglePositions,
            aiClusterGroupTriangleNormals,
            aiClusterGroupTriangleUVs,
            //aBoundaryVertices,
            iClusterGroup,
            aClusterGroupTriEdgePositions,
            aClusterGroupTriEdgeNormals,
            aClusterGroupTriEdgeUVs);

        if(aSortedCollapseInfo.size() <= 0)
        {
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t iMicroseconds0 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        uint32_t iNumSortedCollapseInfo = static_cast<uint32_t>(aSortedCollapseInfo.size());
        uint32_t iNumEdgesToCollapse = std::min(kiNumEdgesToCollapse, iNumSortedCollapseInfo);
        for(uint32_t iStartEdge = 0; iStartEdge < iNumEdgesToCollapse; iStartEdge++)
        {
            std::pair<std::pair<uint32_t, uint32_t>, EdgeCollapseInfo> const* pCollapseInfo = nullptr;
            for(uint32_t iEdge = iStartEdge; iEdge < static_cast<uint32_t>(aSortedCollapseInfo.size()); iEdge++)
            {
                auto const& collapseInfo = aSortedCollapseInfo[iEdge];
                auto const& edge = collapseInfo.first;

                auto iter = std::find(
                    aValidClusterGroupEdgePairs.begin(),
                    aValidClusterGroupEdgePairs.end(),
                    edge);

                // check edge of validity
                if(iter == aValidClusterGroupEdgePairs.end())
                {
                    continue;
                }

                // no edge found
                if(edge.first == 0 && edge.second == 0)
                {
                    continue;
                }

                if(collapseInfo.second.mfCost >= -1.0e-5f)
                {
                    pCollapseInfo = &collapseInfo;
                    break;
                }
            }

            if(pCollapseInfo == nullptr)
            {
                // out of edges to collapse
                break;
            }

            fTotalError += pCollapseInfo->second.mfCost;

            start = std::chrono::high_resolution_clock::now();

            std::pair<uint32_t, uint32_t> const& edge = pCollapseInfo->first;
            EdgeCollapseInfo const& collapseInfo = pCollapseInfo->second;
            float3 const& replaceVertexPosition = collapseInfo.mOptimalVertexPosition;
            float3 const& replaceVertexNormal = collapseInfo.mOptimalNormal;
            float2 const& replaceVertexUV = collapseInfo.mOptimalUV;

            assert(aiClusterGroupTrianglePositions.size() == aiClusterGroupTriangleNormals.size());
            assert(aiClusterGroupTrianglePositions.size() == aiClusterGroupTriangleUVs.size());

            contractEdge(
                aClusterGroupVertexPositions,
                aClusterGroupVertexNormals,
                aClusterGroupVertexUVs,
                aiClusterGroupTrianglePositions,
                aiClusterGroupTriangleNormals,
                aiClusterGroupTriangleUVs,
                aQuadrics,
                aValidClusterGroupEdgePairs,
                edge,
                replaceVertexPosition,
                replaceVertexNormal,
                replaceVertexUV,
                false);

            assert(aiClusterGroupTrianglePositions.size() == aiClusterGroupTriangleNormals.size());
            assert(aiClusterGroupTrianglePositions.size() == aiClusterGroupTriangleUVs.size());

            end = std::chrono::high_resolution_clock::now();
            uint64_t iMicroseconds1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            //DEBUG_PRINTF("contract edge (%d, %d) after num edges: %d total error: %.4f\n", 
            //    edge.first, 
            //    edge.second,
            //    aValidClusterGroupEdgePairs.size(),
            //    fTotalError);

            for(uint32_t iTest = 0; iTest < aiClusterGroupBoundaryVertices.size(); iTest++)
            {
                assert(aiClusterGroupBoundaryVertices[iTest] < aClusterGroupVertexPositions.size());
            }

            for(uint32_t iTest = 0; iTest < aiClusterGroupNonBoundaryVertices.size(); iTest++)
            {
                assert(aiClusterGroupNonBoundaryVertices[iTest] < aClusterGroupVertexPositions.size());
            }

            start = std::chrono::high_resolution_clock::now();
            // add the newly create vertex to non-boundary vertex
            {
                auto checkBoundaryIter = std::find_if(
                    aiClusterGroupBoundaryVertices.begin(),
                    aiClusterGroupBoundaryVertices.end(),
                    [replaceVertexPosition, aClusterGroupVertexPositions](uint32_t iBoundaryVertexIndex)
                    {
                        float fLength = length(aClusterGroupVertexPositions[iBoundaryVertexIndex] - replaceVertexPosition);
                        return (fLength <= 1.0e-5f);
                    });

                uint32_t iNewVertexIndex = static_cast<uint32_t>(aClusterGroupVertexPositions.size() - 1);
                if(checkBoundaryIter == aiClusterGroupBoundaryVertices.end())
                {
                    // non-boundary vertex
                    aiClusterGroupNonBoundaryVertices.push_back(iNewVertexIndex);
                }
                else
                {
                    // boundary vertex
                    aiClusterGroupBoundaryVertices.push_back(iNewVertexIndex);
                }
            }

            end = std::chrono::high_resolution_clock::now();
            uint64_t iMicroseconds2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            start = std::chrono::high_resolution_clock::now();
            // check and remove total boundary edges
            bool bEdgeDeleted = false;
            std::pair<uint32_t, uint32_t> const* paValidClusterGroupEdgePairs = aValidClusterGroupEdgePairs.data();
            for(int32_t iEdge = 0; iEdge < static_cast<int32_t>(aValidClusterGroupEdgePairs.size()); iEdge++)
            {
                if(bEdgeDeleted)
                {
                    iEdge = 0;
                    bEdgeDeleted = false;
                }

                auto const& edge = paValidClusterGroupEdgePairs[iEdge];
                auto boundaryIter0 = std::find_if(
                    aiClusterGroupBoundaryVertices.begin(),
                    aiClusterGroupBoundaryVertices.end(),
                    [edge](uint32_t const& iVertexIndex)
                    {
                        return iVertexIndex == edge.first;
                    });
                if(boundaryIter0 != aiClusterGroupBoundaryVertices.end())
                {
                    auto boundaryIter1 = std::find_if(
                        aiClusterGroupBoundaryVertices.begin(),
                        aiClusterGroupBoundaryVertices.end(),
                        [edge](uint32_t const& iVertexIndex)
                        {
                            return iVertexIndex == edge.second;
                        });

                    if(boundaryIter1 != aiClusterGroupBoundaryVertices.end())
                    {
                        aValidClusterGroupEdgePairs.erase(aValidClusterGroupEdgePairs.begin() + iEdge);

                        iEdge = -1;
                        bEdgeDeleted = true;

                    }   // if edge position 1 is boundary vertex

                }   // if edge position 0 is boundary vertex 

            }   // for edge = 0 to num valid cluster group edges

        }   // for edge = 0 to num edges to collapse at once

        //DEBUG_PRINTF("num edges: %d after contraction\n",
        //    aValidClusterGroupEdgePairs.size());

        end = std::chrono::high_resolution_clock::now();
        uint64_t iMicroseconds3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    }   // while cluster group has more than the maximum given triangles

    auto start = std::chrono::high_resolution_clock::now();

    // get the triangle vertex positions
    std::vector<float3> aClusterGroupTriangleVertexPositions(aiClusterGroupTrianglePositions.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterGroupTrianglePositions.size()); iTri += 3)
    {
        uint32_t iV0 = aiClusterGroupTrianglePositions[iTri];
        uint32_t iV1 = aiClusterGroupTrianglePositions[iTri + 1];
        uint32_t iV2 = aiClusterGroupTrianglePositions[iTri + 2];

        aClusterGroupTriangleVertexPositions[iTri] = aClusterGroupVertexPositions[iV0];
        aClusterGroupTriangleVertexPositions[iTri + 1] = aClusterGroupVertexPositions[iV1];
        aClusterGroupTriangleVertexPositions[iTri + 2] = aClusterGroupVertexPositions[iV2];
    }

    std::vector<float3> aTrimmedClusterGroupVertexPositions;
    std::vector<uint32_t> aiTrimmedClusterGroupTriangleIndices;
    updateClusterGroup(
        aTrimmedClusterGroupVertexPositions,
        aiTrimmedClusterGroupTriangleIndices,
        aiClusterGroupTrianglePositions,
        aClusterGroupTriangleVertexPositions);

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t iMicroseconds4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    // output obj file
    {
        std::ostringstream simplifiedClusterFolderPath;
        {
            simplifiedClusterFolderPath << homeDirectory << "simplified-cluster-groups\\" << meshModelName << "\\";
            std::filesystem::path simplifiedClusterFolderFileSystemPath(simplifiedClusterFolderPath.str());
            if(!std::filesystem::exists(simplifiedClusterFolderFileSystemPath))
            {
                std::filesystem::create_directory(simplifiedClusterFolderFileSystemPath);
            }
        }

        std::ostringstream clusterGroupName;
        clusterGroupName << "simplified-cluster-group-lod";
        clusterGroupName << iLODLevel << "-group";
        clusterGroupName << iClusterGroup;

        std::ostringstream outputFilePath;
        outputFilePath << simplifiedClusterFolderPath.str();
        outputFilePath << clusterGroupName.str() << ".obj";

        FILE* fp = fopen(outputFilePath.str().c_str(), "wb");
        assert(fp);
        fprintf(fp, "o %s\n", clusterGroupName.str().c_str());
        fprintf(fp, "usemtl %s\n", clusterGroupName.str().c_str());
        for(uint32_t i = 0; i < static_cast<uint32_t>(aTrimmedClusterGroupVertexPositions.size()); i++)
        {
            fprintf(fp, "v %.4f %.4f %.4f\n",
                aTrimmedClusterGroupVertexPositions[i].x,
                aTrimmedClusterGroupVertexPositions[i].y,
                aTrimmedClusterGroupVertexPositions[i].z);
        }

        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTrimmedClusterGroupTriangleIndices.size()); iTri += 3)
        {
            fprintf(fp, "f %d// %d// %d//\n",
                aiTrimmedClusterGroupTriangleIndices[iTri] + 1,
                aiTrimmedClusterGroupTriangleIndices[iTri + 1] + 1,
                aiTrimmedClusterGroupTriangleIndices[iTri + 2] + 1);
        }

        fclose(fp);

        // material file
        float fRand0 = float(rand() % 255) / 255.0f;
        float fRand1 = float(rand() % 255) / 255.0f;
        float fRand2 = float(rand() % 255) / 255.0f;
        std::ostringstream clusterGroupMaterialFilePath;
        clusterGroupMaterialFilePath << simplifiedClusterFolderPath.str();
        clusterGroupMaterialFilePath << clusterGroupName.str() << ".mtl";
        fp = fopen(clusterGroupMaterialFilePath.str().c_str(), "wb");
        fprintf(fp, "newmtl %s\n", clusterGroupName.str().c_str());
        fprintf(fp, "Kd %.4f %.4f %.4f\n",
            fRand0,
            fRand1,
            fRand2);
        fclose(fp);
    }

    end = std::chrono::high_resolution_clock::now();
    uint64_t iMicroseconds5 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


#if 0
    // check for lone vertices
    {
        for(uint32_t iV = 0; iV < static_cast<uint32_t>(aiTrimmedClusterGroupTriangleIndices.size()); iV++)
        {
            uint32_t iVIndex = aiTrimmedClusterGroupTriangleIndices[iV];
            uint32_t iNumSame = 0;
            for(uint32_t iCheck = 0; iCheck < static_cast<uint32_t>(aiTrimmedClusterGroupTriangleIndices.size()); iCheck++)
            {
                if(iVIndex == aiTrimmedClusterGroupTriangleIndices[iCheck])
                {
                    ++iNumSame;
                }
            }
        }
    }
#endif // #if 0
}

/*
**
*/
void rebuildSimplifiedMeshData(
    std::vector<float3>& aVertexPositions,
    std::vector<uint32_t>& aiTrianglePositionIndices,
    std::vector<std::vector<float3>> const& aaClusterGroupVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTriangles,
    uint32_t iLODLevel,
    std::string const& homeDirectory)
{
    std::vector<float3> aTotalVertexPositions;
    for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaClusterGroupVertexPositions.size()); iClusterGroup++)
    {
        auto const& aClusterGroupVertexPositions = aaClusterGroupVertexPositions[iClusterGroup];
        auto const& aiClusterTriangleIndices = aaiClusterGroupTriangles[iClusterGroup];
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTriangleIndices.size()); iTri += 3)
        {
            uint32_t iPos0 = aiClusterTriangleIndices[iTri];
            uint32_t iPos1 = aiClusterTriangleIndices[iTri + 1];
            uint32_t iPos2 = aiClusterTriangleIndices[iTri + 2];

            aTotalVertexPositions.push_back(aClusterGroupVertexPositions[iPos0]);
            aTotalVertexPositions.push_back(aClusterGroupVertexPositions[iPos1]);
            aTotalVertexPositions.push_back(aClusterGroupVertexPositions[iPos2]);
        }
    }

    for(auto const& vertexPosition : aTotalVertexPositions)
    {
        auto iter = std::find_if(
            aVertexPositions.begin(),
            aVertexPositions.end(),
            [vertexPosition](float3 const& checkPos)
            {
                return length(checkPos - vertexPosition) < 1.0e-5f;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aVertexPositions.size());
        if(iter == aVertexPositions.end())
        {
            aVertexPositions.push_back(vertexPosition);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aVertexPositions.begin(), iter));
        }

        aiTrianglePositionIndices.push_back(iVertexIndex);
    }

    // write to obj file
    {
        std::ostringstream simplifiedMeshName;
        simplifiedMeshName << "simplified-" << iLODLevel;

        std::ostringstream outputFilePath;
        outputFilePath << homeDirectory << "simplified\\";
        outputFilePath << simplifiedMeshName.str() << ".obj";
        FILE* fp = fopen(outputFilePath.str().c_str(), "wb");
        fprintf(fp, "o %s\n", simplifiedMeshName.str().c_str());
        fprintf(fp, "usemtl %s\n", simplifiedMeshName.str().c_str());
        for(uint32_t i = 0; i < static_cast<uint32_t>(aVertexPositions.size()); i++)
        {
            fprintf(fp, "v %.4f %.4f %.4f\n",
                aVertexPositions[i].x,
                aVertexPositions[i].y,
                aVertexPositions[i].z);
        }

        for(uint32_t i = 0; i < static_cast<uint32_t>(aiTrianglePositionIndices.size()); i += 3)
        {
            fprintf(fp, "f %d// %d// %d//\n",
                aiTrianglePositionIndices[i] + 1,
                aiTrianglePositionIndices[i + 1] + 1,
                aiTrianglePositionIndices[i + 2] + 1);
        }
        fclose(fp);

        float fRand0 = float(rand() % 255) / 255.0f;
        float fRand1 = float(rand() % 255) / 255.0f;
        float fRand2 = float(rand() % 255) / 255.0f;

        std::ostringstream outputMaterialFilePath;
        outputMaterialFilePath << homeDirectory << "simplified\\";
        outputMaterialFilePath << simplifiedMeshName.str() << ".mtl";
        fp = fopen(outputMaterialFilePath.str().c_str(), "wb");
        fprintf(fp, "newmtl %s\n", simplifiedMeshName.str().c_str());
        fprintf(fp, "Kd %.4f %.4f %.4f\n",
            fRand0,
            fRand1,
            fRand2);
        fclose(fp);
    }
}