#include "move_operations.h"
#include <assert.h>

#include <sstream>

#include "obj_helper.h"
#include "LogPrint.h"

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
    std::vector<std::vector<float3>>& aaNewVertexPositions,
    std::vector<std::vector<float3>>& aaNewVertexNormals, 
    std::vector<std::vector<float2>>& aaNewVertexUVs, 
    std::vector<std::vector<uint32_t>>& aaiNewVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiNewVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiNewVertexUVIndices,
    std::vector<std::vector<float3>> const& aaVertexPositions,
    std::vector<std::vector<float3>> const& aaVertexNormals, 
    std::vector<std::vector<float2>> const& aaVertexUVs, 
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiVertexUVIndices,
    uint32_t iCluster,
    uint32_t iCheckTri,
    uint32_t iNewIndex)
{
    for(uint32_t i = 0; i < 3; i++)
    {
        // position
        {
            uint32_t iIndex = static_cast<uint32_t>(aaNewVertexPositions[iNewIndex].size());
            uint32_t iPos = aaiVertexPositionIndices[iCluster][iCheckTri + i];
            float3 const& pos = aaVertexPositions[iCluster][iPos];
            auto iter = std::find_if(
                aaNewVertexPositions[iNewIndex].begin(),
                aaNewVertexPositions[iNewIndex].end(),
                [pos](float3 const& checkPos)
                {
                    return lengthSquared(pos - checkPos) <= 1.0e-7f;
                });
            if(iter == aaNewVertexPositions[iNewIndex].end())
            {
                aaNewVertexPositions[iNewIndex].push_back(pos);
            }
            else
            {
                iIndex = static_cast<uint32_t>(std::distance(aaNewVertexPositions[iNewIndex].begin(), iter));
            }
            aaiNewVertexPositionIndices[iNewIndex].push_back(iIndex);
        }

        // normal
        {
            uint32_t iIndex = static_cast<uint32_t>(aaNewVertexNormals[iNewIndex].size());
            uint32_t iNormal = aaiVertexNormalIndices[iCluster][iCheckTri + i];
            float3 const& normal = aaVertexNormals[iCluster][iNormal];
            auto iter = std::find_if(
                aaNewVertexNormals[iNewIndex].begin(),
                aaNewVertexNormals[iNewIndex].end(),
                [normal](float3 const& checkPos)
                {
                    return lengthSquared(normal - checkPos) <= 1.0e-7f;
                });
            if(iter == aaNewVertexNormals[iNewIndex].end())
            {
                aaNewVertexNormals[iNewIndex].push_back(normal);
            }
            else
            {
                iIndex = static_cast<uint32_t>(std::distance(aaNewVertexNormals[iNewIndex].begin(), iter));
            }
            aaiNewVertexNormalIndices[iNewIndex].push_back(iIndex);
        }

        // uv
        {
            uint32_t iIndex = static_cast<uint32_t>(aaNewVertexUVs[iNewIndex].size());
            uint32_t iUV = aaiVertexUVIndices[iCluster][iCheckTri + i];
            float2 const& uv = aaVertexUVs[iCluster][iUV];
            auto iter = std::find_if(
                aaNewVertexUVs[iNewIndex].begin(),
                aaNewVertexUVs[iNewIndex].end(),
                [uv](float2 const& checkPos)
                {
                    return lengthSquared(uv - checkPos) <= 1.0e-7f;
                });
            if(iter == aaNewVertexUVs[iNewIndex].end())
            {
                aaNewVertexUVs[iNewIndex].push_back(uv);
            }
            else
            {
                iIndex = static_cast<uint32_t>(std::distance(aaNewVertexUVs[iNewIndex].begin(), iter));
            }
            aaiNewVertexUVIndices[iNewIndex].push_back(iIndex);

        }

    }   // for i = 0 to 3
}

/*
**
*/
bool moveTriangles(
    std::vector<std::vector<float3>>& aaVertexPositions,
    std::vector<std::vector<float3>>& aaVertexNormals,
    std::vector<std::vector<float2>>& aaVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexUVIndices,
    uint32_t iSrcCluster,
    uint32_t iMaxTriangleVertexCount)
{
    std::vector<std::vector<float3>> aaNewVertexPositions(2);
    std::vector<std::vector<float3>> aaNewVertexNormals(2);
    std::vector<std::vector<float2>> aaNewVertexUVs(2);
    std::vector<std::vector<uint32_t>> aaiNewVertexPositionIndices(2);
    std::vector<std::vector<uint32_t>> aaiNewVertexNormalIndices(2);
    std::vector<std::vector<uint32_t>> aaiNewVertexUVIndices(2);

    assert(aaiVertexPositionIndices[iSrcCluster].size() > iMaxTriangleVertexCount);
    uint32_t iVerticesToMove = static_cast<uint32_t>(aaiVertexPositionIndices[iSrcCluster].size()) - iMaxTriangleVertexCount;
    uint32_t iFinishDestClusterSize = static_cast<uint32_t>(aaNewVertexPositions[0].size()) + iVerticesToMove;

    std::vector<uint32_t> aiSharedEdge(aaiVertexPositionIndices[iSrcCluster].size());
    
    uint32_t iNumAddedTriangleVertices = 0;
    uint32_t iDestCluster = 0;
    for(iDestCluster = 0; iDestCluster < static_cast<uint32_t>(aaiVertexPositionIndices.size() + 1); iDestCluster++)
    {
        if(iDestCluster == iSrcCluster)
        {
            continue;
        }

        if(iDestCluster == aaiVertexPositionIndices.size())
        {
            // need new list 

            aaVertexPositions.push_back(std::vector<float3>());
            aaVertexNormals.push_back(std::vector<float3>());
            aaVertexUVs.push_back(std::vector<float2>());
            aaiVertexPositionIndices.push_back(std::vector<uint32_t>());
            aaiVertexNormalIndices.push_back(std::vector<uint32_t>());
            aaiVertexUVIndices.push_back(std::vector<uint32_t>());

            iDestCluster = static_cast<uint32_t>(aaVertexPositions.size() - 1);

            // add first triangle
            aaNewVertexPositions[0].push_back(aaVertexPositions[iSrcCluster][aaiVertexPositionIndices[iSrcCluster][0]]);
            aaNewVertexPositions[0].push_back(aaVertexPositions[iSrcCluster][aaiVertexPositionIndices[iSrcCluster][1]]);
            aaNewVertexPositions[0].push_back(aaVertexPositions[iSrcCluster][aaiVertexPositionIndices[iSrcCluster][2]]);

            aaNewVertexNormals[0].push_back(aaVertexNormals[iSrcCluster][aaiVertexNormalIndices[iSrcCluster][0]]);
            aaNewVertexNormals[0].push_back(aaVertexNormals[iSrcCluster][aaiVertexNormalIndices[iSrcCluster][1]]);
            aaNewVertexNormals[0].push_back(aaVertexNormals[iSrcCluster][aaiVertexNormalIndices[iSrcCluster][2]]);

            aaNewVertexUVs[0].push_back(aaVertexUVs[iSrcCluster][aaiVertexUVIndices[iSrcCluster][0]]);
            aaNewVertexUVs[0].push_back(aaVertexUVs[iSrcCluster][aaiVertexUVIndices[iSrcCluster][1]]);
            aaNewVertexUVs[0].push_back(aaVertexUVs[iSrcCluster][aaiVertexUVIndices[iSrcCluster][2]]);

            aaiNewVertexPositionIndices[0].push_back(0);
            aaiNewVertexPositionIndices[0].push_back(1);
            aaiNewVertexPositionIndices[0].push_back(2);

            aaiNewVertexNormalIndices[0].push_back(0);
            aaiNewVertexNormalIndices[0].push_back(1);
            aaiNewVertexNormalIndices[0].push_back(2);

            aaiNewVertexUVIndices[0].push_back(0);
            aaiNewVertexUVIndices[0].push_back(1);
            aaiNewVertexUVIndices[0].push_back(2);
        }
        else if(aaiVertexPositionIndices[iDestCluster].size() + iVerticesToMove <= iMaxTriangleVertexCount)
        {
            aaNewVertexPositions[0].insert(aaNewVertexPositions[0].end(), aaVertexPositions[iDestCluster].begin(), aaVertexPositions[iDestCluster].end());
            aaNewVertexNormals[0].insert(aaNewVertexNormals[0].end(), aaVertexNormals[iDestCluster].begin(), aaVertexNormals[iDestCluster].end());
            aaNewVertexUVs[0].insert(aaNewVertexUVs[0].end(), aaVertexUVs[iDestCluster].begin(), aaVertexUVs[iDestCluster].end());

            aaiNewVertexPositionIndices[0].insert(aaiNewVertexPositionIndices[0].end(), aaiVertexPositionIndices[iDestCluster].begin(), aaiVertexPositionIndices[iDestCluster].end());
            aaiNewVertexNormalIndices[0].insert(aaiNewVertexNormalIndices[0].end(), aaiVertexNormalIndices[iDestCluster].begin(), aaiVertexNormalIndices[iDestCluster].end());
            aaiNewVertexUVIndices[0].insert(aaiNewVertexUVIndices[0].end(), aaiVertexUVIndices[iDestCluster].begin(), aaiVertexUVIndices[iDestCluster].end());
        }
        else
        {
            if(iDestCluster >= static_cast<uint32_t>(aaiVertexPositionIndices.size()) - 1)
            {
                aaVertexPositions.push_back(std::vector<float3>());
                aaVertexNormals.push_back(std::vector<float3>());
                aaVertexUVs.push_back(std::vector<float2>());
                aaiVertexPositionIndices.push_back(std::vector<uint32_t>());
                aaiVertexNormalIndices.push_back(std::vector<uint32_t>());
                aaiVertexUVIndices.push_back(std::vector<uint32_t>());

                iDestCluster = static_cast<uint32_t>(aaVertexPositions.size() - 1);

                // add first triangle
                aaNewVertexPositions[0].push_back(aaVertexPositions[iSrcCluster][aaiVertexPositionIndices[iSrcCluster][0]]);
                aaNewVertexPositions[0].push_back(aaVertexPositions[iSrcCluster][aaiVertexPositionIndices[iSrcCluster][1]]);
                aaNewVertexPositions[0].push_back(aaVertexPositions[iSrcCluster][aaiVertexPositionIndices[iSrcCluster][2]]);

                aaNewVertexNormals[0].push_back(aaVertexNormals[iSrcCluster][aaiVertexNormalIndices[iSrcCluster][0]]);
                aaNewVertexNormals[0].push_back(aaVertexNormals[iSrcCluster][aaiVertexNormalIndices[iSrcCluster][1]]);
                aaNewVertexNormals[0].push_back(aaVertexNormals[iSrcCluster][aaiVertexNormalIndices[iSrcCluster][2]]);

                aaNewVertexUVs[0].push_back(aaVertexUVs[iSrcCluster][aaiVertexUVIndices[iSrcCluster][0]]);
                aaNewVertexUVs[0].push_back(aaVertexUVs[iSrcCluster][aaiVertexUVIndices[iSrcCluster][1]]);
                aaNewVertexUVs[0].push_back(aaVertexUVs[iSrcCluster][aaiVertexUVIndices[iSrcCluster][2]]);

                aaiNewVertexPositionIndices[0].push_back(0);
                aaiNewVertexPositionIndices[0].push_back(1);
                aaiNewVertexPositionIndices[0].push_back(2);

                aaiNewVertexNormalIndices[0].push_back(0);
                aaiNewVertexNormalIndices[0].push_back(1);
                aaiNewVertexNormalIndices[0].push_back(2);

                aaiNewVertexUVIndices[0].push_back(0);
                aaiNewVertexUVIndices[0].push_back(1);
                aaiNewVertexUVIndices[0].push_back(2);
            }
        }

        memset(aiSharedEdge.data(), 0, aiSharedEdge.size() * sizeof(uint32_t));

        // check for adjacent edge and add to new cluster if found
        iNumAddedTriangleVertices = 0;
        bool bResetLoop = false;
        uint32_t iNumTrianglesToCheck = static_cast<uint32_t>(aaiNewVertexPositionIndices[0].size());
        for(uint32_t iTri = 0; iTri < iNumTrianglesToCheck; iTri += 3)
        {
            if(bResetLoop)
            {
                iTri = 0;
                bResetLoop = false;
            }

            if(iNumAddedTriangleVertices >= iFinishDestClusterSize)
            {
                break;
            }

            for(uint32_t iCheckTri = 0; iCheckTri < static_cast<uint32_t>(aaiVertexPositionIndices[iSrcCluster].size()); iCheckTri += 3)
            {
                if(iNumAddedTriangleVertices >= iFinishDestClusterSize)
                {
                    break;
                }

                if(aiSharedEdge[iCheckTri] == 1)
                {
                    continue;
                }

                float3 aSamePos[3];
                uint32_t iNumSamePos = 0;
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = aaiNewVertexPositionIndices[0][iTri + i];
                    float3 const& pos = aaNewVertexPositions[0][iPos];
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        uint32_t iCheckPos = aaiVertexPositionIndices[iSrcCluster][iCheckTri + j];
                        float3 const& checkPos = aaVertexPositions[iSrcCluster][iCheckPos];
                        if(lengthSquared(checkPos - pos) <= 1.0e-7f)
                        {
                            aSamePos[iNumSamePos++] = checkPos;
                        }
                    }
                }

                // shared an edge, add to cluster
                if(iNumSamePos >= 2)
                {
                    addTriangle(
                        aaNewVertexPositions,
                        aaNewVertexNormals,
                        aaNewVertexUVs,
                        aaiNewVertexPositionIndices,
                        aaiNewVertexNormalIndices,
                        aaiNewVertexUVIndices,
                        aaVertexPositions,
                        aaVertexNormals,
                        aaVertexUVs,
                        aaiVertexPositionIndices,
                        aaiVertexNormalIndices,
                        aaiVertexUVIndices,
                        iSrcCluster,
                        iCheckTri,
                        0);

                    aiSharedEdge[iCheckTri] = 1;
                    bResetLoop = true;

                    iNumTrianglesToCheck = static_cast<uint32_t>(aaiNewVertexPositionIndices[0].size());
                    iNumAddedTriangleVertices += 3;

                }   // if shared an edge

            }   // for check tri = 0 to num src position indices


        }   // for tri = 0 to num dest position indices

        // add the rest of the src cluster triangles
        if(iNumAddedTriangleVertices >= iFinishDestClusterSize)
        {
            for(uint32_t i = 0; i < static_cast<uint32_t>(aaiVertexPositionIndices[iSrcCluster].size()); i += 3)
            {
                if(aiSharedEdge[i] == 0)
                {
                    addTriangle(
                        aaNewVertexPositions,
                        aaNewVertexNormals,
                        aaNewVertexUVs,
                        aaiNewVertexPositionIndices,
                        aaiNewVertexNormalIndices,
                        aaiNewVertexUVIndices,
                        aaVertexPositions,
                        aaVertexNormals,
                        aaVertexUVs,
                        aaiVertexPositionIndices,
                        aaiVertexNormalIndices,
                        aaiVertexUVIndices,
                        iSrcCluster,
                        i,
                        1);
                }

            }   // add reset of the src cluster triangle
            
            break;
        }
        else
        {
            aaNewVertexPositions[0].clear();
            aaNewVertexNormals[0].clear();
            aaNewVertexUVs[0].clear();

            aaiNewVertexPositionIndices[0].clear();
            aaiNewVertexNormalIndices[0].clear();
            aaiNewVertexUVIndices[0].clear();

            iNumAddedTriangleVertices = 0;
        }

    }   // for dest cluster = 0 to num clusters

    // copy to destination cluster
    aaVertexPositions[iDestCluster] = aaNewVertexPositions[0];
    aaVertexNormals[iDestCluster] = aaNewVertexNormals[0];
    aaVertexUVs[iDestCluster] = aaNewVertexUVs[0];

    aaiVertexPositionIndices[iDestCluster] = aaiNewVertexPositionIndices[0];
    aaiVertexNormalIndices[iDestCluster] = aaiNewVertexNormalIndices[0];
    aaiVertexUVIndices[iDestCluster] = aaiNewVertexUVIndices[0];

    // copy to source cluster
    aaVertexPositions[iSrcCluster] = aaNewVertexPositions[1];
    aaVertexNormals[iSrcCluster] = aaNewVertexNormals[1];
    aaVertexUVs[iSrcCluster] = aaNewVertexUVs[1];

    aaiVertexPositionIndices[iSrcCluster] = aaiNewVertexPositionIndices[1];
    aaiVertexNormalIndices[iSrcCluster] = aaiNewVertexNormalIndices[1];
    aaiVertexUVIndices[iSrcCluster] = aaiNewVertexUVIndices[1];

    DEBUG_PRINTF("move %d triangle vertices from cluster %d to cluster %d\n", 
        iNumAddedTriangleVertices, 
        iSrcCluster, 
        iDestCluster);

    assert(aaiVertexPositionIndices.size() == aaiVertexNormalIndices.size());

    for(uint32_t i = 0; i < static_cast<uint32_t>(aaNewVertexPositions.size()); i++)
    {
        assert(aaiVertexPositionIndices[i].size() == aaiVertexNormalIndices[i].size());
        assert(aaiVertexPositionIndices[i].size() == aaiVertexUVIndices[i].size());
        
        std::ostringstream clusterName;
        clusterName << "cleaned-cluster" << i;

        std::ostringstream outputClusterFilePath;
        outputClusterFilePath << "c:\\Users\\Dingwings\\demo-models\\debug-output\\" << clusterName.str() << ".obj";

        writeOBJFile(
            aaNewVertexPositions[i],
            aaNewVertexNormals[i],
            aaNewVertexUVs[i],
            aaiNewVertexPositionIndices[i],
            aaiNewVertexNormalIndices[i],
            aaiNewVertexUVIndices[i],
            outputClusterFilePath.str(),
            clusterName.str());
    }

    return iNumAddedTriangleVertices > 0;
}

/*
**
*/
bool mergeTriangles(
    std::vector<std::vector<float3>>& aaVertexPositions,
    std::vector<std::vector<float3>>& aaVertexNormals,
    std::vector<std::vector<float2>>& aaVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiVertexPositionIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiVertexUVIndices,
    uint32_t iSrcCluster)
{
    DEBUG_PRINTF("try to merge cluster %d size: %d\n", iSrcCluster, aaiVertexPositionIndices.size());

    std::vector<uint32_t> aiMerged(aaiVertexPositionIndices[iSrcCluster].size() / 3);
    memset(aiMerged.data(), 0, aiMerged.size() * sizeof(uint32_t));

    std::vector<std::vector<float3>> aaVertexPositionsCopy = aaVertexPositions;
    std::vector<std::vector<float3>> aaVertexNormalsCopy = aaVertexNormals;
    std::vector<std::vector<float2>> aaVertexUVsCopy = aaVertexUVs;
    std::vector<std::vector<uint32_t>> aaiVertexPositionIndicesCopy = aaiVertexPositionIndices;
    std::vector<std::vector<uint32_t>> aaiVertexNormalIndicesCopy = aaiVertexNormalIndices;
    std::vector<std::vector<uint32_t>> aaiVertexUVIndicesCopy = aaiVertexUVIndices;

    uint32_t const kiMaxTriangleVertices = 128 * 3;
    uint32_t iDestCluster = 0;
    for(uint32_t iMergeTri = 0; iMergeTri < static_cast<uint32_t>(aaiVertexPositionIndices[iSrcCluster].size()); iMergeTri += 3)
    {
        for(iDestCluster = 0; iDestCluster < static_cast<uint32_t>(aaiVertexPositionIndices.size()); iDestCluster++)
        {
            if(iDestCluster == iSrcCluster)
            {
                continue;
            }

            if(aaiVertexPositionIndicesCopy[iDestCluster].size() <= kiMaxTriangleVertices - aaiVertexPositionIndices[iSrcCluster].size())
            {
                for(uint32_t iCheckTri = 0; iCheckTri < static_cast<uint32_t>(aaiVertexPositionIndicesCopy[iDestCluster].size()); iCheckTri += 3)
                {
                    float3 aSamePos[3];
                    uint32_t iNumSamePos = 0;
                    for(uint32_t i = 0; i < 3; i++)
                    {
                        uint32_t iPos = aaiVertexPositionIndices[iSrcCluster][iMergeTri + i];
                        float3 const& pos = aaVertexPositions[iSrcCluster][iPos];
                        for(uint32_t j = 0; j < 3; j++)
                        {
                            uint32_t iCheckPos = aaiVertexPositionIndicesCopy[iDestCluster][iCheckTri + j];
                            float3 const& checkPos = aaVertexPositionsCopy[iDestCluster][iCheckPos];
                            if(lengthSquared(checkPos - pos) <= 1.0e-9f)
                            {
                                aSamePos[iNumSamePos++] = checkPos;
                            }
                        }
                    }

                    float3 aAddedPos[3];
                    uint32_t aiAddedIndices[3];
                    if(iNumSamePos >= 2)
                    {
                        for(uint32_t i = 0; i < 3; i++)
                        {
                            // position
                            {
                                uint32_t iIndex = UINT32_MAX;
                                uint32_t iPos = aaiVertexPositionIndices[iSrcCluster][iMergeTri + i];
                                float3 const& pos = aaVertexPositions[iSrcCluster][iPos];
                                auto iter = std::find_if(
                                    aaVertexPositionsCopy[iDestCluster].begin(),
                                    aaVertexPositionsCopy[iDestCluster].end(),
                                    [pos](float3 const& checkPos)
                                    {
                                        return lengthSquared(pos - checkPos) <= 1.0e-7f;
                                    });
                                if(iter == aaVertexPositionsCopy[iDestCluster].end())
                                {
                                    aaVertexPositionsCopy[iDestCluster].push_back(pos);
                                    aAddedPos[i] = pos;
                                    iIndex = static_cast<uint32_t>(aaVertexPositionsCopy[iDestCluster].size() - 1);
                                }
                                else
                                {
                                    iIndex = static_cast<uint32_t>(std::distance(aaVertexPositionsCopy[iDestCluster].begin(), iter));
                                    aAddedPos[i] = aaVertexPositionsCopy[iDestCluster][iIndex];
                                }
                                assert(iIndex != UINT32_MAX);
                                assert(iIndex <= aaVertexPositionsCopy[iDestCluster].size());
                                aaiVertexPositionIndicesCopy[iDestCluster].push_back(iIndex);
                                aiAddedIndices[i] = iIndex;
                            }

                            // normal
                            {
                                uint32_t iIndex = UINT32_MAX;
                                uint32_t iPos = aaiVertexNormalIndices[iSrcCluster][iMergeTri + i];
                                float3 const& pos = aaVertexNormals[iSrcCluster][iPos];
                                auto iter = std::find_if(
                                    aaVertexNormalsCopy[iDestCluster].begin(),
                                    aaVertexNormalsCopy[iDestCluster].end(),
                                    [pos](float3 const& checkPos)
                                    {
                                        return lengthSquared(pos - checkPos) <= 1.0e-7f;
                                    });
                                if(iter == aaVertexNormalsCopy[iDestCluster].end())
                                {
                                    aaVertexNormalsCopy[iDestCluster].push_back(pos);
                                    iIndex = static_cast<uint32_t>(aaVertexNormalsCopy[iDestCluster].size() - 1);
                                }
                                else
                                {
                                    iIndex = static_cast<uint32_t>(std::distance(aaVertexNormalsCopy[iDestCluster].begin(), iter));
                                }
                                assert(iIndex != UINT32_MAX);
                                assert(iIndex <= aaVertexNormalsCopy[iDestCluster].size());
                                aaiVertexNormalIndicesCopy[iDestCluster].push_back(iIndex);
                            }

                            // uv
                            {
                                uint32_t iIndex = UINT32_MAX;
                                uint32_t iPos = aaiVertexUVIndices[iSrcCluster][iMergeTri + i];
                                float2 const& pos = aaVertexUVs[iSrcCluster][iPos];
                                auto iter = std::find_if(
                                    aaVertexUVsCopy[iDestCluster].begin(),
                                    aaVertexUVsCopy[iDestCluster].end(),
                                    [pos](float2 const& checkPos)
                                    {
                                        return lengthSquared(pos - checkPos) <= 1.0e-7f;
                                    });
                                if(iter == aaVertexUVsCopy[iDestCluster].end())
                                {
                                    aaVertexUVsCopy[iDestCluster].push_back(pos);
                                    iIndex = static_cast<uint32_t>(aaVertexUVsCopy[iDestCluster].size() - 1);
                                }
                                else
                                {
                                    iIndex = static_cast<uint32_t>(std::distance(aaVertexUVsCopy[iDestCluster].begin(), iter));
                                }
                                assert(iIndex != UINT32_MAX);
                                assert(iIndex <= aaVertexUVsCopy[iDestCluster].size());
                                aaiVertexUVIndicesCopy[iDestCluster].push_back(iIndex);

                            }

                        }   // for i = 0 to 3

                        DEBUG_PRINTF("merge triangleID %d from cluster %d to cluster %d triangle (%d, %d, %d)\n",
                            iMergeTri / 3,
                            iSrcCluster,
                            iDestCluster,
                            aiAddedIndices[0],
                            aiAddedIndices[1], 
                            aiAddedIndices[2]);

                        aiMerged[iMergeTri / 3] = 1;
                        break;

                    }   // if share edge

                }   // for check tri = 0 to num dest triangles

                if(aiMerged[iMergeTri / 3])
                {
                    break;
                }

            }   // if dest cluster can hold the triangles

        }   // for dest cluster to num clusters
    
        if(aiMerged[iMergeTri / 3] == 0)
        {
            DEBUG_PRINTF("did not merge triangleID %d\n", iMergeTri / 3);
        }

    }   // for tri = 0 to num src triangles

    auto iter = std::find(aiMerged.begin(), aiMerged.end(), 0);
    if(iter == aiMerged.end())
    {
        aaVertexPositionsCopy.erase(aaVertexPositionsCopy.begin() + iSrcCluster);
        aaVertexNormalsCopy.erase(aaVertexNormalsCopy.begin() + iSrcCluster);
        aaVertexUVsCopy.erase(aaVertexUVsCopy.begin() + iSrcCluster);

        aaiVertexPositionIndicesCopy.erase(aaiVertexPositionIndicesCopy.begin() + iSrcCluster);
        aaiVertexNormalIndicesCopy.erase(aaiVertexNormalIndicesCopy.begin() + iSrcCluster);
        aaiVertexUVIndicesCopy.erase(aaiVertexUVIndicesCopy.begin() + iSrcCluster);

        aaVertexPositions = aaVertexPositionsCopy;
        aaVertexNormals = aaVertexNormalsCopy;
        aaVertexUVs = aaVertexUVsCopy;

        aaiVertexPositionIndices = aaiVertexPositionIndicesCopy;
        aaiVertexNormalIndices = aaiVertexNormalIndicesCopy;
        aaiVertexUVIndices = aaiVertexUVIndicesCopy;

        DEBUG_PRINTF("!!! successfully merged cluster %d\n", iSrcCluster);
    }
    else
    {
        DEBUG_PRINTF("!!! did not merge cluster %d\n", iSrcCluster);
    }

    return (iter == aiMerged.end());
}