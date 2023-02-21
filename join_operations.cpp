#include "join_operations.h"

#include <assert.h>

/*
**
*/
bool canJoinClusters(
    std::vector<float3> const& aVertexPositions0,
    std::vector<float3> const& aVertexPositions1,
    std::vector<uint32_t> const& aiTrianglePositionIndices0,
    std::vector<uint32_t> const& aiTrianglePositionIndices1)
{
    float const kiEqualityThreshold = 1.0e-7f;

    float3 aSamePos0[3];
    float3 aSamePos1[3];
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTrianglePositionIndices0.size()); iTri += 3)
    {
        for(uint32_t iCheckTri = 0; iCheckTri < static_cast<uint32_t>(aiTrianglePositionIndices1.size()); iCheckTri += 3)
        {
            uint32_t iNumSameIndex = 0;
            for(uint32_t i = 0; i < 3; i++)
            {
                float3 const& pos0 = aVertexPositions0[aiTrianglePositionIndices0[iTri + i]];
                for(uint32_t j = 0; j < 3; j++)
                {
                    float3 const& pos1 = aVertexPositions1[aiTrianglePositionIndices1[iCheckTri + j]];
                    if(lengthSquared(pos1 - pos0) <= kiEqualityThreshold)
                    {
                        if(iNumSameIndex < 3)
                        {
                            aSamePos0[iNumSameIndex] = pos0;
                            aSamePos1[iNumSameIndex] = pos1;
                        }
                        ++iNumSameIndex;
                    }
                }
            }

            if(iNumSameIndex >= 2)
            {
                return true;
            }

        }   // for check tri = 0 to num triangles 1

    }   // for tri = 0 to num triangles 0 

    return false;
}

/*
**
*/
void joinSharedClusters(
    std::vector<float3>& aTotalVertexPositions,
    std::vector<float3>& aTotalVertexNormals,
    std::vector<float2>& aTotalVertexUVs,
    std::vector<uint32_t>& aiTotalTrianglePositionIndices,
    std::vector<uint32_t>& aiTotalTriangleNormalIndices,
    std::vector<uint32_t>& aiTotalTriangleUVIndices,
    std::vector<float3> const& aVertexPositions0,
    std::vector<float3> const& aVertexPositions1,
    std::vector<float3> const& aVertexNormals0,
    std::vector<float3> const& aVertexNormals1,
    std::vector<float2> const& aVertexUVs0,
    std::vector<float2> const& aVertexUVs1,
    std::vector<uint32_t> const& aiTrianglePositionIndices0,
    std::vector<uint32_t> const& aiTrianglePositionIndices1,
    std::vector<uint32_t> const& aiTriangleNormalIndices0,
    std::vector<uint32_t> const& aiTriangleNormalIndices1,
    std::vector<uint32_t> const& aiTriangleUVIndices0,
    std::vector<uint32_t> const& aiTriangleUVIndices1)
{
    static float const kfEqualityThreshold = 1.0e-7f;

    // check if vertex positions exists in the total vertex position list, push into the total vertex position list if not
    std::vector<float3> const* paVertexPositions[2] = { &aVertexPositions0 , &aVertexPositions1 };
    for(uint32_t i = 0; i < 2; i++)
    {
        for(auto const& vertexPosition : *paVertexPositions[i])
        {
            auto iter = std::find_if(
                aTotalVertexPositions.begin(),
                aTotalVertexPositions.end(),
                [vertexPosition](float3 const& checkVertexPosition)
                {
                    return (lengthSquared(vertexPosition - checkVertexPosition) <= kfEqualityThreshold);
                }
            );

            if(iter == aTotalVertexPositions.end())
            {
                aTotalVertexPositions.push_back(vertexPosition);
            }
        };
    }

    // normals
    std::vector<float3> const* paVertexNormals[2] = { &aVertexNormals0 , &aVertexNormals1 };
    for(uint32_t i = 0; i < 2; i++)
    {
        for(auto const& vertexNormal : *paVertexNormals[i])
        {
            auto iter = std::find_if(
                aTotalVertexNormals.begin(),
                aTotalVertexNormals.end(),
                [vertexNormal](float3 const& checkVertexNormal)
                {
                    return (lengthSquared(vertexNormal - checkVertexNormal) <= kfEqualityThreshold);
                }
            );

            if(iter == aTotalVertexNormals.end())
            {
                aTotalVertexNormals.push_back(vertexNormal);
            }
        };
    }

    // uvs
    std::vector<float2> const* paVertexUVs[2] = { &aVertexUVs0 , &aVertexUVs1 };
    for(uint32_t i = 0; i < 2; i++)
    {
        for(auto const& vertexUV : *paVertexUVs[i])
        {
            auto iter = std::find_if(
                aTotalVertexUVs.begin(),
                aTotalVertexUVs.end(),
                [vertexUV](float2 const& checkVertexUV)
                {
                    return (lengthSquared(vertexUV - checkVertexUV) <= kfEqualityThreshold);
                }
            );

            if(iter == aTotalVertexUVs.end())
            {
                aTotalVertexUVs.push_back(vertexUV);
            }
        };
    }

    // look for matching vertex position from old list to new total vertex position list and add the index
    std::vector<uint32_t> const* paiTrianglePositionIndices[2] = { &aiTrianglePositionIndices0, &aiTrianglePositionIndices1 };
    for(uint32_t i = 0; i < 2; i++)
    {
        std::vector<float3> const* pVertexPositions = paVertexPositions[i];
        for(auto const& iTrianglePositionIndex : *paiTrianglePositionIndices[i])
        {
            auto iter = std::find_if(
                aTotalVertexPositions.begin(),
                aTotalVertexPositions.end(),
                [iTrianglePositionIndex,
                pVertexPositions](float3 const& checkVertexPosition)
                {
                    return (lengthSquared((*pVertexPositions)[iTrianglePositionIndex] - checkVertexPosition) <= kfEqualityThreshold);
                }
            );
            assert(iter != aTotalVertexPositions.end());
            uint32_t iIndex = static_cast<uint32_t>(std::distance(aTotalVertexPositions.begin(), iter));
            aiTotalTrianglePositionIndices.push_back(iIndex);
        };
    }

    // normal
    std::vector<uint32_t> const* paiTriangleNormalIndices[2] = { &aiTriangleNormalIndices0, &aiTriangleNormalIndices1 };
    for(uint32_t i = 0; i < 2; i++)
    {
        std::vector<float3> const* pVertexNormals = paVertexNormals[i];
        for(auto const& iTriangleNormalIndex : *paiTriangleNormalIndices[i])
        {
            auto iter = std::find_if(
                aTotalVertexNormals.begin(),
                aTotalVertexNormals.end(),
                [iTriangleNormalIndex,
                pVertexNormals](float3 const& checkVertexNormal)
                {
                    return (lengthSquared((*pVertexNormals)[iTriangleNormalIndex] - checkVertexNormal) <= kfEqualityThreshold);
                }
            );
            assert(iter != aTotalVertexNormals.end());
            uint32_t iIndex = static_cast<uint32_t>(std::distance(aTotalVertexNormals.begin(), iter));
            aiTotalTriangleNormalIndices.push_back(iIndex);
        };
    }
    
    // uv
    std::vector<uint32_t> const* paiTriangleUVIndices[2] = { &aiTriangleUVIndices0, &aiTriangleUVIndices1 };
    for(uint32_t i = 0; i < 2; i++)
    {
        std::vector<float2> const* pVertexUVs = paVertexUVs[i];
        for(auto const& iTriangleUVIndex : *paiTriangleUVIndices[i])
        {
            auto iter = std::find_if(
                aTotalVertexUVs.begin(),
                aTotalVertexUVs.end(),
                [iTriangleUVIndex,
                pVertexUVs](float2 const& checkVertexUV)
                {
                    return (lengthSquared((*pVertexUVs)[iTriangleUVIndex] - checkVertexUV) <= kfEqualityThreshold);
                }
            );
            assert(iter != aTotalVertexUVs.end());
            uint32_t iIndex = static_cast<uint32_t>(std::distance(aTotalVertexUVs.begin(), iter));
            aiTotalTriangleUVIndices.push_back(iIndex);
        };
    }

    assert(aiTotalTrianglePositionIndices.size() == aiTotalTriangleNormalIndices.size());
    assert(aiTotalTrianglePositionIndices.size() == aiTotalTriangleUVIndices.size());
}