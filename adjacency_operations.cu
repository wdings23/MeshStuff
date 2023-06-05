#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "float3_lib.cuh"

#include "LogPrint.h"

#define NUM_WORKGROUPS      64

#define uint32_t unsigned int
#define int32_t int

__global__
void getNumAdjacentClusters(
    uint32_t* paiRetNumAdjacentClusters,
    float3 const* paaVertexPositions,
    uint32_t const* paiBoundaryVertexIndices, 
    uint32_t const* paiNumBoundaryVertexIndices,
    uint32_t const* paiClusterFloat3Offsets,
    uint32_t iNumClusters)
{
    uint32_t iCluster = blockIdx.x * NUM_WORKGROUPS + threadIdx.x;
    if(iCluster >= iNumClusters)
    {
        return;
    }

    uint32_t iBoundaryVertexOffset = 0;
    for(uint32_t i = 0; i < iCluster; i++)
    {
        iBoundaryVertexOffset += paiNumBoundaryVertexIndices[i];
    }

    uint32_t iFloat3Offset = paiClusterFloat3Offsets[iCluster];
    uint32_t iNumBoundaryVertexIndices = paiNumBoundaryVertexIndices[iCluster];
    float3 const* aVertexPositions = paaVertexPositions + iFloat3Offset;
    uint32_t const* aiBoundaryVertexIndices = paiBoundaryVertexIndices + iBoundaryVertexOffset;

    //printf("cluster %d iNumBoundaryVertexIndices = %d iBoundaryVertexOffset = %d\n", 
    //    iCluster, 
    //    iNumBoundaryVertexIndices,
    //    iBoundaryVertexOffset);

    for(uint32_t i = 0; i < iNumBoundaryVertexIndices; i++)
    {
        uint32_t iVertexIndex = aiBoundaryVertexIndices[i];
        float3 const& vertexPosition = aVertexPositions[iVertexIndex];
        //if(iCluster == 1)
        //{
        //    printf("cluster1 %d (%.4f, %.4f, %.4f)\n",
        //        iVertexIndex,
        //        vertexPosition.x,
        //        vertexPosition.y,
        //        vertexPosition.z);
        //}

        for(uint32_t iCheckCluster = iCluster + 1; iCheckCluster < iNumClusters; iCheckCluster++)
        {
            if(iCheckCluster == iCluster)
            {
                continue;
            }

            uint32_t iCheckBoundaryVertexOffset = 0;
            for(uint32_t j = 0; j < iCheckCluster; j++)
            {
                iCheckBoundaryVertexOffset += paiNumBoundaryVertexIndices[j];
            }

            uint32_t iCheckFloat3Offset = paiClusterFloat3Offsets[iCheckCluster];
            uint32_t iNumCheckBoundaryVertexIndices = paiNumBoundaryVertexIndices[iCheckCluster];
            float3 const* aCheckVertexPositions = paaVertexPositions + iCheckFloat3Offset;
            uint32_t const* aiCheckBoundaryVertexIndices = paiBoundaryVertexIndices + iCheckBoundaryVertexOffset;

            for(uint32_t iCheckBoundaryVertex = 0; iCheckBoundaryVertex < iNumCheckBoundaryVertexIndices; iCheckBoundaryVertex++)
            {
                uint32_t iCheckVertexIndex = aiCheckBoundaryVertexIndices[iCheckBoundaryVertex];
                float3 const& checkVertexPosition = aCheckVertexPositions[iCheckVertexIndex];

                float fLength = lengthSquared(vertexPosition - checkVertexPosition);
                if(fLength <= 1.0e-8f)
                {
                    uint32_t iIndex = iCluster * iNumClusters + iCheckCluster;
                    uint32_t iCheckIndex = iCheckCluster * iNumClusters + iCluster;
                    paiRetNumAdjacentClusters[iIndex] += 1;
                    paiRetNumAdjacentClusters[iCheckIndex] += 1;
                    break;
                }
            }
            
        }   // for check cluster = 0 to num clusters
    
    }   // for i = 0 to num boundary vertex indices   

    if(iCluster % 100 == 0)
    {
        printf("cluster %d (%d) done\n", 
            iCluster,
            iNumClusters);
    }
}

#undef uint32_t
#undef int32_t

#include "adjacency_operations_cuda.h"

/*
**
*/
void buildClusterEdgeAdjacencyCUDA3(
    std::vector<std::vector<uint32_t>>& aaiNumAdjacentClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiBoundaryVertexIndices)
{
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());

    // boundary vertex indices
    uint32_t* paiBoundaryVertexIndices = nullptr;
    uint32_t iNumTotalBoundaryVertexIndices = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        iNumTotalBoundaryVertexIndices += static_cast<uint32_t>(aaiBoundaryVertexIndices[iCluster].size());
    }
    cudaMalloc(
        &paiBoundaryVertexIndices,
        iNumTotalBoundaryVertexIndices * sizeof(uint32_t));
    iNumTotalBoundaryVertexIndices = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            paiBoundaryVertexIndices + iNumTotalBoundaryVertexIndices,
            aaiBoundaryVertexIndices[iCluster].data(),
            aaiBoundaryVertexIndices[iCluster].size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
        iNumTotalBoundaryVertexIndices += static_cast<uint32_t>(aaiBoundaryVertexIndices[iCluster].size());
    }
    
    // num boundary vertex indices
    uint32_t* paiNumBoundaryVertexIndices = nullptr;
    cudaMalloc(
        &paiNumBoundaryVertexIndices,
        iNumClusters * sizeof(uint32_t));
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        uint32_t iNumBoundaryVertices = static_cast<uint32_t>(aaiBoundaryVertexIndices[iCluster].size());
        cudaMemcpy(
            paiNumBoundaryVertexIndices + iCluster,
            &iNumBoundaryVertices,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    }

    // vertex positions
    uint32_t iNumTotalVertexPositions = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        iNumTotalVertexPositions += static_cast<uint32_t>(aaVertexPositions[iCluster].size());
    }
    float3* paaVertexPositions = nullptr;
    cudaMalloc(
        &paaVertexPositions,
        iNumTotalVertexPositions * sizeof(float3));
    iNumTotalVertexPositions = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            paaVertexPositions + iNumTotalVertexPositions,
            aaVertexPositions[iCluster].data(),
            aaVertexPositions[iCluster].size() * sizeof(float3),
            cudaMemcpyHostToDevice);
        iNumTotalVertexPositions += static_cast<uint32_t>(aaVertexPositions[iCluster].size());
    }

    // num cluster vertex positions
    uint32_t* paiRetNumAdjacentClusters = nullptr;
    cudaMalloc(
        &paiRetNumAdjacentClusters,
        iNumClusters * iNumClusters * sizeof(uint32_t));

    // cluster array byte offset
    uint32_t* paiClusterFloat3Offsets = nullptr;
    cudaMalloc(
        &paiClusterFloat3Offsets,
        iNumClusters * sizeof(uint32_t));
    uint32_t iOffsetFloat3 = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            paiClusterFloat3Offsets + iCluster,
            &iOffsetFloat3,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);
        iOffsetFloat3 += static_cast<uint32_t>(aaVertexPositions[iCluster].size());
    }

    //uint32_t const kiTestCluster = 1;
    //for(uint32_t i = 0; i < aaiBoundaryVertexIndices[kiTestCluster].size(); i++)
    //{
    //    uint32_t iBoundaryVertexIndex = aaiBoundaryVertexIndices[kiTestCluster][i];
    //    vec3 const& vertexPosition = aaVertexPositions[kiTestCluster][iBoundaryVertexIndex];
    //    DEBUG_PRINTF("cluster%d %d (%.4f, %.4f, %.4f)\n",
    //        kiTestCluster,
    //        iBoundaryVertexIndex,
    //        vertexPosition.x,
    //        vertexPosition.y,
    //        vertexPosition.z);
    //}

    cudaMemset(
        &paiRetNumAdjacentClusters,
        0,
        iNumClusters * iNumClusters * sizeof(uint32_t));

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumClusters) / float(NUM_WORKGROUPS)));
    getNumAdjacentClusters<<<iNumBlocks, NUM_WORKGROUPS>>>(
        paiRetNumAdjacentClusters,
        paaVertexPositions,
        paiBoundaryVertexIndices,
        paiNumBoundaryVertexIndices,
        paiClusterFloat3Offsets,
        iNumClusters);

    aaiNumAdjacentClusters.resize(iNumClusters);
    for(uint32_t i = 0; i < iNumClusters; i++)
    {
        aaiNumAdjacentClusters[i].resize(iNumClusters);
        cudaMemcpy(
            aaiNumAdjacentClusters[i].data(),
            paiRetNumAdjacentClusters + iNumClusters * i,
            iNumClusters * sizeof(uint32_t),
            cudaMemcpyDeviceToHost);
    }

    cudaFree(paiBoundaryVertexIndices);
    cudaFree(paiNumBoundaryVertexIndices);
    cudaFree(paaVertexPositions);
    cudaFree(paiRetNumAdjacentClusters);
    cudaFree(paiClusterFloat3Offsets);
    
}