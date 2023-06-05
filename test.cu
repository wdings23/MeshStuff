#include <cuda_runtime.h>
#include <algorithm>

#define uint32_t unsigned int
#define int32_t int

#define MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP       3000

#define WORKGROUP_SIZE 64

/*
**
*/
__global__
void getSamePositions(
    float* pOutput,
    float* pData0,
    float* pData1,
    int iNumElements0,
    int iNumElements1)
{
    float fRet = 0.0f;
    for(int i = 0; i < iNumElements0; i++)
    {
        for(int j = 0; j < iNumElements1; j++)
        {
            if(pData0[i] == pData1[j])
            {
                fRet += 1.0f;
            }
        }
    }
    *pOutput = fRet;
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
__global__
void checkClusterGroupBoundaryVertices(
    uint32_t* aiRetClusterGroupBoundaryVertices,
    uint32_t* aiRetNumClusterGroupBoundaryVertices,
    uint32_t iNumClusterGroups,
    float* aaClusterGroupVertexPositions,
    uint32_t* aaiClusterClusterTrianglePositionIndices,
    uint32_t* aiNumClusterGroupVertexPositions,
    uint32_t* aiNumClusterGroupTrianglePositionIndices,
    uint32_t* aiClusterGroupVertexPositionOffsets,
    uint32_t* aiClusterGroupTriangleIndexOffsets)
{
    int iClusterGroup = blockIdx.x * 256 + threadIdx.x;
    if(iClusterGroup >= iNumClusterGroups)
    {
        return;
    }

    uint32_t iNumBoundaryVertices = 0;
    float* paClusterGroupTriangleVertexPositions = &aaClusterGroupVertexPositions[aiClusterGroupVertexPositionOffsets[iClusterGroup]];
    uint32_t* paiClusterGroupTriangleIndices = &aaiClusterClusterTrianglePositionIndices[aiClusterGroupTriangleIndexOffsets[iClusterGroup]];
    uint32_t iNumClusterGroupTriangleIndices = aiNumClusterGroupTrianglePositionIndices[iClusterGroup];

    for(uint32_t iTri = 0; iTri < iNumClusterGroupTriangleIndices; iTri += 3)
    {
        uint32_t aiAdjacentEdges[3] = { 0, 0, 0 };

        for(uint32_t iCheckClusterGroup = 0; iCheckClusterGroup < iNumClusterGroups; iCheckClusterGroup++)
        {
            uint32_t iNumCheckClusterGroupTriangleIndices = aiNumClusterGroupTrianglePositionIndices[iCheckClusterGroup];
            float* paCheckClusterGroupTriangleVertexPositions = &aaClusterGroupVertexPositions[aiClusterGroupVertexPositionOffsets[iCheckClusterGroup]];
            uint32_t* paiCheckClusterGroupTriangleIndices = &aaiClusterClusterTrianglePositionIndices[aiClusterGroupTriangleIndexOffsets[iCheckClusterGroup]];
            
            for(uint32_t iCheckTri = 0; iCheckTri < iNumCheckClusterGroupTriangleIndices; iCheckTri += 3)
            {
                if(iClusterGroup == iCheckClusterGroup && iTri == iCheckTri)
                {
                    continue;
                }

                // check the number of same vertex positions
                uint32_t aiSamePositionIndices[3];
                aiSamePositionIndices[0] = 0; aiSamePositionIndices[1] = 0; aiSamePositionIndices[2] = 0;
                uint32_t iNumSamePositions = 0;
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = paiClusterGroupTriangleIndices[iTri + i];
                    float fX = paClusterGroupTriangleVertexPositions[iPos * 3];
                    float fY = paClusterGroupTriangleVertexPositions[iPos * 3 + 1];
                    float fZ = paClusterGroupTriangleVertexPositions[iPos * 3 + 2];

                    for(uint32_t j = 0; j < 3; j++)
                    {
                        uint32_t iCheckPos = paiCheckClusterGroupTriangleIndices[iCheckTri + j];
                        float fCheckX = paCheckClusterGroupTriangleVertexPositions[iCheckPos * 3];
                        float fCheckY = paCheckClusterGroupTriangleVertexPositions[iCheckPos * 3 + 1];
                        float fCheckZ = paCheckClusterGroupTriangleVertexPositions[iCheckPos * 3 + 2];
                        
                        float fDiffX = fCheckX - fX;
                        float fDiffY = fCheckY - fY;
                        float fDiffZ = fCheckZ - fZ;

                        if((fDiffX * fDiffX + fDiffY * fDiffY + fDiffZ * fDiffZ) < 1.0e-6f)
                        {
                            aiSamePositionIndices[i] = 1;
                            ++iNumSamePositions;
                            break;
                        }
                    }
                }   // for i = 0 to 3, checking same position

                if(iNumSamePositions >= 2)
                {
                    // edge index based on the same vertex positions
                    uint32_t iEdge = 0xffffffff;
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

                    if(iEdge <= 2)
                    {
                        aiAdjacentEdges[iEdge] = 1;
                    }
                }

            }   // for check tri = 0 to num triangles
        
        }   // for check cluster group = 0 to num check cluster groups
    
        if(aiAdjacentEdges[0] == 0)
        {
            uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumBoundaryVertices;
            aiRetClusterGroupBoundaryVertices[iIndex] = paiClusterGroupTriangleIndices[iTri];
            aiRetClusterGroupBoundaryVertices[iIndex + 1] = paiClusterGroupTriangleIndices[iTri+1];
            iNumBoundaryVertices += 2;
        }
        else if(aiAdjacentEdges[1] == 0)
        {
            uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumBoundaryVertices;
            aiRetClusterGroupBoundaryVertices[iIndex] = paiClusterGroupTriangleIndices[iTri];
            aiRetClusterGroupBoundaryVertices[iIndex + 1] = paiClusterGroupTriangleIndices[iTri + 2];
            iNumBoundaryVertices += 2;
        }
        else if(aiAdjacentEdges[2] == 0)
        {
            uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumBoundaryVertices;
            aiRetClusterGroupBoundaryVertices[iIndex] = paiClusterGroupTriangleIndices[iTri + 1];
            aiRetClusterGroupBoundaryVertices[iIndex + 1] = paiClusterGroupTriangleIndices[iTri + 2];
            iNumBoundaryVertices += 2;
        }

    }   // for tri = 0 to num triangles

    aiRetNumClusterGroupBoundaryVertices[iClusterGroup] = iNumBoundaryVertices;
}


/*
**
*/
__device__
void getTriangleIndexFromThreadIndex(
    uint32_t* piRetClusterGroup,
    uint32_t* piRetTri,
    uint32_t iThreadIndex,
    uint32_t iNumClusterGroups,
    uint32_t* aiNumClusterGroupVertexPositions)
{
    uint32_t iClusterGroup = 0;
    uint32_t iTotalTris = 0;
    for(iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        if(iTotalTris + aiNumClusterGroupVertexPositions[iClusterGroup] > iThreadIndex)
        {
            break;
        }

        iTotalTris += aiNumClusterGroupVertexPositions[iClusterGroup];
    }

    *piRetClusterGroup = iClusterGroup;
    *piRetTri = (iThreadIndex >= aiNumClusterGroupVertexPositions[0]) ? iThreadIndex - iTotalTris : iThreadIndex;
}

/*
**
*/
__global__
void checkClusterGroupBoundaryVertices2(
    uint32_t* aiRetClusterGroupBoundaryVertices,
    uint32_t* aiRetNumClusterGroupBoundaryVertices,
    uint32_t iNumClusterGroups,
    float* aaClusterGroupVertexPositions,
    uint32_t* aaiClusterClusterTrianglePositionIndices,
    uint32_t* aiNumClusterGroupVertexPositions,
    uint32_t* aiNumClusterGroupTrianglePositionIndices,
    uint32_t* aiClusterGroupVertexPositionOffsets,
    uint32_t* aiClusterGroupTriangleIndexOffsets,
    uint32_t iNumTotalTriangleIndices)
{
    uint32_t iClusterGroup = 0;
    uint32_t iTri = 0;
    uint32_t iThreadIndex = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    getTriangleIndexFromThreadIndex(
        &iClusterGroup,
        &iTri,
        iThreadIndex * 3,
        iNumClusterGroups,
        aiNumClusterGroupTrianglePositionIndices);

    if(iTri >= iNumTotalTriangleIndices)
    {
        return;
    }

    //uint32_t iNumBoundaryVertices = 0;
    float* paClusterGroupTriangleVertexPositions = &aaClusterGroupVertexPositions[aiClusterGroupVertexPositionOffsets[iClusterGroup]];
    uint32_t* paiClusterGroupTriangleIndices = &aaiClusterClusterTrianglePositionIndices[aiClusterGroupTriangleIndexOffsets[iClusterGroup]];
    //uint32_t iNumClusterGroupTriangleIndices = aiNumClusterGroupTrianglePositionIndices[iClusterGroup];

    //for(uint32_t iTri = 0; iTri < iNumClusterGroupTriangleIndices; iTri += 3)
    {
        uint32_t aiAdjacentEdges[3] = { 0, 0, 0 };

        for(uint32_t iCheckClusterGroup = 0; iCheckClusterGroup < iNumClusterGroups; iCheckClusterGroup++)
        {
            uint32_t iNumCheckClusterGroupTriangleIndices = aiNumClusterGroupTrianglePositionIndices[iCheckClusterGroup];
            float* paCheckClusterGroupTriangleVertexPositions = &aaClusterGroupVertexPositions[aiClusterGroupVertexPositionOffsets[iCheckClusterGroup]];
            uint32_t* paiCheckClusterGroupTriangleIndices = &aaiClusterClusterTrianglePositionIndices[aiClusterGroupTriangleIndexOffsets[iCheckClusterGroup]];

            for(uint32_t iCheckTri = 0; iCheckTri < iNumCheckClusterGroupTriangleIndices; iCheckTri += 3)
            {
                if(iClusterGroup == iCheckClusterGroup && iTri == iCheckTri)
                {
                    continue;
                }

                // check the number of same vertex positions
                uint32_t aiSamePositionIndices[3];
                aiSamePositionIndices[0] = 0; aiSamePositionIndices[1] = 0; aiSamePositionIndices[2] = 0;
                uint32_t iNumSamePositions = 0;
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = paiClusterGroupTriangleIndices[iTri + i];
                    float fX = paClusterGroupTriangleVertexPositions[iPos * 3];
                    float fY = paClusterGroupTriangleVertexPositions[iPos * 3 + 1];
                    float fZ = paClusterGroupTriangleVertexPositions[iPos * 3 + 2];

                    for(uint32_t j = 0; j < 3; j++)
                    {
                        uint32_t iCheckPos = paiCheckClusterGroupTriangleIndices[iCheckTri + j];
                        float fCheckX = paCheckClusterGroupTriangleVertexPositions[iCheckPos * 3];
                        float fCheckY = paCheckClusterGroupTriangleVertexPositions[iCheckPos * 3 + 1];
                        float fCheckZ = paCheckClusterGroupTriangleVertexPositions[iCheckPos * 3 + 2];

                        float fDiffX = fCheckX - fX;
                        float fDiffY = fCheckY - fY;
                        float fDiffZ = fCheckZ - fZ;

                        if((fDiffX * fDiffX + fDiffY * fDiffY + fDiffZ * fDiffZ) < 1.0e-6f)
                        {
                            aiSamePositionIndices[i] = 1;
                            ++iNumSamePositions;
                            break;
                        }
                    }
                }   // for i = 0 to 3, checking same position

                if(iNumSamePositions >= 2)
                {
                    // edge index based on the same vertex positions
                    uint32_t iEdge = 0xffffffff;
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

                    if(iEdge <= 2)
                    {
                        aiAdjacentEdges[iEdge] = 1;
                    }
                }

            }   // for check tri = 0 to num triangles

        }   // for check cluster group = 0 to num check cluster groups

        if(aiAdjacentEdges[0] == 0)
        {
            uint32_t iNumBoundaryVertices = atomicAdd(&aiRetNumClusterGroupBoundaryVertices[iClusterGroup], 2);
            uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumBoundaryVertices;
            aiRetClusterGroupBoundaryVertices[iIndex] = paiClusterGroupTriangleIndices[iTri];
            aiRetClusterGroupBoundaryVertices[iIndex + 1] = paiClusterGroupTriangleIndices[iTri + 1];
        }
        else if(aiAdjacentEdges[1] == 0)
        {
            uint32_t iNumBoundaryVertices = atomicAdd(&aiRetNumClusterGroupBoundaryVertices[iClusterGroup], 2);
            uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumBoundaryVertices;
            aiRetClusterGroupBoundaryVertices[iIndex] = paiClusterGroupTriangleIndices[iTri];
            aiRetClusterGroupBoundaryVertices[iIndex + 1] = paiClusterGroupTriangleIndices[iTri + 2];
        }
        else if(aiAdjacentEdges[2] == 0)
        {
            uint32_t iNumBoundaryVertices = atomicAdd(&aiRetNumClusterGroupBoundaryVertices[iClusterGroup], 2);
            uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumBoundaryVertices;
            aiRetClusterGroupBoundaryVertices[iIndex] = paiClusterGroupTriangleIndices[iTri + 1];
            aiRetClusterGroupBoundaryVertices[iIndex + 1] = paiClusterGroupTriangleIndices[iTri + 2];
        }

    }   // for tri = 0 to num triangles
}

/*
**
*/
__global__
void checkClusterAdjacency(
    uint32_t* aiNumAdjacentVertices,
    float* afTotalClusterVertexPositionComponents,
    uint32_t* aiNumVertexPositionComponents,
    uint32_t* aiClusterVertexPositionComponentOffsets,
    uint32_t iNumTotalClusters,
    bool bOnlyEdgeAdjacent)
{
    uint32_t iCluster = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iCluster >= iNumTotalClusters)
    {
        return;
    }

    float* aClusterVertexPositionComponents = &afTotalClusterVertexPositionComponents[aiClusterVertexPositionComponentOffsets[iCluster]];
    uint32_t iNumVertexPositionComponents = aiNumVertexPositionComponents[iCluster];

    for(uint32_t iCheckCluster = 0; iCheckCluster < iNumTotalClusters; iCheckCluster++)
    {
        if(iCheckCluster == iCluster)
        {
            continue;
        }

        uint32_t iNumAdjacentVertices = 0;
        float* aCheckClusterVertexPositionComponents = &afTotalClusterVertexPositionComponents[aiClusterVertexPositionComponentOffsets[iCheckCluster]];
        uint32_t iNumCheckVertexPositionComponents = aiNumVertexPositionComponents[iCheckCluster];

        for(uint32_t iVertComponent = 0; iVertComponent < iNumVertexPositionComponents; iVertComponent += 3)
        {
            float fX = aClusterVertexPositionComponents[iVertComponent];
            float fY = aClusterVertexPositionComponents[iVertComponent + 1];
            float fZ = aClusterVertexPositionComponents[iVertComponent + 2];

            for(uint32_t iCheckVertComponent = 0; iCheckVertComponent < iNumCheckVertexPositionComponents; iCheckVertComponent += 3)
            {
                float fCheckX = aCheckClusterVertexPositionComponents[iCheckVertComponent];
                float fCheckY = aCheckClusterVertexPositionComponents[iCheckVertComponent + 1];
                float fCheckZ = aCheckClusterVertexPositionComponents[iCheckVertComponent + 2];

                float fDiffX = fX - fCheckX;
                float fDiffY = fY - fCheckY;
                float fDiffZ = fZ - fCheckZ;

                float fLength = fDiffX * fDiffX + fDiffY * fDiffY + fDiffZ * fDiffZ;
                if(fLength <= 1.0e-8f)
                {
                    if(bOnlyEdgeAdjacent && iNumAdjacentVertices >= 2)
                    {
                        break;
                    }
                    else
                    {
                        ++iNumAdjacentVertices;
                        break;
                    }
                    
                }
            }
        }

        uint32_t iIndex = iCluster * iNumTotalClusters + iCheckCluster;
        aiNumAdjacentVertices[iIndex] = iNumAdjacentVertices;
    }
}

/*
**
*/
__device__
void getClusterGroupAndVertexIndex(
    uint32_t* iRetClusterGroup,
    uint32_t* iRetVertexComponentIndex,
    uint32_t iTotalVertexIndex,
    uint32_t* aiNumVertexPositionComponents,
    uint32_t iNumClusterGroups)
{
    uint32_t iTotalVertexComponentIndex = iTotalVertexIndex * 3;
    uint32_t iTotalVertexPositionComponents = 0;
    uint32_t iClusterGroup = 0;
    for(iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        if(iTotalVertexPositionComponents + aiNumVertexPositionComponents[iClusterGroup] > iTotalVertexComponentIndex)
        {
            break;
        }

        iTotalVertexPositionComponents += aiNumVertexPositionComponents[iClusterGroup];
    }

    *iRetClusterGroup = iClusterGroup;
    *iRetVertexComponentIndex = iTotalVertexComponentIndex - iTotalVertexPositionComponents;
}

/*
**
*/
__global__
void checkClusterGroupAdjacency(
    uint32_t* aiAdjacentClusterGroupVertexIndices,
    uint32_t* aiNumAdjacentClusterGroupVertices,
    float* afTotalClusterGroupVertexPositionComponents,
    uint32_t* aiNumVertexPositionComponents,
    uint32_t* aiClusterGroupVertexPositionComponentOffsets,
    uint32_t iNumTotalVertexIndices,
    uint32_t iNumTotalClusterGroups)
{
    uint32_t iTotalVertexIndex = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iTotalVertexIndex >= iNumTotalVertexIndices / 3)
    {
        return;
    }

    uint32_t iClusterGroup = 0, iVertComponent = 0;
    getClusterGroupAndVertexIndex(
        &iClusterGroup,
        &iVertComponent,
        iTotalVertexIndex,
        aiNumVertexPositionComponents,
        iNumTotalClusterGroups);

if(iClusterGroup == 385 && iVertComponent >= 870 * 3)
{
    getClusterGroupAndVertexIndex(
        &iClusterGroup,
        &iVertComponent,
        iTotalVertexIndex,
        aiNumVertexPositionComponents,
        iNumTotalClusterGroups);

    printf("wtf\n");
}

    uint32_t iClusterGroupVertexPositionComponentOffset = aiClusterGroupVertexPositionComponentOffsets[iClusterGroup];
    float* aClusterVertexPositionComponents = &afTotalClusterGroupVertexPositionComponents[iClusterGroupVertexPositionComponentOffset];
//    uint32_t iNumVertexPositionComponents = aiNumVertexPositionComponents[iClusterGroup];

    float fX = aClusterVertexPositionComponents[iVertComponent];
    float fY = aClusterVertexPositionComponents[iVertComponent + 1];
    float fZ = aClusterVertexPositionComponents[iVertComponent + 2];

    for(uint32_t iCheckClusterGroup = 0; iCheckClusterGroup < iNumTotalClusterGroups; iCheckClusterGroup++)
    {
        if(iCheckClusterGroup == iClusterGroup)
        {
            continue;
        }

        uint32_t iCheckClusterGroupVertexPositionComponentOffset = aiClusterGroupVertexPositionComponentOffsets[iCheckClusterGroup];
        float* aCheckClusterVertexPositionComponents = &afTotalClusterGroupVertexPositionComponents[iCheckClusterGroupVertexPositionComponentOffset];
        uint32_t iNumCheckVertexPositionComponents = aiNumVertexPositionComponents[iCheckClusterGroup];

        for(uint32_t iCheckVertComponent = 0; iCheckVertComponent < iNumCheckVertexPositionComponents; iCheckVertComponent += 3)
        {
            float fCheckX = aCheckClusterVertexPositionComponents[iCheckVertComponent];
            float fCheckY = aCheckClusterVertexPositionComponents[iCheckVertComponent + 1];
            float fCheckZ = aCheckClusterVertexPositionComponents[iCheckVertComponent + 2];

            float fDiffX = fX - fCheckX;
            float fDiffY = fY - fCheckY;
            float fDiffZ = fZ - fCheckZ;

            float fLength = fDiffX * fDiffX + fDiffY * fDiffY + fDiffZ * fDiffZ;
            if(fLength <= 1.0e-10f)
            {
                uint32_t iArrayIndex = atomicAdd(&aiNumAdjacentClusterGroupVertices[iClusterGroup], 1);
if(iArrayIndex >= MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP)
{
    printf("wtf\n");
}
if(iClusterGroup == 385)
{
    printf("debug\n");
}

                uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iArrayIndex;
                aiAdjacentClusterGroupVertexIndices[iIndex] = iVertComponent / 3;
            }
            
        }   // for check vertex component = 0 to num vertex components

    }   // for check cluster group = 0 to num cluster groups 
}

/*
**
*/
__global__
void computeEdgeCollapseInfo(
    float* afRetEdgeCollapseCosts,
    uint32_t* aiRetEdgeCollapseVertexIndices0,
    uint32_t* aiRetEdgeCollapseVertexIndices1,
    float* afRetEdgeCollapseVertexPositions,
    float* afRetEdgeCollapseVertexNormals,
    float* afRetEdgeCollapseVertexUVs,
    uint32_t* aiClusterGroupNonBoundaryVertexIndices,
    float* afVertexPositionComponents,
    float* afVertexNormalComponents,
    float* afVertexUVComponents,
    float* afQuadrics,
    float* afVertexNormalPlaneAngles,
    uint32_t* aiClusterGroupEdgePairs,
    uint32_t* aiClusterGroupTrianglePositionIndicesGPU,
    uint32_t* aiClusterGroupTriangleNormalIndicesGPU,
    uint32_t* aiClusterGroupTriangleUVIndicesGPU,
    uint32_t* aiNormalIndexToEdgeMap,
    uint32_t* aiUVIndexToEdgeMap,
    uint32_t iNumClusterGroupTrianglePositionIndices,
    uint32_t iNumClusterGroupNonBoundaryVertices,
    uint32_t iNumEdges)
{
    uint32_t iEdge = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iEdge >= iNumEdges)
    {
        return;
    }

    uint32_t iEdgeIndex = iEdge * 2;
    uint32_t iEdgePos0 = aiClusterGroupEdgePairs[iEdgeIndex];
    uint32_t iEdgePos1 = aiClusterGroupEdgePairs[iEdgeIndex + 1];

    uint32_t iNorm0 = aiNormalIndexToEdgeMap[iEdgeIndex];
    uint32_t iNorm1 = aiNormalIndexToEdgeMap[iEdgeIndex + 1];

    uint32_t iUV0 = aiUVIndexToEdgeMap[iEdgeIndex];
    uint32_t iUV1 = aiUVIndexToEdgeMap[iEdgeIndex + 1];

    aiRetEdgeCollapseVertexIndices0[iEdge] = iEdgePos0;
    aiRetEdgeCollapseVertexIndices1[iEdge] = iEdgePos1;

    // check which edge vertices are non-boundary
    bool bValid0 = false;
    bool bValid1 = false;
    for(uint32_t i = 0; i < iNumClusterGroupNonBoundaryVertices; i++)
    {
        if(aiClusterGroupNonBoundaryVertexIndices[i] == iEdgePos0)
        {
            bValid0 = true;
        }

        if(aiClusterGroupNonBoundaryVertexIndices[i] == iEdgePos1)
        {
            bValid1 = true;
        }

        if(bValid0 && bValid1)
        {
            break;
        }
    }

    if(!bValid0 && !bValid0)
    {
        afRetEdgeCollapseCosts[iEdge] = 1.0e+10f;
        aiRetEdgeCollapseVertexIndices0[iEdge] = 0;
        aiRetEdgeCollapseVertexIndices1[iEdge] = 0;
        return;
    }

    float afQuadrics0[16];
    memcpy(afQuadrics0, &afQuadrics[iEdgePos0 * 16], sizeof(float) * 16);

    float afQuadrics1[16];
    memcpy(afQuadrics1, &afQuadrics[iEdgePos1 * 16], sizeof(float) * 16);

    // normal plane angles for feature value
    float fTotalNormalPlaneAngles0 = afVertexNormalPlaneAngles[iEdgePos0];
    float fTotalNormalPlaneAngles1 = afVertexNormalPlaneAngles[iEdgePos1];

    // feature value
//    float const kfFeatureMult = 1.0f;
    float fDiffX = afVertexPositionComponents[iEdgePos1 * 3] - afVertexPositionComponents[iEdgePos0 * 3];
    float fDiffY = afVertexPositionComponents[iEdgePos1 * 3 + 1] - afVertexPositionComponents[iEdgePos0 * 3 + 1];
    float fDiffZ = afVertexPositionComponents[iEdgePos1 * 3 + 2] - afVertexPositionComponents[iEdgePos0 * 3 + 2];
    float fEdgeLength = fDiffX * fDiffX + fDiffY * fDiffY + fDiffZ * fDiffZ;
    float fFeatureValue = fEdgeLength * (1.0f + 0.5f * (fTotalNormalPlaneAngles0 + fTotalNormalPlaneAngles1));

    float afEdgeQuadrics[16];
    for(uint32_t i = 0; i < 16; i++)
    {
        afEdgeQuadrics[i] = afQuadrics0[i] + afQuadrics1[i];
    }
    afEdgeQuadrics[15] += fFeatureValue;

    //if(iEdgePos0 == 0 && iEdgePos1 == 1)
    //{
    //    printf("\n\n*********\n\n");
    //    for(uint32_t i = 0; i < 16; i++)
    //    {
    //        printf("%.4f\n", afEdgeQuadrics[i]);
    //    }
    //
    //    printf("****\n");
    //}

    if(bValid0 == false)
    {
        // boundary
        afRetEdgeCollapseVertexPositions[iEdge * 3] =     afVertexPositionComponents[iEdgePos0 * 3];
        afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] = afVertexPositionComponents[iEdgePos0 * 3 + 1];
        afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] = afVertexPositionComponents[iEdgePos0 * 3 + 2];

        afRetEdgeCollapseVertexNormals[iEdge * 3] =     afVertexNormalComponents[iNorm0 * 3];
        afRetEdgeCollapseVertexNormals[iEdge * 3 + 1] = afVertexNormalComponents[iNorm0 * 3 + 1];
        afRetEdgeCollapseVertexNormals[iEdge * 3 + 2] = afVertexNormalComponents[iNorm0 * 3 + 2];

        afRetEdgeCollapseVertexUVs[iEdge * 3] =     afVertexUVComponents[iUV0 * 3];
        afRetEdgeCollapseVertexUVs[iEdge * 3 + 1] = afVertexUVComponents[iUV0 * 3 + 1];
    }
    else if(bValid1 == false)
    {
        // boundary
        afRetEdgeCollapseVertexPositions[iEdge * 3] = afVertexPositionComponents[iEdgePos1 * 3];
        afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] = afVertexPositionComponents[iEdgePos1 * 3 + 1];
        afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] = afVertexPositionComponents[iEdgePos1 * 3 + 2];

        afRetEdgeCollapseVertexNormals[iEdge * 3] =     afVertexNormalComponents[iNorm1 * 3];
        afRetEdgeCollapseVertexNormals[iEdge * 3 + 1] = afVertexNormalComponents[iNorm1 * 3 + 1];
        afRetEdgeCollapseVertexNormals[iEdge * 3 + 2] = afVertexNormalComponents[iNorm1 * 3 + 2];

        afRetEdgeCollapseVertexUVs[iEdge * 3] =     afVertexUVComponents[iUV1 * 3];
        afRetEdgeCollapseVertexUVs[iEdge * 3 + 1] = afVertexUVComponents[iUV1 * 3 + 1];
    }
    else
    {
        // mid point
        afRetEdgeCollapseVertexPositions[iEdge * 3] =     (afVertexPositionComponents[iEdgePos0 * 3]     + afVertexPositionComponents[iEdgePos1 * 3]) * 0.5f;
        afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] = (afVertexPositionComponents[iEdgePos0 * 3 + 1] + afVertexPositionComponents[iEdgePos1 * 3 + 1]) * 0.5f;
        afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] = (afVertexPositionComponents[iEdgePos0 * 3 + 2] + afVertexPositionComponents[iEdgePos1 * 3 + 2]) * 0.5f;

        afRetEdgeCollapseVertexNormals[iEdge * 3] =     (afVertexNormalComponents[iNorm0 * 3]     + afVertexNormalComponents[iNorm1 * 3]) * 0.5f;
        afRetEdgeCollapseVertexNormals[iEdge * 3 + 1] = (afVertexNormalComponents[iNorm0 * 3 + 1] + afVertexNormalComponents[iNorm1 * 3 + 1]) * 0.5f;
        afRetEdgeCollapseVertexNormals[iEdge * 3 + 2] = (afVertexNormalComponents[iNorm0 * 3 + 2] + afVertexNormalComponents[iNorm1 * 3 + 2]) * 0.5f;

        afRetEdgeCollapseVertexUVs[iEdge * 3] =     (afVertexUVComponents[iUV0 * 3]     + afVertexUVComponents[iUV1 * 3]) * 0.5f;
        afRetEdgeCollapseVertexUVs[iEdge * 3 + 1] = (afVertexUVComponents[iUV0 * 3 + 1] + afVertexUVComponents[iUV1 * 3 + 1]) * 0.5f;
    }

    // compute the cost of the contraction (transpose(v_optimal) * M * v_optimal)
    afRetEdgeCollapseCosts[iEdge] =
        afEdgeQuadrics[0] * afRetEdgeCollapseVertexPositions[iEdge * 3] * afRetEdgeCollapseVertexPositions[iEdge * 3] +
        2.0f * afEdgeQuadrics[1] * afRetEdgeCollapseVertexPositions[iEdge * 3] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] +
        2.0f * afEdgeQuadrics[2] * afRetEdgeCollapseVertexPositions[iEdge * 3] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] +
        2.0f * afEdgeQuadrics[3] * afRetEdgeCollapseVertexPositions[iEdge * 3] +

        afEdgeQuadrics[5] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] +
        2.0f * afEdgeQuadrics[6] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] +
        2.0f * afEdgeQuadrics[7] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 1] +

        afEdgeQuadrics[10] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] +
        2.0f * afEdgeQuadrics[11] * afRetEdgeCollapseVertexPositions[iEdge * 3 + 2] +

        afEdgeQuadrics[15];

//printf("edge %d collapse cost: %.4f\n", iEdge, afRetEdgeCollapseCosts[iEdge]);
//if(iEdge == 0)
//{
//    printf("edge %d optimal vertex position (%.4f, %.4f, %.4f)\n",
//        iEdge,
//        afRetEdgeCollapseVertexPositions[iEdge * 3],
//        afRetEdgeCollapseVertexPositions[iEdge * 3 + 1],
//        afRetEdgeCollapseVertexPositions[iEdge * 3 + 2]);
//}
}

/*
**
*/
__device__
float _dot(float fX0, float fY0, float fZ0, float fX1, float fY1, float fZ1)
{
    return fX0 * fX1 + fY0 * fY1 + fZ0 * fZ1;
}

/*
**
*/
__device__
float _length(float fX, float fY, float fZ)
{
    return sqrt(fX * fX + fY * fY + fZ * fZ);
}

/*
**
*/
__device__
void _normalize(
    float* fNormalizedX,
    float* fNormalizedY,
    float* fNormalizedZ,
    float fX, 
    float fY, 
    float fZ)
{
    float fLength = _length(fX, fY, fZ);
    *fNormalizedX = fX / fLength;
    *fNormalizedY = fY / fLength;
    *fNormalizedZ = fZ / fLength;
}

/*
**
*/
__device__
void _cross(
    float* fRetX,
    float* fRetY,
    float* fRetZ,
    float fX0,
    float fY0,
    float fZ0,
    float fX1,
    float fY1,
    float fZ1)
{
    *fRetX = fY0 * fZ1 - fZ0 * fY1;
    *fRetY = fZ0 * fX1 - fX0 * fZ1;
    *fRetZ = fX0 * fY1 - fY0 * fX1;
}

/*
**
*/
__global__
void computeQuadrics(
    float* afRetQuadrics,
    float* afRetAdjacentCount,
    uint32_t* aiTriangleVertexPositionIndices,
    float* afVertexPositionComponents,
    uint32_t iNumVertices,
    uint32_t iNumTriangleIndices)
{
    uint32_t iVertex = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iVertex >= iNumVertices)
    {
        return;
    }

//    uint32_t iVertexPositionComponentIndex = iVertex * 3;
//    uint32_t iQuadricComponentIndex = iVertex * 16;

    float fAdjacentCount = 0.0f;
    float afTotalQuadricMatrix[16] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float afAverageVertexNormal[3] = { 0.0f, 0.0f, 0.0f };
    for(uint32_t iTri = 0; iTri < iNumTriangleIndices; iTri += 3)
    {
        uint32_t iV0 = aiTriangleVertexPositionIndices[iTri];
        uint32_t iV1 = aiTriangleVertexPositionIndices[iTri + 1];
        uint32_t iV2 = aiTriangleVertexPositionIndices[iTri + 2];

        float afPlane[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if(iV0 == iVertex)
        {
            float fDiffX0 = afVertexPositionComponents[iV1 * 3] -       afVertexPositionComponents[iV0 * 3];
            float fDiffY0 = afVertexPositionComponents[iV1 * 3 + 1] -   afVertexPositionComponents[iV0 * 3 + 1];
            float fDiffZ0 = afVertexPositionComponents[iV1 * 3 + 2] -   afVertexPositionComponents[iV0 * 3 + 2];
            float fNormalizedX0, fNormalizedY0, fNormalizedZ0;
            _normalize(&fNormalizedX0, &fNormalizedY0, &fNormalizedZ0, fDiffX0, fDiffY0, fDiffZ0);
            
            float fDiffX1 = afVertexPositionComponents[iV2 * 3] -       afVertexPositionComponents[iV0 * 3];
            float fDiffY1 = afVertexPositionComponents[iV2 * 3 + 1] -   afVertexPositionComponents[iV0 * 3 + 1];
            float fDiffZ1 = afVertexPositionComponents[iV2 * 3 + 2] -   afVertexPositionComponents[iV0 * 3 + 2];
            float fNormalizedX1, fNormalizedY1, fNormalizedZ1;
            _normalize(&fNormalizedX1, &fNormalizedY1, &fNormalizedZ1, fDiffX1, fDiffY1, fDiffZ1);
            
            _cross(&afPlane[0], &afPlane[1], &afPlane[2], fNormalizedX1, fNormalizedY1, fNormalizedZ1, fNormalizedX0, fNormalizedY0, fNormalizedZ0);
            afPlane[3] = _dot(
                afPlane[0], 
                afPlane[1], 
                afPlane[2], 
                afVertexPositionComponents[iV0 * 3], 
                afVertexPositionComponents[iV0 * 3 + 1], 
                afVertexPositionComponents[iV0 * 3 + 2]) * -1.0f;

            afTotalQuadricMatrix[0] += afPlane[0] * afPlane[0];
            afTotalQuadricMatrix[1] += afPlane[0] * afPlane[1];
            afTotalQuadricMatrix[2] += afPlane[0] * afPlane[2];
            afTotalQuadricMatrix[3] += afPlane[0] * afPlane[3];

            afTotalQuadricMatrix[4] += afPlane[1] * afPlane[0];
            afTotalQuadricMatrix[5] += afPlane[1] * afPlane[1];
            afTotalQuadricMatrix[6] += afPlane[1] * afPlane[2];
            afTotalQuadricMatrix[7] += afPlane[1] * afPlane[3];

            afTotalQuadricMatrix[8] += afPlane[2] * afPlane[0];
            afTotalQuadricMatrix[9] += afPlane[2] * afPlane[1];
            afTotalQuadricMatrix[10] += afPlane[2] * afPlane[2];
            afTotalQuadricMatrix[11] += afPlane[2] * afPlane[3];

            afTotalQuadricMatrix[12] += afPlane[3] * afPlane[0];
            afTotalQuadricMatrix[13] += afPlane[3] * afPlane[1];
            afTotalQuadricMatrix[14] += afPlane[3] * afPlane[2];
            afTotalQuadricMatrix[15] += afPlane[3] * afPlane[3];

            afAverageVertexNormal[0] += afPlane[0];
            afAverageVertexNormal[1] += afPlane[1];
            afAverageVertexNormal[2] += afPlane[2];

            fAdjacentCount += 1.0f;
        }
        else if(iV1 == iVertex)
        {
            float fDiffX0 = afVertexPositionComponents[iV0 * 3] -       afVertexPositionComponents[iV1 * 3];
            float fDiffY0 = afVertexPositionComponents[iV0 * 3 + 1] -   afVertexPositionComponents[iV1 * 3 + 1];
            float fDiffZ0 = afVertexPositionComponents[iV0 * 3 + 2] -   afVertexPositionComponents[iV1 * 3 + 2];
            float fNormalizedX0, fNormalizedY0, fNormalizedZ0;
            _normalize(&fNormalizedX0, &fNormalizedY0, &fNormalizedZ0, fDiffX0, fDiffY0, fDiffZ0);

            float fDiffX1 = afVertexPositionComponents[iV2 * 3] -       afVertexPositionComponents[iV1 * 3];
            float fDiffY1 = afVertexPositionComponents[iV2 * 3 + 1] -   afVertexPositionComponents[iV1 * 3 + 1];
            float fDiffZ1 = afVertexPositionComponents[iV2 * 3 + 2] -   afVertexPositionComponents[iV1 * 3 + 2];
            float fNormalizedX1, fNormalizedY1, fNormalizedZ1;
            _normalize(&fNormalizedX1, &fNormalizedY1, &fNormalizedZ1, fDiffX1, fDiffY1, fDiffZ1);

            _cross(&afPlane[0], &afPlane[1], &afPlane[2], fNormalizedX1, fNormalizedY1, fNormalizedZ1, fNormalizedX0, fNormalizedY0, fNormalizedZ0);
            afPlane[3] = _dot(
                afPlane[0],
                afPlane[1],
                afPlane[2],
                afVertexPositionComponents[iV0 * 3],
                afVertexPositionComponents[iV0 * 3 + 1],
                afVertexPositionComponents[iV0 * 3 + 2]) * -1.0f;

            afTotalQuadricMatrix[0] += afPlane[0] * afPlane[0];
            afTotalQuadricMatrix[1] += afPlane[0] * afPlane[1];
            afTotalQuadricMatrix[2] += afPlane[0] * afPlane[2];
            afTotalQuadricMatrix[3] += afPlane[0] * afPlane[3];

            afTotalQuadricMatrix[4] += afPlane[1] * afPlane[0];
            afTotalQuadricMatrix[5] += afPlane[1] * afPlane[1];
            afTotalQuadricMatrix[6] += afPlane[1] * afPlane[2];
            afTotalQuadricMatrix[7] += afPlane[1] * afPlane[3];

            afTotalQuadricMatrix[8] += afPlane[2] * afPlane[0];
            afTotalQuadricMatrix[9] += afPlane[2] * afPlane[1];
            afTotalQuadricMatrix[10] += afPlane[2] * afPlane[2];
            afTotalQuadricMatrix[11] += afPlane[2] * afPlane[3];

            afTotalQuadricMatrix[12] += afPlane[3] * afPlane[0];
            afTotalQuadricMatrix[13] += afPlane[3] * afPlane[1];
            afTotalQuadricMatrix[14] += afPlane[3] * afPlane[2];
            afTotalQuadricMatrix[15] += afPlane[3] * afPlane[3];

            afAverageVertexNormal[0] += afPlane[0];
            afAverageVertexNormal[1] += afPlane[1];
            afAverageVertexNormal[2] += afPlane[2];

            fAdjacentCount += 1.0f;
        }
        else if(iV2 == iVertex)
        {
            float fDiffX0 = afVertexPositionComponents[iV0 * 3] -       afVertexPositionComponents[iV2 * 3];
            float fDiffY0 = afVertexPositionComponents[iV0 * 3 + 1] -   afVertexPositionComponents[iV2 * 3 + 1];
            float fDiffZ0 = afVertexPositionComponents[iV0 * 3 + 2] -   afVertexPositionComponents[iV2 * 3 + 2];
            float fNormalizedX0, fNormalizedY0, fNormalizedZ0;
            _normalize(&fNormalizedX0, &fNormalizedY0, &fNormalizedZ0, fDiffX0, fDiffY0, fDiffZ0);

            float fDiffX1 = afVertexPositionComponents[iV1 * 3] -       afVertexPositionComponents[iV2 * 3];
            float fDiffY1 = afVertexPositionComponents[iV1 * 3 + 1] -   afVertexPositionComponents[iV2 * 3 + 1];
            float fDiffZ1 = afVertexPositionComponents[iV1 * 3 + 2] -   afVertexPositionComponents[iV2 * 3 + 2];
            float fNormalizedX1, fNormalizedY1, fNormalizedZ1;
            _normalize(&fNormalizedX1, &fNormalizedY1, &fNormalizedZ1, fDiffX1, fDiffY1, fDiffZ1);

            _cross(&afPlane[0], &afPlane[1], &afPlane[2], fNormalizedX1, fNormalizedY1, fNormalizedZ1, fNormalizedX0, fNormalizedY0, fNormalizedZ0);
            afPlane[3] = _dot(
                afPlane[0],
                afPlane[1],
                afPlane[2],
                afVertexPositionComponents[iV0 * 3],
                afVertexPositionComponents[iV0 * 3 + 1],
                afVertexPositionComponents[iV0 * 3 + 2]) * -1.0f;

            afTotalQuadricMatrix[0] += afPlane[0] * afPlane[0];
            afTotalQuadricMatrix[1] += afPlane[0] * afPlane[1];
            afTotalQuadricMatrix[2] += afPlane[0] * afPlane[2];
            afTotalQuadricMatrix[3] += afPlane[0] * afPlane[3];

            afTotalQuadricMatrix[4] += afPlane[1] * afPlane[0];
            afTotalQuadricMatrix[5] += afPlane[1] * afPlane[1];
            afTotalQuadricMatrix[6] += afPlane[1] * afPlane[2];
            afTotalQuadricMatrix[7] += afPlane[1] * afPlane[3];

            afTotalQuadricMatrix[8] += afPlane[2] * afPlane[0];
            afTotalQuadricMatrix[9] += afPlane[2] * afPlane[1];
            afTotalQuadricMatrix[10] += afPlane[2] * afPlane[2];
            afTotalQuadricMatrix[11] += afPlane[2] * afPlane[3];

            afTotalQuadricMatrix[12] += afPlane[3] * afPlane[0];
            afTotalQuadricMatrix[13] += afPlane[3] * afPlane[1];
            afTotalQuadricMatrix[14] += afPlane[3] * afPlane[2];
            afTotalQuadricMatrix[15] += afPlane[3] * afPlane[3];

            afAverageVertexNormal[0] += afPlane[0];
            afAverageVertexNormal[1] += afPlane[1];
            afAverageVertexNormal[2] += afPlane[2];

            fAdjacentCount += 1.0f;
        }

    }   // for tri = 0 to num vertex components

    memcpy(&afRetQuadrics[iVertex * 16], &afTotalQuadricMatrix[0], sizeof(float) * 16);
    afRetAdjacentCount[iVertex] = fAdjacentCount;
}

#define MAX_NUM_PLANES_PER_VERTEX     100

/*
**
*/
__global__
void computeAverageVertexNormals(
    float* afRetVertexPlanes,
    float* afRetAverageVertexNormals,
    float* afRetQuadricMatrices,
    uint32_t* aiRetNumVertexPlanes,
    uint32_t* aiTriangleVertexPositionIndices,
    float* afVertexPositionComponents,
    uint32_t iNumVertices,
    uint32_t iNumTriangleIndices)
{
    uint32_t iVertex = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iVertex >= iNumVertices)
    {
        return;
    }

    uint32_t iVertexComponent = iVertex * 3;

    float afAverageNormal[3] = { 0.0f, 0.0f, 0.0f };
    float afTotalQuadricMatrix[16] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f };
//    uint32_t iNumPlanes = 0;
    for(uint32_t iTri = 0; iTri < iNumTriangleIndices; iTri += 3)
    {
        uint32_t iV0 = aiTriangleVertexPositionIndices[iTri];
        uint32_t iV1 = aiTriangleVertexPositionIndices[iTri + 1];
        uint32_t iV2 = aiTriangleVertexPositionIndices[iTri + 2];

        if(iV0 == iVertex || iV1 == iVertex || iV2 == iVertex)
        {
            float fDiffX0 = 0.0f, fDiffY0 = 0.0f, fDiffZ0 = 0.0f;
            float fDiffX1 = 0.0f, fDiffY1 = 0.0f, fDiffZ1 = 0.0f;
            if(iV0 == iVertex)
            {
                fDiffX0 = afVertexPositionComponents[iV1 * 3] - afVertexPositionComponents[iV0 * 3];
                fDiffY0 = afVertexPositionComponents[iV1 * 3 + 1] - afVertexPositionComponents[iV0 * 3 + 1];
                fDiffZ0 = afVertexPositionComponents[iV1 * 3 + 2] - afVertexPositionComponents[iV0 * 3 + 2];

                fDiffX1 = afVertexPositionComponents[iV2 * 3] - afVertexPositionComponents[iV0 * 3];
                fDiffY1 = afVertexPositionComponents[iV2 * 3 + 1] - afVertexPositionComponents[iV0 * 3 + 1];
                fDiffZ1 = afVertexPositionComponents[iV2 * 3 + 2] - afVertexPositionComponents[iV0 * 3 + 2];
            }
            else if(iV1 == iVertex)
            {
                fDiffX0 = afVertexPositionComponents[iV0 * 3] -     afVertexPositionComponents[iV1 * 3];
                fDiffY0 = afVertexPositionComponents[iV0 * 3 + 1] - afVertexPositionComponents[iV1 * 3 + 1];
                fDiffZ0 = afVertexPositionComponents[iV0 * 3 + 2] - afVertexPositionComponents[iV1 * 3 + 2];

                fDiffX1 = afVertexPositionComponents[iV2 * 3] -     afVertexPositionComponents[iV1 * 3];
                fDiffY1 = afVertexPositionComponents[iV2 * 3 + 1] - afVertexPositionComponents[iV1 * 3 + 1];
                fDiffZ1 = afVertexPositionComponents[iV2 * 3 + 2] - afVertexPositionComponents[iV1 * 3 + 2];
            }
            else if(iV2 == iVertex)
            {
                fDiffX0 = afVertexPositionComponents[iV0 * 3] -     afVertexPositionComponents[iV2 * 3];
                fDiffY0 = afVertexPositionComponents[iV0 * 3 + 1] - afVertexPositionComponents[iV2 * 3 + 1];
                fDiffZ0 = afVertexPositionComponents[iV0 * 3 + 2] - afVertexPositionComponents[iV2 * 3 + 2];

                fDiffX1 = afVertexPositionComponents[iV1 * 3] -     afVertexPositionComponents[iV2 * 3];
                fDiffY1 = afVertexPositionComponents[iV1 * 3 + 1] - afVertexPositionComponents[iV2 * 3 + 1];
                fDiffZ1 = afVertexPositionComponents[iV1 * 3 + 2] - afVertexPositionComponents[iV2 * 3 + 2];
            }

            float fNormalizedX0, fNormalizedY0, fNormalizedZ0;
            _normalize(&fNormalizedX0, &fNormalizedY0, &fNormalizedZ0, fDiffX0, fDiffY0, fDiffZ0);
                
            float fNormalizedX1, fNormalizedY1, fNormalizedZ1;
            _normalize(&fNormalizedX1, &fNormalizedY1, &fNormalizedZ1, fDiffX1, fDiffY1, fDiffZ1);

            float afPlane[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            _cross(&afPlane[0], &afPlane[1], &afPlane[2], fNormalizedX1, fNormalizedY1, fNormalizedZ1, fNormalizedX0, fNormalizedY0, fNormalizedZ0);
            afPlane[3] = _dot(
                afPlane[0],
                afPlane[1],
                afPlane[2],
                afVertexPositionComponents[iV0 * 3],
                afVertexPositionComponents[iV0 * 3 + 1],
                afVertexPositionComponents[iV0 * 3 + 2]) * -1.0f;

            afAverageNormal[0] += afPlane[0];
            afAverageNormal[1] += afPlane[1];
            afAverageNormal[2] += afPlane[2];

            uint32_t iCurrNumPlanes = atomicAdd(&aiRetNumVertexPlanes[iVertex], 1);
            uint32_t iVertexPlaneIndex = iVertex * MAX_NUM_PLANES_PER_VERTEX + iCurrNumPlanes * 4;
            afRetVertexPlanes[iVertexPlaneIndex] = afPlane[0];
            afRetVertexPlanes[iVertexPlaneIndex + 1] = afPlane[1];
            afRetVertexPlanes[iVertexPlaneIndex + 2] = afPlane[2];
            afRetVertexPlanes[iVertexPlaneIndex + 3] = afPlane[3];

            afTotalQuadricMatrix[0] += afPlane[0] * afPlane[0];
            afTotalQuadricMatrix[1] += afPlane[0] * afPlane[1];
            afTotalQuadricMatrix[2] += afPlane[0] * afPlane[2];
            afTotalQuadricMatrix[3] += afPlane[0] * afPlane[3];

            afTotalQuadricMatrix[4] += afPlane[1] * afPlane[0];
            afTotalQuadricMatrix[5] += afPlane[1] * afPlane[1];
            afTotalQuadricMatrix[6] += afPlane[1] * afPlane[2];
            afTotalQuadricMatrix[7] += afPlane[1] * afPlane[3];

            afTotalQuadricMatrix[8] += afPlane[2] * afPlane[0];
            afTotalQuadricMatrix[9] += afPlane[2] * afPlane[1];
            afTotalQuadricMatrix[10] += afPlane[2] * afPlane[2];
            afTotalQuadricMatrix[11] += afPlane[2] * afPlane[3];

            afTotalQuadricMatrix[12] += afPlane[3] * afPlane[0];
            afTotalQuadricMatrix[13] += afPlane[3] * afPlane[1];
            afTotalQuadricMatrix[14] += afPlane[3] * afPlane[2];
            afTotalQuadricMatrix[15] += afPlane[3] * afPlane[3];

        }   // if contains vertex 
    
    }   // for tri = 0 to num triangles

    if(afAverageNormal[0] == 0.0f && afAverageNormal[1] == 0.0f && afAverageNormal[2] == 0.0f)
    {
        afRetAverageVertexNormals[iVertexComponent] = 0.0f;
        afRetAverageVertexNormals[iVertexComponent + 1] = 0.0f;
        afRetAverageVertexNormals[iVertexComponent + 2] = 0.0f;
    }
    else
    {
        _normalize(
            &afRetAverageVertexNormals[iVertexComponent],
            &afRetAverageVertexNormals[iVertexComponent + 1],
            &afRetAverageVertexNormals[iVertexComponent + 2],
            afAverageNormal[0],
            afAverageNormal[1],
            afAverageNormal[2]);
    }

    //if(iVertex == 0 || iVertex == 1)
    //{
    //    printf("vertex %d\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n%.4f\n",
    //        iVertex,
    //        afTotalQuadricMatrix[0], afTotalQuadricMatrix[1], afTotalQuadricMatrix[2], afTotalQuadricMatrix[3],
    //        afTotalQuadricMatrix[4], afTotalQuadricMatrix[5], afTotalQuadricMatrix[6], afTotalQuadricMatrix[7],
    //        afTotalQuadricMatrix[8], afTotalQuadricMatrix[9], afTotalQuadricMatrix[10], afTotalQuadricMatrix[11],
    //        afTotalQuadricMatrix[12], afTotalQuadricMatrix[13], afTotalQuadricMatrix[14], afTotalQuadricMatrix[15]);
    //}

    memcpy(
        &afRetQuadricMatrices[iVertex * 16],
        &afTotalQuadricMatrix[0],
        16 * sizeof(float));

    //printf("%d average normal (%.4f, %.4f, %.4f)\n", 
    //    iVertex, 
    //    afRetAverageVertexNormals[iVertexComponent],
    //    afRetAverageVertexNormals[iVertexComponent + 1],
    //    afRetAverageVertexNormals[iVertexComponent + 2]);
}

/*
**
*/
__global__ 
void computeTotalNormalPlaneAngles(
    float* afRetTotalNormalPlaneAngles,
    float* afVertexNormals,
    float* afVertexPlanes,
    uint32_t* aiNumVertexPlanes,
    uint32_t iNumVertices)
{
    uint32_t iVertex = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iVertex >= iNumVertices)
    {
        return;
    }

    uint32_t iNumVertexPlanes = aiNumVertexPlanes[iVertex];
    float fNormalX = afVertexNormals[iVertex * 3];
    float fNormalY = afVertexNormals[iVertex * 3 + 1];
    float fNormalZ = afVertexNormals[iVertex * 3 + 2];

    float fTotalAngle = 0.0f;
    for(uint32_t iPlane = 0; iPlane < iNumVertexPlanes; iPlane++)
    {
        uint32_t iPlaneIndex = iVertex * MAX_NUM_PLANES_PER_VERTEX + iPlane * 4;
        float fPlaneX = afVertexPlanes[iPlaneIndex];
        float fPlaneY = afVertexPlanes[iPlaneIndex + 1];
        float fPlaneZ = afVertexPlanes[iPlaneIndex + 2];
        fTotalAngle += abs(_dot(fNormalX, fNormalY, fNormalZ, fPlaneX, fPlaneY, fPlaneZ));
    }

    afRetTotalNormalPlaneAngles[iVertex] = fTotalAngle / max(float(iNumVertexPlanes), 0.001f);

//printf("%d total normal plane angles: %.4f\n", iVertex, afRetTotalNormalPlaneAngles[iVertex]);
}

/*
**
*/
__global__
void getMatchingTriangleNormalAndUV(
    uint32_t* aiRetNormalIndices,
    uint32_t* aiRetUVIndices,
    uint32_t* aiTriangleVertexPositionIndices,
    uint32_t* aiTriangleVertexNormalIndices,
    uint32_t* aiTriangleVertexUVIndices,
    uint32_t* aiEdges,
    uint32_t iNumEdges,
    uint32_t iNumTriangleVertexPositionIndices)
{
    uint32_t iEdge = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iEdge >= iNumEdges)
    {
        return;
    }

    uint32_t iEdgeComponent = iEdge * 2;

    uint32_t iEdgePos0 = aiEdges[iEdgeComponent];
    uint32_t iEdgePos1 = aiEdges[iEdgeComponent + 1];

    uint32_t iNorm0 = UINT32_MAX, iNorm1 = UINT32_MAX;
    uint32_t iUV0 = UINT32_MAX;
    uint32_t iUV1 = UINT32_MAX;
    {
        uint32_t iMatchingTri = 0;
        uint32_t aiTriIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
        for(iMatchingTri = 0; iMatchingTri < iNumTriangleVertexPositionIndices; iMatchingTri += 3)
        {
            uint32_t iNumSamePosition = 0;
            aiTriIndices[0] = aiTriIndices[1] = aiTriIndices[2] = UINT32_MAX;
            for(uint32_t i = 0; i < 3; i++)
            {
                if(aiTriangleVertexPositionIndices[iMatchingTri + i] == iEdgePos0 ||
                    aiTriangleVertexPositionIndices[iMatchingTri + i] == iEdgePos1)
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
        if(iMatchingTri >= iNumTriangleVertexPositionIndices)
        {
            aiRetNormalIndices[iEdgeComponent] = UINT32_MAX;
            aiRetNormalIndices[iEdgeComponent + 1] = UINT32_MAX;

            aiRetUVIndices[iEdgeComponent] = UINT32_MAX;
            aiRetUVIndices[iEdgeComponent + 1] = UINT32_MAX;

            return;
        }

        iNorm0 = aiTriangleVertexNormalIndices[iMatchingTri + aiTriIndices[0]];
        iNorm1 = aiTriangleVertexNormalIndices[iMatchingTri + aiTriIndices[1]];

        iUV0 = aiTriangleVertexUVIndices[iMatchingTri + aiTriIndices[0]];
        iUV1 = aiTriangleVertexUVIndices[iMatchingTri + aiTriIndices[1]];

        aiRetNormalIndices[iEdgeComponent] = iNorm0;
        aiRetNormalIndices[iEdgeComponent + 1] = iNorm1;

        aiRetUVIndices[iEdgeComponent] = iUV0;
        aiRetUVIndices[iEdgeComponent + 1] = iUV1;
    }
}

/*
**
*/
__global__
void getShortestVertexDistances(
    float* afRetShortestDistances,
    uint32_t* aiRetShortestVertexPositionIndices,
    float* afVertexPositions0,
    float* afVertexPositions1,
    uint32_t iNumVertexPositions0,
    uint32_t iNumVertexPositions1)
{
    uint32_t iVertex = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iVertex >= iNumVertexPositions0)
    {
        return;
    }

    uint32_t iVertexComponentIndex = iVertex * 3;
    float fX = afVertexPositions0[iVertexComponentIndex];
    float fY = afVertexPositions0[iVertexComponentIndex + 1];
    float fZ = afVertexPositions0[iVertexComponentIndex + 2];

    float fShortestLength = 1.0e+10f;
    uint32_t iNumVertexPositionComponents1 = iNumVertexPositions1 * 3;
    for(uint32_t iPos = 0; iPos < iNumVertexPositionComponents1; iPos += 3)
    {
        float fCheckX = afVertexPositions1[iPos];
        float fCheckY = afVertexPositions1[iPos + 1];
        float fCheckZ = afVertexPositions1[iPos + 2];

        float fLength = _length(fCheckX - fX, fCheckY - fY, fCheckZ - fZ);
        
        if(fLength < fShortestLength)
        {
            afRetShortestDistances[iVertex] = fLength;
            aiRetShortestVertexPositionIndices[iVertex] = iPos / 3;

            fShortestLength = fLength;
        }
    }
}


/*
**
*/
__global__
void buildClusterEdgeAdjacency(
    uint32_t* paaiRetAdjacentEdgeClusters,
    uint32_t* paaiRetNumAdjacentEdgeClusters,
    float* pafTotalClusterVertexPositions,
    uint32_t* paaiVertexPositionIndices,
    uint32_t* paiNumVertexPositionComponents,
    uint32_t* paiNumVertexPositionIndices,
    uint32_t* paiVertexPositionComponentOffsets,
    uint32_t* paiVertexPositionIndexOffsets,
    uint32_t* paiDistanceSortedCluster,
    uint32_t iNumClusters)
{
    uint32_t iCluster = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iCluster >= iNumClusters)
    {
        return;
    }

//printf("start buildClusterEdgeAdjacency processing cluster %d of total %d clusters\n", iCluster, iNumClusters);

    uint32_t iVertexPositionComponentOffset = paiVertexPositionComponentOffsets[iCluster];
    uint32_t iVertexPositionIndexOffset = paiVertexPositionIndexOffsets[iCluster];

    uint32_t iNumTri = paiNumVertexPositionIndices[iCluster];
    for(uint32_t iTri = 0; iTri < iNumTri; iTri += 3)
    {
//        uint32_t iTriComponent = iVertexPositionIndexOffset + iTri * 3;
        //for(uint32_t iCheckCluster = iCluster + 1; iCheckCluster < iNumClusters; iCheckCluster++)

        for(uint32_t iCheckCluster = 0; iCheckCluster < 10; iCheckCluster++)
        {
            uint32_t iCheckClusterID = paiDistanceSortedCluster[iCluster * iNumClusters + iCheckCluster];
            if(iCheckClusterID == iCluster)
            {
                continue;
            }

            uint32_t iCheckVertexPositionComponentOffset = paiVertexPositionComponentOffsets[iCheckClusterID];
            uint32_t iCheckVertexPositionIndexOffset = paiVertexPositionIndexOffsets[iCheckClusterID];

            uint32_t iNumCheckTri = paiNumVertexPositionIndices[iCheckClusterID];
            for(uint32_t iCheckTri = 0; iCheckTri < iNumCheckTri; iCheckTri += 3)
            {
                // check same positions for the triangles
//                uint32_t iCheckTriComponent = iCheckTri * 3;
                //uint32_t aiSamePos[3];
                //uint32_t aiPos[3];
                uint32_t iNumSamePos = 0;
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = paaiVertexPositionIndices[iVertexPositionIndexOffset + iTri];
                    uint32_t iCheckPos = paaiVertexPositionIndices[iCheckVertexPositionIndexOffset + iCheckTri];

                    float fX = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + (iPos + i) * 3];
                    float fY = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + (iPos + i) * 3 + 1];
                    float fZ = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + (iPos + i) * 3 + 2];
                    
//if(iTri == 3)
//{
//    printf("cluster %d vertex %d (%.4f, %.4f, %.4f) tri: %d local index: %d : total index: %d\n",
//        iCluster,
//        iPos,
//        fX, fY, fZ,
//        iTri,
//        iPos,
//        iVertexPositionComponentOffset + iPos * 3);
//}
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        float fCheckX = pafTotalClusterVertexPositions[iCheckVertexPositionComponentOffset + (iCheckPos + j) * 3];
                        float fCheckY = pafTotalClusterVertexPositions[iCheckVertexPositionComponentOffset + (iCheckPos + j) * 3 + 1];
                        float fCheckZ = pafTotalClusterVertexPositions[iCheckVertexPositionComponentOffset + (iCheckPos + j) * 3 + 2];

                        float fLength = _length(fX - fCheckX, fY - fCheckY, fZ - fCheckZ);
                        if(fLength <= 1.0e-7f)
                        {
                            //aiPos[i] = iPos + i;
                            //aiSamePos[i] = iCheckPos + j;
                            ++iNumSamePos;
                        }
                    }
                }

                // check if 2 or more positions are the same, ie. same edge
                if(iNumSamePos >= 2)
                {
                    uint32_t iNumAdjacentEdgeClusters = atomicAdd(&paaiRetNumAdjacentEdgeClusters[iCluster], 1);
                    uint32_t iNumCheckAdjacentEdgeClusters = atomicAdd(&paaiRetNumAdjacentEdgeClusters[iCheckClusterID], 1);

                    paaiRetAdjacentEdgeClusters[iCluster * iNumClusters + iCheckClusterID] = 1;
                    paaiRetAdjacentEdgeClusters[iCheckClusterID * iNumClusters + iCluster] = 1;

//printf("cluster %d is adjacent to cluster %d vertex id (%d, %d, %d) (%d, %d, %d)\n",
//    iCluster,
//    iCheckCluster,
//    aiPos[0],
//    aiPos[1],
//    aiPos[2],
//    aiSamePos[0],
//    aiSamePos[1],
//    aiSamePos[2]);

                    break;
                }

            }   // for check tri

        }   // for check cluster

    }   // for tri

//printf("end buildClusterEdgeAdjacency processing cluster %d of total %d clusters\n", iCluster, iNumClusters);
    
}

/*
**
*/
__global__
void getClusterBounds(
    float* pafRetMinMaxCenterPositions,
    float* pafTotalClusterVertexPositions,
    uint32_t* paiVertexPositionComponentOffsets,
    uint32_t* paiNumVertexPositionComponents,
    uint32_t iNumClusters)
{
    uint32_t iCluster = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iCluster >= iNumClusters)
    {
        return;
    }

    float fLargestX = -1.0e+10f, fLargestY = -1.0e+10f, fLargestZ = -1.0e+10f;
    float fSmallestX = 1.0e+10f, fSmallestY = 1.0e+10f, fSmallestZ = 1.0e+10f;

    uint32_t iVertexPositionComponentOffset = paiVertexPositionComponentOffsets[iCluster];
    for(uint32_t i = 0; i < paiNumVertexPositionComponents[iCluster]; i += 3)
    {
        float fX = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + i];
        float fY = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + i + 1];
        float fZ = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + i + 2];

        fLargestX = max(fLargestX, fX);
        fLargestY = max(fLargestY, fY);
        fLargestZ = max(fLargestZ, fZ);

        fSmallestX = min(fSmallestX, fX);
        fSmallestY = min(fSmallestY, fY);
        fSmallestZ = min(fSmallestZ, fZ);
    }
    
    uint32_t iIndex = iCluster * 10;
    pafRetMinMaxCenterPositions[iIndex]     = fSmallestX;
    pafRetMinMaxCenterPositions[iIndex + 1] = fSmallestY;
    pafRetMinMaxCenterPositions[iIndex + 2] = fSmallestZ;

    pafRetMinMaxCenterPositions[iIndex + 3] = fLargestX;
    pafRetMinMaxCenterPositions[iIndex + 4] = fLargestY;
    pafRetMinMaxCenterPositions[iIndex + 5] = fLargestZ;

    pafRetMinMaxCenterPositions[iIndex + 6] = (fLargestX + fSmallestX) * 0.5f;
    pafRetMinMaxCenterPositions[iIndex + 7] = (fLargestY + fSmallestY) * 0.5f;
    pafRetMinMaxCenterPositions[iIndex + 8] = (fLargestZ + fSmallestZ) * 0.5f;

    pafRetMinMaxCenterPositions[iIndex + 9] = _length(fLargestX - fSmallestX, fLargestY - fSmallestY, fLargestZ - fSmallestZ) * 0.5f;
}

/*
**
*/
__global__
void getClusterDistances(
    float* pafRetDistances,
    float* pafClusterCenters,
    uint32_t iNumClusters)
{
    uint32_t iCluster = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iCluster >= iNumClusters)
    {
        return;
    }

    float fX = pafClusterCenters[iCluster * 3];
    float fY = pafClusterCenters[iCluster * 3 + 1];
    float fZ = pafClusterCenters[iCluster * 3 + 2];

    for(uint32_t iCheckCluster = 0; iCheckCluster < iNumClusters; iCheckCluster++)
    {
        if(iCheckCluster == iCluster)
        {
            continue;
        }

        float fCheckX = pafClusterCenters[iCheckCluster * 3];
        float fCheckY = pafClusterCenters[iCheckCluster * 3 + 1];
        float fCheckZ = pafClusterCenters[iCheckCluster * 3 + 2];
        float fLength = _length(fX - fCheckX, fY - fCheckY, fZ - fCheckZ);

        uint32_t iIndex = iCluster * iNumClusters + iCheckCluster;
        pafRetDistances[iIndex] = fLength;
    }
}

#include "float3_lib.cuh"

/*
**
*/
__global__
void projectVertices(
    float* afRetProjectedPositions,
    float* afTriangleVertexPositions0,
    float* afTriangleVertexPositions1,
    float* afIntersectInfo,
    uint32_t iNumVertices0,
    uint32_t iNumVertices1)
{
    uint32_t iNumTriangles = iNumVertices0 / 3;
    uint32_t iTriangle = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iTriangle >= iNumTriangles)
    {
        return;
    }

    float3 pos0 = make_float3(
        afTriangleVertexPositions0[iTriangle * 9], 
        afTriangleVertexPositions0[iTriangle * 9 + 1], 
        afTriangleVertexPositions0[iTriangle * 9 + 2]);
    float3 pos1 = make_float3(
        afTriangleVertexPositions0[iTriangle * 9 + 3], 
        afTriangleVertexPositions0[iTriangle * 9 + 4], 
        afTriangleVertexPositions0[iTriangle * 9 + 5]);
    float3 pos2 = make_float3(
        afTriangleVertexPositions0[iTriangle * 9 + 6], 
        afTriangleVertexPositions0[iTriangle * 9 + 7], 
        afTriangleVertexPositions0[iTriangle * 9 + 8]);
    
    float3 diff0 = normalize(pos2 - pos0);
    float3 diff1 = normalize(pos1 - pos0);

    float3 faceNormal = cross(diff0, diff1);

    uint32_t iNumCheckTriangles = iNumVertices1 / 3;
    for(uint32_t i = 0; i < 3; i++)
    {
        float3 ret = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);

        uint32_t iTriangleVertexIndex = iTriangle * 9 + i * 3;
        float3 pos = make_float3(
            afTriangleVertexPositions0[iTriangleVertexIndex],
            afTriangleVertexPositions0[iTriangleVertexIndex + 1],
            afTriangleVertexPositions0[iTriangleVertexIndex + 2]);
        
        float fT = FLT_MAX;
        float fIntersectTriangle = FLT_MAX;
        for(uint32_t iCheckTriangle = 0; iCheckTriangle < iNumCheckTriangles; iCheckTriangle++)
        {
            float3 checkPos0 = make_float3(
                afTriangleVertexPositions1[iCheckTriangle * 9],
                afTriangleVertexPositions1[iCheckTriangle * 9 + 1],
                afTriangleVertexPositions1[iCheckTriangle * 9 + 2]);
            float3 checkPos1 = make_float3(
                afTriangleVertexPositions1[iCheckTriangle * 9 + 3],
                afTriangleVertexPositions1[iCheckTriangle * 9 + 4],
                afTriangleVertexPositions1[iCheckTriangle * 9 + 5]);
            float3 checkPos2 = make_float3(
                afTriangleVertexPositions1[iCheckTriangle * 9 + 6],
                afTriangleVertexPositions1[iCheckTriangle * 9 + 7],
                afTriangleVertexPositions1[iCheckTriangle * 9 + 8]);

            float3 diff0 = normalize(checkPos2 - checkPos0);
            float3 diff1 = normalize(checkPos1 - checkPos0);

            // normal of the plane
            float3 checkNormal = cross(diff0, diff1);
            float fPlaneD = dot(checkNormal, checkPos0) * -1.0f;

            float3 pt1 = pos + faceNormal * 100.0f;
            fT = rayPlaneIntersection(
                pos,
                pt1,
                checkNormal,
                fPlaneD);

            // account for forward and backward direction
            if(fT >= -1.0f && fT <= 1.0f)
            {
                float3 intersectionPt = pos + (pt1 - pos) * fT;
                float3 barycentricPt = barycentric(intersectionPt, checkPos0, checkPos1, checkPos2);
                bool bProjected =
                    (barycentricPt.x >= -0.01f && barycentricPt.x <= 1.01f &&
                        barycentricPt.y >= -0.01f && barycentricPt.y <= 1.01f &&
                        barycentricPt.z >= -0.01f && barycentricPt.z <= 1.01f);
                if(bProjected)
                {
                    ret = checkPos0 * barycentricPt.x + checkPos1 * barycentricPt.y + checkPos2 * barycentricPt.z;
                    fIntersectTriangle = float(iCheckTriangle);
                }
            }

            if(ret.x != FLT_MAX)
            {
                break;
            }
        }

        if(ret.x == FLT_MAX)
        {
            ret = pos + faceNormal * 10.0f;
            afRetProjectedPositions[iTriangleVertexIndex] = ret.x;
            afRetProjectedPositions[iTriangleVertexIndex+1] = ret.y;
            afRetProjectedPositions[iTriangleVertexIndex+2] = ret.z;
        }
        else
        {
            afRetProjectedPositions[iTriangleVertexIndex] = ret.x;
            afRetProjectedPositions[iTriangleVertexIndex + 1] = ret.y;
            afRetProjectedPositions[iTriangleVertexIndex + 2] = ret.z;
        }

        afIntersectInfo[iTriangle * 3 * 2 + i * 2] = fT;
        afIntersectInfo[iTriangle * 3 * 2 + i * 2 + 1] = fIntersectTriangle;

    }   // for i = 0 to 3
}

/*
**
*/
__global__
void buildClusterEdgeAdjacency2(
    uint32_t* paaiRetAdjacentEdgeClusters,
    uint32_t* paaiRetNumAdjacentEdgeClusters,
    float* pafTotalClusterVertexPositions,
    uint32_t* paaiVertexPositionIndices,
    uint32_t* paiNumVertexPositionComponents,
    uint32_t* paiNumVertexPositionIndices,
    uint32_t* paiVertexPositionComponentOffsets,
    uint32_t* paiVertexPositionIndexOffsets,
    uint32_t* paiDistanceSortedCluster,
    uint32_t iNumClusters)
{
    uint32_t iCluster = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iCluster >= iNumClusters)
    {
        return;
    }

    uint32_t iVertexPositionComponentOffset = paiVertexPositionComponentOffsets[iCluster];
    uint32_t iVertexPositionIndexOffset = paiVertexPositionIndexOffsets[iCluster];

    uint32_t iNumTri = paiNumVertexPositionIndices[iCluster];
    for(uint32_t iTri = 0; iTri < iNumTri; iTri += 3)
    {
        for(uint32_t iCheckCluster = 0; iCheckCluster < 10; iCheckCluster++)
        {
            uint32_t iCheckClusterID = paiDistanceSortedCluster[iCluster * iNumClusters + iCheckCluster];
            if(iCheckClusterID == iCluster)
            {
                continue;
            }

            uint32_t iCheckVertexPositionComponentOffset = paiVertexPositionComponentOffsets[iCheckClusterID];
            uint32_t iCheckVertexPositionIndexOffset = paiVertexPositionIndexOffsets[iCheckClusterID];

            uint32_t iNumSamePos = 0;
            uint32_t iNumCheckTri = paiNumVertexPositionIndices[iCheckClusterID];
            for(uint32_t iCheckTri = 0; iCheckTri < iNumCheckTri; iCheckTri += 3)
            {
                // check same positions for the triangles
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = paaiVertexPositionIndices[iVertexPositionIndexOffset + iTri];
                    uint32_t iCheckPos = paaiVertexPositionIndices[iCheckVertexPositionIndexOffset + iCheckTri];

                    float fX = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + (iPos + i) * 3];
                    float fY = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + (iPos + i) * 3 + 1];
                    float fZ = pafTotalClusterVertexPositions[iVertexPositionComponentOffset + (iPos + i) * 3 + 2];

                    bool bSame = false;
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        float fCheckX = pafTotalClusterVertexPositions[iCheckVertexPositionComponentOffset + (iCheckPos + j) * 3];
                        float fCheckY = pafTotalClusterVertexPositions[iCheckVertexPositionComponentOffset + (iCheckPos + j) * 3 + 1];
                        float fCheckZ = pafTotalClusterVertexPositions[iCheckVertexPositionComponentOffset + (iCheckPos + j) * 3 + 2];

                        float fLength = _length(fX - fCheckX, fY - fCheckY, fZ - fCheckZ);
                        if(fLength <= 1.0e-8f)
                        {
                            ++iNumSamePos;
                            bSame = true;
                            break;
                        }
                    }

                    if(bSame)
                    {
                        break;
                    }
                }

            }   // for check tri

            if(iNumSamePos > 0)
            {
                paaiRetAdjacentEdgeClusters[iCluster * iNumClusters + iCheckClusterID] = iNumSamePos;
            }

        }   // for check cluster

    }   // for tri
}

/*
**
*/
__global__
void checkClusterGroupAdjacency2(
    uint32_t* aiAdjacentClusterGroupVertexIndices,
    uint32_t* aiNumAdjacentClusterGroupVertices,
    float3 const* afTotalClusterGroupVertexPositionComponents,
    uint32_t const* aiNumVertexPositions,
    uint32_t const* aiClusterGroupVertexPositionArrayByteOffsets,
    uint32_t iNumTotalClusterGroups)
{
    uint32_t iClusterGroup = blockIdx.x * WORKGROUP_SIZE + threadIdx.x;
    if(iClusterGroup >= iNumTotalClusterGroups)
    {
        return;
    }
    
    uint32_t iVertexPositionByteOffset = aiClusterGroupVertexPositionArrayByteOffsets[iClusterGroup];
    float3 const* aClusterVertexPositions = (float3 const*)((char*)afTotalClusterGroupVertexPositionComponents + iVertexPositionByteOffset);
    uint32_t iNumVertexPositions = aiNumVertexPositions[iClusterGroup];
    
    uint32_t iNumAdjacentClusterGroupVertices = 0;
    for(uint32_t iVertex = 0; iVertex < iNumVertexPositions; iVertex++)
    {
        float3 const& position = aClusterVertexPositions[iVertex];
        bool bHasAdjacentVertex = false;
        for(uint32_t iCheckClusterGroup = 0; iCheckClusterGroup < iNumTotalClusterGroups; iCheckClusterGroup++)
        {
            if(iCheckClusterGroup == iClusterGroup)
            {
                continue;
            }

            uint32_t iCheckVertexPositionByteOffset = aiClusterGroupVertexPositionArrayByteOffsets[iCheckClusterGroup];
            float3 const* aCheckClusterVertexPositions = (float3 const*)((char*)afTotalClusterGroupVertexPositionComponents + iCheckVertexPositionByteOffset);
            uint32_t iNumCheckVertexPositions = aiNumVertexPositions[iCheckClusterGroup];

            for(uint32_t iCheckVertex = 0; iCheckVertex < iNumCheckVertexPositions; iCheckVertex++)
            {
                float3 const& checkPosition = aCheckClusterVertexPositions[iCheckVertex];
                float3 diff = position - checkPosition;

                if(lengthSquared(diff) <= 1.0e-10f)
                {
if(iNumAdjacentClusterGroupVertices >= MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP)
{
    printf("wtf\n");
}
                    
                    uint32_t iIndex = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP + iNumAdjacentClusterGroupVertices;
                    aiAdjacentClusterGroupVertexIndices[iIndex] = iVertex;

                    ++iNumAdjacentClusterGroupVertices;
                    bHasAdjacentVertex = true;

                    break;
                }

            }   // for check vertex component = 0 to num vertex components

            if(bHasAdjacentVertex)
            {
                break;
            }

        }   // for check cluster group = 0 to num cluster groups 
    
    }   // for vertex = 0 to num vertices in cluster group

    aiNumAdjacentClusterGroupVertices[iClusterGroup] = iNumAdjacentClusterGroupVertices;
}

#undef uint32_t
#undef int32_t

#include "test.h"
#include "LogPrint.h"

#include <chrono>
#include <assert.h>

/*
**
*/
void checkClusterGroupBoundaryVerticesCUDA(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices, 
    std::vector<std::vector<vec3>> const& aaClusterGroupVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTrianglePositionIndices)
{
DEBUG_PRINTF("*** checkClusterGroupBoundaryVerticesCUDA ***\n");
auto start = std::chrono::high_resolution_clock::now();

    PrintOptions printOptions;
    printOptions.mbDisplayTime = false;
    setPrintOptions(printOptions);

    // prepare data to be passed into device, getting data offsets and the number of vertex position for clusters and number of triangle indices
    uint32_t iCurrVertexPositionDataOffset = 0;
    uint32_t iCurrTriangleIndexDataOffset = 0;
    uint32_t iNumClusterGroups = static_cast<uint32_t>(aaClusterGroupVertexPositions.size());
    uint32_t iNumTotalTriangleIndices = 0;
    std::vector<uint32_t> aiVertexPositionDataOffsets(iNumClusterGroups);
    std::vector<uint32_t> aiTriangleIndexOffsets(iNumClusterGroups);
    std::vector<uint32_t> aiNumVertexPositions(iNumClusterGroups);
    std::vector<uint32_t> aiNumTriangleIndices(iNumClusterGroups);
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        aiVertexPositionDataOffsets[iClusterGroup] = iCurrVertexPositionDataOffset;
        aiTriangleIndexOffsets[iClusterGroup] = iCurrTriangleIndexDataOffset;

        aiNumVertexPositions[iClusterGroup] = static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size());
        aiNumTriangleIndices[iClusterGroup] = static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size());

        //DEBUG_PRINTF("position offset cluster group %d : %d\n", iClusterGroup, iCurrVertexPositionDataOffset);
        //DEBUG_PRINTF("triangle index offset cluster group %d : %d\n", iClusterGroup, iCurrTriangleIndexDataOffset);

        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * 3);
        iCurrTriangleIndexDataOffset += static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size());

        iNumTotalTriangleIndices += aiNumTriangleIndices[iClusterGroup];
    }
    
    //DEBUG_PRINTF("total position offset : %d\n", iCurrVertexPositionDataOffset);
    //DEBUG_PRINTF("total triangle index offset : %d\n", iCurrVertexPositionDataOffset);
    //DEBUG_PRINTF("total num triangle indices: %d\n", iNumTotalTriangleIndices);

    // allocate device memory
    float* pafTotalClusterGroupVertexPositions;
    cudaMalloc(&pafTotalClusterGroupVertexPositions, iCurrVertexPositionDataOffset * sizeof(float));
    
    unsigned int* paiTotalClusterGroupTriangleIndices;
    cudaMalloc(&paiTotalClusterGroupTriangleIndices, iCurrTriangleIndexDataOffset * sizeof(int));

    unsigned int* paiNumClusterGroupVertexPositions;
    cudaMalloc(&paiNumClusterGroupVertexPositions, iNumClusterGroups * sizeof(int));

    unsigned int* paiNumClusterGroupTriangleIndices;
    cudaMalloc(&paiNumClusterGroupTriangleIndices, iNumClusterGroups * sizeof(int));

    unsigned int* paiClusterGroupVertexPositionDataOffsets;
    cudaMalloc(&paiClusterGroupVertexPositionDataOffsets, iNumClusterGroups * sizeof(int));

    unsigned int* paiClusterGroupTriangleIndexDataOffsets;
    cudaMalloc(&paiClusterGroupTriangleIndexDataOffsets, iNumClusterGroups * sizeof(int));

    unsigned int* aiRetNumClusterGroupBoundaryVertices;
    cudaMalloc(&aiRetNumClusterGroupBoundaryVertices, iNumClusterGroups * sizeof(int));

    unsigned int* aiRetClusterGroupBoundaryVertexIndices;
    cudaMalloc(&aiRetClusterGroupBoundaryVertexIndices, iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(int));

    // copy over data (vertex positions, triangle indices) for all the cluster groups
    iCurrVertexPositionDataOffset = 0;
    iCurrTriangleIndexDataOffset = 0;
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        cudaMemcpy(
            pafTotalClusterGroupVertexPositions + iCurrVertexPositionDataOffset,
            aaClusterGroupVertexPositions[iClusterGroup].data(),
            aaClusterGroupVertexPositions[iClusterGroup].size() * sizeof(float3),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            paiTotalClusterGroupTriangleIndices + iCurrTriangleIndexDataOffset,
            aaiClusterGroupTrianglePositionIndices[iClusterGroup].data(),
            aaiClusterGroupTrianglePositionIndices[iClusterGroup].size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * 3);
        iCurrTriangleIndexDataOffset += static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size());
    }

    // vertex positions
    cudaMemcpy(
        paiNumClusterGroupVertexPositions,
        aiNumVertexPositions.data(),
        aiNumVertexPositions.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    // triangle indices
    cudaMemcpy(
        paiNumClusterGroupTriangleIndices,
        aiNumTriangleIndices.data(),
        aiNumTriangleIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    // vertex position offset for all the cluster groups
    cudaMemcpy(
        paiClusterGroupVertexPositionDataOffsets,
        aiVertexPositionDataOffsets.data(),
        aiVertexPositionDataOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    // triangle index offsets for all the cluster groups
    cudaMemcpy(
        paiClusterGroupTriangleIndexDataOffsets,
        aiTriangleIndexOffsets.data(),
        aiTriangleIndexOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    // run on device
    //uint32_t iNumBlocks = static_cast<uint32_t>(std::max(1, int32_t(iNumClusterGroups) / 256));
    //checkClusterGroupBoundaryVertices<<<iNumBlocks, 256>>>(
    //    aiRetClusterGroupBoundaryVertexIndices,
    //    aiRetNumClusterGroupBoundaryVertices,
    //    iNumClusterGroups,
    //    pafTotalClusterGroupVertexPositions,
    //    paiTotalClusterGroupTriangleIndices,
    //    paiNumClusterGroupVertexPositions,
    //    paiNumClusterGroupTriangleIndices,
    //    paiClusterGroupVertexPositionDataOffsets,
    //    paiClusterGroupTriangleIndexDataOffsets);

    // initialize
    std::vector<uint32_t> aiDefaultValues(iNumClusterGroups);
    cudaMemcpy(
        aiRetNumClusterGroupBoundaryVertices,
        aiDefaultValues.data(),
        iNumClusterGroups * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(float(iNumTotalTriangleIndices / 3) / float(WORKGROUP_SIZE)));
    checkClusterGroupBoundaryVertices2<<<iNumBlocks, WORKGROUP_SIZE>>>(
        aiRetClusterGroupBoundaryVertexIndices,
        aiRetNumClusterGroupBoundaryVertices,
        iNumClusterGroups,
        pafTotalClusterGroupVertexPositions,
        paiTotalClusterGroupTriangleIndices,
        paiNumClusterGroupVertexPositions,
        paiNumClusterGroupTriangleIndices,
        paiClusterGroupVertexPositionDataOffsets,
        paiClusterGroupTriangleIndexDataOffsets,
        iNumTotalTriangleIndices);

    // copy output
    std::vector<uint32_t> aiRetClusterGroupBoundaryVertexIndicesCPU(iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP);
    std::vector<uint32_t> aiRetNumClusterGroupBoundaryVerticesCPU(iNumClusterGroups);
    cudaMemcpy(
        aiRetClusterGroupBoundaryVertexIndicesCPU.data(), 
        aiRetClusterGroupBoundaryVertexIndices, 
        iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        aiRetNumClusterGroupBoundaryVerticesCPU.data(), 
        aiRetNumClusterGroupBoundaryVertices, 
        iNumClusterGroups * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);

    cudaFree(pafTotalClusterGroupVertexPositions);
    cudaFree(paiTotalClusterGroupTriangleIndices);
    cudaFree(paiNumClusterGroupVertexPositions);
    cudaFree(paiNumClusterGroupTriangleIndices);
    cudaFree(paiClusterGroupVertexPositionDataOffsets);
    cudaFree(paiClusterGroupTriangleIndexDataOffsets);
    cudaFree(aiRetNumClusterGroupBoundaryVertices);
    cudaFree(aiRetClusterGroupBoundaryVertexIndices);

    // output cluster group boundary vertex indices
    aaiClusterGroupBoundaryVertices.resize(iNumClusterGroups);
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        uint32_t iDataOffset = iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP;
        for(uint32_t i = 0; i < aiRetNumClusterGroupBoundaryVerticesCPU[iClusterGroup]; i++)
        {
            uint32_t iPos = aiRetClusterGroupBoundaryVertexIndicesCPU[iDataOffset + i];
            auto iter = std::find(
                aaiClusterGroupBoundaryVertices[iClusterGroup].begin(),
                aaiClusterGroupBoundaryVertices[iClusterGroup].end(),
                iPos);
            if(iter == aaiClusterGroupBoundaryVertices[iClusterGroup].end())
            {
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos);

                //vec3 const& pos = aaClusterGroupVertexPositions[iClusterGroup][iPos];
                //DEBUG_PRINTF("\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                //    pos.x, pos.y, pos.z);
            }
        }

        std::sort(aaiClusterGroupBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupBoundaryVertices[iClusterGroup].end());
    }

    printOptions.mbDisplayTime = true;
    setPrintOptions(printOptions);

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("*** took %lld seconds for checkClusterGroupBoundaryVerticesCUDA to complete ***\n", iSeconds);

}

/*
**
*/
void buildClusterAdjacencyCUDA(
    std::vector<std::vector<uint32_t>>& aaiNumAdjacentVertices,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    bool bOnlyEdgeAdjacent)
{
DEBUG_PRINTF("*** start buildClusterAdjacencyCUDA ***\n");
auto start = std::chrono::high_resolution_clock::now();

    uint32_t iCurrVertexPositionDataOffset = 0;
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());
    std::vector<uint32_t> aiNumVertexPositionComponents(iNumClusters);
    std::vector<uint32_t> aiVertexPositionComponentArrayOffsets(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aiNumVertexPositionComponents[iCluster] = static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
        aiVertexPositionComponentArrayOffsets[iCluster] = iCurrVertexPositionDataOffset;
        
        //DEBUG_PRINTF("cluster %d num vertices: %d num vertex components: %d data offset: %d\n",
        //    iCluster,
        //    aaVertexPositions[iCluster].size(),
        //    aiNumVertexPositionComponents[iCluster],
        //    iCurrVertexPositionDataOffset);
        
        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
    }

    // allocate device memory
    float* pafTotalClusterVertexPositions;
    uint32_t iArrayIndexOffset = 0;
    cudaMalloc(&pafTotalClusterVertexPositions, iCurrVertexPositionDataOffset * sizeof(float));
    
    uint32_t* paiNumAdjacentClusterVertices;
    cudaMalloc(&paiNumAdjacentClusterVertices, iNumClusters * iNumClusters * sizeof(int));
    
    // initialize
    //std::vector<uint32_t> aiDefaultValues(iNumClusters * iNumClusters);
    //memset(aiDefaultValues.data(), 0, iNumClusters * iNumClusters * sizeof(int));
    //cudaMemcpy(
    //    paiNumAdjacentClusterVertices,
    //    aiDefaultValues.data(),
    //    iNumClusters * iNumClusters * sizeof(int),
    //    cudaMemcpyHostToDevice);

    cudaMemset(
        paiNumAdjacentClusterVertices,
        0,
        iNumClusters * iNumClusters * sizeof(int));

    uint32_t* paiNumVertexPositionComponents;
    cudaMalloc(&paiNumVertexPositionComponents, iNumClusters * sizeof(int));
    
    uint32_t* paiNumVertexPositionComponentOffsets;
    cudaMalloc(&paiNumVertexPositionComponentOffsets, iNumClusters * sizeof(int));

    // copy to device memory
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            pafTotalClusterVertexPositions + iArrayIndexOffset,
            aaVertexPositions[iCluster].data(),
            aaVertexPositions[iCluster].size() * sizeof(float) * 3,
            cudaMemcpyHostToDevice);
        iArrayIndexOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
    }

    cudaMemcpy(
        paiNumVertexPositionComponents,
        aiNumVertexPositionComponents.data(),
        aiNumVertexPositionComponents.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        paiNumVertexPositionComponentOffsets,
        aiVertexPositionComponentArrayOffsets.data(),
        aiVertexPositionComponentArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);


    //void checkClusterAdjacency(
//    float* afTotalClusterVertexPositionComponents,
//    uint32_t * aiNumAdjacentVertices,
//    uint32_t * aiNumVertexPositionComponents,
//    uint32_t * aiClusterVertexPositionComponentOffsets,
//    uint32_t iNumTotalClusters)

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumClusters) / float(WORKGROUP_SIZE)));
    checkClusterAdjacency<<<iNumBlocks, WORKGROUP_SIZE>>>(
        paiNumAdjacentClusterVertices,
        pafTotalClusterVertexPositions,
        paiNumVertexPositionComponents,
        paiNumVertexPositionComponentOffsets,
        iNumClusters,
        bOnlyEdgeAdjacent);

    std::vector<uint32_t> aiNumAdjacentClusterVerticesCPU(iNumClusters * iNumClusters);
    cudaMemcpy(
        aiNumAdjacentClusterVerticesCPU.data(),
        paiNumAdjacentClusterVertices,
        iNumClusters * iNumClusters * sizeof(int),
        cudaMemcpyDeviceToHost);

    aaiNumAdjacentVertices.resize(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aaiNumAdjacentVertices[iCluster].resize(iNumClusters);
        for(uint32_t iCheckCluster = 0; iCheckCluster < iNumClusters; iCheckCluster++)
        {
            uint32_t iIndex = iCluster * iNumClusters + iCheckCluster;
            aaiNumAdjacentVertices[iCluster][iCheckCluster] = aiNumAdjacentClusterVerticesCPU[iIndex];
        }
    }

    cudaFree(paiNumVertexPositionComponentOffsets);
    cudaFree(paiNumVertexPositionComponents);
    cudaFree(pafTotalClusterVertexPositions);
    cudaFree(paiNumAdjacentClusterVertices);

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("*** took %lld seconds for buildClusterAdjacencyCUDA to finish ***\n",
    iSeconds);

}

/*
**
*/
void getClusterGroupBoundaryVerticesCUDA2(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<vec3>> const& aaClusterGroupVertexPositions)
{
    DEBUG_PRINTF("*** start getClusterGroupBoundaryVerticesCUDA2 ***\n");
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t iNumClusterGroups = static_cast<uint32_t>(aaClusterGroupVertexPositions.size());

    uint32_t iCurrVertexPositionDataOffset = 0;
    std::vector<uint32_t> aiNumVertexPositions(iNumClusterGroups);

    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        aiNumVertexPositions[iClusterGroup] = static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size());
        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * sizeof(float3));
    }

    // copy to device memory
    uint32_t* paiRetClusterGroupAdjacentVertexIndices = nullptr;
    cudaMalloc(
        &paiRetClusterGroupAdjacentVertexIndices,
        iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(uint32_t));
    
    uint32_t* paiRetNumAdjacentClusterGroupVertices = nullptr;
    cudaMalloc(
        &paiRetNumAdjacentClusterGroupVertices,
        iNumClusterGroups * sizeof(uint32_t));
    
    uint32_t iTotalVertexPositionSize = 0;
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        iTotalVertexPositionSize += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * sizeof(float3));
    }

    float3* paClusterGroupVertexPositions = nullptr;
    cudaMalloc(
        &paClusterGroupVertexPositions,
        iTotalVertexPositionSize);
    uint32_t iByteOffset = 0;
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        cudaMemcpy(
            (char *)paClusterGroupVertexPositions + iByteOffset,
            aaClusterGroupVertexPositions[iClusterGroup].data(),
            aaClusterGroupVertexPositions[iClusterGroup].size() * sizeof(float3),
            cudaMemcpyHostToDevice);
        iByteOffset += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * sizeof(float3));
    }

    uint32_t* paiNumClusterGroupVertexPositions = nullptr;
    cudaMalloc(
        &paiNumClusterGroupVertexPositions,
        iNumClusterGroups * sizeof(uint32_t));
    iByteOffset = 0;
    for(uint32_t i = 0; i < iNumClusterGroups; i++)
    {
        uint32_t iNumVertexPositions = static_cast<uint32_t>(aaClusterGroupVertexPositions[i].size());
        cudaMemcpy(
            paiNumClusterGroupVertexPositions + i,
            &iNumVertexPositions,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    }

    uint32_t* paiClusterGroupVertexPositionArrayByteOffsets = nullptr;
    cudaMalloc(
        &paiClusterGroupVertexPositionArrayByteOffsets,
        iNumClusterGroups * sizeof(uint32_t));
    iByteOffset = 0;
    for(uint32_t i = 0; i < iNumClusterGroups; i++)
    {
        cudaMemcpy(
            paiClusterGroupVertexPositionArrayByteOffsets + i,
            &iByteOffset,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);
        uint32_t iDataSize = static_cast<uint32_t>(aaClusterGroupVertexPositions[i].size() * sizeof(float3));
        iByteOffset += iDataSize;
    }

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumClusterGroups) / float(WORKGROUP_SIZE)));
    checkClusterGroupAdjacency2<<<iNumBlocks, WORKGROUP_SIZE>>>(
        paiRetClusterGroupAdjacentVertexIndices,
        paiRetNumAdjacentClusterGroupVertices,
        paClusterGroupVertexPositions,
        paiNumClusterGroupVertexPositions,
        paiClusterGroupVertexPositionArrayByteOffsets,
        iNumClusterGroups);

    std::vector<uint32_t> aiTotalClusterGroupAdjacentVerticesCPU(MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * iNumClusterGroups);
    cudaMemcpy(
        aiTotalClusterGroupAdjacentVerticesCPU.data(),
        paiRetClusterGroupAdjacentVertexIndices,
        iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);

    std::vector<uint32_t> aiNumAdjacentVertices(iNumClusterGroups);
    cudaMemcpy(
        aiNumAdjacentVertices.data(),
        paiRetNumAdjacentClusterGroupVertices,
        iNumClusterGroups * sizeof(int),
        cudaMemcpyDeviceToHost);
    
    iByteOffset = 0;
    aaiClusterGroupBoundaryVertices.resize(iNumClusterGroups);
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        uint32_t iNumBoundaryVertices = aiNumAdjacentVertices[iClusterGroup];
        aaiClusterGroupBoundaryVertices[iClusterGroup].resize(iNumBoundaryVertices);
        memcpy(
            aaiClusterGroupBoundaryVertices[iClusterGroup].data(),
            (char*)aiTotalClusterGroupAdjacentVerticesCPU.data() + iByteOffset,
            iNumBoundaryVertices * sizeof(uint32_t));

        iByteOffset += static_cast<uint32_t>(MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(uint32_t));
    }

    cudaFree(paiRetClusterGroupAdjacentVertexIndices);
    cudaFree(paiRetNumAdjacentClusterGroupVertices);
    cudaFree(paClusterGroupVertexPositions);
    cudaFree(paiNumClusterGroupVertexPositions);
}

/*
**
*/
void getClusterGroupBoundaryVerticesCUDA(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<vec3>> const& aaClusterGroupVertexPositions)
{
DEBUG_PRINTF("*** start getClusterGroupBoundaryVerticesCUDA ***\n");
auto start = std::chrono::high_resolution_clock::now();

    uint32_t iNumClusterGroups = static_cast<uint32_t>(aaClusterGroupVertexPositions.size());

    uint32_t iCurrVertexPositionDataOffset = 0;
    std::vector<uint32_t> aiNumVertexPositionComponents(iNumClusterGroups);
    std::vector<uint32_t> aiVertexPositionComponentArrayOffsets(iNumClusterGroups);
    
    uint32_t iNumTotalVertices = 0;
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        aiNumVertexPositionComponents[iClusterGroup] = static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * 3);
        aiVertexPositionComponentArrayOffsets[iClusterGroup] = iCurrVertexPositionDataOffset;

        //DEBUG_PRINTF("cluster group %d num vertices: %d num vertex components: %d data offset: %d\n",
        //    iClusterGroup,
        //    aaClusterGroupVertexPositions[iClusterGroup].size(),
        //    aiNumVertexPositionComponents[iClusterGroup],
        //    iCurrVertexPositionDataOffset);

        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * 3);

        iNumTotalVertices += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size());
    }

    //DEBUG_PRINTF("num total vertices: %d\n", iNumTotalVertices);

    // allocate device memory
    float* pafTotalClusterVertexPositions;
    uint32_t iArrayIndexOffset = 0;
    cudaMalloc(&pafTotalClusterVertexPositions, iCurrVertexPositionDataOffset * sizeof(float));
    
    uint32_t* paiNumAdjacentClusterVertices;
    cudaMalloc(&paiNumAdjacentClusterVertices, iNumClusterGroups * iNumClusterGroups * sizeof(int));

    uint32_t* paiNumVertexPositionComponents;
    cudaMalloc(&paiNumVertexPositionComponents, iNumClusterGroups * sizeof(int));

    uint32_t* paiNumVertexPositionComponentOffsets;
    cudaMalloc(&paiNumVertexPositionComponentOffsets, iNumClusterGroups * sizeof(int));

    uint32_t* paiClusterGroupAdjacentVertexIndices;
    cudaMalloc(&paiClusterGroupAdjacentVertexIndices, iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(int));

    // copy to device memory
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        cudaMemcpy(
            pafTotalClusterVertexPositions + iArrayIndexOffset,
            aaClusterGroupVertexPositions[iClusterGroup].data(),
            aaClusterGroupVertexPositions[iClusterGroup].size() * sizeof(float) * 3,
            cudaMemcpyHostToDevice);
        iArrayIndexOffset += static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size() * 3);
    }

    cudaMemcpy(
        paiNumVertexPositionComponents,
        aiNumVertexPositionComponents.data(),
        aiNumVertexPositionComponents.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        paiNumVertexPositionComponentOffsets,
        aiVertexPositionComponentArrayOffsets.data(),
        aiVertexPositionComponentArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);


    //void checkClusterGroupAdjacency(
    //    uint32_t * aiAdjacentClusterGroupVertexIndices,
    //    float* afTotalClusterGroupVertexPositionComponents,
    //    uint32_t * aiNumAdjacentClusterGroupVertices,
    //    uint32_t * aiNumVertexPositionComponents,
    //    uint32_t * aiClusterGroupVertexPositionComponentOffsets,
    //    uint32_t iNumTotalVertexIndices,
    //    uint32_t iNumTotalClusterGroups)

    std::vector<uint32_t> aiDefaultValues(iNumClusterGroups * iNumClusterGroups);
    memset(aiDefaultValues.data(), 0, aiDefaultValues.size() * sizeof(int));
    cudaMemcpy(
        paiNumAdjacentClusterVertices, 
        aiDefaultValues.data(), 
        iNumClusterGroups * iNumClusterGroups * sizeof(int), 
        cudaMemcpyHostToDevice);

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumTotalVertices) / float(WORKGROUP_SIZE)));
    checkClusterGroupAdjacency <<<iNumBlocks, WORKGROUP_SIZE>>>(
        paiClusterGroupAdjacentVertexIndices,
        paiNumAdjacentClusterVertices,
        pafTotalClusterVertexPositions,
        paiNumVertexPositionComponents,
        paiNumVertexPositionComponentOffsets,
        iCurrVertexPositionDataOffset,
        iNumClusterGroups);

    std::vector<uint32_t> aiTotalClusterGroupAdjacentVerticesCPU(iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP);
    cudaMemcpy(
        aiTotalClusterGroupAdjacentVerticesCPU.data(),
        paiClusterGroupAdjacentVertexIndices,
        iNumClusterGroups * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP * sizeof(int),
        cudaMemcpyDeviceToHost);

    std::vector<uint32_t> aiNumAdjacentVertices(iNumClusterGroups);
    cudaMemcpy(
        aiNumAdjacentVertices.data(),
        paiNumAdjacentClusterVertices,
        iNumClusterGroups * sizeof(int),
        cudaMemcpyDeviceToHost);

    aaiClusterGroupBoundaryVertices.resize(iNumClusterGroups);
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        uint32_t iNumAdjacentVertices = aiNumAdjacentVertices[iClusterGroup];
        aaiClusterGroupBoundaryVertices[iClusterGroup].resize(iNumAdjacentVertices);
        memcpy(
            aaiClusterGroupBoundaryVertices[iClusterGroup].data(),
            aiTotalClusterGroupAdjacentVerticesCPU.data() + iClusterGroup * MAX_BOUNDARY_VERTICES_PER_CLUSTER_GROUP,
            sizeof(int) * iNumAdjacentVertices);
        for(auto const& iVertex : aaiClusterGroupBoundaryVertices[iClusterGroup])
        {
            assert(iVertex < aaClusterGroupVertexPositions[iClusterGroup].size());
        }
    }

    cudaFree(paiNumAdjacentClusterVertices);
    cudaFree(pafTotalClusterVertexPositions);
    cudaFree(paiNumVertexPositionComponents);
    cudaFree(paiNumVertexPositionComponentOffsets);
    cudaFree(paiClusterGroupAdjacentVertexIndices);

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("*** took %lld seconds for getClusterGroupBoundaryVerticesCUDA to finish ***\n",
    iSeconds);

}

/*
**
*/
void computeEdgeCollapseInfoCUDA(
    std::vector<float>& afCollapseCosts,
    std::vector<vec3>& aOptimalVertexPositions,
    std::vector<vec3>& aOptimalVertexNormals,
    std::vector<vec2>& aOptimalVertexUVs,
    std::vector<std::pair<uint32_t, uint32_t>>& aEdges,
    std::vector<vec3> const& aClusterGroupVertexPositions,
    std::vector<vec3> const& aClusterGroupVertexNormals,
    std::vector<vec2> const& aClusterGroupVertexUVs,
    std::vector<std::pair<uint32_t, uint32_t>> const& aiValidClusterGroupEdgePairs,
    std::vector<uint32_t> const& aiClusterGroupNonBoundaryVertices,
    std::vector<uint32_t> const& aiClusterGroupTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterGroupTriangleUVIndices,
    std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices)
{
//DEBUG_PRINTF("*** start computeEdgeCollapseInfoCUDA ***\n");
//auto start = std::chrono::high_resolution_clock::now();
    
    uint32_t iNumEdges = static_cast<uint32_t>(aiValidClusterGroupEdgePairs.size());
    uint32_t iNumClusteGroupTrianglePositionIndices = static_cast<uint32_t>(aiClusterGroupTrianglePositionIndices.size());
    uint32_t iNumClusterGroupNonBoundaryVertices = static_cast<uint32_t>(aiClusterGroupNonBoundaryVertices.size());
    uint32_t iNumVertices = static_cast<uint32_t>(aClusterGroupVertexPositions.size());
    uint32_t iNumTriangleIndices = static_cast<uint32_t>(aiClusterGroupTrianglePositionIndices.size());

    float* afVertexPositionComponents;
    cudaMalloc(
        &afVertexPositionComponents,
        aClusterGroupVertexPositions.size() * 3 * sizeof(float));
    cudaMemcpy(
        afVertexPositionComponents,
        aClusterGroupVertexPositions.data(),
        aClusterGroupVertexPositions.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* afVertexNormalComponents;
    cudaMalloc(
        &afVertexNormalComponents,
        aClusterGroupVertexNormals.size() * 3 * sizeof(float));
    cudaMemcpy(
        afVertexNormalComponents,
        aClusterGroupVertexNormals.data(),
        aClusterGroupVertexNormals.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* afVertexUVComponents;
    cudaMalloc(
        &afVertexUVComponents,
        aClusterGroupVertexUVs.size() * 2 * sizeof(float));
    cudaMemcpy(
        afVertexUVComponents,
        aClusterGroupVertexUVs.data(),
        aClusterGroupVertexUVs.size() * 2 * sizeof(float),
        cudaMemcpyHostToDevice);

    uint32_t* aiClusterGroupEdgePairs;
    cudaMalloc(
        &aiClusterGroupEdgePairs,
        aiValidClusterGroupEdgePairs.size() * 2 * sizeof(int));
    std::vector<uint32_t> aiEdgePairs;
    for(uint32_t i = 0; i < static_cast<uint32_t>(aiValidClusterGroupEdgePairs.size()); i++)
    {
        aiEdgePairs.push_back(aiValidClusterGroupEdgePairs[i].first);
        aiEdgePairs.push_back(aiValidClusterGroupEdgePairs[i].second);
    }
    cudaMemcpy(
        aiClusterGroupEdgePairs,
        aiEdgePairs.data(),
        aiEdgePairs.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* aiClusterGroupNonBoundaryVertexIndices;
    cudaMalloc(
        &aiClusterGroupNonBoundaryVertexIndices,
        aiClusterGroupNonBoundaryVertices.size() * sizeof(int));
    cudaMemcpy(
        aiClusterGroupNonBoundaryVertexIndices,
        aiClusterGroupNonBoundaryVertices.data(),
        aiClusterGroupNonBoundaryVertices.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* aiClusterGroupTrianglePositionIndicesGPU;
    cudaMalloc(
        &aiClusterGroupTrianglePositionIndicesGPU,
        aiClusterGroupTrianglePositionIndices.size() * sizeof(int));
    cudaMemcpy(
        aiClusterGroupTrianglePositionIndicesGPU,
        aiClusterGroupTrianglePositionIndices.data(),
        aiClusterGroupTrianglePositionIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* aiClusterGroupTriangleNormalIndicesGPU;
    cudaMalloc(
        &aiClusterGroupTriangleNormalIndicesGPU,
        aiClusterGroupTriangleNormalIndices.size() * sizeof(int));
    cudaMemcpy(
        aiClusterGroupTriangleNormalIndicesGPU,
        aiClusterGroupTriangleNormalIndices.data(),
        aiClusterGroupTriangleNormalIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* aiClusterGroupTriangleUVIndicesGPU;
    cudaMalloc(
        &aiClusterGroupTriangleUVIndicesGPU,
        aiClusterGroupTriangleUVIndices.size() * sizeof(int));
    cudaMemcpy(
        aiClusterGroupTriangleUVIndicesGPU,
        aiClusterGroupTriangleUVIndices.data(),
        aiClusterGroupTriangleUVIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    float* afEdgeCollapseCosts;
    cudaMalloc(
        &afEdgeCollapseCosts,
        iNumEdges * sizeof(float));

    uint32_t* aiEdgeCollapseVertexIndices0;
    cudaMalloc(
        &aiEdgeCollapseVertexIndices0,
        iNumEdges * sizeof(int));

    uint32_t* aiEdgeCollapseVertexIndices1;
    cudaMalloc(
        &aiEdgeCollapseVertexIndices1,
        iNumEdges * sizeof(int));

    float* afEdgeCollapseVertexPositions;
    cudaMalloc(
        &afEdgeCollapseVertexPositions,
        iNumEdges * 3 * sizeof(float));

    float* afEdgeCollapseVertexNormals;
    cudaMalloc(
        &afEdgeCollapseVertexNormals,
        iNumEdges * 3 * sizeof(float));

    float* afEdgeCollapseVertexUVs;
    cudaMalloc(
        &afEdgeCollapseVertexUVs,
        iNumEdges * 3 * sizeof(float));

    float* afQuadrics;
    cudaMalloc(
        &afQuadrics,
        aClusterGroupVertexPositions.size() * 16 * sizeof(float));

    float* afTotalNormalPlaneAngles;
    cudaMalloc(
        &afTotalNormalPlaneAngles,
        aClusterGroupVertexPositions.size() * sizeof(float));

    uint32_t* aiEdgeNormalIndices;
    cudaMalloc(
        &aiEdgeNormalIndices,
        aiEdgePairs.size() * sizeof(int));
    
    uint32_t* aiEdgeUVIndices;
    cudaMalloc(
        &aiEdgeUVIndices,
        aiEdgePairs.size() * sizeof(int));

    // build normal and uv map to position for edges
    std::vector<uint32_t> aiEdgeNormalIndicesCPU(iNumEdges * 2);
    std::vector<uint32_t> aiEdgeUVIndicesCPU(iNumEdges * 2);
    {
        //void getMatchingTriangleNormalAndUV(
        //    uint32_t * aiRetNormalIndices,
        //    uint32_t * aiRetUVIndices,
        //    uint32_t * aiTriangleVertexPositionIndices,
        //    uint32_t * aiTriangleVertexNormalIndices,
        //    uint32_t * aiTriangleVertexUVIndices,
        //    uint32_t * aiEdges,
        //    uint32_t iNumEdges,
        //    uint32_t iNumTriangleVertexPositionIndices)

        uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumEdges) / float(WORKGROUP_SIZE)));
        getMatchingTriangleNormalAndUV<<<iNumBlocks, WORKGROUP_SIZE>>>(
            aiEdgeNormalIndices,
            aiEdgeUVIndices,
            aiClusterGroupTrianglePositionIndicesGPU,
            aiClusterGroupTriangleNormalIndicesGPU,
            aiClusterGroupTriangleUVIndicesGPU,
            aiClusterGroupEdgePairs,
            iNumEdges,
            iNumClusteGroupTrianglePositionIndices);

        cudaDeviceSynchronize();

        cudaMemcpy(
            aiEdgeNormalIndicesCPU.data(),
            aiEdgeNormalIndices,
            sizeof(int) * iNumEdges * 2,
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            aiEdgeUVIndicesCPU.data(),
            aiEdgeUVIndices,
            sizeof(int)* iNumEdges * 2,
            cudaMemcpyDeviceToHost);
    }

    float* afVertexPlanes;
    float* afAverageVertexNormals;
    uint32_t* aiNumVertexPlanes;
    float* afVertexNormalPlaneAngles;
    {
        cudaMalloc(
            &afAverageVertexNormals,
            iNumVertices * sizeof(float) * 3);
        cudaMemset(afAverageVertexNormals, 0, iNumVertices * sizeof(float) * 3);

        cudaMalloc(
            &afVertexPlanes,
            iNumVertices* MAX_NUM_PLANES_PER_VERTEX * 4 * sizeof(float));
        cudaMemset(afVertexPlanes, 0, iNumVertices* MAX_NUM_PLANES_PER_VERTEX * 4 * sizeof(float));

        cudaMalloc(
            &aiNumVertexPlanes,
            iNumVertices * sizeof(int));
        cudaMemset(aiNumVertexPlanes, 0, iNumVertices * sizeof(int));

        uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumVertices) / float(WORKGROUP_SIZE)));
        iNumBlocks = std::max(iNumBlocks, 1u);
        computeAverageVertexNormals<<<iNumBlocks, WORKGROUP_SIZE>>>(
            afVertexPlanes,
            afAverageVertexNormals,
            afQuadrics,
            aiNumVertexPlanes,
            aiClusterGroupTrianglePositionIndicesGPU,
            afVertexPositionComponents,
            iNumVertices,
            iNumTriangleIndices);

        cudaDeviceSynchronize();

        cudaMalloc(
            &afVertexNormalPlaneAngles, 
            iNumVertices * sizeof(float));
        
        computeTotalNormalPlaneAngles<<<iNumBlocks, WORKGROUP_SIZE>>>(
            afVertexNormalPlaneAngles,
            afAverageVertexNormals,
            afVertexPlanes,
            aiNumVertexPlanes,
            iNumVertices);

        cudaDeviceSynchronize();

        std::vector<float> afVertexPlanesCPU(iNumVertices * MAX_NUM_PLANES_PER_VERTEX * 4);
        std::vector<float> afAverageVertexNormalCPU(iNumVertices * 3);
        std::vector<uint32_t> aiNumVertexPlanesCPU(iNumVertices);
        std::vector<float> afVertexNormalPlaneAnglesCPU(iNumVertices);
        std::vector<float> afQuadricsCPU(iNumVertices * 16);
        cudaMemcpy(
            afVertexPlanesCPU.data(),
            afVertexPlanes,
            afVertexPlanesCPU.size() * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            afAverageVertexNormalCPU.data(),
            afAverageVertexNormals,
            afAverageVertexNormalCPU.size() * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            aiNumVertexPlanesCPU.data(),
            aiNumVertexPlanes,
            aiNumVertexPlanesCPU.size() * sizeof(int),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            afVertexNormalPlaneAnglesCPU.data(),
            afVertexNormalPlaneAngles,
            afVertexNormalPlaneAnglesCPU.size() * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            afQuadricsCPU.data(),
            afQuadrics,
            afQuadricsCPU.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
    }


    float* afVertexAdjacentTriangleCounts;
    cudaMalloc(
        &afVertexAdjacentTriangleCounts,
        iNumVertices * sizeof(float));
    cudaMemset(
        afVertexAdjacentTriangleCounts,
        0,
        iNumVertices * sizeof(float));

#if 0
    {
        uint32_t iNumVertices = static_cast<uint32_t>(aClusterGroupVertexPositions.size());
        uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumVertices) / float(WORKGROUP_SIZE)));
        computeQuadrics<<<iNumBlocks, WORKGROUP_SIZE>>>(
            afQuadrics,
            afVertexAdjacentTriangleCounts,
            aiClusterGroupTrianglePositionIndicesGPU,
            afVertexPositionComponents,
            iNumVertices,
            iNumTriangleIndices);

        cudaDeviceSynchronize();

        std::vector<float> afVertexAdjacentTriangleCountsCPU(iNumVertices);
        cudaMemcpy(
            afVertexAdjacentTriangleCountsCPU.data(),
            afVertexAdjacentTriangleCounts,
            iNumVertices * sizeof(float),
            cudaMemcpyDeviceToHost);

        std::vector<float> afQuadricsCPU(iNumVertices * 16);
        cudaMemcpy(
            afQuadricsCPU.data(),
            afQuadrics,
            iNumVertices * 16 * sizeof(float),
            cudaMemcpyDeviceToHost);

        std::vector<float> afAverageVertexNormalsCPU(iNumVertices * 3);
        cudaMemcpy(
            afAverageVertexNormalsCPU.data(),
            afAverageVertexNormals,
            iNumVertices * 3 * sizeof(float),
            cudaMemcpyDeviceToHost);

        int iDebug = 1;
    }
#endif // #if 0

    //void computeEdgeCollapseInfo(
        //    float* afRetEdgeCollapseCosts,
        //    uint32_t* aiRetEdgeCollapseVertexIndices0,
        //    uint32_t* aiRetEdgeCollapseVertexIndices1,
        //    float* afRetEdgeCollapseVertexPositions,
        //    float* afRetEdgeCollapseVertexNormals,
        //    float* afRetEdgeCollapseVertexUVs,
        //    uint32_t* aiClusterGroupNonBoundaryVertexIndices,
        //    float* afVertexPositionComponents,
        //    float* afVertexNormalComponents,
        //    float* afVertexUVComponents,
        //    float* afQuadrics,
        //    float* afTotalNormalPlaneAngles,
        //    uint32_t* aiClusterGroupEdgePairs,
        //    uint32_t* aiClusterGroupTrianglePositionIndicesGPU,
        //    uint32_t* aiClusterGroupTriangleNormalIndicesGPU,
        //    uint32_t* aiClusterGroupTriangleUVIndicesGPU,
        //    uint32_t* aiNormalIndexToEdgeMap,
        //    uint32_t* aiUVIndexToEdgeMap,
        //    uint32_t iNumClusterGroupTrianglePositionIndices,
        //    uint32_t iNumClusterGroupNonBoundaryVertices,
        //    uint32_t iNumEdges)


    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(aiEdgePairs.size() / 2) / float(WORKGROUP_SIZE)));
    iNumBlocks = std::max(iNumBlocks, 1u);
    computeEdgeCollapseInfo<<<iNumBlocks, WORKGROUP_SIZE>>>(
        afEdgeCollapseCosts,
        aiEdgeCollapseVertexIndices0,
        aiEdgeCollapseVertexIndices1,
        afEdgeCollapseVertexPositions,
        afEdgeCollapseVertexNormals,
        afEdgeCollapseVertexUVs,
        aiClusterGroupNonBoundaryVertexIndices,
        afVertexPositionComponents,
        afVertexNormalComponents,
        afVertexUVComponents,
        afQuadrics,
        afVertexNormalPlaneAngles,
        aiClusterGroupEdgePairs,
        aiClusterGroupTrianglePositionIndicesGPU,
        aiClusterGroupTriangleNormalIndicesGPU,
        aiClusterGroupTriangleUVIndicesGPU,
        aiEdgeNormalIndices,
        aiEdgeUVIndices,
        iNumClusteGroupTrianglePositionIndices,
        iNumClusterGroupNonBoundaryVertices,
        iNumEdges);

    cudaDeviceSynchronize();

    afCollapseCosts.resize(iNumEdges);
    aOptimalVertexPositions.resize(iNumEdges);
    aOptimalVertexNormals.resize(iNumEdges);
    aOptimalVertexUVs.resize(iNumEdges);
    aEdges.resize(iNumEdges);

    cudaMemcpy(
        afCollapseCosts.data(),
        afEdgeCollapseCosts,
        afCollapseCosts.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        aOptimalVertexPositions.data(),
        afEdgeCollapseVertexPositions,
        aOptimalVertexPositions.size() * 3 * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        aOptimalVertexNormals.data(),
        afEdgeCollapseVertexNormals,
        aOptimalVertexNormals.size() * 3 * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        aOptimalVertexUVs.data(),
        afEdgeCollapseVertexUVs,
        aOptimalVertexUVs.size() * 2 * sizeof(float),
        cudaMemcpyDeviceToHost);

    std::vector<uint32_t> aiEdgeIndex0(iNumEdges);
    std::vector<uint32_t> aiEdgeIndex1(iNumEdges);
    cudaMemcpy(
        aiEdgeIndex0.data(),
        aiEdgeCollapseVertexIndices0,
        aiEdgeIndex0.size() * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        aiEdgeIndex1.data(),
        aiEdgeCollapseVertexIndices1,
        aiEdgeIndex1.size() * sizeof(int),
        cudaMemcpyDeviceToHost);

    aEdges.resize(iNumEdges);
    for(uint32_t i = 0; i < iNumEdges; i++)
    {
        aEdges[i] = std::make_pair(aiEdgeIndex0[i], aiEdgeIndex1[i]);
    }

    cudaFree(afVertexPositionComponents);
    cudaFree(afVertexNormalComponents);
    cudaFree(afVertexUVComponents);
    cudaFree(aiClusterGroupEdgePairs);
    cudaFree(aiClusterGroupNonBoundaryVertexIndices);
    cudaFree(aiClusterGroupTrianglePositionIndicesGPU);
    cudaFree(aiClusterGroupTriangleNormalIndicesGPU);
    cudaFree(aiClusterGroupTriangleUVIndicesGPU);
    cudaFree(afEdgeCollapseCosts);
    cudaFree(aiEdgeCollapseVertexIndices0);
    cudaFree(aiEdgeCollapseVertexIndices1);
    cudaFree(afEdgeCollapseVertexPositions);
    cudaFree(afEdgeCollapseVertexNormals);
    cudaFree(afEdgeCollapseVertexUVs);
    cudaFree(afQuadrics);
    cudaFree(afTotalNormalPlaneAngles);

    cudaFree(aiEdgeNormalIndices);
    cudaFree(aiEdgeUVIndices);
    cudaFree(afVertexAdjacentTriangleCounts);
    cudaFree(afAverageVertexNormals);
    cudaFree(afVertexPlanes);
    cudaFree(aiNumVertexPlanes);
    cudaFree(afVertexNormalPlaneAngles);

//auto end = std::chrono::high_resolution_clock::now();
//uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
//DEBUG_PRINTF("*** took %d seconds for computeEdgeCollapseInfoCUDA to finish ***\n",
//    iSeconds);

}

/*
**
*/
void getProjectVertexDistancesCUDA(
    std::vector<vec3>& aProjectedPositions,
    std::vector<vec3> const& aTriangleVertexPositions0,
    std::vector<vec3> const& aTriangleVertexPositions1)
{
    uint32_t iNumVertices0 = static_cast<uint32_t>(aTriangleVertexPositions0.size());
    uint32_t iNumVertices1 = static_cast<uint32_t>(aTriangleVertexPositions1.size());

    assert(iNumVertices0 % 3 == 0);
    assert(iNumVertices1 % 3 == 0);

    float* afRetProjectedPositions;
    cudaMalloc(&afRetProjectedPositions, iNumVertices0 * sizeof(float) * 3);
    cudaMemset(afRetProjectedPositions, 0, iNumVertices0 * sizeof(float) * 3);

    float* afTriangleVertexPositions0;
    cudaMalloc(&afTriangleVertexPositions0, iNumVertices0 * 3 * sizeof(float));
    cudaMemcpy(
        afTriangleVertexPositions0,
        aTriangleVertexPositions0.data(),
        aTriangleVertexPositions0.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* afTriangleVertexPositions1;
    cudaMalloc(&afTriangleVertexPositions1, iNumVertices1 * 3 * sizeof(float));
    cudaMemcpy(
        afTriangleVertexPositions1,
        aTriangleVertexPositions1.data(),
        aTriangleVertexPositions1.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* afIntersectInfo;
    cudaMalloc(&afIntersectInfo, iNumVertices0 * 2 * sizeof(float));
    cudaMemset(&afIntersectInfo, 0, iNumVertices0 * 2 * sizeof(float));

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumVertices0) / float(WORKGROUP_SIZE)));
    iNumBlocks = std::max(iNumBlocks, 1u);
    projectVertices<<<iNumBlocks, WORKGROUP_SIZE>>>(
        afRetProjectedPositions,
        afTriangleVertexPositions0,
        afTriangleVertexPositions1,
        afIntersectInfo,
        iNumVertices0,
        iNumVertices1);

    aProjectedPositions.resize(iNumVertices0);
    cudaMemcpy(
        aProjectedPositions.data(),
        afRetProjectedPositions,
        iNumVertices0 * sizeof(float) * 3,
        cudaMemcpyKind::cudaMemcpyDeviceToHost);

    struct IntersectInfo
    {
        float mfT;
        float mfTriangle;
    };

    std::vector<IntersectInfo> afIntersectT(iNumVertices0);
    cudaMemcpy(
        afIntersectT.data(),
        afIntersectInfo,
        iNumVertices0 * 2 * sizeof(float),
        cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(afRetProjectedPositions);
    cudaFree(afTriangleVertexPositions1);
    cudaFree(afTriangleVertexPositions0);
}

/*
**
*/
void getShortestVertexDistancesCUDA(
    std::vector<float>& afClosestDistances,
    std::vector<uint32_t>& aiClosestVertexPositionIndices,
    std::vector<vec3> const& aVertexPositions0,
    std::vector<vec3> const& aVertexPositions1)
{
    uint32_t iNumVertices0 = static_cast<uint32_t>(aVertexPositions0.size());
    uint32_t iNumVertices1 = static_cast<uint32_t>(aVertexPositions1.size());

    //float* afRetShortestDistances,
    //    uint32_t* aiRetShortestVertexPositionIndices,
    //    float* afVertexPositions0,
    //    float* afVertexPositions1,
    //    uint32_t iNumVertexPositions0,
    //    uint32_t iNumVertexPositions1

    float* afRetShortestDistances;
    cudaMalloc(&afRetShortestDistances, iNumVertices0 * sizeof(float));
    cudaMemset(afRetShortestDistances, 0, iNumVertices0 * sizeof(float));

    uint32_t* aiRetShortestVertexPositionIndices;
    cudaMalloc(&aiRetShortestVertexPositionIndices, iNumVertices0 * sizeof(uint32_t));
    cudaMemset(aiRetShortestVertexPositionIndices, 0, iNumVertices0 * sizeof(uint32_t));

    float* afVertexPositions0;
    cudaMalloc(&afVertexPositions0, iNumVertices0 * 3 * sizeof(float));
    cudaMemcpy(
        afVertexPositions0,
        aVertexPositions0.data(),
        aVertexPositions0.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* afVertexPositions1;
    cudaMalloc(&afVertexPositions1, iNumVertices1 * 3 * sizeof(float));
    cudaMemcpy(
        afVertexPositions1,
        aVertexPositions1.data(),
        aVertexPositions1.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumVertices0) / float(WORKGROUP_SIZE)));
    iNumBlocks = std::max(iNumBlocks, 1u);
    getShortestVertexDistances<<<iNumBlocks, WORKGROUP_SIZE>>>(
        afRetShortestDistances,
        aiRetShortestVertexPositionIndices,
        afVertexPositions0,
        afVertexPositions1,
        iNumVertices0,
        iNumVertices1);

    afClosestDistances.resize(aVertexPositions0.size());
    cudaMemcpy(
        afClosestDistances.data(),
        afRetShortestDistances,
        aVertexPositions0.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    aiClosestVertexPositionIndices.resize(aVertexPositions0.size());
    cudaMemcpy(
        aiClosestVertexPositionIndices.data(),
        aiRetShortestVertexPositionIndices,
        aiClosestVertexPositionIndices.size() * sizeof(int),
        cudaMemcpyDeviceToHost);

    cudaFree(afRetShortestDistances);
    cudaFree(aiRetShortestVertexPositionIndices);
    cudaFree(afVertexPositions0);
    cudaFree(afVertexPositions1);
}

#define MAX_NUM_ADJACENT_EDGES 50
#define MAX_NUM_CLUSTER_POSITION_INDICES 1000

/*
**
*/
void getSortedEdgeAdjacentClustersCUDA(
    std::vector<std::vector<uint32_t>>& aaiSortedAdjacentEdgeClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices)
{
    DEBUG_PRINTF("*** start getSortedEdgeAdjacentClustersCUDA ***\n");
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t iCurrVertexPositionIndexDataOffset = 0;
    uint32_t iCurrVertexPositionDataOffset = 0;
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());
    std::vector<uint32_t> aiNumVertexPositionComponents(iNumClusters);
    std::vector<uint32_t> aiVertexPositionComponentArrayOffsets(iNumClusters);
    std::vector<uint32_t> aiVertexPositionIndexArrayOffsets(iNumClusters);
    std::vector<uint32_t> aiNumVertexPositionIndices(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        //DEBUG_PRINTF("cluster %d vertex position offset: %d vertex position index offset: %d\n",
        //    iCluster,
        //    iCurrVertexPositionDataOffset,
        //    iCurrVertexPositionIndexDataOffset);

        aiNumVertexPositionComponents[iCluster] = static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
        aiVertexPositionComponentArrayOffsets[iCluster] = iCurrVertexPositionDataOffset;
        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);

        aiNumVertexPositionIndices[iCluster] = static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
        aiVertexPositionIndexArrayOffsets[iCluster] = iCurrVertexPositionIndexDataOffset;
        iCurrVertexPositionIndexDataOffset += static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
    }

    // allocate device memory
    uint32_t* paaiRetAdjacentEdgeClusters;
    cudaMalloc(&paaiRetAdjacentEdgeClusters, iNumClusters * iNumClusters * sizeof(uint32_t));
    cudaMemset(paaiRetAdjacentEdgeClusters, 0xff, iNumClusters * iNumClusters * sizeof(uint32_t));

    uint32_t* paiRetNumAdjacentEdgeClusters;
    cudaMalloc(&paiRetNumAdjacentEdgeClusters, iNumClusters * sizeof(uint32_t));
    cudaMemset(paiRetNumAdjacentEdgeClusters, 0, iNumClusters * sizeof(uint32_t));

    uint32_t* paiNumVertexPositionComponents;
    cudaMalloc(&paiNumVertexPositionComponents, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiNumVertexPositionComponents,
        aiNumVertexPositionComponents.data(),
        aiNumVertexPositionComponents.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionComponentOffsets;
    cudaMalloc(&paiVertexPositionComponentOffsets, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionComponentOffsets,
        aiVertexPositionComponentArrayOffsets.data(),
        aiVertexPositionComponentArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionIndexOffsets;
    cudaMalloc(&paiVertexPositionIndexOffsets, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionIndexOffsets,
        aiVertexPositionIndexArrayOffsets.data(),
        aiVertexPositionIndexArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiNumVertexPositionIndices;
    cudaMalloc(&paiNumVertexPositionIndices, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiNumVertexPositionIndices,
        aiNumVertexPositionIndices.data(),
        aiNumVertexPositionIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);


    // copy vertex positions
    float* pafTotalClusterVertexPositions;
    cudaMalloc(&pafTotalClusterVertexPositions, iCurrVertexPositionDataOffset * sizeof(float));
    uint32_t iArrayIndexOffset = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            pafTotalClusterVertexPositions + iArrayIndexOffset,
            aaVertexPositions[iCluster].data(),
            aaVertexPositions[iCluster].size() * sizeof(float) * 3,
            cudaMemcpyHostToDevice);
        iArrayIndexOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
    }

    // copy vertex indices
    uint32_t* paaiVertexPositionIndices;
    cudaMalloc(&paaiVertexPositionIndices, iCurrVertexPositionIndexDataOffset * sizeof(uint32_t));
    iArrayIndexOffset = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            paaiVertexPositionIndices + iArrayIndexOffset,
            aaiVertexPositionIndices[iCluster].data(),
            aaiVertexPositionIndices[iCluster].size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        iArrayIndexOffset += static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
    }

    float* pafClusterMinMaxCenterRadius;
    cudaMalloc(&pafClusterMinMaxCenterRadius, iNumClusters * 10 * sizeof(float));

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumClusters) / float(WORKGROUP_SIZE)));
    getClusterBounds<<<iNumBlocks, WORKGROUP_SIZE>>>(
        pafClusterMinMaxCenterRadius,
        pafTotalClusterVertexPositions,
        paiVertexPositionComponentOffsets,
        paiNumVertexPositionComponents,
        iNumClusters);

    std::vector<float> afClusterMinMaxCenterRadiusCPU(iNumClusters * 10);
    cudaMemcpy(
        afClusterMinMaxCenterRadiusCPU.data(),
        pafClusterMinMaxCenterRadius,
        afClusterMinMaxCenterRadiusCPU.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    std::vector<vec3> aMinBounds(iNumClusters);
    std::vector<vec3> aMaxBounds(iNumClusters);
    std::vector<vec3> aCenter(iNumClusters);
    std::vector<float> afRadius(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aMinBounds[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10];
        aMinBounds[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 1];
        aMinBounds[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 2];

        aMaxBounds[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 3];
        aMaxBounds[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 4];
        aMaxBounds[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 5];

        aCenter[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 6];
        aCenter[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 7];
        aCenter[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 8];

        afRadius[iCluster] = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 9];
    }

    float* pafRetDistances;
    cudaMalloc(&pafRetDistances, iNumClusters * iNumClusters * sizeof(float));
    cudaMemset(pafRetDistances, 0, iNumClusters * iNumClusters * sizeof(float));

    float* pafClusterCenters;
    cudaMalloc(&pafClusterCenters, iNumClusters * 3 * sizeof(float));
    cudaMemcpy(
        pafClusterCenters,
        aCenter.data(),
        iNumClusters * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    getClusterDistances<<<iNumBlocks, WORKGROUP_SIZE>>>(
        pafRetDistances,
        pafClusterCenters,
        iNumClusters);

    std::vector<float> afRetDistancesCPU(iNumClusters * iNumClusters);
    cudaMemcpy(
        afRetDistancesCPU.data(),
        pafRetDistances,
        iNumClusters * iNumClusters * sizeof(float),
        cudaMemcpyDeviceToHost);

    struct ClusterDistanceInfo
    {
        uint32_t        miCluster;
        float           mfDistance;
    };

    std::vector<std::vector< ClusterDistanceInfo>> aaClusterDistanceInfo(iNumClusters);

    std::vector<std::vector<float>> aafDistances(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aaClusterDistanceInfo[iCluster].resize(iNumClusters);

        aafDistances[iCluster].resize(iNumClusters);
        for(uint32_t iCheckCluster = 0; iCheckCluster < iNumClusters; iCheckCluster++)
        {
            uint32_t iIndex = iCluster * iNumClusters + iCheckCluster;
            aafDistances[iCluster][iCheckCluster] = afRetDistancesCPU[iIndex];

            aaClusterDistanceInfo[iCluster][iCheckCluster].miCluster = iCheckCluster;
            aaClusterDistanceInfo[iCluster][iCheckCluster].mfDistance = aafDistances[iCluster][iCheckCluster];
        }

        std::sort(
            aaClusterDistanceInfo[iCluster].begin(),
            aaClusterDistanceInfo[iCluster].end(),
            [](ClusterDistanceInfo const& checkInfo0, ClusterDistanceInfo const& checkInfo1)
            {
                return checkInfo0.mfDistance < checkInfo1.mfDistance;
            }
        );
    }

    aaiSortedAdjacentEdgeClusters.resize(iNumClusters);
    for(uint32_t i = 0; i < iNumClusters; i++)
    {
        aaiSortedAdjacentEdgeClusters[i].resize(iNumClusters);
        for(uint32_t j = 0; j < iNumClusters; j++)
        {
            aaiSortedAdjacentEdgeClusters[i][j] = aaClusterDistanceInfo[i][j].miCluster;
        }
    }

    cudaFree(pafClusterMinMaxCenterRadius);
    cudaFree(pafRetDistances);
    cudaFree(pafClusterCenters);
    cudaFree(paaiRetAdjacentEdgeClusters);
    cudaFree(paiRetNumAdjacentEdgeClusters);
    cudaFree(paiNumVertexPositionComponents);
    cudaFree(paiVertexPositionComponentOffsets);
    cudaFree(paiVertexPositionIndexOffsets);
    cudaFree(paiNumVertexPositionIndices);
    cudaFree(pafTotalClusterVertexPositions);
    cudaFree(paaiVertexPositionIndices);

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    DEBUG_PRINTF("*** took %lld seconds for getSortedEdgeAdjacentClustersCUDA to finish ***\n",
        iSeconds);
}

/*
**  TODO: finish this
*/
void buildClusterEdgeAdjacencyCUDA(
    std::vector<std::vector<uint32_t>>& aaiAdjacentEdgeClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices)
{
    DEBUG_PRINTF("*** start buildClusterEdgeAdjacencyCUDA ***\n");
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t iCurrVertexPositionIndexDataOffset = 0;
    uint32_t iCurrVertexPositionDataOffset = 0;
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());
    std::vector<uint32_t> aiNumVertexPositionComponents(iNumClusters);
    std::vector<uint32_t> aiVertexPositionComponentArrayOffsets(iNumClusters);
    std::vector<uint32_t> aiVertexPositionIndexArrayOffsets(iNumClusters);
    std::vector<uint32_t> aiNumVertexPositionIndices(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        //DEBUG_PRINTF("cluster %d vertex position offset: %d vertex position index offset: %d\n",
        //    iCluster,
        //    iCurrVertexPositionDataOffset,
        //    iCurrVertexPositionIndexDataOffset);

        aiNumVertexPositionComponents[iCluster] = static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
        aiVertexPositionComponentArrayOffsets[iCluster] = iCurrVertexPositionDataOffset;
        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);

        aiNumVertexPositionIndices[iCluster] = static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
        aiVertexPositionIndexArrayOffsets[iCluster] = iCurrVertexPositionIndexDataOffset;
        iCurrVertexPositionIndexDataOffset += static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
    }

    // allocate device memory
    uint32_t* paaiRetAdjacentEdgeClusters;
    cudaMalloc(&paaiRetAdjacentEdgeClusters, iNumClusters * iNumClusters * sizeof(uint32_t));
    cudaMemset(paaiRetAdjacentEdgeClusters, 0xff, iNumClusters * iNumClusters * sizeof(uint32_t));

    uint32_t* paiRetNumAdjacentEdgeClusters;
    cudaMalloc(&paiRetNumAdjacentEdgeClusters, iNumClusters * sizeof(uint32_t));
    cudaMemset(paiRetNumAdjacentEdgeClusters, 0, iNumClusters * sizeof(uint32_t));

    uint32_t* paiNumVertexPositionComponents;
    cudaMalloc(&paiNumVertexPositionComponents, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiNumVertexPositionComponents,
        aiNumVertexPositionComponents.data(),
        aiNumVertexPositionComponents.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionComponentOffsets;
    cudaMalloc(&paiVertexPositionComponentOffsets, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionComponentOffsets,
        aiVertexPositionComponentArrayOffsets.data(),
        aiVertexPositionComponentArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionIndexOffsets;
    cudaMalloc(&paiVertexPositionIndexOffsets, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionIndexOffsets,
        aiVertexPositionIndexArrayOffsets.data(),
        aiVertexPositionIndexArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiNumVertexPositionIndices;
    cudaMalloc(&paiNumVertexPositionIndices, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiNumVertexPositionIndices,
        aiNumVertexPositionIndices.data(),
        aiNumVertexPositionIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);

       

    // copy vertex positions
    float* pafTotalClusterVertexPositions;
    cudaMalloc(&pafTotalClusterVertexPositions, iCurrVertexPositionDataOffset * sizeof(float));
    uint32_t iArrayIndexOffset = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            pafTotalClusterVertexPositions + iArrayIndexOffset,
            aaVertexPositions[iCluster].data(),
            aaVertexPositions[iCluster].size() * sizeof(float) * 3,
            cudaMemcpyHostToDevice);
        iArrayIndexOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
    }

    // copy vertex indices
    uint32_t* paaiVertexPositionIndices;
    cudaMalloc(&paaiVertexPositionIndices, iCurrVertexPositionIndexDataOffset * sizeof(uint32_t));
    iArrayIndexOffset = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            paaiVertexPositionIndices + iArrayIndexOffset,
            aaiVertexPositionIndices[iCluster].data(),
            aaiVertexPositionIndices[iCluster].size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        iArrayIndexOffset += static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
    }

    float* pafClusterMinMaxCenterRadius;
    cudaMalloc(&pafClusterMinMaxCenterRadius, iNumClusters * 10 * sizeof(float));

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumClusters) / float(WORKGROUP_SIZE)));
    getClusterBounds<<<iNumBlocks, WORKGROUP_SIZE>>>(
        pafClusterMinMaxCenterRadius,
        pafTotalClusterVertexPositions,
        paiVertexPositionComponentOffsets,
        paiNumVertexPositionComponents,
        iNumClusters);
    
    std::vector<float> afClusterMinMaxCenterRadiusCPU(iNumClusters * 10);
    cudaMemcpy(
        afClusterMinMaxCenterRadiusCPU.data(),
        pafClusterMinMaxCenterRadius,
        afClusterMinMaxCenterRadiusCPU.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    std::vector<vec3> aMinBounds(iNumClusters);
    std::vector<vec3> aMaxBounds(iNumClusters);
    std::vector<vec3> aCenter(iNumClusters);
    std::vector<float> afRadius(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aMinBounds[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10];
        aMinBounds[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 1];
        aMinBounds[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 2];

        aMaxBounds[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 3];
        aMaxBounds[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 4];
        aMaxBounds[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 5];

        aCenter[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 6];
        aCenter[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 7];
        aCenter[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 8];

        afRadius[iCluster] = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 9];
    }
    
    float* pafRetDistances;
    cudaMalloc(&pafRetDistances, iNumClusters * iNumClusters * sizeof(float));
    cudaMemset(pafRetDistances, 0, iNumClusters * iNumClusters * sizeof(float));
    
    float* pafClusterCenters;
    cudaMalloc(&pafClusterCenters, iNumClusters * 3 * sizeof(float));
    cudaMemcpy(
        pafClusterCenters,
        aCenter.data(),
        iNumClusters * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    getClusterDistances<<<iNumBlocks,WORKGROUP_SIZE>>>(
        pafRetDistances,
        pafClusterCenters,
        iNumClusters);

    std::vector<float> afRetDistancesCPU(iNumClusters * iNumClusters);
    cudaMemcpy(
        afRetDistancesCPU.data(),
        pafRetDistances,
        iNumClusters * iNumClusters * sizeof(float),
        cudaMemcpyDeviceToHost);

    struct ClusterDistanceInfo
    {
        uint32_t        miCluster;
        float           mfDistance;
    };

    std::vector<std::vector< ClusterDistanceInfo>> aaClusterDistanceInfo(iNumClusters);

    std::vector<std::vector<float>> aafDistances(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aaClusterDistanceInfo[iCluster].resize(iNumClusters);

        aafDistances[iCluster].resize(iNumClusters);
        for(uint32_t iCheckCluster = 0; iCheckCluster < iNumClusters; iCheckCluster++)
        {
            uint32_t iIndex = iCluster * iNumClusters + iCheckCluster;
            aafDistances[iCluster][iCheckCluster] = afRetDistancesCPU[iIndex];

            aaClusterDistanceInfo[iCluster][iCheckCluster].miCluster = iCheckCluster;
            aaClusterDistanceInfo[iCluster][iCheckCluster].mfDistance = aafDistances[iCluster][iCheckCluster];
        }

        std::sort(
            aaClusterDistanceInfo[iCluster].begin(),
            aaClusterDistanceInfo[iCluster].end(),
            [](ClusterDistanceInfo const& checkInfo0, ClusterDistanceInfo const& checkInfo1)
            {
                return checkInfo0.mfDistance < checkInfo1.mfDistance;
            }
        );
    }

    std::vector<uint32_t> aiSortedClusters(iNumClusters * iNumClusters);
    for(uint32_t i = 0; i < iNumClusters; i++)
    {
        for(uint32_t j = 0; j < iNumClusters; j++)
        {
            uint32_t iIndex = i * iNumClusters + j;
            aiSortedClusters[iIndex] = aaClusterDistanceInfo[i][j].miCluster;
        }
    }

    cudaFree(pafClusterMinMaxCenterRadius);
    cudaFree(pafRetDistances);
    cudaFree(pafClusterCenters);

    uint32_t* paiDistanceSortedClusterID;
    cudaMalloc(&paiDistanceSortedClusterID, iNumClusters* iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiDistanceSortedClusterID,
        aiSortedClusters.data(),
        aiSortedClusters.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    buildClusterEdgeAdjacency<<<iNumBlocks, WORKGROUP_SIZE>>>(
        paaiRetAdjacentEdgeClusters,
        paiRetNumAdjacentEdgeClusters,
        pafTotalClusterVertexPositions,
        paaiVertexPositionIndices,
        paiNumVertexPositionComponents,
        paiNumVertexPositionIndices,
        paiVertexPositionComponentOffsets,
        paiVertexPositionIndexOffsets,
        paiDistanceSortedClusterID,
        iNumClusters);

    std::vector<uint32_t> aaiRetAdjacentEdgeClustersCPU(iNumClusters * iNumClusters);
    cudaMemcpy(
        aaiRetAdjacentEdgeClustersCPU.data(),
        paaiRetAdjacentEdgeClusters,
        iNumClusters * iNumClusters * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);

    std::vector<uint32_t> aiRetNumAdjacentEdgeClustersCPU(iNumClusters);
    cudaMemcpy(
        aiRetNumAdjacentEdgeClustersCPU.data(),
        paiRetNumAdjacentEdgeClusters,
        iNumClusters * sizeof(int),
        cudaMemcpyDeviceToHost);

    aaiAdjacentEdgeClusters.resize(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        for(uint32_t i = 0; i < iNumClusters; i++)
        {
            uint32_t iIndex = iCluster * iNumClusters + i;
            if(aaiRetAdjacentEdgeClustersCPU[iIndex] != UINT32_MAX)
            {
                aaiAdjacentEdgeClusters[iCluster].push_back(i);
            }
        }
    }

    cudaFree(paaiRetAdjacentEdgeClusters);
    cudaFree(paiRetNumAdjacentEdgeClusters);
    cudaFree(paiNumVertexPositionComponents);
    cudaFree(paiVertexPositionComponentOffsets);
    cudaFree(paiVertexPositionIndexOffsets);
    cudaFree(paiNumVertexPositionIndices);
    cudaFree(pafTotalClusterVertexPositions);
    cudaFree(paaiVertexPositionIndices);
    cudaFree(paiDistanceSortedClusterID);

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    DEBUG_PRINTF("*** took %lld seconds for buildClusterEdgeAdjacencyCUDA to finish ***\n",
        iSeconds);

}

/*
**
*/
void buildClusterEdgeAdjacencyCUDA2(
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaiAdjacentEdgeClusters,
    std::vector<std::vector<vec3>> const& aaVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiVertexPositionIndices)
{
    DEBUG_PRINTF("*** start buildClusterEdgeAdjacencyCUDA ***\n");
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t iCurrVertexPositionIndexDataOffset = 0;
    uint32_t iCurrVertexPositionDataOffset = 0;
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());
    std::vector<uint32_t> aiNumVertexPositionComponents(iNumClusters);
    std::vector<uint32_t> aiVertexPositionComponentArrayOffsets(iNumClusters);
    std::vector<uint32_t> aiVertexPositionIndexArrayOffsets(iNumClusters);
    std::vector<uint32_t> aiNumVertexPositionIndices(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        //DEBUG_PRINTF("cluster %d vertex position offset: %d vertex position index offset: %d\n",
        //    iCluster,
        //    iCurrVertexPositionDataOffset,
        //    iCurrVertexPositionIndexDataOffset);

        aiNumVertexPositionComponents[iCluster] = static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
        aiVertexPositionComponentArrayOffsets[iCluster] = iCurrVertexPositionDataOffset;
        iCurrVertexPositionDataOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);

        aiNumVertexPositionIndices[iCluster] = static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
        aiVertexPositionIndexArrayOffsets[iCluster] = iCurrVertexPositionIndexDataOffset;
        iCurrVertexPositionIndexDataOffset += static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
    }

    // allocate device memory
    uint32_t* paaiRetAdjacentEdgeClusters;
    cudaMalloc(&paaiRetAdjacentEdgeClusters, iNumClusters * iNumClusters * sizeof(uint32_t));
    cudaMemset(paaiRetAdjacentEdgeClusters, 0xff, iNumClusters * iNumClusters * sizeof(uint32_t));

    uint32_t* paiRetNumAdjacentEdgeClusters;
    cudaMalloc(&paiRetNumAdjacentEdgeClusters, iNumClusters * sizeof(uint32_t));
    cudaMemset(paiRetNumAdjacentEdgeClusters, 0, iNumClusters * sizeof(uint32_t));

    uint32_t* paiNumVertexPositionComponents;
    cudaMalloc(&paiNumVertexPositionComponents, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiNumVertexPositionComponents,
        aiNumVertexPositionComponents.data(),
        aiNumVertexPositionComponents.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionComponentOffsets;
    cudaMalloc(&paiVertexPositionComponentOffsets, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionComponentOffsets,
        aiVertexPositionComponentArrayOffsets.data(),
        aiVertexPositionComponentArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionIndexOffsets;
    cudaMalloc(&paiVertexPositionIndexOffsets, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionIndexOffsets,
        aiVertexPositionIndexArrayOffsets.data(),
        aiVertexPositionIndexArrayOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    uint32_t* paiNumVertexPositionIndices;
    cudaMalloc(&paiNumVertexPositionIndices, iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiNumVertexPositionIndices,
        aiNumVertexPositionIndices.data(),
        aiNumVertexPositionIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice);



    // copy vertex positions
    float* pafTotalClusterVertexPositions;
    cudaMalloc(&pafTotalClusterVertexPositions, iCurrVertexPositionDataOffset * sizeof(float));
    uint32_t iArrayIndexOffset = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            pafTotalClusterVertexPositions + iArrayIndexOffset,
            aaVertexPositions[iCluster].data(),
            aaVertexPositions[iCluster].size() * sizeof(float) * 3,
            cudaMemcpyHostToDevice);
        iArrayIndexOffset += static_cast<uint32_t>(aaVertexPositions[iCluster].size() * 3);
    }

    // copy vertex indices
    uint32_t* paaiVertexPositionIndices;
    cudaMalloc(&paaiVertexPositionIndices, iCurrVertexPositionIndexDataOffset * sizeof(uint32_t));
    iArrayIndexOffset = 0;
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        cudaMemcpy(
            paaiVertexPositionIndices + iArrayIndexOffset,
            aaiVertexPositionIndices[iCluster].data(),
            aaiVertexPositionIndices[iCluster].size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        iArrayIndexOffset += static_cast<uint32_t>(aaiVertexPositionIndices[iCluster].size());
    }

    float* pafClusterMinMaxCenterRadius;
    cudaMalloc(&pafClusterMinMaxCenterRadius, iNumClusters * 10 * sizeof(float));

    uint32_t iNumBlocks = static_cast<uint32_t>(ceilf(static_cast<float>(iNumClusters) / float(WORKGROUP_SIZE)));
    getClusterBounds << <iNumBlocks, WORKGROUP_SIZE >> > (
        pafClusterMinMaxCenterRadius,
        pafTotalClusterVertexPositions,
        paiVertexPositionComponentOffsets,
        paiNumVertexPositionComponents,
        iNumClusters);

    std::vector<float> afClusterMinMaxCenterRadiusCPU(iNumClusters * 10);
    cudaMemcpy(
        afClusterMinMaxCenterRadiusCPU.data(),
        pafClusterMinMaxCenterRadius,
        afClusterMinMaxCenterRadiusCPU.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    std::vector<vec3> aMinBounds(iNumClusters);
    std::vector<vec3> aMaxBounds(iNumClusters);
    std::vector<vec3> aCenter(iNumClusters);
    std::vector<float> afRadius(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aMinBounds[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10];
        aMinBounds[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 1];
        aMinBounds[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 2];

        aMaxBounds[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 3];
        aMaxBounds[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 4];
        aMaxBounds[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 5];

        aCenter[iCluster].x = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 6];
        aCenter[iCluster].y = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 7];
        aCenter[iCluster].z = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 8];

        afRadius[iCluster] = afClusterMinMaxCenterRadiusCPU[iCluster * 10 + 9];
    }

    float* pafRetDistances;
    cudaMalloc(&pafRetDistances, iNumClusters * iNumClusters * sizeof(float));
    cudaMemset(pafRetDistances, 0, iNumClusters * iNumClusters * sizeof(float));

    float* pafClusterCenters;
    cudaMalloc(&pafClusterCenters, iNumClusters * 3 * sizeof(float));
    cudaMemcpy(
        pafClusterCenters,
        aCenter.data(),
        iNumClusters * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    getClusterDistances << <iNumBlocks, WORKGROUP_SIZE >> > (
        pafRetDistances,
        pafClusterCenters,
        iNumClusters);

    std::vector<float> afRetDistancesCPU(iNumClusters * iNumClusters);
    cudaMemcpy(
        afRetDistancesCPU.data(),
        pafRetDistances,
        iNumClusters * iNumClusters * sizeof(float),
        cudaMemcpyDeviceToHost);

    struct ClusterDistanceInfo
    {
        uint32_t        miCluster;
        float           mfDistance;
    };

    std::vector<std::vector< ClusterDistanceInfo>> aaClusterDistanceInfo(iNumClusters);

    std::vector<std::vector<float>> aafDistances(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aaClusterDistanceInfo[iCluster].resize(iNumClusters);

        aafDistances[iCluster].resize(iNumClusters);
        for(uint32_t iCheckCluster = 0; iCheckCluster < iNumClusters; iCheckCluster++)
        {
            uint32_t iIndex = iCluster * iNumClusters + iCheckCluster;
            aafDistances[iCluster][iCheckCluster] = afRetDistancesCPU[iIndex];

            aaClusterDistanceInfo[iCluster][iCheckCluster].miCluster = iCheckCluster;
            aaClusterDistanceInfo[iCluster][iCheckCluster].mfDistance = aafDistances[iCluster][iCheckCluster];
        }

        std::sort(
            aaClusterDistanceInfo[iCluster].begin(),
            aaClusterDistanceInfo[iCluster].end(),
            [](ClusterDistanceInfo const& checkInfo0, ClusterDistanceInfo const& checkInfo1)
            {
                return checkInfo0.mfDistance < checkInfo1.mfDistance;
            }
        );
    }

    std::vector<uint32_t> aiSortedClusters(iNumClusters * iNumClusters);
    for(uint32_t i = 0; i < iNumClusters; i++)
    {
        for(uint32_t j = 0; j < iNumClusters; j++)
        {
            uint32_t iIndex = i * iNumClusters + j;
            aiSortedClusters[iIndex] = aaClusterDistanceInfo[i][j].miCluster;
        }
    }

    cudaFree(pafClusterMinMaxCenterRadius);
    cudaFree(pafRetDistances);
    cudaFree(pafClusterCenters);

    uint32_t* paiDistanceSortedClusterID;
    cudaMalloc(&paiDistanceSortedClusterID, iNumClusters * iNumClusters * sizeof(uint32_t));
    cudaMemcpy(
        paiDistanceSortedClusterID,
        aiSortedClusters.data(),
        aiSortedClusters.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    buildClusterEdgeAdjacency2<<<iNumBlocks, WORKGROUP_SIZE>>>(
        paaiRetAdjacentEdgeClusters,
        paiRetNumAdjacentEdgeClusters,
        pafTotalClusterVertexPositions,
        paaiVertexPositionIndices,
        paiNumVertexPositionComponents,
        paiNumVertexPositionIndices,
        paiVertexPositionComponentOffsets,
        paiVertexPositionIndexOffsets,
        paiDistanceSortedClusterID,
        iNumClusters);

    std::vector<uint32_t> aaiRetAdjacentEdgeClustersCPU(iNumClusters * iNumClusters);
    cudaMemcpy(
        aaiRetAdjacentEdgeClustersCPU.data(),
        paaiRetAdjacentEdgeClusters,
        iNumClusters * iNumClusters * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);

    std::vector<uint32_t> aiRetNumAdjacentEdgeClustersCPU(iNumClusters);
    cudaMemcpy(
        aiRetNumAdjacentEdgeClustersCPU.data(),
        paiRetNumAdjacentEdgeClusters,
        iNumClusters * sizeof(int),
        cudaMemcpyDeviceToHost);

    aaiAdjacentEdgeClusters.resize(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        for(uint32_t i = 0; i < iNumClusters; i++)
        {
            uint32_t iIndex = iCluster * iNumClusters + i;
            if(aaiRetAdjacentEdgeClustersCPU[iIndex] != UINT32_MAX)
            {
                aaiAdjacentEdgeClusters[iCluster].push_back(std::make_pair(i, aaiRetAdjacentEdgeClustersCPU[iIndex]));
            }
        }
    }

    cudaFree(paaiRetAdjacentEdgeClusters);
    cudaFree(paiRetNumAdjacentEdgeClusters);
    cudaFree(paiNumVertexPositionComponents);
    cudaFree(paiVertexPositionComponentOffsets);
    cudaFree(paiVertexPositionIndexOffsets);
    cudaFree(paiNumVertexPositionIndices);
    cudaFree(pafTotalClusterVertexPositions);
    cudaFree(paaiVertexPositionIndices);
    cudaFree(paiDistanceSortedClusterID);

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    DEBUG_PRINTF("*** took %lld seconds for buildClusterEdgeAdjacencyCUDA2 to finish ***\n",
        iSeconds);
}