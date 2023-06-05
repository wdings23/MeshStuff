#include "test_raster.h"

#include <algorithm>
#include <string>
#include <sstream>

#include "rasterizer.h"
#include "Camera.h"
#include "LogPrint.h"
#include "utils.h"

#include "rasterizerCUDA.h"

#include "tinyexr/tinyexr.h"
#include "stb_image_write.h"

#include <assert.h>
#include "mesh_cluster.h"

/*
**
*/
void loadClusterGroup(
    MeshClusterGroup& clusterGroup,
    uint32_t iAddress,
    std::vector<uint8_t> const& clusterGroupBuffer)
{
    uint8_t const* pData = clusterGroupBuffer.data() + iAddress * sizeof(MeshClusterGroup);
    memcpy(&clusterGroup, pData, sizeof(MeshClusterGroup));
}

/*
**
*/
void loadMeshCluster(
    MeshCluster& cluster,
    uint32_t iAddress,
    std::vector<uint8_t> const& clusterBuffer)
{
    uint8_t const* pData = clusterBuffer.data() + iAddress * sizeof(MeshCluster);
    memcpy(&cluster, pData, sizeof(MeshCluster));
}

/*
**
*/
uint32_t getNumClusterGroupsAtLOD(
    std::vector<uint8_t> const& clusterGroupBuffer,
    uint32_t iLODLevel)
{
    uint32_t iNumMeshClusterGroups = 0;
    uint32_t iOffset = 0;
    uint8_t const* pAddress = reinterpret_cast<uint8_t const*>(clusterGroupBuffer.data());
    for(;;)
    {
        MeshClusterGroup const* pMeshClusterGroup = reinterpret_cast<MeshClusterGroup const*>(pAddress + iOffset);
        if(pMeshClusterGroup->miLODLevel == iLODLevel)
        {
            ++iNumMeshClusterGroups;
            for(uint32_t i = 1; i < 128; i++)
            {
                pMeshClusterGroup = reinterpret_cast<MeshClusterGroup const*>(pAddress + iOffset + i * sizeof(MeshClusterGroup));
                if(pMeshClusterGroup->miLODLevel != iLODLevel)
                {
                    break;
                }

                ++iNumMeshClusterGroups;
            }

            break;
        }

        iOffset += sizeof(MeshClusterGroup);
    }

    return iNumMeshClusterGroups;
}

/*
**
*/
void testClusterLOD(
    std::vector<uint8_t>& aMeshClusterData,
    std::vector<uint8_t>& aMeshClusterGroupData,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<uint8_t>& trianglePositionIndexBuffer,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    ClusterTreeNode const& startTreeNode,
    std::vector<uint8_t> const& clusterGroupBuffer,
    std::vector<uint8_t> const& clusterBuffer,
    float fPixelErrorThreshold)
{
    float const kfCameraNear = 1.0f;
    float const kfCameraFar = 100.0f;

    float3 direction = normalize(cameraLookAt - cameraPosition);
    float3 up = (fabsf(direction.z) > fabsf(direction.x) && fabsf(direction.z) > fabsf(direction.y)) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 binormal = normalize(cross(up, direction));

    // camera 
    CCamera camera;
    camera.setFar(kfCameraFar);
    camera.setNear(kfCameraNear);
    camera.setLookAt(cameraLookAt);
    camera.setPosition(cameraPosition);
    CameraUpdateInfo cameraUpdateInfo =
    {
        /* .mfViewWidth      */  1000.0f,
        /* .mfViewHeight     */  1000.0f,
        /* .mfFieldOfView    */  3.14159f * 0.5f,
        /* .mUp              */  up,
        /* .mfNear           */  kfCameraNear,
        /* .mfFar            */  kfCameraFar,
    };
    camera.update(cameraUpdateInfo);

    mat4 const& viewMatrix = camera.getViewMatrix();
    mat4 const& projectionMatrix = camera.getProjectionMatrix();
    mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

    // cluster group root
    uint32_t iClusterGroupAddress = startTreeNode.miClusterGroupAddress;
    uint32_t iCurrLODLevel = UINT32_MAX;
    int32_t aiClusterGroupStack[128] = { 0 };
    int32_t iClusterGroupStackTop = 0;
    int32_t aiLODStack[128] = { 0 };
    int32_t iLODStackTop = 0;

    MeshClusterGroup clusterGroup;
    loadClusterGroup(
        clusterGroup,
        iClusterGroupAddress,
        clusterGroupBuffer);
    iCurrLODLevel = clusterGroup.miLODLevel;
    int32_t iNumClusterGroupsAtLOD = static_cast<int32_t>(getNumClusterGroupsAtLOD(
        clusterGroupBuffer,
        clusterGroup.miLODLevel));
    iClusterGroupAddress = getMeshClusterGroupAddress(
        aMeshClusterGroupData,
        iCurrLODLevel, 
        0);

    aiLODStack[iLODStackTop] = iCurrLODLevel;

    struct RenderClusterInfo
    {
        uint32_t        miCluster;
        uint32_t        miClusterGroup;
        uint32_t        miLOD;
        uint32_t        miMIP;
    };

    uint32_t iNumRenderClusters = 0;
    RenderClusterInfo aRenderClusters[128];
    for(uint32_t i = 0; i < sizeof(aRenderClusters) / sizeof(*aRenderClusters); i++)
    {
        aRenderClusters[i].miCluster = UINT32_MAX;
        aRenderClusters[i].miClusterGroup = UINT32_MAX;
        aRenderClusters[i].miLOD = UINT32_MAX;
    }


    for(;;)
    {
        for(int32_t iClusterGroup = aiClusterGroupStack[iClusterGroupStackTop]; iClusterGroup < iNumClusterGroupsAtLOD; iClusterGroup++)
        {
            iClusterGroupAddress = getMeshClusterGroupAddress(
                aMeshClusterGroupData,
                iCurrLODLevel, 
                iClusterGroup);
            MeshClusterGroup clusterGroup;
            loadClusterGroup(
                clusterGroup,
                iClusterGroupAddress,
                clusterGroupBuffer);
            assert(clusterGroup.miLODLevel == iCurrLODLevel);

            float3 clusterGroupPosition = (clusterGroup.mMaxBounds + clusterGroup.mMinBounds) * 0.5f;
            uint32_t iClusterAddress = clusterGroup.maiClusters[0][0]; //  getMeshClusterAddress(iCurrLODLevel, clusterGroup.maiClusters[0][0]);

            MeshCluster cluster;
            loadMeshCluster(cluster, iClusterAddress, clusterBuffer);
            assert(cluster.miLODLevel == iCurrLODLevel);

            DEBUG_PRINTF("LOD %d cluster group %d\n",
                cluster.miLODLevel,
                clusterGroup.miIndex);

            // project the error length to screen space
            // MIP 0
            //float4 clipSpace0 = viewProjectionMatrix * float4(clusterGroupPosition - binormal * clusterGroup.mafMaxErrors[0] * 0.5f, 1.0f);
            //float4 clipSpace1 = viewProjectionMatrix * float4(clusterGroupPosition + binormal * clusterGroup.mafMaxErrors[0] * 0.5f, 1.0f);
            float4 clipSpace0 = viewProjectionMatrix * float4(clusterGroup.maMaxErrorPositions[0][0], 1.0f);
            float4 clipSpace1 = viewProjectionMatrix * float4(clusterGroup.maMaxErrorPositions[0][1], 1.0f);


            // MIP 1
            //float4 clipSpace2 = viewProjectionMatrix * float4(clusterGroupPosition - binormal * clusterGroup.mafMaxErrors[1] * 0.5f, 1.0f);
            //float4 clipSpace3 = viewProjectionMatrix * float4(clusterGroupPosition + binormal * clusterGroup.mafMaxErrors[1] * 0.5f, 1.0f);
            float4 clipSpace2 = viewProjectionMatrix * float4(clusterGroup.maMaxErrorPositions[1][0], 1.0f);
            float4 clipSpace3 = viewProjectionMatrix * float4(clusterGroup.maMaxErrorPositions[1][1], 1.0f);

            //float4 clipSpace0 = viewProjectionMatrix * float4(cluster.mMaxErrorPosition0, 1.0f);
            //float4 clipSpace1 = viewProjectionMatrix * float4(cluster.mMaxErrorPosition1, 1.0f);


            clipSpace0.x /= clipSpace0.w; clipSpace0.y /= clipSpace0.w; clipSpace0.z /= clipSpace0.w;
            clipSpace1.x /= clipSpace1.w; clipSpace1.y /= clipSpace1.w; clipSpace1.z /= clipSpace1.w;
            clipSpace2.x /= clipSpace2.w; clipSpace2.y /= clipSpace2.w; clipSpace2.z /= clipSpace2.w;
            clipSpace3.x /= clipSpace3.w; clipSpace3.y /= clipSpace3.w; clipSpace3.z /= clipSpace3.w;

            clipSpace0 = clipSpace0 * 0.5f + 0.5f;
            clipSpace1 = clipSpace1 * 0.5f + 0.5f;
            clipSpace2 = clipSpace2 * 0.5f + 0.5f;
            clipSpace3 = clipSpace3 * 0.5f + 0.5f;

            float3 diff0 = float3(clipSpace1.x, clipSpace1.y, clipSpace1.z) - float3(clipSpace0.x, clipSpace0.y, clipSpace0.z);
            float3 diff1 = float3(clipSpace3.x, clipSpace3.y, clipSpace3.z) - float3(clipSpace2.x, clipSpace2.y, clipSpace2.z);
            //float fScreenSpacePixelError = maxf(diff.x, maxf(diff.y, diff.z)) * float(iOutputWidth);
            float fScreenSpacePixelError0 = length(diff0) * float(iOutputWidth);
            float fScreenSpacePixelError1 = length(diff1) * float(iOutputWidth);

            DEBUG_PRINTF("error (%.4f, %.4f) screen space pixel error: (%.4fpx, %.4fpx)\n",
                clusterGroup.mafMaxErrors[0],
                clusterGroup.mafMaxErrors[1],
                fScreenSpacePixelError0,
                fScreenSpacePixelError1);

            // update cluster group stack
            aiClusterGroupStack[iClusterGroupStackTop] = iClusterGroup + 1;

            if(fScreenSpacePixelError0 >= fPixelErrorThreshold && fScreenSpacePixelError1 >= fPixelErrorThreshold)
            {
                // go down a LOD level for higher resolution mesh, visiting the children cluster groups
                if(iCurrLODLevel == 0)
                {
                    // at LOD 0, just render all the clusters
                    for(uint32_t iCluster = 0; iCluster < clusterGroup.maiNumClusters[0]; iCluster++)
                    {
                        MeshCluster cluster;
                        loadMeshCluster(cluster, clusterGroup.maiClusters[0][iCluster], clusterBuffer);

                        aRenderClusters[iNumRenderClusters++].miCluster = cluster.miIndex;
                        aRenderClusters[iNumRenderClusters++].miClusterGroup = clusterGroup.miIndex;
                        aRenderClusters[iNumRenderClusters++].miLOD = iCurrLODLevel;
                    }

                    continue;
                }

                iCurrLODLevel = (iCurrLODLevel > 0) ? iCurrLODLevel - 1 : 0;
                iClusterGroupAddress = getMeshClusterGroupAddress(
                    aMeshClusterGroupData,
                    iCurrLODLevel, 
                    0);

                // start at the beginning of the next LOD level
                ++iClusterGroupStackTop;
                aiClusterGroupStack[iClusterGroupStackTop] = 0;
                iNumClusterGroupsAtLOD = getNumClusterGroupsAtLOD(
                    clusterGroupBuffer,
                    iCurrLODLevel);

                ++iLODStackTop;
                aiLODStack[iLODStackTop] = iCurrLODLevel;

                iClusterGroup = -1;
            }
            else
            {
                // found cluster group within the error threshold

                // use MIP 1 if the screen space pixel error is still less than threshold
                uint32_t iMIP = 0;
                if(fScreenSpacePixelError1 <= fPixelErrorThreshold)
                {
                    iMIP = 1;
                }

                for(uint32_t iCluster = 0; iCluster < clusterGroup.maiNumClusters[iMIP]; iCluster++)
                {
                    MeshCluster cluster;
                    loadMeshCluster(cluster, clusterGroup.maiClusters[iMIP][iCluster], clusterBuffer);
                    aRenderClusters[cluster.miIndex].miCluster = cluster.miIndex;
                    aRenderClusters[cluster.miIndex].miClusterGroup = clusterGroup.miIndex;
                    aRenderClusters[cluster.miIndex].miLOD = iCurrLODLevel;
                    aRenderClusters[cluster.miIndex].miMIP = iMIP;
                }

            }

        }   // for cluster group = 0 to num cluster groups at lod level

        --iClusterGroupStackTop;
        --iLODStackTop;

        // done with all the LODs
        if(iClusterGroupStackTop < 0 || iLODStackTop < 0)
        {
            break;
        }

        iCurrLODLevel = (iLODStackTop >= 0) ? aiLODStack[iLODStackTop] : 0;
        iNumClusterGroupsAtLOD = static_cast<int32_t>(getNumClusterGroupsAtLOD(
            clusterGroupBuffer,
            iCurrLODLevel));


    }

    std::vector<std::string> aClusterNames;
    std::vector<uint32_t> aiClusterIndices;
    for(uint32_t i = 0; i < 128; i++)
    {
        if(aRenderClusters[i].miCluster != UINT32_MAX)
        {
            DEBUG_PRINTF("render cluster %d from cluster group %d with MIP %d\n",
                aRenderClusters[i].miCluster,
                aRenderClusters[i].miClusterGroup,
                aRenderClusters[i].miMIP);

            aiClusterIndices.push_back(i);

            MeshCluster cluster;
            loadMeshCluster(
                cluster,
                aRenderClusters[i].miCluster,
                clusterBuffer);

            // cluster culling
            // TODO: verify this
            float fDP = dot(float3(cluster.mNormalCone), direction);
            if(fDP + cluster.mNormalCone.w < 0.0f && fDP - cluster.mNormalCone.w < 0.0f)
            {
                continue;
            }

            {
                std::vector<float3> aVertexPositions(cluster.miNumVertexPositions);
                uint64_t iVertexPositionBufferAddress = cluster.miVertexPositionStartArrayAddress * sizeof(float3);
                float3 const* pVertexPositionBuffer = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + iVertexPositionBufferAddress);
                memcpy(aVertexPositions.data(), pVertexPositionBuffer, sizeof(float3) * aVertexPositions.size());

                std::vector<uint32_t> aiTrianglePositionIndices(cluster.miNumTrianglePositionIndices);
                uint64_t iTriangleIndexAddress = cluster.miTrianglePositionIndexArrayAddress * sizeof(uint32_t);
                uint32_t const* pTrianglePositionIndexBuffer = reinterpret_cast<uint32_t const*>(trianglePositionIndexBuffer.data() + iTriangleIndexAddress);
                memcpy(aiTrianglePositionIndices.data(), pTrianglePositionIndexBuffer, sizeof(uint32_t) * aiTrianglePositionIndices.size());

                std::ostringstream clusterName;
                clusterName << "cluster-lod" << aRenderClusters[i].miLOD << "-mip" << aRenderClusters[i].miMIP << "-cluster" << aRenderClusters[i].miCluster;

                std::string outputDirectory = "c:\\Users\\Dingwings\\demo-models\\render-clusters";
                std::ostringstream outputFilePath;
                outputFilePath << outputDirectory << "\\" << clusterName.str() << ".obj";

                FILE* fp = fopen(outputFilePath.str().c_str(), "wb");
                fprintf(fp, "o %s\n", clusterName.str().c_str());
                fprintf(fp, "usemtl %s\n", clusterName.str().c_str());
                for(uint32_t iV = 0; iV < static_cast<uint32_t>(aVertexPositions.size()); iV++)
                {
                    fprintf(fp, "v %.4f %.4f %.4f\n", aVertexPositions[iV].x, aVertexPositions[iV].y, aVertexPositions[iV].z);
                }
                for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTrianglePositionIndices.size()); iTri += 3)
                {
                    fprintf(fp, "f %d// %d// %d//\n", aiTrianglePositionIndices[iTri] + 1, aiTrianglePositionIndices[iTri + 1] + 1, aiTrianglePositionIndices[iTri + 2] + 1);
                }
                fclose(fp);

                float fRand0 = float(rand() % 255) / 255.0f;
                float fRand1 = float(rand() % 255) / 255.0f;
                float fRand2 = float(rand() % 255) / 255.0f;
                std::ostringstream outputMaterialFilePath;
                outputMaterialFilePath << outputDirectory << "\\" << clusterName.str() << ".mtl";
                fp = fopen(outputMaterialFilePath.str().c_str(), "wb");
                fprintf(fp, "newmtl %s\n", clusterName.str().c_str());
                fprintf(fp, "Kd %.4f %.4f %.4f\n", fRand0, fRand1, fRand2);
                fclose(fp);

                outputMeshToImage(
                    "c:\\Users\\Dingwings\\demo-models\\rasterized-output-images",
                    clusterName.str(),
                    aVertexPositions,
                    aiTrianglePositionIndices,
                    camera,
                    256,
                    256);

                aClusterNames.push_back(clusterName.str());
            }
        }
    }   // for i = 0 to 128

    static std::vector<float3> saColors;
    if(saColors.size() <= 0)
    {
        for(uint32_t i = 0; i < 64; i++)
        {
            float fRand0 = float(rand() % 255) / 255.0f;
            float fRand1 = float(rand() % 255) / 255.0f;
            float fRand2 = float(rand() % 255) / 255.0f;

            saColors.push_back(float3(fRand0, fRand1, fRand2));
        }
    }

    std::vector<float4> aLightingOutput(256 * 256);
    memset(aLightingOutput.data(), 0, sizeof(float4) * 256 * 256);
    std::vector<float4> aDepthOutput(256 * 256);
    for(uint32_t i = 0; i < 256 * 256; i++)
    {
        aDepthOutput[i] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterNames.size()); i++)
    {
        std::string depthImageFilePath = std::string("c:\\Users\\Dingwings\\demo-models\\rasterized-output-images\\") + aClusterNames[i] + "-depth.exr";
        char const* pError = nullptr;
        float* afDepth = nullptr;
        int32_t iWidth = 0, iHeight = 0;
        LoadEXR(&afDepth, &iWidth, &iHeight, depthImageFilePath.c_str(), &pError);

        float* afLighting = nullptr;
        std::string lightingImageFilePath = std::string("c:\\Users\\Dingwings\\demo-models\\rasterized-output-images\\") + aClusterNames[i] + "-lighting.exr";
        LoadEXR(&afLighting, &iWidth, &iHeight, lightingImageFilePath.c_str(), &pError);

        for(int32_t iY = 0; iY < iHeight; iY++)
        {
            for(int32_t iX = 0; iX < iWidth; iX++)
            {
                int32_t iImageIndex = iY * iWidth + iX;
                float fIncomingDepth = 1.0f - afDepth[iImageIndex * 4];
                float4 incomingLighting = float4(afLighting[iImageIndex * 4], afLighting[iImageIndex * 4 + 1], afLighting[iImageIndex * 4 + 2], afLighting[iImageIndex * 4 + 3]);
                incomingLighting.x *= saColors[aiClusterIndices[i]].x;
                incomingLighting.y *= saColors[aiClusterIndices[i]].y;
                incomingLighting.z *= saColors[aiClusterIndices[i]].z;
                if(aDepthOutput[iImageIndex].x > fIncomingDepth)
                {
                    aDepthOutput[iImageIndex] = float4(fIncomingDepth, fIncomingDepth, fIncomingDepth, 1.0f);
                    aLightingOutput[iImageIndex] = incomingLighting;
                }
            }
        }

        free(afDepth);
        free(afLighting);
    }

    char const* pError = nullptr;
    SaveEXR(reinterpret_cast<float const*>(aLightingOutput.data()), 256, 256, 4, 0, "c:\\Users\\Dingwings\\demo-models\\rasterized-output-images\\total-output.exr", &pError);



    std::vector<uint8_t> acImageData(256 * 256 * 4);
    for(int32_t iY = 0; iY < 256; iY++)
    {
        for(int32_t iX = 0; iX < 256; iX++)
        {
            uint32_t iImageIndex = iY * 256 + iX;
            float4 const& lighting = aLightingOutput[iImageIndex];
            acImageData[iImageIndex * 4] = clamp(uint8_t(lighting.x * 255.0f), 0, 255);
            acImageData[iImageIndex * 4 + 1] = clamp(uint8_t(lighting.y * 255.0f), 0, 255);
            acImageData[iImageIndex * 4 + 2] = clamp(uint8_t(lighting.z * 255.0f), 0, 255);
            acImageData[iImageIndex * 4 + 3] = 255;
        }
    }

    static uint32_t siImageIndex = 0;
    std::ostringstream outputLDRImageFilePath;
    outputLDRImageFilePath << "c:\\Users\\Dingwings\\demo-models\\rasterized-output-images\\ldr\\output-" << siImageIndex << ".png";
    stbi_write_png(outputLDRImageFilePath.str().c_str(), 256, 256, 4, acImageData.data(), 256 * 4 * sizeof(char));

    DEBUG_PRINTF("%s\n", outputLDRImageFilePath.str().c_str());

    ++siImageIndex;
}

/*
**
*/
void setClusterTreeNodeErrorTerm(
    std::vector<ClusterTreeNode>& aClusterNodes,
    uint32_t iAddress,
    float fErrorTerm)
{
    auto iter = std::find_if(
        aClusterNodes.begin(),
        aClusterNodes.end(),
        [iAddress](ClusterTreeNode const& checkNode)
        {
            return checkNode.miClusterAddress == iAddress;
        }
    );
    assert(iter != aClusterNodes.end());

    iter->mfScreenSpaceError = fErrorTerm;
}

/*
**
*/
void getCluster(
    ClusterTreeNode& cluster,
    uint32_t iAddress,
    std::vector< ClusterTreeNode> const& aNodes)
{
    auto iter = std::find_if(
        aNodes.begin(),
        aNodes.end(),
        [iAddress](ClusterTreeNode const& checkNode)
        {
            return checkNode.miClusterAddress == iAddress;
        }
    );
    assert(iter != aNodes.end());

    cluster = *iter;
}

/*
**
*/
void setDecendantErrorTerms(
    std::vector<ClusterTreeNode>& aClusterNodes,
    uint32_t iClusterAddress,
    float fErrorTerm)
{
    int32_t aiStack[64] = { 0 };
    int32_t iStackTop = 1;
    aiStack[0] = iClusterAddress;
    for(;;)
    {
        --iStackTop;
        if(iStackTop < 0)
        {
            break;
        }
        uint32_t iCurrClusterAddress = aiStack[iStackTop];
        

        auto iter = std::find_if(
            aClusterNodes.begin(),
            aClusterNodes.end(),
            [iCurrClusterAddress](ClusterTreeNode const& cluster)
            {
                return cluster.miClusterAddress == iCurrClusterAddress;
            }
        );

        if(iter->mfScreenSpaceError == FLT_MAX)
        {
            iter->mfScreenSpaceError = fErrorTerm;
            for(uint32_t iChild = 0; iChild < iter->miNumChildren; iChild++)
            {
                uint32_t iChildAddress = iter->maiChildrenAddress[iChild];
                auto childIter = std::find_if(
                    aClusterNodes.begin(),
                    aClusterNodes.end(),
                    [iChildAddress](ClusterTreeNode const& cluster)
                    {
                        return cluster.miClusterAddress == iChildAddress;
                    }
                );

                if(childIter->mfScreenSpaceError == FLT_MAX)
                {
                    aiStack[iStackTop] = iChildAddress;
                    ++iStackTop;

                    DEBUG_PRINTF("push cluster %d on stack\n", iChildAddress);
                }
            }
        }
    }
}

/*
**
*/
void setClusterErrorTerm(
    std::vector<ClusterTreeNode>& aClusterNodes,
    uint32_t iAddress,
    float fScreenSpaceError,
    bool bCheckExisting)
{
    auto iter = std::find_if(
        aClusterNodes.begin(),
        aClusterNodes.end(),
        [iAddress](ClusterTreeNode const& checkNode)
        {
            return checkNode.miClusterAddress == iAddress;
        }
    );
    assert(iter != aClusterNodes.end());
    if(!bCheckExisting || iter->mfScreenSpaceError == FLT_MAX)
    {
        iter->mfScreenSpaceError = fScreenSpaceError;
    }
}

/*
**
*/
void setClusterErrorTerm2(
    uint8_t* aClusterNodes,
    uint32_t iAddress,
    float fScreenSpaceError,
    bool bCheckExisting)
{
    bool bFound = false;
    uint32_t iNumClusterNodes = *(reinterpret_cast<uint32_t*>(aClusterNodes));
    ClusterTreeNode* paClusterNodes = reinterpret_cast<ClusterTreeNode*>(aClusterNodes + sizeof(uint32_t));
    for(uint32_t iCluster = 0; iCluster < iNumClusterNodes; iCluster++)
    {
        if(paClusterNodes[iCluster].miClusterAddress == iAddress)
        {
            paClusterNodes[iCluster].mfScreenSpaceError = fScreenSpaceError;
            bFound = true;
            break;
        }
    }

    assert(bFound);
}


/*
**
*/
void testClusterLOD2(
    std::vector<ClusterTreeNode>& aClusterNodes,
    std::vector<ClusterGroupTreeNode>& aClusterGroupNodes,
    std::vector<uint32_t> const& aiLevelStartGroupIndices,
    std::vector<uint32_t> const& aiNumLevelGroupNodes,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    float fPixelErrorThreshold)
{
    float const kfCameraNear = 1.0f;
    float const kfCameraFar = 100.0f;

    float3 direction = normalize(cameraLookAt - cameraPosition);
    float3 up = (fabsf(direction.z) > fabsf(direction.x) && fabsf(direction.z) > fabsf(direction.y)) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 binormal = normalize(cross(up, direction));

    // camera 
    CCamera camera;
    camera.setFar(kfCameraFar);
    camera.setNear(kfCameraNear);
    camera.setLookAt(cameraLookAt);
    camera.setPosition(cameraPosition);
    CameraUpdateInfo cameraUpdateInfo =
    {
        /* .mfViewWidth      */  1000.0f,
        /* .mfViewHeight     */  1000.0f,
        /* .mfFieldOfView    */  3.14159f * 0.5f,
        /* .mUp              */  up,
        /* .mfNear           */  kfCameraNear,
        /* .mfFar            */  kfCameraFar,
    };
    camera.update(cameraUpdateInfo);

    mat4 const& viewMatrix = camera.getViewMatrix();
    mat4 const& projectionMatrix = camera.getProjectionMatrix();
    mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;
    
    float3 cameraBinormal = float3(0.0f, 0.0f, 0.0f);
    {
        float3 cameraLookAt = normalize(camera.getLookAt() - camera.getPosition());
        float3 up = (fabsf(cameraLookAt.y) > fabsf(cameraLookAt.x) && fabsf(cameraLookAt.y) > fabsf(cameraLookAt.z)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        cameraBinormal = cross(up, cameraLookAt);
    }

    int32_t aiClusterGroupStack[128] = { 0 };
    int32_t iClusterGroupStackTop = 0;

    std::vector<uint32_t> aiRenderClusterCandidates;

    struct ScreenSpaceErrorInfo
    {
        uint32_t        miCluster;
        uint32_t        miClusterGroup;
        float           mfError;
    };

    std::vector<ScreenSpaceErrorInfo> aScreenSpaceErrors;

    int32_t iNumLevels = static_cast<uint32_t>(aiNumLevelGroupNodes.size());
    for(int32_t iLevel = iNumLevels - 1; iLevel >= 0; iLevel--)
    {
        uint32_t iNumClusterGroupsAtLevel = aiNumLevelGroupNodes[iLevel];
        uint32_t iStartIndex = aiLevelStartGroupIndices[iLevel];
        for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroupsAtLevel; iClusterGroup++)
        {
            float fScreenSpacePixelError = 0.0f;
            ClusterGroupTreeNode& clusterGroup = aClusterGroupNodes[iClusterGroup + iStartIndex];
            DEBUG_PRINTF("check cluster group %d\n", clusterGroup.miClusterGroupAddress);
            {
                for(uint32_t i = 0; i < clusterGroup.miNumChildClusters; i++)
                {
                    ClusterTreeNode cluster;
                    getCluster(cluster, clusterGroup.maiClusterAddress[i], aClusterNodes);

                    //float4 clipSpace0 = viewProjectionMatrix * float4(cluster.mMaxDistanceCurrLODClusterPosition, 1.0f);
                    //float4 clipSpace1 = viewProjectionMatrix * float4(cluster.mMaxDistanceLOD0ClusterPosition, 1.0f);
                    //
                    //clipSpace0.x /= clipSpace0.w; clipSpace0.y /= clipSpace0.w; clipSpace0.z /= clipSpace0.w;
                    //clipSpace1.x /= clipSpace1.w; clipSpace1.y /= clipSpace1.w; clipSpace1.z /= clipSpace1.w;
                    //
                    //clipSpace0 = clipSpace0 * 0.5f + 0.5f;
                    //clipSpace1 = clipSpace1 * 0.5f + 0.5f;
                    //
                    //float3 diff = float3(clipSpace1.x, clipSpace1.y, clipSpace1.z) - float3(clipSpace0.x, clipSpace0.y, clipSpace0.z);
                    //float fClusterError = length(diff) * float(iOutputWidth);
                    //fScreenSpacePixelError = maxf(fClusterError, fScreenSpacePixelError);

                    // compute the screen projected average error distance
                    {
                        float4 pt0 = float4(cameraBinormal * cluster.mfAverageDistanceFromLOD0 * -0.5f, 1.0f);
                        float4 pt1 = float4(cameraBinormal * cluster.mfAverageDistanceFromLOD0 * 0.5f, 1.0f);

                        float4 clipSpacePt0 = viewProjectionMatrix * pt0;
                        float4 clipSpacePt1 = viewProjectionMatrix * pt1;

                        clipSpacePt0.x /= clipSpacePt0.w; clipSpacePt0.y /= clipSpacePt0.w; clipSpacePt0.z /= clipSpacePt0.w;
                        clipSpacePt1.x /= clipSpacePt1.w; clipSpacePt1.y /= clipSpacePt1.w; clipSpacePt1.z /= clipSpacePt1.w;

                        clipSpacePt0 = clipSpacePt0 * 0.5f + 0.5f;
                        clipSpacePt1 = clipSpacePt1 * 0.5f + 0.5f;

                        float3 diff = float3(clipSpacePt1.x, clipSpacePt1.y, clipSpacePt1.z) - float3(clipSpacePt0.x, clipSpacePt0.y, clipSpacePt0.z);
                        float fAverageClusterError = length(diff) * float(iOutputWidth);
                        fScreenSpacePixelError = maxf(fAverageClusterError, fScreenSpacePixelError);
                    }

#if 0
                    // TODO: needs to make sure the error is always smaller in the lower LOD
                    // check smaller parent cluster error, set to a percentage of it if possible
                    if(cluster.miNumParents > 0)
                    {
                        float fParentMaxError = FLT_MAX;
                        for(uint32_t iParent = 0; iParent < cluster.miNumParents; iParent++)
                        {
                            ClusterTreeNode parentCluster;
                            getCluster(parentCluster, cluster.maiParentAddress[iParent], aClusterNodes);
                            if(parentCluster.mfScreenSpaceError < fScreenSpacePixelError)
                            {
                                if(fParentMaxError == FLT_MAX)
                                {
                                    fParentMaxError = parentCluster.mfScreenSpaceError;
                                }
                                else
                                {
                                    fParentMaxError = maxf(fParentMaxError, parentCluster.mfScreenSpaceError);
                                }
                            }
                        }

                        if(fParentMaxError < fScreenSpacePixelError)
                        {
                            fClusterError = fParentMaxError * 0.8f;
                            fScreenSpacePixelError = fClusterError;
                        }
                    }   // for i = 0 to num children in cluster group
#endif // #if 0

                    setClusterErrorTerm(
                        aClusterNodes,
                        clusterGroup.maiClusterAddress[i],
                        fScreenSpacePixelError,
                        true);

                    ScreenSpaceErrorInfo errorInfo;
                    errorInfo.miCluster = cluster.miClusterAddress;
                    errorInfo.miClusterGroup = clusterGroup.miClusterGroupAddress;
                    errorInfo.mfError = fScreenSpacePixelError;

                    aScreenSpaceErrors.push_back(errorInfo);
                }

                for(uint32_t i = 0; i < clusterGroup.miNumChildClusters; i++)
                {
                    setClusterErrorTerm(
                        aClusterNodes,
                        clusterGroup.maiClusterAddress[i],
                        fScreenSpacePixelError,
                        true);
                }
            }
            
            clusterGroup.mfScreenSpacePixelError = fScreenSpacePixelError;

            if(fScreenSpacePixelError <= fPixelErrorThreshold)
            {
                // small enough error, render
                for(uint32_t i = 0; i < clusterGroup.miNumChildClusters; i++)
                {
                    uint32_t iClusterAddress = clusterGroup.maiClusterAddress[i];
                    aiRenderClusterCandidates.push_back(iClusterAddress);
                    
                    auto iter = std::find_if(
                        aClusterNodes.begin(),
                        aClusterNodes.end(),
                        [iClusterAddress](ClusterTreeNode const& checkNode)
                        {
                            return checkNode.miClusterAddress == iClusterAddress;
                        }
                    );
                    assert(iter != aClusterNodes.end());

                    // set all the decendants to the error which in essence culls them away
                    setDecendantErrorTerms(
                        aClusterNodes, 
                        iClusterAddress, 
                        fScreenSpacePixelError);
                }
            }
            
        }   // for cluster group = 0 to num cluster groups at level

    }   // for level = 0 to num cluster group levels

    std::sort(
        aScreenSpaceErrors.begin(),
        aScreenSpaceErrors.end(),
        [](ScreenSpaceErrorInfo const& left, ScreenSpaceErrorInfo const& right)
        {
            return left.miCluster < right.miCluster;
        }
    );

    // cull out clusters
    std::vector<uint32_t> aiClustersToRender;
    for(auto const& iClusterAddress : aiRenderClusterCandidates)
    {
        ClusterTreeNode cluster;
        getCluster(cluster, iClusterAddress, aClusterNodes);

        DEBUG_PRINTF("check cluster %d with error %.4f\n", iClusterAddress, cluster.mfScreenSpaceError);

        float fParentErrorTerm = FLT_MAX;
        for(uint32_t iParent = 0; iParent < cluster.miNumParents; iParent++)
        {
            uint32_t iParentAddress = cluster.maiParentAddress[iParent];
            ClusterTreeNode parentCluster;
            getCluster(parentCluster, iParentAddress, aClusterNodes);
            fParentErrorTerm = (iParent == 0) ? parentCluster.mfScreenSpaceError : maxf(fParentErrorTerm, parentCluster.mfScreenSpaceError);

            DEBUG_PRINTF("\tparent cluster %d with error %.4f\n", iParentAddress, fParentErrorTerm);
        }

        if(fParentErrorTerm > fPixelErrorThreshold && cluster.mfScreenSpaceError <= fPixelErrorThreshold)
        {
            aiClustersToRender.push_back(iClusterAddress);
            DEBUG_PRINTF("\t!!! add cluster %d !!!\n", iClusterAddress);
        }
    }

    
    FILE* fp = fopen("c:\\Users\\Dingwings\\demo-models\\test-render-clusters\\draw_clusters.py", "wb");
    fprintf(fp, "import bpy\n");
    for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aiClustersToRender.size()); iCluster++)
    {
        uint32_t iClusterAddress = aiClustersToRender[iCluster];
        auto iter = std::find_if(
            aClusterNodes.begin(),
            aClusterNodes.end(),
            [iClusterAddress](ClusterTreeNode const& checkNode)
            {
                return checkNode.miClusterAddress == iClusterAddress;
            }
        );
        assert(iter != aClusterNodes.end());

        std::ostringstream fullPath;
        fullPath << "c:\\\\Users\\\\Dingwings\\\\demo-models\\\\test-render-clusters\\\\cluster-" << iter->miClusterAddress << ".obj";

        fprintf(fp, "bpy.ops.import_scene.obj(filepath=\'%s\', axis_forward='-Z', axis_up='Y', use_split_groups = True)\n", fullPath.str().c_str());
    }
    fclose(fp);
}

/*
**
*/
bool _hasVisited(uint32_t const* aiVisited, uint32_t iClusterAddress)
{
    return (aiVisited[iClusterAddress] > 0);
}

/*
**
*/
void _traverseClusterNodes(
    std::vector<uint32_t>& aiDrawClusterAddress,
    std::vector<uint32_t>& aiVisited,
    ClusterTreeNode const& rootCluster,
    std::vector<ClusterTreeNode> const& aClusterNodes,
    float fDrawScreenErrorThreshold)
{
    uint32_t* paiVisited = aiVisited.data();
    uint32_t iNumVisited = 0;
    
    int32_t aiStack[128] = { 0 };
    aiStack[0] = rootCluster.miClusterAddress;
    int32_t iStackTop = 1;
    for(;;)
    {
        --iStackTop;
        if(iStackTop < 0)
        {
            break;
        }
        uint32_t iClusterAddress = aiStack[iStackTop];
        ClusterTreeNode clusterNode;
        getCluster(clusterNode, iClusterAddress, aClusterNodes);
        
        DEBUG_PRINTF("cluster %d error: %.4f\n", iClusterAddress, clusterNode.mfScreenSpaceError);
        if(clusterNode.mfScreenSpaceError > fDrawScreenErrorThreshold)
        {
            if(!_hasVisited(paiVisited, iClusterAddress))
            {
                for(uint32_t iChild = 0; iChild < clusterNode.miNumChildren; iChild++)
                {
                    if(!_hasVisited(paiVisited, clusterNode.maiChildrenAddress[iChild]))
                    {
                        DEBUG_PRINTF("\tadd child cluster %d\n", clusterNode.maiChildrenAddress[iChild]);
                        
                        assert(iStackTop < 128);
                        aiStack[iStackTop] = clusterNode.maiChildrenAddress[iChild];
                        ++iStackTop;
                    }
                }
            }
        }
        else
        {
            aiDrawClusterAddress.push_back(clusterNode.miClusterAddress);

            DEBUG_PRINTF("\t!!! draw this cluster %d (%lld) error: %.4f !!!\n", 
                iClusterAddress, 
                aiDrawClusterAddress.size(),
                clusterNode.mfScreenSpaceError);
        }

        assert(iClusterAddress < 65536);
        aiVisited[iClusterAddress] = 1;
        DEBUG_PRINTF("\n");
    }

    DEBUG_PRINTF("!!! done with root cluster %d !!!\n", rootCluster.miClusterAddress);
}

/*
**
*/
void getCluster2(
    ClusterTreeNode& cluster,
    uint32_t iAddress,
    uint8_t const* aNodes)
{
    uint32_t iNumNodes = *(reinterpret_cast<uint32_t const*>(aNodes));
    ClusterTreeNode const* paNodes = reinterpret_cast<ClusterTreeNode const*>(aNodes + sizeof(uint32_t));

    ClusterTreeNode const* pNode = paNodes;
    for(uint32_t i = 0; i < iNumNodes; i++)
    {
        if(pNode->miClusterAddress == iAddress)
        {
            cluster = *pNode;
            break;
        }
        ++pNode;
    }
}

/*
**
*/
void _traverseClusterNodes2(
    uint8_t* aiDrawClusterAddress,
    uint8_t* aiVisited,
    ClusterTreeNode const& rootCluster,
    uint8_t const* aClusterNodes,
    float fDrawScreenErrorThreshold)
{
    uint32_t* paiVisited = reinterpret_cast<uint32_t*>(aiVisited);
    uint32_t* pStart = paiVisited;
    *pStart = 65536;
    paiVisited = pStart + 1;
    uint32_t iNumVisited = 0;

    uint32_t* paiDrawClusterAddressStart = reinterpret_cast<uint32_t*>(aiDrawClusterAddress);
    uint32_t* paiDrawClusterAddress = paiDrawClusterAddressStart + 1;

    int32_t aiStack[128] = { 0 };
    aiStack[0] = rootCluster.miClusterAddress;
    int32_t iStackTop = 1;
    for(;;)
    {
        --iStackTop;
        if(iStackTop < 0)
        {
            break;
        }
        uint32_t iClusterAddress = aiStack[iStackTop];
        ClusterTreeNode clusterNode;
        getCluster2(clusterNode, iClusterAddress, aClusterNodes);

        DEBUG_PRINTF("cluster %d error: %.4f\n", iClusterAddress, clusterNode.mfScreenSpaceError);
        if(clusterNode.mfScreenSpaceError > fDrawScreenErrorThreshold)
        {
            if(!_hasVisited(paiVisited, iClusterAddress))
            {
                for(uint32_t iChild = 0; iChild < clusterNode.miNumChildren; iChild++)
                {
                    if(!_hasVisited(paiVisited, clusterNode.maiChildrenAddress[iChild]))
                    {
                        DEBUG_PRINTF("\tadd child cluster %d\n", clusterNode.maiChildrenAddress[iChild]);

                        assert(iStackTop < 128);
                        aiStack[iStackTop] = clusterNode.maiChildrenAddress[iChild];
                        ++iStackTop;
                    }
                }
            }
        }
        else
        {
            //aiDrawClusterAddress.push_back(clusterNode.miClusterAddress);
            uint32_t iSize = *paiDrawClusterAddressStart;
            *(paiDrawClusterAddress + iSize) = clusterNode.miClusterAddress;
            *paiDrawClusterAddressStart = iSize + 1;

            DEBUG_PRINTF("\t!!! draw this cluster %d (%d) error: %.4f !!!\n", 
                iClusterAddress, 
                *paiDrawClusterAddressStart,
                clusterNode.mfScreenSpaceError);
        }

        assert(iClusterAddress < 65536);
        paiVisited[iClusterAddress] = 1;
        assert(*pStart == 65536);
        DEBUG_PRINTF("\n");
    }

    DEBUG_PRINTF("!!! done with root cluster %d !!!\n", rootCluster.miClusterAddress);
}

/*
**
*/
void testClusterLOD3(
    std::vector<uint32_t>& aiDrawClusterAddress,
    std::vector<ClusterTreeNode>& aClusterNodes,
    std::vector<ClusterGroupTreeNode>& aClusterGroupNodes,
    std::vector<uint32_t> const& aiLevelStartGroupIndices,
    std::vector<uint32_t> const& aiNumLevelGroupNodes,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    float fPixelErrorThreshold)
{
    PrintOptions option;
    option.mbDisplayTime = false;
    setPrintOptions(option);

    float const kfCameraNear = 1.0f;
    float const kfCameraFar = 100.0f;

    float3 direction = normalize(cameraLookAt - cameraPosition);
    float3 up = (fabsf(direction.z) > fabsf(direction.x) && fabsf(direction.z) > fabsf(direction.y)) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 binormal = normalize(cross(up, direction));

    // camera 
    CCamera camera;
    camera.setFar(kfCameraFar);
    camera.setNear(kfCameraNear);
    camera.setLookAt(cameraLookAt);
    camera.setPosition(cameraPosition);
    CameraUpdateInfo cameraUpdateInfo =
    {
        /* .mfViewWidth      */  1000.0f,
        /* .mfViewHeight     */  1000.0f,
        /* .mfFieldOfView    */  3.14159f * 0.5f,
        /* .mUp              */  up,
        /* .mfNear           */  kfCameraNear,
        /* .mfFar            */  kfCameraFar,
    };
    camera.update(cameraUpdateInfo);

    mat4 const& viewMatrix = camera.getViewMatrix();
    mat4 const& projectionMatrix = camera.getProjectionMatrix();
    mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

    float3 cameraBinormal = float3(0.0f, 0.0f, 0.0f);
    {
        float3 cameraLookAt = normalize(camera.getLookAt() - camera.getPosition());
        float3 up = (fabsf(cameraLookAt.y) > fabsf(cameraLookAt.x) && fabsf(cameraLookAt.y) > fabsf(cameraLookAt.z)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        cameraBinormal = cross(up, cameraLookAt);
    }

    int32_t aiClusterGroupStack[128] = { 0 };
    int32_t iClusterGroupStackTop = 0;

    uint32_t const kiMaxDrawClusters = 1 << 16;
    std::vector<uint8_t> aiDrawClusterAddressCopy(kiMaxDrawClusters * sizeof(uint32_t) + sizeof(uint32_t));
    uint8_t* aiDrawClusterAddress2 = aiDrawClusterAddressCopy.data();
    
    std::vector<uint8_t> aClusterNodesCopy(aClusterNodes.size() * sizeof(ClusterTreeNode) + sizeof(uint32_t));
    uint8_t* aClusterNodes2 = aClusterNodesCopy.data();
    uint32_t* paiClusterNodes2 = reinterpret_cast<uint32_t*>(aClusterNodes2);
    *paiClusterNodes2 = static_cast<uint32_t>(aClusterNodes.size());
    memcpy(aClusterNodes2 + sizeof(uint32_t), aClusterNodes.data(), aClusterNodes.size() * sizeof(ClusterTreeNode));
    
    // compute group node screen pixel erros
    int32_t iNumLevels = static_cast<uint32_t>(aiNumLevelGroupNodes.size());
    for(int32_t iLevel = iNumLevels - 1; iLevel >= 0; iLevel--)
    {
        uint32_t iNumClusterGroupsAtLevel = aiNumLevelGroupNodes[iLevel];
        uint32_t iStartIndex = aiLevelStartGroupIndices[iLevel];
        for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroupsAtLevel; iClusterGroup++)
        {
            float fScreenSpacePixelError = 0.0f;
            ClusterGroupTreeNode& clusterGroup = aClusterGroupNodes[iClusterGroup + iStartIndex];
            DEBUG_PRINTF("check cluster group %d\n", clusterGroup.miClusterGroupAddress);
            {
                for(uint32_t i = 0; i < clusterGroup.miNumChildClusters; i++)
                {
                    ClusterTreeNode cluster;
                    //getCluster(cluster, clusterGroup.maiClusterAddress[i], aClusterNodes);
                    getCluster2(cluster, clusterGroup.maiClusterAddress[i], aClusterNodes2);

                    //float4 clipSpace0 = viewProjectionMatrix * float4(cluster.mMaxDistanceCurrLODClusterPosition, 1.0f);
                    //float4 clipSpace1 = viewProjectionMatrix * float4(cluster.mMaxDistanceLOD0ClusterPosition, 1.0f);
                    //
                    //clipSpace0.x /= clipSpace0.w; clipSpace0.y /= clipSpace0.w; clipSpace0.z /= clipSpace0.w;
                    //clipSpace1.x /= clipSpace1.w; clipSpace1.y /= clipSpace1.w; clipSpace1.z /= clipSpace1.w;
                    //
                    //clipSpace0 = clipSpace0 * 0.5f + 0.5f;
                    //clipSpace1 = clipSpace1 * 0.5f + 0.5f;
                    //
                    //float3 diff = float3(clipSpace1.x, clipSpace1.y, clipSpace1.z) - float3(clipSpace0.x, clipSpace0.y, clipSpace0.z);
                    //float fClusterError = length(diff) * float(iOutputWidth);
                    //fScreenSpacePixelError = maxf(fClusterError, fScreenSpacePixelError);

                    // compute the screen projected average error distance
                    {
                        float4 pt0 = float4(cameraBinormal * cluster.mfAverageDistanceFromLOD0 * -0.5f, 1.0f);
                        float4 pt1 = float4(cameraBinormal * cluster.mfAverageDistanceFromLOD0 * 0.5f, 1.0f);

                        float4 clipSpacePt0 = viewProjectionMatrix * pt0;
                        float4 clipSpacePt1 = viewProjectionMatrix * pt1;

                        clipSpacePt0.x /= clipSpacePt0.w; clipSpacePt0.y /= clipSpacePt0.w; clipSpacePt0.z /= clipSpacePt0.w;
                        clipSpacePt1.x /= clipSpacePt1.w; clipSpacePt1.y /= clipSpacePt1.w; clipSpacePt1.z /= clipSpacePt1.w;

                        clipSpacePt0 = clipSpacePt0 * 0.5f + 0.5f;
                        clipSpacePt1 = clipSpacePt1 * 0.5f + 0.5f;

                        float3 diff = float3(clipSpacePt1.x, clipSpacePt1.y, clipSpacePt1.z) - float3(clipSpacePt0.x, clipSpacePt0.y, clipSpacePt0.z);
                        float fAverageClusterError = length(diff) * float(iOutputWidth);
                        fScreenSpacePixelError = maxf(fAverageClusterError, fScreenSpacePixelError);
                    }

                }   // for i = 0 to num child clusters

            }

            DEBUG_PRINTF("cluster group %d error: %.4f\n", clusterGroup.miClusterGroupAddress, fScreenSpacePixelError);
            clusterGroup.mfScreenSpacePixelError = fScreenSpacePixelError;

            for(uint32_t iCluster = 0; iCluster < clusterGroup.miNumChildClusters; iCluster++)
            {
                //setClusterErrorTerm(
                //    aClusterNodes,
                //    clusterGroup.maiClusterAddress[iCluster],
                //    fScreenSpacePixelError,
                //    false);
                setClusterErrorTerm2(
                    aClusterNodes2,
                    clusterGroup.maiClusterAddress[iCluster],
                    fScreenSpacePixelError,
                    false);
            }

        }   // for cluster group = 0 to num cluster groups at level

    }   // for level = 0 to num cluster group levels

    uint32_t const kiMaxVisitedClusterFlags = 65536;

    
    std::vector<uint32_t> aiVisited(kiMaxVisitedClusterFlags);
    memset(aiVisited.data(), 0, sizeof(uint32_t) * aiVisited.size());

    std::vector<uint32_t> aiVisited2(kiMaxVisitedClusterFlags);
    memset(aiVisited2.data(), 0, sizeof(uint32_t) * aiVisited2.size());

    uint32_t iLevel = iNumLevels - 1;
    uint32_t iNumClusterGroupsAtLevel = aiNumLevelGroupNodes[iLevel];
    uint32_t iStartIndex = aiLevelStartGroupIndices[iLevel];
    //for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroupsAtLevel; iClusterGroup++)
    for(uint32_t iClusterGroup = 0; iClusterGroup < aClusterGroupNodes.size(); iClusterGroup++)
    {
        auto const& clusterGroup = aClusterGroupNodes[iClusterGroup];
        if(clusterGroup.miLevel != iLevel)
        {
            continue;
        }

        uint32_t iNumRootClusters = clusterGroup.miNumChildClusters;
        for(uint32_t iCluster = 0; iCluster < iNumRootClusters; iCluster++)
        {
            ClusterTreeNode rootCluster;
            //getCluster(rootCluster, clusterGroup.maiClusterAddress[iCluster], aClusterNodes);
            //_traverseClusterNodes(
            //    aiDrawClusterAddress,
            //    aiVisited,
            //    rootCluster,
            //    aClusterNodes,
            //    fPixelErrorThreshold);
            getCluster2(rootCluster, clusterGroup.maiClusterAddress[iCluster], aClusterNodes2);
            _traverseClusterNodes2(
                aiDrawClusterAddress2,
                reinterpret_cast<uint8_t*>(aiVisited2.data()),
                rootCluster,
                aClusterNodes2,
                fPixelErrorThreshold);
        }
    }
    
    uint32_t iNumClusters = *(reinterpret_cast<uint32_t*>(aiDrawClusterAddress2));
    aiDrawClusterAddress.resize(iNumClusters);
    memcpy(aiDrawClusterAddress.data(), aiDrawClusterAddress2 + sizeof(uint32_t), iNumClusters * sizeof(uint32_t));

    // python script to load clusters
    FILE* fp = fopen("c:\\Users\\Dingwings\\demo-models\\test-render-clusters\\draw_clusters.py", "wb");
    fprintf(fp, "import bpy\n");
    for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aiDrawClusterAddress.size()); iCluster++)
    {
        uint32_t iClusterAddress = aiDrawClusterAddress[iCluster];
        auto iter = std::find_if(
            aClusterNodes.begin(),
            aClusterNodes.end(),
            [iClusterAddress](ClusterTreeNode const& checkNode)
            {
                return checkNode.miClusterAddress == iClusterAddress;
            }
        );
        assert(iter != aClusterNodes.end());

        std::ostringstream fullPath;
        fullPath << "c:\\\\Users\\\\Dingwings\\\\demo-models\\\\test-render-clusters\\\\cluster-" << iter->miClusterAddress << ".obj";

        fprintf(fp, "bpy.ops.import_scene.obj(filepath=\'%s\', axis_forward='-Z', axis_up='Y', use_split_groups = True)\n", fullPath.str().c_str());
    }
    fclose(fp);

    option.mbDisplayTime = true;
    setPrintOptions(option);
}



/*
**
*/
void drawMeshClusterImage2(
    std::vector<float3>& aLightIntensityBuffer,
    std::vector<float3>& aPositionBuffer,
    std::vector<float3>& aNormalBuffer,
    std::vector<float>& afDepthBuffer,
    std::vector<float3>& aColorBuffer,
    std::vector<uint32_t> const& aiClusterAddress,
    std::vector<float3> const& aClusterColors,
    std::vector<MeshCluster*> const& aMeshClusters,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<uint8_t>& vertexNormalBuffer,
    std::vector<uint8_t>& trianglePositionIndexBuffer,
    std::vector<uint8_t>& triangleNormalIndexBuffer,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight)
{
    float const kfCameraNear = 1.0f;
    float const kfCameraFar = 100.0f;

    uint32_t const kiImageWidth = 1024;
    uint32_t const kiImageHeight = 1024;

    float3 direction = normalize(cameraLookAt - cameraPosition);
    float3 up = (fabsf(direction.z) > fabsf(direction.x) && fabsf(direction.z) > fabsf(direction.y)) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

    // camera 
    CCamera camera;
    camera.setFar(kfCameraFar);
    camera.setNear(kfCameraNear);
    camera.setLookAt(cameraLookAt);
    camera.setPosition(cameraPosition);
    CameraUpdateInfo cameraUpdateInfo =
    {
        /* .mfViewWidth      */  1000.0f,
        /* .mfViewHeight     */  1000.0f,
        /* .mfFieldOfView    */  3.14159f * 0.5f,
        /* .mUp              */  up,
        /* .mfNear           */  kfCameraNear,
        /* .mfFar            */  kfCameraFar,
    };
    camera.update(cameraUpdateInfo);
    mat4 const& viewMatrix = camera.getViewMatrix();
    mat4 const& projectionMatrix = camera.getProjectionMatrix();
    mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

    std::vector<float3> aTotalTrianglePositions;
    std::vector<float3> aTotalTriangleNormals;
    std::vector<float3> aTotalTriangleColors;

    std::vector<std::string> aClusterNames;
    for(auto const& iClusterAddress : aiClusterAddress)
    {
        auto iter = std::find_if(
            aMeshClusters.begin(),
            aMeshClusters.end(),
            [iClusterAddress](MeshCluster const* pMeshCluster)
            {
                return pMeshCluster->miIndex == iClusterAddress;
            }
        );
        assert(iter != aMeshClusters.end());

        std::vector<float3> aVertexPositions((*iter)->miNumVertexPositions);
        uint64_t iVertexPositionBufferAddress = (*iter)->miVertexPositionStartArrayAddress * sizeof(float3);
        float3 const* pVertexPositionBuffer = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + iVertexPositionBufferAddress);
        memcpy(aVertexPositions.data(), pVertexPositionBuffer, sizeof(float3) * aVertexPositions.size());

        std::vector<float3> aVertexNormals((*iter)->miNumVertexNormals);
        uint64_t iVertexNormalBufferAddress = (*iter)->miVertexNormalStartArrayAddress * sizeof(float3);
        float3 const* pVertexNormalBuffer = reinterpret_cast<float3 const*>(vertexNormalBuffer.data() + iVertexNormalBufferAddress);
        memcpy(aVertexNormals.data(), pVertexNormalBuffer, sizeof(float3) * aVertexNormals.size());

        std::vector<uint32_t> aiTrianglePositionIndices((*iter)->miNumTrianglePositionIndices);
        uint64_t iTriangleIndexAddress = (*iter)->miTrianglePositionIndexArrayAddress * sizeof(uint32_t);
        uint32_t const* pTrianglePositionIndexBuffer = reinterpret_cast<uint32_t const*>(trianglePositionIndexBuffer.data() + iTriangleIndexAddress);
        memcpy(aiTrianglePositionIndices.data(), pTrianglePositionIndexBuffer, sizeof(uint32_t) * aiTrianglePositionIndices.size());

        std::vector<uint32_t> aiTriangleNormalIndices((*iter)->miNumTriangleNormalIndices);
        uint64_t iTriangleNormalIndexAddress = (*iter)->miTriangleNormalIndexArrayAddress * sizeof(uint32_t);
        uint32_t const* pTriangleNormalIndexBuffer = reinterpret_cast<uint32_t const*>(triangleNormalIndexBuffer.data() + iTriangleNormalIndexAddress);
        memcpy(aiTriangleNormalIndices.data(), pTriangleNormalIndexBuffer, sizeof(uint32_t) * aiTriangleNormalIndices.size());

        // transform into camera space (view projection)
        std::vector<float4> aXFormVertexPositions(aVertexPositions.size());
        for(uint32_t i = 0; i < static_cast<uint32_t>(aVertexPositions.size()); i++)
        {
            aXFormVertexPositions[i] = viewProjectionMatrix * float4(aVertexPositions[i], 1.0f);
        }

        // clip space positions
        float3 minClipSpacePosition = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxClipSpacePosition = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        std::vector<float3> aClipSpaceVertexPositions(aXFormVertexPositions.size());
        for(uint32_t iV = 0; iV < static_cast<uint32_t>(aXFormVertexPositions.size()); iV++)
        {
            aClipSpaceVertexPositions[iV].x = aXFormVertexPositions[iV].x / aXFormVertexPositions[iV].w;
            aClipSpaceVertexPositions[iV].y = aXFormVertexPositions[iV].y / aXFormVertexPositions[iV].w;
            aClipSpaceVertexPositions[iV].z = aXFormVertexPositions[iV].z / aXFormVertexPositions[iV].w;

            aClipSpaceVertexPositions[iV].x = aClipSpaceVertexPositions[iV].x * 0.5f + 0.5f;
            aClipSpaceVertexPositions[iV].y = 1.0f - (aClipSpaceVertexPositions[iV].y * 0.5f + 0.5f);
            aClipSpaceVertexPositions[iV].z = aClipSpaceVertexPositions[iV].z * 0.5f + 0.5f;
        }

        // triangle clip space positions and normals
        std::vector<float3> aTriangleClipSpaceVertexPositions(aiTrianglePositionIndices.size());
        std::vector<float3> aTriangleVertexNormals(aiTrianglePositionIndices.size());
        std::vector<float3> aTriangleVertexColor(aiTrianglePositionIndices.size());
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTrianglePositionIndices.size()); iTri += 3)
        {
            uint32_t iPos0 = aiTrianglePositionIndices[iTri];
            uint32_t iPos1 = aiTrianglePositionIndices[iTri + 1];
            uint32_t iPos2 = aiTrianglePositionIndices[iTri + 2];

            aTriangleClipSpaceVertexPositions[iTri] = aClipSpaceVertexPositions[iPos0];
            aTriangleClipSpaceVertexPositions[iTri + 1] = aClipSpaceVertexPositions[iPos1];
            aTriangleClipSpaceVertexPositions[iTri + 2] = aClipSpaceVertexPositions[iPos2];

            uint32_t iNorm0 = aiTriangleNormalIndices[iTri];
            uint32_t iNorm1 = aiTriangleNormalIndices[iTri + 1];
            uint32_t iNorm2 = aiTriangleNormalIndices[iTri + 2];

            aTriangleVertexNormals[iTri] = aVertexNormals[iNorm0];
            aTriangleVertexNormals[iTri + 1] = aVertexNormals[iNorm1];
            aTriangleVertexNormals[iTri + 2] = aVertexNormals[iNorm2];

            aTriangleVertexColor[iTri] = aClusterColors[iClusterAddress];
            aTriangleVertexColor[iTri + 1] = aClusterColors[iClusterAddress];
            aTriangleVertexColor[iTri + 2] = aClusterColors[iClusterAddress];
        }

        aTotalTrianglePositions.insert(aTotalTrianglePositions.end(), aTriangleClipSpaceVertexPositions.begin(), aTriangleClipSpaceVertexPositions.end());
        aTotalTriangleNormals.insert(aTotalTriangleNormals.end(), aTriangleVertexNormals.begin(), aTriangleVertexNormals.end());
        aTotalTriangleColors.insert(aTotalTriangleColors.end(), aTriangleVertexColor.begin(), aTriangleVertexColor.end());

    }   // for cluster = 0 to num clusters

    //rasterizeMeshCUDA2(
    //    aLightIntensityBuffer,
    //    aPositionBuffer,
    //    aNormalBuffer,
    //    afDepthBuffer,
    //    aColorBuffer,
    //    aTotalTrianglePositions,
    //    aTotalTriangleNormals,
    //    aTotalTriangleColors,
    //    iOutputWidth,
    //    iOutputHeight,
    //    3);
}

/*
**
*/
void drawMeshClusterImage(
    std::vector<uint32_t> const& aiClusterAddress,
    std::vector<MeshCluster*> const& aMeshClusters,
    std::vector<uint8_t>& vertexPositionBuffer,
    std::vector<uint8_t>& trianglePositionIndexBuffer,
    float3 const& cameraPosition,
    float3 const& cameraLookAt,
    uint32_t iOutputWidth,
    uint32_t iOutputHeight,
    std::string const& outputDirectory,
    std::string const& outputFileName)
{
    float const kfCameraNear = 1.0f;
    float const kfCameraFar = 100.0f;

    uint32_t const kiImageWidth = 1024;
    uint32_t const kiImageHeight = 1024;

    float3 direction = normalize(cameraLookAt - cameraPosition);
    float3 up = (fabsf(direction.z) > fabsf(direction.x) && fabsf(direction.z) > fabsf(direction.y)) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);

    // camera 
    CCamera camera;
    camera.setFar(kfCameraFar);
    camera.setNear(kfCameraNear);
    camera.setLookAt(cameraLookAt);
    camera.setPosition(cameraPosition);
    CameraUpdateInfo cameraUpdateInfo =
    {
        /* .mfViewWidth      */  1000.0f,
        /* .mfViewHeight     */  1000.0f,
        /* .mfFieldOfView    */  3.14159f * 0.5f,
        /* .mUp              */  up,
        /* .mfNear           */  kfCameraNear,
        /* .mfFar            */  kfCameraFar,
    };
    camera.update(cameraUpdateInfo);

    std::vector<std::string> aClusterNames;
    for(auto const& iClusterAddress : aiClusterAddress)
    {
        auto iter = std::find_if(
            aMeshClusters.begin(),
            aMeshClusters.end(),
            [iClusterAddress](MeshCluster const* pMeshCluster)
            {
                return pMeshCluster->miIndex == iClusterAddress;
            }
        );
        assert(iter != aMeshClusters.end());
        
        std::vector<float3> aVertexPositions((*iter)->miNumVertexPositions);
        uint64_t iVertexPositionBufferAddress = (*iter)->miVertexPositionStartArrayAddress * sizeof(float3);
        float3 const* pVertexPositionBuffer = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + iVertexPositionBufferAddress);
        memcpy(aVertexPositions.data(), pVertexPositionBuffer, sizeof(float3) * aVertexPositions.size());

        std::vector<uint32_t> aiTrianglePositionIndices((*iter)->miNumTrianglePositionIndices);
        uint64_t iTriangleIndexAddress = (*iter)->miTrianglePositionIndexArrayAddress * sizeof(uint32_t);
        uint32_t const* pTrianglePositionIndexBuffer = reinterpret_cast<uint32_t const*>(trianglePositionIndexBuffer.data() + iTriangleIndexAddress);
        memcpy(aiTrianglePositionIndices.data(), pTrianglePositionIndexBuffer, sizeof(uint32_t) * aiTrianglePositionIndices.size());

        std::ostringstream clusterName;
        clusterName << "cluster-" << iClusterAddress;
        outputMeshToImage(
            outputDirectory,
            clusterName.str(),
            aVertexPositions,
            aiTrianglePositionIndices,
            camera,
            kiImageWidth,
            kiImageHeight);

        aClusterNames.push_back(clusterName.str());
    }

    static std::vector<float3> saColors;
    if(saColors.size() <= 0)
    {
        for(uint32_t i = 0; i < 64; i++)
        {
            float fRand0 = float(rand() % 255) / 255.0f;
            float fRand1 = float(rand() % 255) / 255.0f;
            float fRand2 = float(rand() % 255) / 255.0f;

            saColors.push_back(float3(fRand0, fRand1, fRand2));
        }
    }

    std::vector<float4> aLightingOutput(kiImageWidth * kiImageHeight);
    memset(aLightingOutput.data(), 0, sizeof(float4) * kiImageWidth * kiImageHeight);
    std::vector<float4> aDepthOutput(kiImageWidth * kiImageHeight);
    for(uint32_t i = 0; i < kiImageWidth * kiImageHeight; i++)
    {
        aDepthOutput[i] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterNames.size()); i++)
    {
        std::string depthImageFilePath = std::string("c:\\Users\\Dingwings\\demo-models\\cluster-images\\") + aClusterNames[i] + "-depth.exr";
        char const* pError = nullptr;
        float* afDepth = nullptr;
        int32_t iWidth = 0, iHeight = 0;
        LoadEXR(&afDepth, &iWidth, &iHeight, depthImageFilePath.c_str(), &pError);

        float* afLighting = nullptr;
        std::string lightingImageFilePath = std::string("c:\\Users\\Dingwings\\demo-models\\cluster-images\\") + aClusterNames[i] + "-lighting.exr";
        LoadEXR(&afLighting, &iWidth, &iHeight, lightingImageFilePath.c_str(), &pError);

        for(int32_t iY = 0; iY < iHeight; iY++)
        {
            for(int32_t iX = 0; iX < iWidth; iX++)
            {
                int32_t iImageIndex = iY * iWidth + iX;
                float fIncomingDepth = 1.0f - afDepth[iImageIndex * 4];
                float4 incomingLighting = float4(afLighting[iImageIndex * 4], afLighting[iImageIndex * 4 + 1], afLighting[iImageIndex * 4 + 2], afLighting[iImageIndex * 4 + 3]);
                incomingLighting.x *= saColors[i].x;
                incomingLighting.y *= saColors[i].y;
                incomingLighting.z *= saColors[i].z;
                if(aDepthOutput[iImageIndex].x > fIncomingDepth)
                {
                    aDepthOutput[iImageIndex] = float4(fIncomingDepth, fIncomingDepth, fIncomingDepth, 1.0f);
                    aLightingOutput[iImageIndex] = incomingLighting;
                }
            }
        }

        free(afDepth);
        free(afLighting);
    }

    char const* pError = nullptr;
    SaveEXR(reinterpret_cast<float const*>(aLightingOutput.data()), kiImageWidth, kiImageHeight, 4, 0, "c:\\Users\\Dingwings\\demo-models\\cluster-images\\total-output.exr", &pError);

    std::vector<uint8_t> acImageData(kiImageWidth* kiImageHeight * 4);
    for(int32_t iY = 0; iY < kiImageHeight; iY++)
    {
        for(int32_t iX = 0; iX < kiImageWidth; iX++)
        {
            uint32_t iImageIndex = iY * kiImageWidth + iX;
            float4 const& lighting = aLightingOutput[iImageIndex];
            acImageData[iImageIndex * 4] = clamp(uint8_t(lighting.x * 255.0f), 0, 255);
            acImageData[iImageIndex * 4 + 1] = clamp(uint8_t(lighting.y * 255.0f), 0, 255);
            acImageData[iImageIndex * 4 + 2] = clamp(uint8_t(lighting.z * 255.0f), 0, 255);
            acImageData[iImageIndex * 4 + 3] = 255;
        }
    }

    static uint32_t siImageIndex = 0;
    std::ostringstream outputLDRImageFilePath;
    outputLDRImageFilePath << "c:\\Users\\Dingwings\\demo-models\\cluster-images\\" << outputFileName;
    stbi_write_png(outputLDRImageFilePath.str().c_str(), kiImageWidth, kiImageHeight, 4, acImageData.data(), kiImageWidth * 4 * sizeof(char));

    DEBUG_PRINTF("%s\n", outputLDRImageFilePath.str().c_str());
    
}