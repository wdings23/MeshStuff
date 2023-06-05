#include "cluster_tree.h"

#include <algorithm>
#include <assert.h>

/*
**
*/
uint32_t getMeshClusterGroupAddress(
    std::vector<uint8_t>& meshClusterGroupBuffer,
    uint32_t iLODLevel,
    uint32_t iClusterGroup)
{
    uint32_t iOffset = 0;
    uint8_t* pAddress = reinterpret_cast<uint8_t*>(meshClusterGroupBuffer.data());
    for(;;)
    {
        MeshClusterGroup const* pMeshClusterGroup = reinterpret_cast<MeshClusterGroup const*>(pAddress + iOffset);
        if(pMeshClusterGroup->miLODLevel == iLODLevel)
        {
            iOffset += iClusterGroup * sizeof(MeshClusterGroup);
            break;
        }

        iOffset += sizeof(MeshClusterGroup);
    }

    return iOffset / sizeof(MeshClusterGroup);
}

/*
**
*/
uint32_t getMeshClusterAddress(
    std::vector<uint8_t>& meshClusterBuffer,
    uint32_t iLODLevel,
    uint32_t iCluster)
{
    uint32_t iOffset = 0;
    uint8_t* pAddress = reinterpret_cast<uint8_t*>(meshClusterBuffer.data());
    for(;;)
    {
        MeshCluster const* pMeshCluster = reinterpret_cast<MeshCluster const*>(pAddress + iOffset);
        if(pMeshCluster->miLODLevel == iLODLevel)
        {
            iOffset += iCluster * sizeof(MeshCluster);
            break;
        }

        iOffset += sizeof(MeshCluster);
    }

    return iOffset / sizeof(MeshCluster);
}

/*
**
*/
void createTreeNodes(
    std::vector<ClusterTreeNode>& aNodes,
    uint32_t iNumLODLevels,
    std::vector<uint8_t>& aMeshClusterData,
    std::vector<uint8_t>& aMeshClusterGroupData,
    std::vector<std::vector<MeshClusterGroup>> const& aaMeshClusterGroups,
    std::vector<std::vector<MeshCluster>> const& aaMeshClusters)
{
    uint32_t iCurrNumClusters = 0;
    for(int32_t iLODLevel = static_cast<int32_t>(iNumLODLevels) - 1; iLODLevel >= 0; iLODLevel--)
    {
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaMeshClusters[iLODLevel].size()); iCluster++)
        {
            uint32_t iNumChildren = 0;
            ClusterTreeNode node;
            node.miLevel = iNumLODLevels - iLODLevel;
            node.miClusterAddress = iCurrNumClusters;
            node.miNumChildren = 0;
            ++iCurrNumClusters;

            auto const& cluster = aaMeshClusters[iLODLevel][iCluster];
            uint32_t iClusterGroup = iCluster / 2;

            node.miClusterGroupAddress = getMeshClusterGroupAddress(
                aMeshClusterGroupData,
                iLODLevel,
                iClusterGroup);

            if(iLODLevel > 0)
            {
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaMeshClusters[iLODLevel - 1].size()); i++)
                {
                    if(aaMeshClusters[iLODLevel - 1][i].miClusterGroup == iClusterGroup)
                    {
                        node.maiChildrenAddress[node.miNumChildren] = getMeshClusterAddress(
                            aMeshClusterData,
                            iLODLevel - 1,
                            i);
                        ++node.miNumChildren;
                    }
                }
            }

            aNodes.push_back(node);
        }
    }
}

/*
**
*/
void createTreeNodes2(
    std::vector<ClusterTreeNode>& aNodes,
    uint32_t iNumLODLevels,
    std::vector<uint8_t>& aMeshClusterData,
    std::vector<uint8_t>& aMeshClusterGroupData,
    std::vector<std::vector<MeshClusterGroup>> const& aaMeshClusterGroups,
    std::vector<std::vector<MeshCluster>> const& aaMeshClusters,
    std::vector<std::pair<float3, float3>> const& aTotalMaxClusterDistancePositionFromLOD0)
{
    std::vector<uint32_t> aiStartClusterGroupIndex(iNumLODLevels);
    memset(aiStartClusterGroupIndex.data(), 0, iNumLODLevels * sizeof(uint32_t));
    uint32_t iCurrTotalClusters = 0;
    for(uint32_t iLOD = 0; iLOD < iNumLODLevels; iLOD++)
    {
        aiStartClusterGroupIndex[iLOD] = iCurrTotalClusters;
        iCurrTotalClusters += static_cast<uint32_t>(aaMeshClusterGroups[iLOD].size());
    }

    uint32_t iCurrLevel = iNumLODLevels;
    for(int32_t iLODLevel = static_cast<int32_t>(iNumLODLevels - 1); iLODLevel >= 0; iLODLevel--)
    {
        uint32_t iNumClusterGroups = static_cast<uint32_t>(aaMeshClusterGroups[iLODLevel].size());
        for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
        {
            // use MIP 1 of LOD - 1 as the cluster group
            for(int32_t iMIP = 1; iMIP >= 0; iMIP--)
            {
                if(iMIP == 0 && iLODLevel > 0)
                {
                    break;
                }

                // create node with children cluster
                MeshClusterGroup const& clusterGroup = aaMeshClusterGroups[iLODLevel][iClusterGroup];
                ClusterTreeNode node;
                //node.miLevel = (iLODLevel == 0) ? iLODLevel + iMIP : iLODLevel; // use MIP 1 from LOD > 0, MIP 0 for LOD 0
                node.miLevel = (iMIP == 0) ? iCurrLevel - 1 : iCurrLevel;
                uint32_t iNumClusters = clusterGroup.maiNumClusters[iMIP];
                for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
                {
                    uint32_t iClusterID = clusterGroup.maiClusters[iMIP][iCluster];
                    node.miClusterAddress = iClusterID;
                    node.miClusterGroupAddress = clusterGroup.miIndex;

                    if(node.miLevel > 0)
                    {
                        node.miClusterGroupAddress += aiStartClusterGroupIndex[1];
                    }
                    memset(node.maiChildrenAddress, 0xff, MAX_CLUSTER_TREE_NODE_CHILDREN * sizeof(uint32_t));
                    memset(node.maiParentAddress, 0xff, MAX_CLUSTER_TREE_NODE_PARENTS * sizeof(uint32_t));

                    // use clusters from MIP 0 as children
                    if(iMIP > 0)
                    {
                        uint32_t iNumChildClusters = clusterGroup.maiNumClusters[iMIP - 1];
                        for(uint32_t iChildCluster = 0; iChildCluster < iNumChildClusters; iChildCluster++)
                        {
                            uint32_t iChildClusterID = clusterGroup.maiClusters[iMIP - 1][iChildCluster];
                            node.maiChildrenAddress[iChildCluster] = iChildClusterID;
                        }
                        node.miNumChildren = iNumChildClusters;
                    }
                    else
                    {
                        node.miNumChildren = 0;
                    }

                    // max distance positions from LOD 0 
                    // 1) current cluster closest position to the LOD 0 cluster position
                    // 2) LOD 0 cluster position corresponding to the current cluster position 
                    //node.mMaxDistanceCurrLODClusterPosition = aTotalMaxClusterDistancePositionFromLOD0[iClusterID].first;
                    //node.mMaxDistanceLOD0ClusterPosition = aTotalMaxClusterDistancePositionFromLOD0[iClusterID].second;

                    // get the average error distance from LOD 0
                    float fAverageDistanceFromLOD0 = FLT_MAX;
                    float4 normalCone = float4(0.0f, 0.0f, 0.0f, 0.0f);
                    float3 minBounds = float3(0.0f, 0.0f, 0.0f);
                    float3 maxBounds = float3(0.0f, 0.0f, 0.0f);
                    for(uint32_t i = 0; i < static_cast<uint32_t>(aaMeshClusters.size()); i++)
                    {
                        auto iter = std::find_if(
                            aaMeshClusters[i].begin(),
                            aaMeshClusters[i].end(),
                            [iClusterID](MeshCluster const& checkMeshCluster)
                            {
                                return checkMeshCluster.miIndex == iClusterID;
                            });
                        if(iter != aaMeshClusters[i].end())
                        {
                            fAverageDistanceFromLOD0 = iter->mfAverageDistanceFromLOD0;
                            normalCone = iter->mNormalCone;
                            minBounds = iter->mMaxBounds;
                            maxBounds = iter->mMinBounds;
                            break;
                        }
                    }
                    assert(fAverageDistanceFromLOD0 != FLT_MAX);
                    node.mfAverageDistanceFromLOD0 = fAverageDistanceFromLOD0;
                    node.mNormalCone = normalCone;
                    node.mMinBounds = minBounds;
                    node.mMaxBounds = maxBounds;

                    aNodes.push_back(node);

                }   // for cluster = 0 to num clusters in group 
            }

        }   // for cluster group = 0 to num cluster groups at LOD

        --iCurrLevel;

    }   // for LOD = num lod levels to 0

    // set parents
    uint32_t iNumNodes = static_cast<uint32_t>(aNodes.size());
    for(uint32_t i = 0; i < iNumNodes; i++)
    {
        auto& node = aNodes[i];
        for(uint32_t j = 0; j < node.miNumChildren; j++)
        {
            uint32_t iChildAddress = node.maiChildrenAddress[j];
            auto childIter = std::find_if(
                aNodes.begin(),
                aNodes.end(),
                [iChildAddress](ClusterTreeNode const& checkNode)
                {
                    return checkNode.miClusterAddress == iChildAddress;
                }
            );
            assert(childIter != aNodes.end());

            //assert(childIter->miNumParents < MAX_CLUSTER_TREE_NODE_PARENTS);
            if(childIter->miNumParents < MAX_CLUSTER_TREE_NODE_PARENTS)
            {
                childIter->maiParentAddress[childIter->miNumParents] = node.miClusterAddress;
                ++childIter->miNumParents;
            }
        }
    }

    std::sort(
        aNodes.begin(),
        aNodes.end(),
        [](ClusterTreeNode const& nodeLeft, ClusterTreeNode const& nodeRight)
        {
            return nodeLeft.miClusterAddress < nodeRight.miClusterAddress;
        }
    );

    // make sure the average distance error of LOD n is smaller than LOD n + 1
    std::vector<float> afMaxAverageErrorDistanceLOD(iNumLODLevels + 1);
    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels + 1; iLODLevel++)
    {
        for(auto const& node : aNodes)
        {
            if(node.miLevel == iLODLevel)
            {
                afMaxAverageErrorDistanceLOD[iLODLevel] = maxf(afMaxAverageErrorDistanceLOD[iLODLevel], node.mfAverageDistanceFromLOD0);
            }
        }
    }

    for(uint32_t iLODLevel = 1; iLODLevel < iNumLODLevels + 1; iLODLevel++)
    {
        for(auto& node : aNodes)
        {
            if(node.miLevel == iLODLevel && node.mfAverageDistanceFromLOD0 < afMaxAverageErrorDistanceLOD[iLODLevel - 1])
            {
                node.mfAverageDistanceFromLOD0 = afMaxAverageErrorDistanceLOD[iLODLevel - 1];
            }
        }
    }
}

/*
**
*/
void saveClusterGroupTreeNodes(
    std::string const& outputFilePath,
    std::vector<ClusterGroupTreeNode> const& aClusterGroupNodes)
{
    std::vector<uint8_t> aFileData(aClusterGroupNodes.size() * sizeof(ClusterGroupTreeNode));
    uint64_t iCurrDataIndex = 0;
    for(auto const& clusterGroup : aClusterGroupNodes)
    {
        uint8_t* pStart = aFileData.data() + iCurrDataIndex;

        uint32_t* pStartUINT32 = reinterpret_cast<uint32_t*>(pStart);
        uint32_t* pDataUINT32 = pStartUINT32;
        *pDataUINT32++ = 0;                                                  // cluster group data size
        *pDataUINT32++ = clusterGroup.miClusterGroupAddress;
        *pDataUINT32++ = clusterGroup.miLevel;
        *pDataUINT32++ = clusterGroup.miNumChildClusters;
        for(uint32_t i = 0; i < clusterGroup.miNumChildClusters; i++)
        {
            *pDataUINT32++ = clusterGroup.maiClusterAddress[i];
        }
        float3* pDataFLOAT3 = reinterpret_cast<float3*>(pDataUINT32);
        *pDataFLOAT3++ = clusterGroup.mMaxDistanceCurrClusterPosition;
        *pDataFLOAT3++ = clusterGroup.mMaxDistanceLOD0ClusterPosition;
        float* pDataFLOAT = reinterpret_cast<float*>(pDataFLOAT3);
        *pDataFLOAT++ = clusterGroup.mfScreenSpacePixelError;
        *pStartUINT32 = static_cast<uint32_t>(reinterpret_cast<uint64_t>(pDataFLOAT) - reinterpret_cast<uint64_t>(pStart));        // cluster group size

        iCurrDataIndex += *pStartUINT32;
    }

    aFileData.resize(iCurrDataIndex);

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fwrite(aFileData.data(), sizeof(char), iCurrDataIndex, fp);
    fclose(fp);
}

/*
**
*/
void loadClusterGroupTreeNodes(
    std::vector<ClusterGroupTreeNode>& aClusterGroupNodes,
    std::string const& filePath)
{
    FILE* fp = fopen(filePath.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    uint64_t iFileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> acFileContent(iFileSize);
    fread(acFileContent.data(), sizeof(char), iFileSize, fp);
    fclose(fp);

    for(uint32_t iDataIndex = 0; iDataIndex < static_cast<uint32_t>(acFileContent.size());)
    {
        uint8_t const* pStart = acFileContent.data() + iDataIndex;
        ClusterGroupTreeNode node;
        uint32_t const* piFileContentUINT32 = reinterpret_cast<uint32_t const*>(acFileContent.data() + iDataIndex);
        uint32_t iNodeSize = *piFileContentUINT32++;
        node.miClusterGroupAddress = *piFileContentUINT32++;
        node.miLevel = *piFileContentUINT32++;
        node.miNumChildClusters = *piFileContentUINT32++;
        memset(node.maiClusterAddress, 0xff, MAX_CLUSTER_TREE_NODE_CHILDREN * sizeof(uint32_t));
        uint32_t iNumChildClusters = (node.miNumChildClusters < MAX_CLUSTER_TREE_NODE_CHILDREN) ? node.miNumChildClusters : MAX_CLUSTER_TREE_NODE_CHILDREN - 1;
        memcpy(node.maiClusterAddress, piFileContentUINT32, iNumChildClusters * sizeof(uint32_t));
        piFileContentUINT32 += node.miNumChildClusters;
        float3 const* pFileContentFLOAT3 = reinterpret_cast<float3 const*>(piFileContentUINT32);
        node.mMaxDistanceCurrClusterPosition = *pFileContentFLOAT3++;
        node.mMaxDistanceLOD0ClusterPosition = *pFileContentFLOAT3++;
        float const* pFileContentFLOAT = reinterpret_cast<float const*>(pFileContentFLOAT3);
        node.mfScreenSpacePixelError = *pFileContentFLOAT++;

        aClusterGroupNodes.push_back(node);

        uint32_t iDataSize = static_cast<uint32_t>(reinterpret_cast<uint64_t>(pFileContentFLOAT) - reinterpret_cast<uint64_t>(pStart));
        assert(iDataSize == iNodeSize);
        iDataIndex += iDataSize;
    }
}

/*
**
*/
void saveClusterTreeNodes(
    std::string const& outputFilePath,
    std::vector<ClusterTreeNode> const& aClusterNodes)
{
    uint64_t iCurrDataIndex = 0;
    std::vector<uint8_t> acFileContent(aClusterNodes.size() * sizeof(ClusterTreeNode));
    for(auto const& clusterNode : aClusterNodes)
    {
        uint8_t* pStart = acFileContent.data() + iCurrDataIndex;

        uint32_t* pStartUINT32 = reinterpret_cast<uint32_t*>(pStart);
        uint32_t* pDataUINT32 = pStartUINT32;
        *pDataUINT32++ = 0;                                                  // cluster group data size
        *pDataUINT32++ = clusterNode.miClusterAddress;
        *pDataUINT32++ = clusterNode.miClusterGroupAddress;
        *pDataUINT32++ = clusterNode.miLevel;
        *pDataUINT32++ = clusterNode.miNumChildren;
        *pDataUINT32++ = clusterNode.miNumParents;
        for(uint32_t i = 0; i < clusterNode.miNumChildren; i++)
        {
            *pDataUINT32++ = clusterNode.maiChildrenAddress[i];
        }
        for(uint32_t i = 0; i < clusterNode.miNumParents; i++)
        {
            *pDataUINT32++ = clusterNode.maiParentAddress[i];
        }

        float3* pDataFLOAT3 = reinterpret_cast<float3*>(pDataUINT32);
        *pDataFLOAT3++ = clusterNode.mMinBounds;
        *pDataFLOAT3++ = clusterNode.mMaxBounds;

        float* pDataFLOAT = reinterpret_cast<float*>(pDataFLOAT3);
        *pDataFLOAT++ = clusterNode.mfScreenSpaceError;
        *pDataFLOAT++ = clusterNode.mfAverageDistanceFromLOD0;
        *pDataFLOAT++ = clusterNode.mNormalCone.x;
        *pDataFLOAT++ = clusterNode.mNormalCone.y;
        *pDataFLOAT++ = clusterNode.mNormalCone.z;
        *pDataFLOAT++ = clusterNode.mNormalCone.w;

        uint32_t iDataSize = static_cast<uint32_t>(reinterpret_cast<uint64_t>(pDataFLOAT) - reinterpret_cast<uint64_t>(pStartUINT32));

        pDataUINT32 = pStartUINT32;
        *pDataUINT32 = iDataSize;

        iCurrDataIndex += iDataSize;
    }

    acFileContent.resize(iCurrDataIndex);

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fwrite(acFileContent.data(), sizeof(char), acFileContent.size(), fp);
    fclose(fp);
}

/*
**
*/
void loadClusterTreeNodes(
    std::vector<ClusterTreeNode>& aClusterNodes,
    std::string const& filePath)
{
    FILE* fp = fopen(filePath.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    uint64_t iFileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> acFileContent(iFileSize);
    fread(acFileContent.data(), sizeof(char), iFileSize, fp);
    fclose(fp);

    for(uint32_t iDataIndex = 0; iDataIndex < static_cast<uint32_t>(acFileContent.size());)
    {
        uint8_t const* pStart = acFileContent.data() + iDataIndex;
        ClusterTreeNode node;
        uint32_t const* piFileContentUINT32 = reinterpret_cast<uint32_t const*>(acFileContent.data() + iDataIndex);
        uint32_t iNodeSize = *piFileContentUINT32++;
        node.miClusterAddress = *piFileContentUINT32++;
        node.miClusterGroupAddress = *piFileContentUINT32++;
        node.miLevel = *piFileContentUINT32++;
        node.miNumChildren = *piFileContentUINT32++;
        node.miNumParents = *piFileContentUINT32++;
        memset(node.maiChildrenAddress, 0xff, MAX_CLUSTER_TREE_NODE_CHILDREN * sizeof(uint32_t));
        memset(node.maiParentAddress, 0xff, MAX_CLUSTER_TREE_NODE_PARENTS * sizeof(uint32_t));
        memcpy(node.maiChildrenAddress, piFileContentUINT32, sizeof(uint32_t) * node.miNumChildren);
        piFileContentUINT32 += node.miNumChildren;
        memcpy(node.maiParentAddress, piFileContentUINT32, sizeof(uint32_t) * node.miNumParents);
        piFileContentUINT32 += node.miNumParents;

        float3 const* pFileContentFLOAT3 = reinterpret_cast<float3 const*>(piFileContentUINT32);
        node.mMinBounds = *pFileContentFLOAT3++;
        node.mMaxBounds = *pFileContentFLOAT3++;

        float const* pFileContentFLOAT = reinterpret_cast<float const*>(pFileContentFLOAT3);
        node.mfScreenSpaceError = *pFileContentFLOAT++;
        node.mfAverageDistanceFromLOD0 = *pFileContentFLOAT++;
        node.mNormalCone.x = *pFileContentFLOAT++;
        node.mNormalCone.y = *pFileContentFLOAT++;
        node.mNormalCone.z = *pFileContentFLOAT++;
        node.mNormalCone.w = *pFileContentFLOAT++;

        aClusterNodes.push_back(node);

        uint32_t iDataSize = static_cast<uint32_t>(reinterpret_cast<uint64_t>(pFileContentFLOAT) - reinterpret_cast<uint64_t>(pStart));
        assert(iDataSize == iNodeSize);
        iDataIndex += iDataSize;
    }
}