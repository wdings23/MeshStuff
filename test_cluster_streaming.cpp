#include "test_cluster_streaming.h"

#include <chrono>
#include <assert.h>

#include "LogPrint.h"

struct RequestClusterInfo
{
    uint32_t        miMesh;
    uint32_t        miCluster;
    uint64_t        miTimeAccessed = 0;
    uint32_t        miVertexBufferAddress;
    uint32_t        miIndexBufferAddress;
    uint32_t        miVertexBufferSize;
    uint32_t        miIndexBufferSize;
    uint32_t        miAllocatedVertexBufferSize;
    uint32_t        miAllocatedIndexBufferSize;
    uint32_t        miLoaded = 0;
};

static std::vector<uint8_t> saClusterInfoRequest(1 << 20);
static std::vector<uint8_t> saVertexDataBuffer(1 << 23);
static std::vector<uint8_t> saIndexDataBuffer(1 << 23);

/*
**
*/
void loadClusterInfo(
    RequestClusterInfo& clusterInfo,
    void const* pBuffer,
    uint32_t iIndex,
    uint32_t iSrcOffset)
{
    uint8_t const* pPtr = reinterpret_cast<uint8_t const*>(pBuffer) + iSrcOffset + iIndex * sizeof(RequestClusterInfo);
    memcpy(&clusterInfo, pPtr, sizeof(RequestClusterInfo));
}

/*
**
*/
void testClusterRequests(
    std::vector<uint32_t>& aiNumClusterVertices,
    std::vector<uint32_t>& aiNumClusterIndices,
    std::vector<uint64_t>& saiVertexBufferArrayOffsets,
    std::vector<uint64_t>& saiIndexBufferArrayOffsets,
    std::vector<uint32_t> const& aiDrawClusters)
{
    //static std::vector<uint8_t> saReadWriteBuffer(1 << 24);

    static bool sbStarted = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> sStartTime;
    if(!sbStarted)
    {
        sStartTime = std::chrono::high_resolution_clock::now();
        memset(saClusterInfoRequest.data(), 0, saClusterInfoRequest.size() * sizeof(uint8_t));
        memset(saVertexDataBuffer.data(), 0, saVertexDataBuffer.size() * sizeof(uint8_t));
        memset(saIndexDataBuffer.data(), 0, saIndexDataBuffer.size() * sizeof(uint8_t));
        sbStarted = true;
    }

    // get free space
    uint32_t iClusterInfoAddress = 0;
    uint32_t iClusterInfoBufferSize = 1 << 18;
    uint32_t iVertexBufferSize = static_cast<uint32_t>(saVertexDataBuffer.size());
    uint32_t iIndexBufferSize = static_cast<uint32_t>(saIndexDataBuffer.size());
    uint32_t iMesh = 0;
    uint32_t iVertexBufferAddress = 0;
    uint32_t iIndexBufferAddress = 0;


    DEBUG_PRINTF("\n*** VERTEX BUFFER ADDRESS %d ***\n", iVertexBufferAddress);
    DEBUG_PRINTF("*** INDEX BUFFER ADDRESS %d ***\n", iIndexBufferAddress);
    
    //uint8_t* readWriteBuffer = saReadWriteBuffer.data();
    uint8_t* clusterRequestInfoBuffer = saClusterInfoRequest.data();
    uint8_t* vertexDataBuffer = saVertexDataBuffer.data();
    uint8_t* indexDataBuffer = saIndexDataBuffer.data();
    {
        
        auto loadUInt32 = [](uint8_t* pBuffer, uint32_t& iCurrAddress)
        {
            uint32_t iRet = *(reinterpret_cast<uint32_t*>(pBuffer + iCurrAddress));
            iCurrAddress += 4;
            return iRet;
        };

        auto loadUInt64 = [](uint8_t* pBuffer, uint32_t& iCurrAddress)
        {
            uint64_t iRet = *(reinterpret_cast<uint64_t*>(pBuffer + iCurrAddress));
            iCurrAddress += 8;
            return iRet;
        };

        auto saveUInt32 = [](uint8_t* pBuffer, uint32_t iValue, uint32_t& iCurrAddress)
        {
            *(reinterpret_cast<uint32_t*>(pBuffer + iCurrAddress)) = iValue;
            iCurrAddress += 4;
        };

        auto saveUInt64 = [](uint8_t* pBuffer, uint64_t iValue, uint32_t& iCurrAddress)
        {
            *(reinterpret_cast<uint64_t*>(pBuffer + iCurrAddress)) = iValue;
            iCurrAddress += 8;
        };

        auto gpuMemcpy = [](
            void* pDest,
            void const* pSrc,
            uint32_t iDestOffset,
            uint32_t iSrcOffset,
            uint64_t iSize)
        {
            uint8_t* pDest8 = reinterpret_cast<uint8_t*>(pDest) + iDestOffset;
            uint8_t const* pSrc8 = reinterpret_cast<uint8_t const*>(pSrc) + iSrcOffset;
            memcpy(pDest8, pSrc8, iSize);
        };

        uint32_t const kiMaxVertexBufferSize = sizeof(ConvertedMeshVertexFormat) * 180;
        uint32_t const kiMaxIndexBufferSize = sizeof(uint32_t) * 128 * 3;

        uint64_t iCurrTimeUS = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sStartTime).count() + 1;
        uint32_t iCurrRequestClusterInfoAddress = iClusterInfoAddress;
        uint32_t iNumLoadedClusters = loadUInt32(clusterRequestInfoBuffer, iCurrRequestClusterInfoAddress);
        for(uint32_t iCluster = 0; iCluster < aiDrawClusters.size(); iCluster++)
        {
            uint32_t iDrawCluster = aiDrawClusters[iCluster];
            uint32_t iCurrClusterInfoAddress = sizeof(uint32_t);
            uint32_t iLastVertexAddressEnd = 0, iLastIndexAddressEnd = 0;
            bool bFound = false;
            uint64_t iOldestAccessed = UINT64_MAX;
            uint32_t iOldestLoadedIndex = UINT32_MAX;

            uint32_t iClusterVertexBufferSize = aiNumClusterVertices[iDrawCluster] * sizeof(ConvertedMeshVertexFormat);
            uint32_t iClusterIndexBufferSize = aiNumClusterIndices[iDrawCluster] * sizeof(uint32_t);

            assert(iClusterVertexBufferSize <= kiMaxVertexBufferSize);
            assert(iClusterIndexBufferSize <= kiMaxIndexBufferSize);

            for(uint32_t iLoadedCluster = 0;; iLoadedCluster++)
            {
                // reached end of cluster info buffer
                if(iCurrClusterInfoAddress >= iClusterInfoBufferSize)
                {
                    break;
                }

                if(iVertexBufferAddress + iLastVertexAddressEnd + kiMaxVertexBufferSize >= iVertexBufferSize)
                {
                    break;
                }

                if(iIndexBufferAddress + iLastIndexAddressEnd + kiMaxIndexBufferSize >= iIndexBufferSize)
                {
                    break;
                }

                // load cluster info
                uint32_t iLoadAddress = iCurrClusterInfoAddress;
                RequestClusterInfo cluster;
                cluster.miMesh =                        loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miCluster =                     loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miTimeAccessed =                loadUInt64(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miVertexBufferAddress =         loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miIndexBufferAddress =          loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miVertexBufferSize =            loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miIndexBufferSize =             loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miAllocatedVertexBufferSize =   loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
                cluster.miAllocatedIndexBufferSize =    loadUInt32(clusterRequestInfoBuffer, iLoadAddress);

                // save old access time of cluster
                if(cluster.miTimeAccessed > 0 &&
                    cluster.miTimeAccessed < iOldestAccessed &&
                    cluster.miAllocatedVertexBufferSize >= iClusterVertexBufferSize &&
                    cluster.miAllocatedIndexBufferSize >= iClusterIndexBufferSize)
                {
                    iOldestAccessed = cluster.miTimeAccessed;
                    iOldestLoadedIndex = iLoadedCluster;
                }

                // update accessed time for valid cluster
                if(cluster.miMesh == iMesh && cluster.miCluster == iDrawCluster)
                {
                    uint32_t iSaveAddress = iCurrClusterInfoAddress + 8;
                    saveUInt64(clusterRequestInfoBuffer, iCurrTimeUS, iSaveAddress);
                }

                if(iDrawCluster == cluster.miCluster && iMesh == cluster.miMesh && cluster.miTimeAccessed > 0)
                {
                    uint32_t iSaveAddress = iCurrClusterInfoAddress + 8;
                    saveUInt64(clusterRequestInfoBuffer, iCurrTimeUS, iSaveAddress);
                    bFound = true;

                    DEBUG_PRINTF("FOUND cluster %d at cluster info address: %d vertex buffer address: %d index buffer address: %d\n",
                        iDrawCluster,
                        iSaveAddress,
                        iLastVertexAddressEnd + iVertexBufferAddress,
                        iLastIndexAddressEnd + iIndexBufferAddress);

                    break;
                }
                else if(cluster.miTimeAccessed == 0 && cluster.miVertexBufferAddress == 0)
                {
                    // new cluster free space

                    uint32_t iSaveAddress = iCurrClusterInfoAddress;
                    saveUInt32(clusterRequestInfoBuffer, iMesh, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, iDrawCluster, iSaveAddress);
                    saveUInt64(clusterRequestInfoBuffer, iCurrTimeUS, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, iLastVertexAddressEnd, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, iLastIndexAddressEnd, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, iClusterVertexBufferSize, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, iClusterIndexBufferSize, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, kiMaxVertexBufferSize, iSaveAddress);
                    saveUInt32(clusterRequestInfoBuffer, kiMaxIndexBufferSize, iSaveAddress);

                    iCurrRequestClusterInfoAddress = 0;
                    iNumLoadedClusters += 1;
                    saveUInt32(clusterRequestInfoBuffer, iNumLoadedClusters, iCurrRequestClusterInfoAddress);

                    bFound = true;

#if 0
                    gpuMemcpy(
                        vertexDataBuffer,
                        aaClusterTriangleVertices[iCluster].data(),
                        iLastVertexAddressEnd + iVertexBufferAddress,
                        0,
                        aaClusterTriangleVertices[iCluster].size() * sizeof(MeshVertexFormat));

                    gpuMemcpy(
                        indexDataBuffer,
                        aaiClusterTriangleVertexIndices[iCluster].data(),
                        iLastIndexAddressEnd + iIndexBufferAddress,
                        0,
                        aaiClusterTriangleVertexIndices[iCluster].size() * sizeof(uint32_t));
#endif // #if 0

                    DEBUG_PRINTF("ADD cluster %d at cluster info address: %d vertex buffer address: %d index buffer address: %d\n",
                        iDrawCluster,
                        iSaveAddress,
                        iLastVertexAddressEnd + iVertexBufferAddress,
                        iLastIndexAddressEnd + iIndexBufferAddress);


                    break;
                }
                else
                {
                    // next cluster info
                    iLastVertexAddressEnd = cluster.miVertexBufferAddress + kiMaxVertexBufferSize;
                    iLastIndexAddressEnd = cluster.miIndexBufferAddress + kiMaxIndexBufferSize;
                }

                iCurrClusterInfoAddress += sizeof(RequestClusterInfo);
            }

            // use oldest accessed cluster
            if(!bFound)
            {
                assert(iOldestLoadedIndex != UINT32_MAX);
                
                RequestClusterInfo clusterInfo;
                loadClusterInfo(
                    clusterInfo,
                    clusterRequestInfoBuffer,
                    iOldestLoadedIndex,
                    sizeof(uint32_t));

                DEBUG_PRINTF("!!! replace cluster %d (%d) with cluster %d !!!\n",
                    clusterInfo.miCluster,
                    iOldestLoadedIndex,
                    iDrawCluster);

                uint32_t iSaveAddress = iClusterInfoAddress + sizeof(uint32_t) + static_cast<uint32_t>(iOldestLoadedIndex * sizeof(RequestClusterInfo));
                saveUInt32(clusterRequestInfoBuffer, iMesh, iSaveAddress);
                saveUInt32(clusterRequestInfoBuffer, iDrawCluster, iSaveAddress);
                saveUInt64(clusterRequestInfoBuffer, iCurrTimeUS, iSaveAddress);
                saveUInt32(clusterRequestInfoBuffer, clusterInfo.miVertexBufferAddress, iSaveAddress);
                saveUInt32(clusterRequestInfoBuffer, clusterInfo.miIndexBufferAddress, iSaveAddress);
                saveUInt32(clusterRequestInfoBuffer, iClusterVertexBufferSize, iSaveAddress);
                saveUInt32(clusterRequestInfoBuffer, iClusterIndexBufferSize, iSaveAddress);

#if 0
                gpuMemcpy(
                    vertexDataBuffer,
                    aaClusterTriangleVertices[iCluster].data(),
                    clusterInfo.miVertexBufferAddress + iVertexBufferAddress,
                    0,
                    aaClusterTriangleVertices[iCluster].size() * sizeof(MeshVertexFormat));

                gpuMemcpy(
                    indexDataBuffer,
                    aaiClusterTriangleVertexIndices[iCluster].data(),
                    clusterInfo.miIndexBufferAddress + iIndexBufferAddress,
                    0,
                    aaiClusterTriangleVertexIndices[iCluster].size() * sizeof(uint32_t));
#endif // #if 0

                loadClusterInfo(
                    clusterInfo,
                    clusterRequestInfoBuffer,
                    iOldestLoadedIndex,
                    sizeof(uint32_t));

                int iDebug = 1;
            }
        }

        iCurrRequestClusterInfoAddress = iClusterInfoAddress;
        uint32_t iCurrClusterInfoAddress = 0;
        iNumLoadedClusters = loadUInt32(clusterRequestInfoBuffer, iCurrClusterInfoAddress);
        for(uint32_t iLoadedCluster = 0; iLoadedCluster < iNumLoadedClusters; iLoadedCluster++)
        {
            // load cluster info
            uint32_t iLoadAddress = iCurrClusterInfoAddress;
            RequestClusterInfo cluster;
            cluster.miMesh =                        loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miCluster =                     loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miTimeAccessed =                loadUInt64(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miVertexBufferAddress =         loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miIndexBufferAddress =          loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miVertexBufferSize =            loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miIndexBufferSize =             loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miAllocatedVertexBufferSize =   loadUInt32(clusterRequestInfoBuffer, iLoadAddress);
            cluster.miAllocatedIndexBufferSize =    loadUInt32(clusterRequestInfoBuffer, iLoadAddress);

            iCurrClusterInfoAddress += sizeof(RequestClusterInfo);

            DEBUG_PRINTF("%d loaded cluster mesh: %d cluster: %d time accessed: %lld vertex buffer address: %d index buffer address: %d vertex buffer size: %d index buffer size: %d\n",
                iLoadedCluster,
                cluster.miMesh,
                cluster.miCluster,
                cluster.miTimeAccessed,
                cluster.miVertexBufferAddress + iVertexBufferAddress,
                cluster.miIndexBufferAddress + iIndexBufferAddress,
                cluster.miVertexBufferSize,
                cluster.miIndexBufferSize);
        }

        // swapped out cluster will conflict with the input cluster data, need to search for the draw cluster ID instead of just using the index
        for(uint32_t iDrawCluster = 0; iDrawCluster < aiDrawClusters.size(); iDrawCluster++)
        {
            uint32_t iClusterID = aiDrawClusters[iDrawCluster];

            RequestClusterInfo clusterInfo;
            uint32_t iClusterInfo = 0;
            for(iClusterInfo = 0; iClusterInfo < iNumLoadedClusters; iClusterInfo++)
            {
                loadClusterInfo(
                    clusterInfo,
                    clusterRequestInfoBuffer,
                    iClusterInfo,
                    sizeof(uint32_t));

                if(clusterInfo.miCluster == iClusterID)
                {
                    break;
                }
            }
            
            if(iClusterInfo >= iNumLoadedClusters)
            {
                // swapped out
                DEBUG_PRINTF("!!! can\'t find cluster %d !!!\n", iClusterID);
                for(uint32_t i = 0; i < iNumLoadedClusters; i++)
                {
                    loadClusterInfo(
                        clusterInfo,
                        clusterRequestInfoBuffer,
                        i,
                        sizeof(uint32_t));

                    if(clusterInfo.miCluster == iClusterID)
                    {
                        int iDebug = 1;
                    }
                    DEBUG_PRINTF("%d cluster %d\n", i, clusterInfo.miCluster);
                }

                continue;
            }

#if 0
            uint32_t iNumVertices = clusterInfo.miVertexBufferSize / sizeof(MeshVertexFormat);
            std::vector<uint8_t> aVertexBuffer(clusterInfo.miVertexBufferSize);
            memcpy(aVertexBuffer.data(), vertexDataBuffer + clusterInfo.miVertexBufferAddress + iVertexBufferAddress, clusterInfo.miVertexBufferSize);

            uint32_t iNumIndices = clusterInfo.miIndexBufferSize / sizeof(uint32_t);
            std::vector<uint8_t> aIndexBuffer(clusterInfo.miIndexBufferSize);
            memcpy(aIndexBuffer.data(), indexDataBuffer + clusterInfo.miIndexBufferAddress + iIndexBufferAddress, clusterInfo.miIndexBufferSize);

            MeshVertexFormat const* aVertices = reinterpret_cast<MeshVertexFormat const*>(aVertexBuffer.data());
            uint32_t const* aiIndices = reinterpret_cast<uint32_t const*>(aIndexBuffer.data());
            for(uint32_t iTri = 0; iTri < iNumIndices; iTri += 3)
            {
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iV = aiIndices[iTri + i];
                    assert(iV < iNumVertices);
                    MeshVertexFormat const& vertex = aVertices[iV];

                    MeshVertexFormat const& checkVertex = aaClusterTriangleVertices[iDrawCluster][aaiClusterTriangleVertexIndices[iDrawCluster][iTri + i]];
                    float fLength0 = length(vertex.mPosition - checkVertex.mPosition);
                    float fLength1 = length(vertex.mNormal - checkVertex.mNormal);
                    float fLength2 = length(vertex.mUV - checkVertex.mUV);

                    assert(fLength0 <= 1.0e-8f && fLength1 <= 1.0e-8f && fLength2 <= 1.0e-8f);
                }
            }
#endif // #if 0
        }

        int iDebug = 1;

    }
}

/*
**
*/
void testGetClusterRequests(
    std::vector<uint8_t>& aClusterRequestInfo)
{
    memcpy(aClusterRequestInfo.data(),
        saClusterInfoRequest.data(),
        aClusterRequestInfo.size());
}

/*
**
*/
void testUploadClusterData(
    void* paClusterRequestInfo,
    std::vector<uint32_t> const& aiDrawList,
    std::vector<std::vector<ConvertedMeshVertexFormat>> const& aaVertices,
    std::vector<std::vector<uint32_t>> const& aaiIndices,
    uint32_t iRequestClusterDataSize)
{
    uint8_t* pVertexDataBuffer = reinterpret_cast<uint8_t*>(testGetVertexDataBuffer());
    uint8_t* pIndexDataBuffer = reinterpret_cast<uint8_t*>(testGetIndexDataBuffer());

    uint8_t* pcStart = reinterpret_cast<uint8_t*>(paClusterRequestInfo);
    uint32_t* piDataUInt32 = reinterpret_cast<uint32_t*>(paClusterRequestInfo);
    uint32_t iNumClusterRequestInfo = *piDataUInt32++;

    uint32_t* piTest = reinterpret_cast<uint32_t*>(saClusterInfoRequest.data());
    uint32_t iTestNumClusterRequestInfo = *piTest++;

    for(uint32_t iCluster = 0; iCluster < aiDrawList.size(); iCluster++)
    {
        RequestClusterInfo clusterInfo;
        uint32_t iClusterID = aiDrawList[iCluster];
        uint32_t iRequestIndex = 0;
        for(iRequestIndex = 0; iRequestIndex < iNumClusterRequestInfo; iRequestIndex++)
        {
            loadClusterInfo(
                clusterInfo,
                paClusterRequestInfo,
                iRequestIndex,
                sizeof(uint32_t));

            if(clusterInfo.miCluster == iClusterID)
            {
                break;
            }
        }
        if(iRequestIndex >= iNumClusterRequestInfo)
        {
            // didn't find it, probably swapped out
            continue;
        }

        memcpy(
            pVertexDataBuffer + clusterInfo.miVertexBufferAddress,
            aaVertices[iCluster].data(),
            clusterInfo.miVertexBufferSize);
        memcpy(
            pIndexDataBuffer + clusterInfo.miIndexBufferAddress,
            aaiIndices[iCluster].data(),
            clusterInfo.miIndexBufferSize);

        // mark as loaded
        RequestClusterInfo* pClusterInfo = reinterpret_cast<RequestClusterInfo*>(pcStart + sizeof(uint32_t) + sizeof(RequestClusterInfo) * iCluster);
        pClusterInfo->miLoaded = 1;
    }

    for(uint32_t iCluster = 0; iCluster < iNumClusterRequestInfo; iCluster++)
    {
        RequestClusterInfo clusterInfo;
        loadClusterInfo(
            clusterInfo,
            paClusterRequestInfo,
            iCluster,
            sizeof(uint32_t));
    }

    void* pDest = testGetRequestClusterInfo();
    memcpy(
        pDest,
        paClusterRequestInfo,
        iRequestClusterDataSize);

    piDataUInt32 = reinterpret_cast<uint32_t*>(pDest);
    iNumClusterRequestInfo = *piDataUInt32++;
    for(uint32_t iCluster = 0; iCluster < iNumClusterRequestInfo; iCluster++)
    {
        RequestClusterInfo clusterInfo;
        loadClusterInfo(
            clusterInfo,
            pDest,
            iCluster,
            sizeof(uint32_t));
        int iDebug = 1;
    }
}

/*
**
*/
void testVerifyStreamClusterData(
    void const* aClusterInfoRequestBuffer,
    std::vector<uint32_t> const& aiDrawClusters,
    std::vector<std::vector<ConvertedMeshVertexFormat>> const& aaClusterTriangleVertices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleVertexIndices)
{
    uint32_t iVertexBufferAddress = 0;
    uint32_t iIndexBufferAddress = 0;
    
    uint8_t const* vertexDataBuffer = reinterpret_cast<uint8_t*>(testGetVertexDataBuffer());
    uint8_t const* indexDataBuffer =  reinterpret_cast<uint8_t*>(testGetIndexDataBuffer());

    uint32_t const* piDataUInt32 = reinterpret_cast<uint32_t const*>(aClusterInfoRequestBuffer);
    uint32_t iNumLoadedClusters = *piDataUInt32;
    for(uint32_t iDrawCluster = 0; iDrawCluster < aiDrawClusters.size(); iDrawCluster++)
    {
        uint32_t iClusterID = aiDrawClusters[iDrawCluster];

        RequestClusterInfo clusterInfo;
        uint32_t iClusterInfo = 0;
        for(iClusterInfo = 0; iClusterInfo < iNumLoadedClusters; iClusterInfo++)
        {
            loadClusterInfo(
                clusterInfo,
                aClusterInfoRequestBuffer,
                iClusterInfo,
                sizeof(uint32_t));

            if(clusterInfo.miCluster == iClusterID)
            {
                break;
            }
        }

        if(iClusterInfo >= iNumLoadedClusters)
        {
            // swapped out
            DEBUG_PRINTF("!!! can\'t find cluster %d !!!\n", iClusterID);
            for(uint32_t i = 0; i < iNumLoadedClusters; i++)
            {
                loadClusterInfo(
                    clusterInfo,
                    aClusterInfoRequestBuffer,
                    i,
                    sizeof(uint32_t));

                if(clusterInfo.miCluster == iClusterID)
                {
                    int iDebug = 1;
                }
                DEBUG_PRINTF("%d cluster %d\n", i, clusterInfo.miCluster);
            }

            continue;
        }

        uint32_t iNumVertices = clusterInfo.miVertexBufferSize / sizeof(ConvertedMeshVertexFormat);
        std::vector<uint8_t> aVertexBuffer(clusterInfo.miVertexBufferSize);
        memcpy(aVertexBuffer.data(), vertexDataBuffer + clusterInfo.miVertexBufferAddress + iVertexBufferAddress, clusterInfo.miVertexBufferSize);

        uint32_t iNumIndices = clusterInfo.miIndexBufferSize / sizeof(uint32_t);
        std::vector<uint8_t> aIndexBuffer(clusterInfo.miIndexBufferSize);
        memcpy(aIndexBuffer.data(), indexDataBuffer + clusterInfo.miIndexBufferAddress + iIndexBufferAddress, clusterInfo.miIndexBufferSize);

        ConvertedMeshVertexFormat const* aVertices = reinterpret_cast<ConvertedMeshVertexFormat const*>(aVertexBuffer.data());
        uint32_t const* aiIndices = reinterpret_cast<uint32_t const*>(aIndexBuffer.data());
        for(uint32_t iTri = 0; iTri < iNumIndices; iTri += 3)
        {
            for(uint32_t i = 0; i < 3; i++)
            {
                uint32_t iV = aiIndices[iTri + i];
                assert(iV < iNumVertices);
                ConvertedMeshVertexFormat const& vertex = aVertices[iV];

                ConvertedMeshVertexFormat const& checkVertex = aaClusterTriangleVertices[iDrawCluster][aaiClusterTriangleVertexIndices[iDrawCluster][iTri + i]];
                float fLength0 = length(vertex.mPosition - checkVertex.mPosition);
                float fLength1 = length(vertex.mNormal - checkVertex.mNormal);
                float fLength2 = length(vertex.mUV - checkVertex.mUV);

                assert(fLength0 <= 1.0e-8f && fLength1 <= 1.0e-8f && fLength2 <= 1.0e-8f);
            }
        }
    }
}

/*
**
*/
void* testGetVertexDataBuffer()
{
    return saVertexDataBuffer.data();
}

/*
**
*/
void* testGetIndexDataBuffer()
{
    return saIndexDataBuffer.data();
}

/*
**
*/
void* testGetRequestClusterInfo()
{
    return saClusterInfoRequest.data();
}