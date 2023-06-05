#include "mesh_cluster.h"
#include <algorithm>
#include <assert.h>

/*
**
*/
void saveMeshClusters(
    std::string const& outputFilePath,
    std::vector<MeshCluster*> const& apMeshClusters)
{
    std::vector<uint8_t> acFileContent(apMeshClusters.size() * sizeof(MeshCluster));
    uint32_t iCurrFileContentSize = 0;
    for(auto const* pCluster : apMeshClusters)
    {
        uint8_t* pFileContent = acFileContent.data() + iCurrFileContentSize;
        memcpy(pFileContent, pCluster, sizeof(MeshCluster));
        iCurrFileContentSize += static_cast<uint32_t>(sizeof(MeshCluster));
    }

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fwrite(acFileContent.data(), sizeof(char), iCurrFileContentSize, fp);
    fclose(fp);
}

/*
**
*/
void loadMeshClusters(
    std::vector<MeshCluster>& aMeshClusters,
    std::string const& filePath)
{
    FILE* fp = fopen(filePath.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    uint64_t iFileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> acFileContent(iFileSize);
    fread(acFileContent.data(), iFileSize, sizeof(char), fp);
    fclose(fp);

    uint64_t iNumClusters = iFileSize / sizeof(MeshCluster);
    aMeshClusters.resize(iNumClusters);
    memcpy(aMeshClusters.data(), acFileContent.data(), sizeof(MeshCluster) * aMeshClusters.size());
}

/*
**
*/
void saveMeshClusterData(
    std::vector<uint8_t> const& aVertexPositionBuffer,
    std::vector<uint8_t> const& aVertexNormalBuffer,
    std::vector<uint8_t> const& aVertexUVBuffer,
    std::vector<uint8_t> const& aiTrianglePositionIndexBuffer,
    std::vector<uint8_t> const& aiTriangleNormalIndexBuffer,
    std::vector<uint8_t> const& aiTriangleUVIndexBuffer,
    std::vector<MeshCluster*> const& apMeshClusters,
    std::string const& outputFilePath)
{
    uint32_t iVertexPositionDataEnd = 0;
    uint32_t iVertexNormalDataEnd = 0;
    uint32_t iVertexUVDataEnd = 0;

    uint32_t iPositionIndexDataEnd = 0;
    uint32_t iNormalIndexDataEnd = 0;
    uint32_t iUVIndexDataEnd = 0;

    for(auto const* pMeshCluster : apMeshClusters)
    {
        iVertexPositionDataEnd =
            (iVertexPositionDataEnd < pMeshCluster->miVertexPositionStartArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumVertexPositions * sizeof(float3))) ?
            static_cast<uint32_t>(pMeshCluster->miVertexPositionStartArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumVertexPositions * sizeof(float3))) :
            iVertexPositionDataEnd;

        iVertexNormalDataEnd =
            (iVertexNormalDataEnd < pMeshCluster->miVertexNormalStartArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumVertexNormals * sizeof(float3))) ?
            static_cast<uint32_t>(pMeshCluster->miVertexNormalStartArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumVertexNormals * sizeof(float3))) :
            iVertexNormalDataEnd;

        iVertexUVDataEnd =
            (iVertexUVDataEnd < pMeshCluster->miVertexUVStartArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumVertexUVs * sizeof(float2))) ?
            static_cast<uint32_t>(pMeshCluster->miVertexUVStartArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumVertexUVs * sizeof(float2))) :
            iVertexUVDataEnd;

        iPositionIndexDataEnd =
            (iPositionIndexDataEnd < pMeshCluster->miTrianglePositionIndexArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumTrianglePositionIndices * sizeof(uint32_t))) ?
            static_cast<uint32_t>(pMeshCluster->miTrianglePositionIndexArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumTrianglePositionIndices * sizeof(uint32_t))) :
            iPositionIndexDataEnd;

        iNormalIndexDataEnd =
            (iNormalIndexDataEnd < pMeshCluster->miTriangleNormalIndexArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumTriangleNormalIndices * sizeof(uint32_t))) ?
            static_cast<uint32_t>(pMeshCluster->miTriangleNormalIndexArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumTriangleNormalIndices * sizeof(uint32_t))) :
            iNormalIndexDataEnd;

        iUVIndexDataEnd =
            (iUVIndexDataEnd < pMeshCluster->miTriangleUVIndexArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumTriangleUVIndices * sizeof(uint32_t))) ?
            static_cast<uint32_t>(pMeshCluster->miTriangleUVIndexArrayAddress + static_cast<uint64_t>(pMeshCluster->miNumTriangleUVIndices * sizeof(uint32_t))) :
            iUVIndexDataEnd;
    }

    uint32_t iTotalFileSize = iVertexPositionDataEnd + iVertexNormalDataEnd + iVertexUVDataEnd + iPositionIndexDataEnd + iNormalIndexDataEnd + iUVIndexDataEnd;
    std::vector<uint8_t> acFileContent(iTotalFileSize + 6 * sizeof(uint32_t));
    uint32_t* piData = reinterpret_cast<uint32_t*>(acFileContent.data());
    *piData++ = iVertexPositionDataEnd + sizeof(uint32_t) * 6;
    *piData++ = *(piData - 1) + iVertexNormalDataEnd;
    *piData++ = *(piData - 1) + iVertexUVDataEnd;
    *piData++ = *(piData - 1) + iPositionIndexDataEnd;
    *piData++ = *(piData - 1) + iNormalIndexDataEnd;
    *piData++ = *(piData - 1) + iUVIndexDataEnd;
    
    uint8_t* pcData = reinterpret_cast<uint8_t*>(piData);
    uint8_t* pStart = reinterpret_cast<uint8_t*>(acFileContent.data());
    uint64_t iStartAddress = reinterpret_cast<uint64_t>(pStart);

    piData = reinterpret_cast<uint32_t*>(acFileContent.data());
    uint64_t iCheckVertexPosition = static_cast<uint64_t>(*piData++);
    uint64_t iCheckVertexNormal =   static_cast<uint64_t>(*piData++);
    uint64_t iCheckVertexUV =       static_cast<uint64_t>(*piData++);
    uint64_t iCheckPositionIndex =  static_cast<uint64_t>(*piData++);
    uint64_t iCheckNormalIndex =    static_cast<uint64_t>(*piData++);
    uint64_t iCheckUVIndex =        static_cast<uint64_t>(*piData++);
    
    memcpy(pcData, aVertexPositionBuffer.data(), iVertexPositionDataEnd);
    pcData += iVertexPositionDataEnd;
    assert(reinterpret_cast<uint64_t>(pcData) - iStartAddress == iCheckVertexPosition);

    memcpy(pcData, aVertexNormalBuffer.data(), iVertexNormalDataEnd);
    pcData += iVertexNormalDataEnd;
    assert(reinterpret_cast<uint64_t>(pcData) - iStartAddress == iCheckVertexNormal);

    memcpy(pcData, aVertexUVBuffer.data(), iVertexUVDataEnd);
    pcData += iVertexUVDataEnd;
    assert(reinterpret_cast<uint64_t>(pcData) - iStartAddress == iCheckVertexUV);

    memcpy(pcData, aiTrianglePositionIndexBuffer.data(), iPositionIndexDataEnd);
    pcData += iPositionIndexDataEnd;
    assert(reinterpret_cast<uint64_t>(pcData) - iStartAddress == iCheckPositionIndex);

    memcpy(pcData, aiTriangleNormalIndexBuffer.data(), iNormalIndexDataEnd);
    pcData += iNormalIndexDataEnd;
    assert(reinterpret_cast<uint64_t>(pcData) - iStartAddress == iCheckNormalIndex);

    memcpy(pcData, aiTriangleUVIndexBuffer.data(), iUVIndexDataEnd);
    pcData += iUVIndexDataEnd;
    assert(reinterpret_cast<uint64_t>(pcData) - iStartAddress == iCheckUVIndex);

    assert(acFileContent.size() == iCheckUVIndex);

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fwrite(acFileContent.data(), sizeof(char), acFileContent.size(), fp);
    fclose(fp);
}

/*
**
*/
void saveMeshClusterTriangleData(
    std::vector<uint8_t> const& aVertexPositionBuffer,
    std::vector<uint8_t> const& aVertexNormalBuffer,
    std::vector<uint8_t> const& aVertexUVBuffer,
    std::vector<uint8_t> const& aiTrianglePositionIndexBuffer,
    std::vector<uint8_t> const& aiTriangleNormalIndexBuffer,
    std::vector<uint8_t> const& aiTriangleUVIndexBuffer,
    std::vector<MeshCluster*> const& apMeshClusters,
    std::string const& outputFilePath,
    std::string const& outputVertexDataFilePath,
    std::string const& outputIndexDataFilePath)
{
    std::vector<MeshCluster*> apSortedMeshClusters = apMeshClusters;
    std::sort(
        apSortedMeshClusters.begin(),
        apSortedMeshClusters.end(),
        [](MeshCluster* pLeft, MeshCluster* pRight)
        {
            return pLeft->miIndex < pRight->miIndex;
        }
    );
    
    std::vector<std::vector<MeshVertexFormat>> aaClusterVertices;
    std::vector<std::vector<uint32_t>> aaiClusterVertexIndices;
    for(auto const& pMeshCluster : apSortedMeshClusters)
    {
        std::vector<float3> aClusterVertexPositions(pMeshCluster->miNumVertexPositions);
        memcpy(
            aClusterVertexPositions.data(),
            aVertexPositionBuffer.data() + pMeshCluster->miVertexPositionStartArrayAddress * sizeof(float3),
            pMeshCluster->miNumVertexPositions * sizeof(float3));

        std::vector<float3> aClusterVertexNormals(pMeshCluster->miNumVertexNormals);
        memcpy(
            aClusterVertexNormals.data(),
            aVertexNormalBuffer.data() + pMeshCluster->miVertexNormalStartArrayAddress * sizeof(float3),
            pMeshCluster->miNumVertexNormals * sizeof(float3));

        std::vector<float2> aClusterVertexUVs(pMeshCluster->miNumVertexUVs);
        memcpy(
            aClusterVertexUVs.data(),
            aVertexUVBuffer.data() + pMeshCluster->miVertexUVStartArrayAddress * sizeof(float2),
            pMeshCluster->miNumVertexUVs * sizeof(float2));

        std::vector<uint32_t> aClusterPositionIndices(pMeshCluster->miNumTrianglePositionIndices);
        memcpy(
            aClusterPositionIndices.data(),
            aiTrianglePositionIndexBuffer.data() + pMeshCluster->miTrianglePositionIndexArrayAddress * sizeof(uint32_t),
            pMeshCluster->miNumTrianglePositionIndices * sizeof(uint32_t));

        std::vector<uint32_t> aClusterNormalIndices(pMeshCluster->miNumTriangleNormalIndices);
        memcpy(
            aClusterNormalIndices.data(),
            aiTriangleNormalIndexBuffer.data() + pMeshCluster->miTriangleNormalIndexArrayAddress * sizeof(uint32_t),
            pMeshCluster->miNumTriangleNormalIndices * sizeof(uint32_t));

        std::vector<uint32_t> aClusterUVIndices(pMeshCluster->miNumTriangleUVIndices);
        memcpy(
            aClusterUVIndices.data(),
            aiTriangleUVIndexBuffer.data() + pMeshCluster->miTriangleUVIndexArrayAddress * sizeof(uint32_t),
            pMeshCluster->miNumTriangleUVIndices * sizeof(uint32_t));

        std::vector<MeshVertexFormat> aVertices;
        std::vector<uint32_t> aiVertexIndices;
        for(uint32_t iTri = 0; iTri < pMeshCluster->miNumTrianglePositionIndices; iTri += 3)
        {
            for(uint32_t i = 0; i < 3; i++)
            {
                uint32_t iPos = aClusterPositionIndices[iTri + i];
                assert(iPos < aClusterVertexPositions.size());
                float3 const& pos = aClusterVertexPositions[iPos];

                uint32_t iNorm = aClusterNormalIndices[iTri + i];
                assert(iNorm < aClusterVertexNormals.size());
                float3 const& norm = aClusterVertexNormals[iNorm];

                uint32_t iUV = aClusterUVIndices[iTri + i];
                assert(iUV < aClusterVertexUVs.size());
                float2 const& uv = aClusterVertexUVs[iUV];

                auto existIter = std::find_if(
                    aVertices.begin(),
                    aVertices.end(),
                    [pos, norm, uv](MeshVertexFormat const& v)
                    {
                        return
                            lengthSquared(v.mPosition - pos) <= 1.0e-8f &&
                            lengthSquared(v.mNormal - norm) <= 1.0e-8f &&
                            lengthSquared(v.mUV - uv) <= 1.0e-8f;
                    }
                );

                if(existIter != aVertices.end())
                {
                    uint32_t iVertexIndex = static_cast<uint32_t>(std::distance(aVertices.begin(), existIter));
                    aiVertexIndices.push_back(iVertexIndex);
                }
                else
                {
                    aVertices.emplace_back(pos, norm, uv);
                    aiVertexIndices.push_back(static_cast<uint32_t>(aVertices.size() - 1));
                }

            }   // for i = 0 to 3

        }   // for tri = 0 to num triangles

        aaClusterVertices.push_back(aVertices);
        aaiClusterVertexIndices.push_back(aiVertexIndices);

    }   // for cluster to all clusters

    uint64_t iTotalSize = 0;
    for(auto const& aClusterVertices : aaClusterVertices)
    {
        iTotalSize += (aClusterVertices.size() * sizeof(MeshVertexFormat));
    }

    for(auto const& aiClusterVertexIndices : aaiClusterVertexIndices)
    {
        iTotalSize += (aiClusterVertexIndices.size() * sizeof(uint32_t));
    }

    {
        // vertex buffer
        {
#if 0
            std::vector<uint8_t> acVertexBufferFileContent((aaClusterVertices.size() + 1) * sizeof(uint32_t));
            uint64_t iStartAddress = reinterpret_cast<uint64_t>(acVertexBufferFileContent.data());
            uint32_t* piData = reinterpret_cast<uint32_t*>(acVertexBufferFileContent.data());
            *piData++ = static_cast<uint32_t>(aaClusterVertices.size());
            for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertices.size()); i++)
            {
                *piData++ = static_cast<uint32_t>(aaClusterVertices[i].size());
            }
            uint32_t iDataOffset = static_cast<uint32_t>(acVertexBufferFileContent.size());
            MeshVertexFormat* pVertexData = reinterpret_cast<MeshVertexFormat*>(piData);
            for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertices.size()); i++)
            {
                uint32_t iSize = static_cast<uint32_t>(aaClusterVertices[i].size() * sizeof(MeshVertexFormat));
                acVertexBufferFileContent.resize(acVertexBufferFileContent.size() + iSize);
                iStartAddress = reinterpret_cast<uint64_t>(acVertexBufferFileContent.data());
                pVertexData = reinterpret_cast<MeshVertexFormat*>(acVertexBufferFileContent.data() + iDataOffset);

                memcpy(
                    pVertexData,
                    aaClusterVertices[i].data(),
                    iSize);
                iDataOffset += iSize;
            }

            FILE* fp = fopen(outputVertexDataFilePath.c_str(), "wb");
            fwrite(acVertexBufferFileContent.data(), sizeof(char), acVertexBufferFileContent.size(), fp);
            fclose(fp);
#endif // #if 0

            // CHANGE TO float4, float4, float4 format
            {
                std::vector<uint8_t> acVertexBufferFileContent((aaClusterVertices.size() + 1) * sizeof(uint32_t));
                uint64_t iStartAddress = reinterpret_cast<uint64_t>(acVertexBufferFileContent.data());
                uint32_t* piData = reinterpret_cast<uint32_t*>(acVertexBufferFileContent.data());
                *piData++ = static_cast<uint32_t>(aaClusterVertices.size());
                ConvertedMeshVertexFormat* pVertexData = reinterpret_cast<ConvertedMeshVertexFormat*>(piData);
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertices.size()); i++)
                {
                    *piData++ = static_cast<uint32_t>(aaClusterVertices[i].size());
                }
                uint32_t iDataOffset = static_cast<uint32_t>(acVertexBufferFileContent.size());
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertices.size()); i++)
                {
                    uint32_t iNumClusterVertices = static_cast<uint32_t>(aaClusterVertices[i].size());
                    uint32_t iSize = static_cast<uint32_t>(iNumClusterVertices * sizeof(ConvertedMeshVertexFormat));
                    acVertexBufferFileContent.resize(acVertexBufferFileContent.size() + iSize);
                    iStartAddress = reinterpret_cast<uint64_t>(acVertexBufferFileContent.data());
                    pVertexData = reinterpret_cast<ConvertedMeshVertexFormat*>(acVertexBufferFileContent.data() + iDataOffset);
                    for(uint32_t j = 0; j < iNumClusterVertices; j++)
                    {
                        MeshVertexFormat const& vert = aaClusterVertices[i][j];
                        pVertexData->mPosition = float4(vert.mPosition, 1.0f);
                        pVertexData->mNormal = float4(vert.mNormal, 1.0f);
                        pVertexData->mUV = float4(vert.mUV.x, vert.mUV.y, 0.0f, 0.0f);
                        ++pVertexData;

                        iDataOffset += sizeof(ConvertedMeshVertexFormat);
                    }
                }

                FILE* fp = fopen(outputVertexDataFilePath.c_str(), "wb");
                fwrite(acVertexBufferFileContent.data(), sizeof(char), acVertexBufferFileContent.size(), fp);
                fclose(fp);

            }   // change to new vertex format


            
        }

        // index buffer
        {
            std::vector<uint8_t> acIndexBufferFileContent((aaiClusterVertexIndices.size() + 1) * sizeof(uint32_t));
            uint64_t iStartAddress = reinterpret_cast<uint64_t>(acIndexBufferFileContent.data());
            uint32_t* piData = reinterpret_cast<uint32_t*>(acIndexBufferFileContent.data());
            *piData++ = static_cast<uint32_t>(aaiClusterVertexIndices.size());
            for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterVertexIndices.size()); i++)
            {
                *piData++ = static_cast<uint32_t>(aaiClusterVertexIndices[i].size());
            }
            uint32_t iDataOffset = static_cast<uint32_t>(acIndexBufferFileContent.size());
            for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterVertexIndices.size()); i++)
            {
                uint32_t iSize = static_cast<uint32_t>(aaiClusterVertexIndices[i].size() * sizeof(uint32_t));
                acIndexBufferFileContent.resize(acIndexBufferFileContent.size() + iSize);
                iStartAddress = reinterpret_cast<uint64_t>(acIndexBufferFileContent.data());
                piData = reinterpret_cast<uint32_t*>(acIndexBufferFileContent.data() + iDataOffset);

                memcpy(
                    piData,
                    aaiClusterVertexIndices[i].data(),
                    iSize);
                iDataOffset += iSize;
            }

            FILE* fp = fopen(outputIndexDataFilePath.c_str(), "wb");
            fwrite(acIndexBufferFileContent.data(), sizeof(char), acIndexBufferFileContent.size(), fp);
            fclose(fp);

        }   // index buffer
    }




    // temp values for now
    std::vector<uint8_t> acFileContent(aaClusterVertices.size() * 2 * sizeof(uint32_t) + 2 * sizeof(uint32_t));
    uint64_t iStartAddress = reinterpret_cast<uint64_t>(acFileContent.data());
    uint32_t* piData = reinterpret_cast<uint32_t*>(acFileContent.data());
    *piData++ = static_cast<uint32_t>(aaClusterVertices.size());
    *piData++ = static_cast<uint32_t>(aaiClusterVertexIndices.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertices.size()); i++)
    {
        *piData++ = static_cast<uint32_t>(aaClusterVertices[i].size());
        *piData++ = static_cast<uint32_t>(aaiClusterVertexIndices[i].size());
    }

    // vertices
    uint32_t iDataOffset = static_cast<uint32_t>(acFileContent.size());
    std::vector<uint32_t> aiVertexDataOffset(aaClusterVertices.size());
    ConvertedMeshVertexFormat* pVertexData = reinterpret_cast<ConvertedMeshVertexFormat*>(piData);
    for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertices.size()); i++)
    {
        uint64_t iCurrDataOffset = reinterpret_cast<uint64_t>(pVertexData) - iStartAddress;
        assert(iCurrDataOffset <= acFileContent.size() * sizeof(char));

        aiVertexDataOffset[i] = iDataOffset;
        uint32_t iSize = static_cast<uint32_t>(aaClusterVertices[i].size() * sizeof(ConvertedMeshVertexFormat));
        acFileContent.resize(acFileContent.size() + iSize);
        iStartAddress = reinterpret_cast<uint64_t>(acFileContent.data());
        pVertexData = reinterpret_cast<ConvertedMeshVertexFormat*>(acFileContent.data() + iDataOffset);

        for(uint32_t j = 0; j < static_cast<uint32_t>(aaClusterVertices[i].size()); j++)
        {
            pVertexData->mPosition = float4(aaClusterVertices[i][j].mPosition, 1.0f);
            pVertexData->mNormal = float4(aaClusterVertices[i][j].mNormal, 1.0f);
            pVertexData->mUV = float4(aaClusterVertices[i][j].mUV.x, aaClusterVertices[i][j].mUV.y, 0.0f, 0.0f);
            
            pVertexData += 1;
            iDataOffset += static_cast<uint32_t>(sizeof(ConvertedMeshVertexFormat));
        }

        //memcpy(
        //    pVertexData,
        //    aaClusterVertices[i].data(),
        //    iSize);
        //iDataOffset += iSize;
    }

    // vertex indices
    std::vector<uint32_t> aiVertexIndexOffset(aaiClusterVertexIndices.size());
    uint32_t* piIndexData = reinterpret_cast<uint32_t*>(acFileContent.data() + iDataOffset);
    for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterVertexIndices.size()); i++)
    {
        uint64_t iCurrDataOffset = reinterpret_cast<uint64_t>(piIndexData) - iStartAddress;
        assert(iCurrDataOffset <= acFileContent.size() * sizeof(char));

        aiVertexIndexOffset[i] = iDataOffset;
        uint32_t iSize = static_cast<uint32_t>(aaiClusterVertexIndices[i].size() * sizeof(uint32_t));
        acFileContent.resize(acFileContent.size() + iSize);
        iStartAddress = reinterpret_cast<uint64_t>(acFileContent.data());
        piIndexData = reinterpret_cast<uint32_t*>(acFileContent.data() + iDataOffset);

        memcpy(
            piIndexData,
            aaiClusterVertexIndices[i].data(),
            iSize);
        iDataOffset += iSize;
    }

    // data offset 
    //piData = reinterpret_cast<uint32_t*>(acFileContent.data());
    //piData += 2;
    //for(uint32_t i = 0; i < aiVertexDataOffset.size(); i++)
    //{
    //    *piData++ = aiVertexDataOffset[i];
    //    *piData++ = aiVertexIndexOffset[i];
    //}

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fwrite(acFileContent.data(), sizeof(char), acFileContent.size(), fp);
    fclose(fp);

}

/*
**
*/
void loadMeshClusterTriangleData(
    std::string const& filePath,
    std::vector<std::vector<MeshVertexFormat>>& aaVertices,
    std::vector<std::vector<uint32_t>>& aaiTriangleVertexIndices)
{
    FILE* fp = fopen(filePath.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    uint64_t iFileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> acFileContent(iFileSize);
    fread(acFileContent.data(), sizeof(char), iFileSize, fp);
    fclose(fp);

    uint32_t* piData = reinterpret_cast<uint32_t*>(acFileContent.data());
    uint32_t iNumClusterVertexList = *piData++;
    uint32_t iNumClusterVertexIndexList = *piData++;
    uint64_t iStartAddress = reinterpret_cast<uint64_t>(acFileContent.data());

    aaVertices.resize(iNumClusterVertexList);
    aaiTriangleVertexIndices.resize(iNumClusterVertexIndexList);

    std::vector<uint32_t> aiDataVertexSize(iNumClusterVertexList);
    for(uint32_t i = 0; i < iNumClusterVertexList; i++)
    {
        uint32_t iNumVertices = *piData;
        aaVertices[i].resize(iNumVertices);

        uint32_t iNumIndices = *(piData + 1);
        aaiTriangleVertexIndices[i].resize(iNumIndices);

        piData += 2;
    }

    MeshVertexFormat* pVertexFormat = reinterpret_cast<MeshVertexFormat*>(piData);
    uint64_t iCurrDataOffset = reinterpret_cast<uint64_t>(pVertexFormat) - iStartAddress;
    for(uint32_t i = 0; i < iNumClusterVertexList; i++)
    {
        memcpy(
            aaVertices[i].data(),
            pVertexFormat,
            aaVertices[i].size() * sizeof(MeshVertexFormat));
        pVertexFormat += aaVertices[i].size();
    }

    iCurrDataOffset = reinterpret_cast<uint64_t>(pVertexFormat) - iStartAddress;

    uint32_t* piVertexIndices = reinterpret_cast<uint32_t*>(pVertexFormat);
    for(uint32_t i = 0; i < iNumClusterVertexList; i++)
    {
        memcpy(
            aaiTriangleVertexIndices[i].data(),
            piVertexIndices,
            aaiTriangleVertexIndices[i].size() * sizeof(uint32_t));
        piVertexIndices += aaiTriangleVertexIndices[i].size();
    }
}

/*
**
*/
void loadMeshClusterTriangleDataTableOfContent(
    std::vector<uint32_t>& aiNumClusterVertices,
    std::vector<uint32_t>& aiNumClusterIndices,
    std::vector<uint64_t>& aiVertexBufferArrayOffsets,
    std::vector<uint64_t>& aiIndexBufferArrayOffset,
    std::string const& vertexDataFilePath,
    std::string const& indexDataFilePath)
{
    {
        std::vector<uint8_t> acContentBuffer(1 << 20);
        FILE* fp = fopen(vertexDataFilePath.c_str(), "rb");
        fread(acContentBuffer.data(), sizeof(char), sizeof(uint32_t), fp);
        uint32_t* piIntData = reinterpret_cast<uint32_t*>(acContentBuffer.data());
        uint32_t iNumClusters = *piIntData++;
        acContentBuffer.resize(sizeof(uint32_t) * iNumClusters * sizeof(uint32_t));
        piIntData = reinterpret_cast<uint32_t*>(acContentBuffer.data() + sizeof(uint32_t));
        fread(piIntData, sizeof(char), sizeof(uint32_t) * iNumClusters, fp);
        fclose(fp);

        aiNumClusterVertices.resize(iNumClusters);
        aiVertexBufferArrayOffsets.resize(iNumClusters);
        uint32_t iVertexArrayOffset = 0;
        for(uint32_t i = 0; i < iNumClusters; i++)
        {
            aiNumClusterVertices[i] = *piIntData++;
            aiVertexBufferArrayOffsets[i] = iVertexArrayOffset;
            iVertexArrayOffset += aiNumClusterVertices[i];
        }
    }

    {
        std::vector<uint8_t> acContentBuffer(1 << 20);
        FILE* fp = fopen(indexDataFilePath.c_str(), "rb");
        fread(acContentBuffer.data(), sizeof(char), sizeof(uint32_t), fp);
        uint32_t* piIntData = reinterpret_cast<uint32_t*>(acContentBuffer.data());
        uint32_t iNumClusters = *piIntData++;
        acContentBuffer.resize(sizeof(uint32_t) * iNumClusters * sizeof(uint32_t));
        piIntData = reinterpret_cast<uint32_t*>(acContentBuffer.data() + sizeof(uint32_t));
        fread(piIntData, sizeof(char), sizeof(uint32_t) * iNumClusters, fp);
        fclose(fp);

        aiNumClusterIndices.resize(iNumClusters);
        aiIndexBufferArrayOffset.resize(iNumClusters);
        uint32_t iIndexArrayOffset = 0;
        for(uint32_t i = 0; i < iNumClusters; i++)
        {
            aiNumClusterIndices[i] = *piIntData++;
            aiIndexBufferArrayOffset[i] = iIndexArrayOffset;
            iIndexArrayOffset += aiNumClusterIndices[i];
        }
        
    }
}

/*
**
*/
void loadMeshClusterTriangleDataChunk(
    std::vector<ConvertedMeshVertexFormat>& aClusterTriangleVertices,
    std::vector<uint32_t>& aiClusterTriangleVertexIndices,
    std::string const& vertexDataFilePath,
    std::string const& indexDataFilePath,
    std::vector<uint32_t> const& aiNumClusterVertices,
    std::vector<uint32_t> const& aiNumClusterIndices,
    std::vector<uint64_t> const& aiVertexBufferArrayOffsets,
    std::vector<uint64_t> const& aiIndexBufferArrayOffsets,
    uint32_t iClusterIndex)
{
    assert(iClusterIndex < aiVertexBufferArrayOffsets.size());
    
    FILE* fp = fopen(vertexDataFilePath.c_str(), "rb");
    uint64_t iVertexDataBufferSize = aiNumClusterVertices[iClusterIndex] * sizeof(ConvertedMeshVertexFormat);
    aClusterTriangleVertices.resize(aiNumClusterVertices[iClusterIndex]);
    uint64_t iStartVertexDataOffset = sizeof(uint32_t) + sizeof(uint32_t) * aiNumClusterVertices.size() + aiVertexBufferArrayOffsets[iClusterIndex] * sizeof(ConvertedMeshVertexFormat);
    fseek(fp, static_cast<long>(iStartVertexDataOffset), SEEK_SET);
    fread(aClusterTriangleVertices.data(), sizeof(char), iVertexDataBufferSize, fp);
    fclose(fp);

    fp = fopen(indexDataFilePath.c_str(), "rb");
    uint64_t iIndexDataBufferSize = aiNumClusterIndices[iClusterIndex] * sizeof(uint32_t);
    uint64_t iStartIndexDataOffset = sizeof(uint32_t) + sizeof(uint32_t) * aiNumClusterIndices.size() + aiIndexBufferArrayOffsets[iClusterIndex] * sizeof(uint32_t);
    aiClusterTriangleVertexIndices.resize(aiNumClusterIndices[iClusterIndex]);
    fseek(fp, static_cast<long>(iStartIndexDataOffset), SEEK_SET);
    fread(aiClusterTriangleVertexIndices.data(), sizeof(char), iIndexDataBufferSize, fp);
    fclose(fp);
}

