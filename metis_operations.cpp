#include "tiny_obj_loader.h"
#include "metis_operations.h"

#include "test.h"
#include "LogPrint.h"

#include <atomic>
#include <cassert>
#include <thread>

/*
**
*/
bool readMetisClusterFile(
    std::vector<uint32_t>& aiClusters,
    std::string const& filePath)
{
    bool bRet = false;

    std::vector<char> acFileData;
    FILE* fp = nullptr;
    if(filePath.length() > 0)
    {
        fp = fopen(filePath.c_str(), "rb");
        assert(fp);

        fseek(fp, 0, SEEK_END);
        uint64_t iFileSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        acFileData.resize(iFileSize + 1);
        fread(acFileData.data(), sizeof(char), iFileSize, fp);
        bRet = true;
        fclose(fp);
    }
    else
    {
        bRet = false;
    }

    std::string fileContent;
    if(acFileData.size() > 0)
    {
        fileContent = acFileData.data();
    }

    uint64_t iDataOffset = 0;
    for(iDataOffset = 0; iDataOffset < fileContent.length();)
    {
        if(iDataOffset < fileContent.length())
        {
            uint64_t iEnd = fileContent.find("\n", iDataOffset);
            if(iEnd == std::string::npos)
            {
                break;
            }
            else
            {
                std::string clusterNumber = fileContent.substr(iDataOffset, iEnd - iDataOffset);
                uint32_t iCluster = atoi(clusterNumber.c_str());
                aiClusters.push_back(iCluster);
            }

            iDataOffset = iEnd + 1;
        }
    }

    return bRet;
}

/*
**
*/
void outputMeshClusters2(
    std::vector<float3>& aVertexPositions,
    std::vector<float3>& aVertexNormals,
    std::vector<float2>& aVertexUVs,
    std::vector<uint32_t>& aiRetTrianglePositionIndices,
    std::vector<uint32_t>& aiRetTriangleNormalIndices,
    std::vector<uint32_t>& aiRetTriangleUVIndices,
    std::vector<uint32_t> const& aiClusterIndices,
    std::vector<float3> const& aTriangleVertexPositions,
    std::vector<float3> const& aTriangleVertexNormals,
    std::vector<float2> const& aTriangleVertexUVs,
    std::vector<uint32_t> const& aiTrianglePositionIndices,
    std::vector<uint32_t> const& aiTriangleNormalIndices,
    std::vector<uint32_t> const& aiTriangleUVIndices,
    uint32_t iLODLevel,
    uint32_t iOutputClusterIndex)
{
    //std::vector<uint32_t> aiClusters;
    //readMetisClusterFile(aiClusters, elementPartFilePath);

    // go through the list of element (position, normal, or uv) and add them to list if the cluster index from the metis element file matches with the given cluster
    //for(uint32_t iElementIndex = 0; iElementIndex < static_cast<uint32_t>(aiClusters.size()); iElementIndex++)
    for(uint32_t i = 0; i < static_cast<uint32_t>(aiClusterIndices.size()); i++)
    {
        bool bReading = false;
        uint32_t const& iElementIndex = aiClusterIndices[i];
        {
            // same cluster from the file as given cluster index
            //if(iCluster == iClusterIndex)
            {
                uint32_t iVertex0 = aiTrianglePositionIndices[iElementIndex * 3];
                uint32_t iVertex1 = aiTrianglePositionIndices[iElementIndex * 3 + 1];
                uint32_t iVertex2 = aiTrianglePositionIndices[iElementIndex * 3 + 2];

                assert(iVertex0 != iVertex1 && iVertex0 != iVertex2 && iVertex1 != iVertex2);

                uint32_t iNormal0 = aiTriangleNormalIndices[iElementIndex * 3];
                uint32_t iNormal1 = aiTriangleNormalIndices[iElementIndex * 3 + 1];
                uint32_t iNormal2 = aiTriangleNormalIndices[iElementIndex * 3 + 2];

                uint32_t iUV0 = (aiTriangleUVIndices.size() > 0) ? aiTriangleUVIndices[iElementIndex * 3] : 0;
                uint32_t iUV1 = (aiTriangleUVIndices.size() > 0) ? aiTriangleUVIndices[iElementIndex * 3 + 1] : 0;
                uint32_t iUV2 = (aiTriangleUVIndices.size() > 0) ? aiTriangleUVIndices[iElementIndex * 3 + 2] : 0;

                float3 const& position0 = aTriangleVertexPositions[iVertex0];
                float3 const& position1 = aTriangleVertexPositions[iVertex1];
                float3 const& position2 = aTriangleVertexPositions[iVertex2];

                float3 const& normal0 = (aTriangleVertexNormals.size() > 0) ? aTriangleVertexNormals[iNormal0] : float3(0.0f, 0.0f, 0.0f);
                float3 const& normal1 = (aTriangleVertexNormals.size() > 0) ? aTriangleVertexNormals[iNormal1] : float3(0.0f, 0.0f, 0.0f);
                float3 const& normal2 = (aTriangleVertexNormals.size() > 0) ? aTriangleVertexNormals[iNormal2] : float3(0.0f, 0.0f, 0.0f);

                float2 const& uv0 = (aiTriangleUVIndices.size() > 0 && iUV0 != UINT32_MAX) ? aTriangleVertexUVs[iUV0] : float2(0.0f, 0.0f);
                float2 const& uv1 = (aiTriangleUVIndices.size() > 0 && iUV1 != UINT32_MAX) ? aTriangleVertexUVs[iUV1] : float2(0.0f, 0.0f);
                float2 const& uv2 = (aiTriangleUVIndices.size() > 0 && iUV2 != UINT32_MAX) ? aTriangleVertexUVs[iUV2] : float2(0.0f, 0.0f);

                float kfDifferenceThreshold = 1.0e-9f;

                bool bAddedTriangle = false;
                // remap vertex positions
                {
                    auto iter0 = std::find_if(
                        aVertexPositions.begin(),
                        aVertexPositions.end(),
                        [position0, kfDifferenceThreshold](float3 const& checkPosition)
                        {
                            return (length(checkPosition - position0) <= kfDifferenceThreshold);
                        });
                    bool bFound0 = (iter0 != aVertexPositions.end());
                    uint32_t iIndex0 = (bFound0 == true) ? static_cast<uint32_t>(std::distance(aVertexPositions.begin(), iter0)) : UINT32_MAX;

                    auto iter1 = std::find_if(
                        aVertexPositions.begin(),
                        aVertexPositions.end(),
                        [position1, kfDifferenceThreshold](float3 const& checkPosition)
                        {
                            return (length(checkPosition - position1) <= kfDifferenceThreshold);
                        });
                    bool bFound1 = (iter1 != aVertexPositions.end());
                    uint32_t iIndex1 = (bFound1 == true) ? static_cast<uint32_t>(std::distance(aVertexPositions.begin(), iter1)) : UINT32_MAX;

                    auto iter2 = std::find_if(
                        aVertexPositions.begin(),
                        aVertexPositions.end(),
                        [position2, kfDifferenceThreshold](float3 const& checkPosition)
                        {
                            return (length(checkPosition - position2) <= kfDifferenceThreshold);
                        });
                    bool bFound2 = (iter2 != aVertexPositions.end());
                    uint32_t iIndex2 = (bFound2 == true) ? static_cast<uint32_t>(std::distance(aVertexPositions.begin(), iter2)) : UINT32_MAX;

                    uint32_t iRemap0 = static_cast<uint32_t>(aVertexPositions.size());
                    if(bFound0)
                    {
                        iRemap0 = iIndex0;
                    }
                    else
                    {
                        aVertexPositions.push_back(position0);
                    }

                    uint32_t iRemap1 = static_cast<uint32_t>(aVertexPositions.size());
                    if(bFound1)
                    {
                        iRemap1 = iIndex1;
                    }
                    else
                    {
                        aVertexPositions.push_back(position1);
                    }

                    uint32_t iRemap2 = static_cast<uint32_t>(aVertexPositions.size());
                    if(bFound2)
                    {
                        iRemap2 = iIndex2;
                    }
                    else
                    {
                        aVertexPositions.push_back(position2);
                    }

                    //assert(iRemap0 != iRemap1 && iRemap0 != iRemap2 && iRemap1 != iRemap2);

                    if((iRemap0 != iRemap1 && iRemap0 != iRemap2 && iRemap1 != iRemap2))
                    {
                        aiRetTrianglePositionIndices.push_back(iRemap0);
                        aiRetTrianglePositionIndices.push_back(iRemap1);
                        aiRetTrianglePositionIndices.push_back(iRemap2);

                        bAddedTriangle = true;
                    }
                }

                // re-map of normal index
                if(bAddedTriangle)
                {
                    uint32_t iNormalIndex0 = UINT32_MAX, iNormalIndex1 = UINT32_MAX, iNormalIndex2 = UINT32_MAX;

                    auto iter0 = std::find_if(
                        aVertexNormals.begin(),
                        aVertexNormals.end(),
                        [normal0, kfDifferenceThreshold](float3 const& checkNormal)
                        {
                            return (length(checkNormal - normal0) <= kfDifferenceThreshold);
                        });
                    bool bFound0 = (iter0 != aVertexNormals.end());
                    iNormalIndex0 = (bFound0 == true) ? static_cast<uint32_t>(std::distance(aVertexNormals.begin(), iter0)) : UINT32_MAX;

                    auto iter1 = std::find_if(
                        aVertexNormals.begin(),
                        aVertexNormals.end(),
                        [normal1, kfDifferenceThreshold](float3 const& checkNormal)
                        {
                            return (length(checkNormal - normal1) <= kfDifferenceThreshold);
                        });
                    bool bFound1 = (iter1 != aVertexNormals.end());
                    iNormalIndex1 = (bFound1 == true) ? static_cast<uint32_t>(std::distance(aVertexNormals.begin(), iter1)) : UINT32_MAX;

                    auto iter2 = std::find_if(
                        aVertexNormals.begin(),
                        aVertexNormals.end(),
                        [normal2, kfDifferenceThreshold](float3 const& checkNormal)
                        {
                            return (length(checkNormal - normal2) <= kfDifferenceThreshold);
                        });
                    bool bFound2 = (iter2 != aVertexNormals.end());
                    iNormalIndex2 = (bFound2 == true) ? static_cast<uint32_t>(std::distance(aVertexNormals.begin(), iter2)) : UINT32_MAX;

                    uint32_t iRemap0 = static_cast<uint32_t>(aVertexNormals.size());
                    if(bFound0)
                    {
                        iRemap0 = iNormalIndex0;
                    }
                    else
                    {
                        aVertexNormals.push_back(normal0);
                    }

                    uint32_t iRemap1 = static_cast<uint32_t>(aVertexNormals.size());
                    if(bFound1)
                    {
                        iRemap1 = iNormalIndex1;
                    }
                    else
                    {
                        aVertexNormals.push_back(normal1);
                    }

                    uint32_t iRemap2 = static_cast<uint32_t>(aVertexNormals.size());
                    if(bFound2)
                    {
                        iRemap2 = iNormalIndex2;
                    }
                    else
                    {
                        aVertexNormals.push_back(normal2);
                    }

                    aiRetTriangleNormalIndices.push_back(iRemap0);
                    aiRetTriangleNormalIndices.push_back(iRemap1);
                    aiRetTriangleNormalIndices.push_back(iRemap2);

                }   // remap normals


                // re-map uv indices
                if(bAddedTriangle)
                {
                    uint32_t iUVIndex0 = UINT32_MAX, iUVIndex1 = UINT32_MAX, iUVIndex2 = UINT32_MAX;

                    auto iter0 = std::find_if(
                        aVertexUVs.begin(),
                        aVertexUVs.end(),
                        [uv0, kfDifferenceThreshold](float2 const& checkUV)
                        {
                            return (length(checkUV - uv0) <= kfDifferenceThreshold);
                        });
                    bool bFound0 = (iter0 != aVertexUVs.end());
                    iUVIndex0 = (bFound0 == true) ? static_cast<uint32_t>(std::distance(aVertexUVs.begin(), iter0)) : UINT32_MAX;

                    auto iter1 = std::find_if(
                        aVertexUVs.begin(),
                        aVertexUVs.end(),
                        [uv1, kfDifferenceThreshold](float2 const& checkUV)
                        {
                            return (length(checkUV - uv1) <= kfDifferenceThreshold);
                        });
                    bool bFound1 = (iter1 != aVertexUVs.end());
                    iUVIndex1 = (bFound1 == true) ? static_cast<uint32_t>(std::distance(aVertexUVs.begin(), iter1)) : UINT32_MAX;

                    auto iter2 = std::find_if(
                        aVertexUVs.begin(),
                        aVertexUVs.end(),
                        [uv2, kfDifferenceThreshold](float2 const& checkUV)
                        {
                            return (length(checkUV - uv2) <= kfDifferenceThreshold);
                        });
                    bool bFound2 = (iter2 != aVertexUVs.end());
                    iUVIndex2 = (bFound2 == true) ? static_cast<uint32_t>(std::distance(aVertexUVs.begin(), iter2)) : UINT32_MAX;

                    uint32_t iRemap0 = static_cast<uint32_t>(aVertexUVs.size());
                    if(bFound0)
                    {
                        iRemap0 = iUVIndex0;
                    }
                    else
                    {
                        aVertexUVs.push_back(uv0);
                    }

                    uint32_t iRemap1 = static_cast<uint32_t>(aVertexUVs.size());
                    if(bFound1)
                    {
                        iRemap1 = iUVIndex1;
                    }
                    else
                    {
                        aVertexUVs.push_back(uv1);
                    }

                    uint32_t iRemap2 = static_cast<uint32_t>(aVertexUVs.size());
                    if(bFound2)
                    {
                        iRemap2 = iUVIndex2;
                    }
                    else
                    {
                        aVertexUVs.push_back(uv2);
                    }

                    aiRetTriangleUVIndices.push_back(iRemap0);
                    aiRetTriangleUVIndices.push_back(iRemap1);
                    aiRetTriangleUVIndices.push_back(iRemap2);

                }   // re-map uv indices

                //assert(aVertexPositions.size() == aVertexNormals.size());
                //assert(aVertexPositions.size() == aVertexUVs.size());

            }   // if cluster from file == given cluster index
        }

    }

    assert(aiRetTrianglePositionIndices.size() % 3 == 0);
    assert(aiRetTriangleNormalIndices.size() % 3 == 0);
    assert(aiRetTriangleUVIndices.size() % 3 == 0);
}

/*
**
*/
void buildMETISMeshFile2(
    std::string const& outputFilePath,
    std::vector<uint32_t> const& aiTrianglePositionIndices)
{
    FILE* fp = fopen(outputFilePath.c_str(), "wb");

    fprintf(fp, "%zd 1\n", aiTrianglePositionIndices.size() / 3);
    for(uint32_t i = 0; i < static_cast<uint32_t>(aiTrianglePositionIndices.size()); i += 3)
    {
        fprintf(fp, "%d %d %d\n",
            static_cast<uint32_t>(aiTrianglePositionIndices[i] + 1),
            static_cast<uint32_t>(aiTrianglePositionIndices[i + 1] + 1),
            static_cast<uint32_t>(aiTrianglePositionIndices[i + 2] + 1));
    }

    fclose(fp);
}

/*
**
*/
void buildMETISMeshFile(
    std::string const& outputFilePath,
    std::vector<tinyobj::shape_t> aShapes,
    tinyobj::attrib_t const& attrib)
{
    tinyobj::shape_t const& shape = aShapes[0];
    FILE* fp = fopen(outputFilePath.c_str(), "wb");

    fprintf(fp, "%zd 1\n", shape.mesh.indices.size() / 3);
    for(uint32_t i = 0; i < static_cast<uint32_t>(shape.mesh.indices.size()); i += 3)
    {
        fprintf(fp, "%d %d %d\n",
            static_cast<uint32_t>(shape.mesh.indices[i].vertex_index + 1),
            static_cast<uint32_t>(shape.mesh.indices[i + 1].vertex_index + 1),
            static_cast<uint32_t>(shape.mesh.indices[i + 2].vertex_index + 1));
    }

    fclose(fp);
}

/*
**
*/
void buildMETISGraphFile(
    std::string const& outputFilePath,
    std::vector<std::vector<float3>> const& aaVertexPositions,
    bool bUseCUDA,
    bool bOnlyEdgeAdjacent)
{
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());
    std::vector<std::vector<uint32_t>> aaiNumAdjacentVertices(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aaiNumAdjacentVertices[iCluster].resize(iNumClusters);
    }

    auto start = std::chrono::high_resolution_clock::now();
    if(bUseCUDA)
    {
        auto start0 = std::chrono::high_resolution_clock::now();
        buildClusterAdjacencyCUDA(
            aaiNumAdjacentVertices,
            aaVertexPositions,
            bOnlyEdgeAdjacent);
        auto end0 = std::chrono::high_resolution_clock::now();
        uint64_t iSeconds0 = std::chrono::duration_cast<std::chrono::seconds>(end0 - start0).count();
        DEBUG_PRINTF("took %lld seconds to build all cluster adjacency list\n", iSeconds0);
    }
    else
    {
        uint32_t const kiMaxThreads = 8;
        std::unique_ptr<std::thread> apThreads[kiMaxThreads];
        std::atomic<uint32_t> iCurrCluster{ 0 };
        for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
        {
            apThreads[iThread] = std::make_unique<std::thread>(
                [&iCurrCluster,
                &aaiNumAdjacentVertices,
                aaVertexPositions,
                iNumClusters,
                start,
                bOnlyEdgeAdjacent,
                iThread]()
                {
                    for(;;)
                    {
                        auto start0 = std::chrono::high_resolution_clock::now();

                        uint32_t iThreadCluster = iCurrCluster.fetch_add(1);
                        if(iThreadCluster >= iNumClusters)
                        {
                            break;
                        }

                        std::vector<float3> const& aVertexPositions = aaVertexPositions[iThreadCluster];
                        float3 const* paVertexPositions = aaVertexPositions[iThreadCluster].data();

                        uint32_t iNumVertexPositions = static_cast<uint32_t>(aaVertexPositions[iThreadCluster].size());
                        for(uint32_t iCheckCluster = iThreadCluster + 1; iCheckCluster < iNumClusters; iCheckCluster++)
                        {
                            uint32_t iNumAdjacentVertices = 0;
                            std::vector<float3> const& aCheckVertexPositions = aaVertexPositions[iCheckCluster];
                            float3 const* paCheckVertexPositions = aaVertexPositions[iCheckCluster].data();

                            uint32_t iNumCheckVertexPositions = static_cast<uint32_t>(aCheckVertexPositions.size());

                            for(uint32_t iVert = 0; iVert < iNumVertexPositions; iVert++)
                            {
                                for(uint32_t iCheckVert = 0; iCheckVert < iNumCheckVertexPositions; iCheckVert++)
                                {
                                    //float fLength = lengthSquared(aVertexPositions[iVert] - aCheckVertexPositions[iCheckVert]);
                                    float fLength = lengthSquared(paVertexPositions[iVert] - paCheckVertexPositions[iCheckVert]);
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

                                        //++iNumAdjacentVertices;
                                        //break;
                                    }
                                }
                            }

                            aaiNumAdjacentVertices[iThreadCluster][iCheckCluster] = iNumAdjacentVertices;
                            aaiNumAdjacentVertices[iCheckCluster][iThreadCluster] = iNumAdjacentVertices;

                        }   // for check cluster = cluster + 1 to num clusters

                        auto end = std::chrono::high_resolution_clock::now();
                        uint64_t iMilliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count();
                        uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                        if(iCurrCluster % 100 == 0)
                        {
                            DEBUG_PRINTF("thread %d took %lld milliseconds (total %lld seconds) to build adjacency for cluster %d out of %d\n",
                                iThread,
                                iMilliSeconds,
                                iSeconds,
                                iCurrCluster.load(),
                                iNumClusters);
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

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        DEBUG_PRINTF("took %lld seconds to build all cluster adjacency list\n", iSeconds);

    }   // bUseCUDA

#if 0
    auto start0 = std::chrono::high_resolution_clock::now();
    buildClusterAdjacencyCUDA(
        aaiNumAdjacentVertices,
        aaVertexPositions);
    auto end0 = std::chrono::high_resolution_clock::now();
    uint64_t iSeconds0 = std::chrono::duration_cast<std::chrono::seconds>(end0 - start0).count();
    DEBUG_PRINTF("took %d seconds to build all cluster adjacency list\n", iSeconds0);
#endif // #if 0

    //for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    //{
    //    for(uint32_t iCheckCluster = 0; iCheckCluster < iNumClusters; iCheckCluster++)
    //    {
    //        assert(aaiTestNumAdjacentVertices[iCluster][iCheckCluster] == aaiNumAdjacentVertices[iCluster][iCheckCluster]);
    //    }
    //}


#if 0
    // check the same vertex positions
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<float3> const& aVertexPositions = aaVertexPositions[iCluster];
        float3 const* paVertexPositions = aaVertexPositions[iCluster].data();

        uint32_t iNumVertexPositions = static_cast<uint32_t>(aaVertexPositions[iCluster].size());
        for(uint32_t iCheckCluster = iCluster + 1; iCheckCluster < iNumClusters; iCheckCluster++)
        {
            uint32_t iNumAdjacentVertices = 0;
            std::vector<float3> const& aCheckVertexPositions = aaVertexPositions[iCheckCluster];
            float3 const* paCheckVertexPositions = aaVertexPositions[iCheckCluster].data();

            uint32_t iNumCheckVertexPositions = static_cast<uint32_t>(aCheckVertexPositions.size());

            for(uint32_t iVert = 0; iVert < iNumVertexPositions; iVert++)
            {
                for(uint32_t iCheckVert = 0; iCheckVert < iNumCheckVertexPositions; iCheckVert++)
                {
                    //float fLength = lengthSquared(aVertexPositions[iVert] - aCheckVertexPositions[iCheckVert]);
                    float fLength = lengthSquared(paVertexPositions[iVert] - paCheckVertexPositions[iCheckVert]);
                    if(fLength <= 1.0e-5f)
                    {
                        ++iNumAdjacentVertices;
                        break;
                    }
                }
            }

            aaiNumAdjacentVertices[iCluster][iCheckCluster] = iNumAdjacentVertices;
            aaiNumAdjacentVertices[iCheckCluster][iCluster] = iNumAdjacentVertices;


        }   // for check cluster = cluster + 1 to num clusters

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t iMilliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if(iCluster % 100 == 0)
        {
            DEBUG_PRINTF("took %d milliseconds to build adjacency for cluster %d out of %d\n", iMilliSeconds, iCluster, iNumClusters);
        }

    }   // for cluster = 0 to num clusters
#endif // #if 0

    uint32_t iNumEdges = 0;
    std::vector<std::pair<uint32_t, uint32_t>> aEdges;
    for(uint32_t i = 0; i < iNumClusters; i++)
    {
        for(uint32_t j = i + 1; j < static_cast<uint32_t>(aaiNumAdjacentVertices[i].size()); j++)
        {
            if(aaiNumAdjacentVertices[i][j] > 0)
            {
                aEdges.push_back(std::make_pair(i, j));
                ++iNumEdges;
            }
        }
    }

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fprintf(fp, "%d %d 001\n", iNumClusters, iNumEdges);
    for(uint32_t i = 0; i < iNumClusters; i++)
    {
        for(uint32_t j = 0; j < iNumClusters; j++)
        {
            if(aaiNumAdjacentVertices[i][j] > 0)
            {
                fprintf(fp, "%d %d", j + 1, aaiNumAdjacentVertices[i][j]);
                if(j < iNumClusters - 1)
                {
                    fprintf(fp, " ");
                }
            }
        }

        fprintf(fp, "\n");
    }

    fclose(fp);
}