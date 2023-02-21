#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <sstream>
#include <map>

#include "WaveFrontReader.h"
#include "DirectXMath.h"
#include "DirectXMesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define IDXTYPEWIDTH            32
#define REALTYPEWIDTH           32
#include "metis.h"

#include "tinyexr/tinyexr.h"
#include "stb_image_write.h"

#include "LogPrint.h"

#include "vec.h"
#include "mat4.h"
#include "barycentric.h"

using namespace DirectX;

#include <atomic>
#include <mutex>

std::mutex gMutex;

#include <time.h>

#include "rasterizer.h"

#include "Camera.h"



#include "test.h"
#include "split_operations.h"
#include "join_operations.h"
#include "move_operations.h"

#include "utils.h"

#include "obj_helper.h"

#include "mesh_cluster.h"
#include "test_raster.h"

struct BoundaryEdgeInfo
{
    uint32_t            miClusterGroup;
    uint32_t            miPos0;
    uint32_t            miPos1;
    float3              mPos0;
    float3              mPos1;
};

uint64_t giTotalVertexPositionDataOffset = 0;
uint64_t giTotalVertexNormalDataOffset = 0;
uint64_t giTotalVertexUVDataOffset = 0;
uint64_t giTotalTrianglePositionIndexDataOffset = 0;
uint64_t giTotalTriangleNormalIndexDataOffset = 0;
uint64_t giTotalTriangleUVIndexDataOffset = 0;

std::vector<uint8_t> vertexPositionBuffer(1 << 26);
std::vector<uint8_t> vertexNormalBuffer(1 << 26);
std::vector<uint8_t> vertexUVBuffer(1 << 26);
std::vector<uint8_t> trianglePositionIndexBuffer(1 << 26);
std::vector<uint8_t> triangleNormalIndexBuffer(1 << 26);
std::vector<uint8_t> triangleUVIndexBuffer(1 << 26);

std::vector<uint8_t> gMeshClusterGroupBuffer(1 << 26);
std::vector<uint8_t> gMeshClusterBuffer(1 << 26);

std::string execCommand(std::string const& command, bool bEchoCommand);

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
void buildAdjacencyList(
    tinyobj::attrib_t& attrib,
    std::vector<tinyobj::shape_t>& aShapes,
    std::vector<tinyobj::material_t>& aMaterials,
    std::vector<std::vector<uint32_t>>& aaiAdjacencyList,
    std::string const& outputFilePath,
    std::string const& fullOBJFilePath)
{
    
    std::string warnings;
    std::string errors;

    bool bRet = tinyobj::LoadObj(
        &attrib,
        &aShapes,
        &aMaterials,
        &warnings,
        &errors,
        fullOBJFilePath.c_str());

    // save the faces in map of map to build adjacency
    std::map<uint32_t, std::map<uint32_t, uint32_t>> aaiVertexAdjacencyMap;
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aShapes[0].mesh.indices.size()); iTri += 3)
    {
        uint32_t iPos0 = aShapes[0].mesh.indices[iTri].vertex_index;
        uint32_t iPos1 = aShapes[0].mesh.indices[iTri + 1].vertex_index;
        uint32_t iPos2 = aShapes[0].mesh.indices[iTri + 2].vertex_index;

        aaiVertexAdjacencyMap[iPos0][iPos1] = 1;
        aaiVertexAdjacencyMap[iPos0][iPos2] = 1;

        aaiVertexAdjacencyMap[iPos1][iPos0] = 1;
        aaiVertexAdjacencyMap[iPos1][iPos2] = 1;

        aaiVertexAdjacencyMap[iPos2][iPos0] = 1;
        aaiVertexAdjacencyMap[iPos2][iPos1] = 1;

        DEBUG_PRINTF("%d (%.4f, %.4f, %.4f)\n",
            iPos0,
            attrib.vertices[iPos0 * 3],
            attrib.vertices[iPos0 * 3 + 1],
            attrib.vertices[iPos0 * 3 + 2]);

        DEBUG_PRINTF("%d (%.4f, %.4f, %.4f)\n",
            iPos1,
            attrib.vertices[iPos1 * 3],
            attrib.vertices[iPos1 * 3 + 1],
            attrib.vertices[iPos1 * 3 + 2]);

        DEBUG_PRINTF("%d (%.4f, %.4f, %.4f)\n\n",
            iPos2,
            attrib.vertices[iPos2 * 3],
            attrib.vertices[iPos2 * 3 + 1],
            attrib.vertices[iPos2 * 3 + 2]);
    }

    // build the actual list using keys of maps from above
    for(auto const& keyValue : aaiVertexAdjacencyMap)
    {
        std::vector<uint32_t> aAdjacentVertices;
        auto const& adjacentKeys = keyValue.second;
        for(auto const& key : adjacentKeys)
        {
            aAdjacentVertices.push_back(key.first);
        }

        if(keyValue.first >= aaiAdjacencyList.size())
        {
            aaiAdjacencyList.resize(keyValue.first + 1);
        }

        aaiAdjacencyList[keyValue.first] = aAdjacentVertices;
    }

    {
        FILE* fp = fopen(outputFilePath.c_str(), "wb");

        fprintf(fp, "%d\n", static_cast<uint32_t>(aaiAdjacencyList.size()));
        for(auto const& aiAdjacency : aaiAdjacencyList)
        {
            for(uint32_t i = 0; i < static_cast<uint32_t>(aiAdjacency.size()); i++)
            {
                uint32_t iIndex = aiAdjacency[i];
                fprintf(fp, "%d", iIndex + 1);
                if(i < aiAdjacency.size() - 1)
                {
                    fprintf(fp, " ");
                }
                else
                {
                    fprintf(fp, "\n");
                }
            }
        }

        fclose(fp);
    }
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

                float kfDifferenceThreshold = 1.0e-6f;

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

                    aiRetTrianglePositionIndices.push_back(iRemap0);
                    aiRetTrianglePositionIndices.push_back(iRemap1);
                    aiRetTrianglePositionIndices.push_back(iRemap2);
                }

                // re-map of normal index
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
void buildMETISGraphFile(
    std::string const& outputFilePath,
    std::vector<std::vector<float3>> const& aaVertexPositions)
{
    uint32_t iNumClusters = static_cast<uint32_t>(aaVertexPositions.size());
    std::vector<std::vector<uint32_t>> aaiNumAdjacentVertices(iNumClusters);
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        aaiNumAdjacentVertices[iCluster].resize(iNumClusters);
    }

#if 0
auto start = std::chrono::high_resolution_clock::now();

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
                                if(fLength <= 1.0e-5f)
                                {
                                    ++iNumAdjacentVertices;
                                    break;
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
    DEBUG_PRINTF("thread %d took %d milliseconds (total %d seconds) to build adjacency for cluster %d out of %d\n", 
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
DEBUG_PRINTF("took %d seconds to build all cluster adjacency list\n", iSeconds);
#endif // #if 0


auto start0 = std::chrono::high_resolution_clock::now();
    buildClusterAdjacencyCUDA(
        aaiNumAdjacentVertices,
        aaVertexPositions);
auto end0 = std::chrono::high_resolution_clock::now();
uint64_t iSeconds0 = std::chrono::duration_cast<std::chrono::seconds>(end0 - start0).count();
DEBUG_PRINTF("took %d seconds to build all cluster adjacency list\n", iSeconds0);


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

/*
**
*/
void buildClusterGroups(
    std::vector<std::vector<float3>>& aaClusterGroupVertexPositions,
    std::vector<std::vector<float3>>& aaClusterGroupVertexNormals,
    std::vector<std::vector<float2>>& aaClusterGroupVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupTriangleUVIndices,
    std::vector<uint32_t>& aiClusterGroupMap,
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices,
    std::string const& clusterMeshDirectory,
    uint32_t iNumClusterGroups,
    uint32_t iNumClusters,
    uint32_t iLODLevel,
    std::string const& meshClusterOutputName)
{
    for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
    {
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
    }

    aaClusterGroupVertexPositions.resize(iNumClusterGroups);
    aaiClusterGroupTrianglePositionIndices.resize(iNumClusterGroups);
    
    aaClusterGroupVertexNormals.resize(iNumClusterGroups);
    aaiClusterGroupTriangleNormalIndices.resize(iNumClusterGroups);

    aaClusterGroupVertexUVs.resize(iNumClusterGroups);
    aaiClusterGroupTriangleUVIndices.resize(iNumClusterGroups);
    
    aiClusterGroupMap.resize(iNumClusters);

    // group clusters
    if(iNumClusterGroups > 1)
    {
        // place clusters into the groups specified by metis
        std::ostringstream clusterGroupPartitionFilePath;
        clusterGroupPartitionFilePath << meshClusterOutputName << ".part.";
        clusterGroupPartitionFilePath << iNumClusterGroups;
       
        std::vector<uint32_t> aiClusterGroups;
        readMetisClusterFile(aiClusterGroups, clusterGroupPartitionFilePath.str());

        assert(aiClusterGroups.size() == aaClusterVertexPositions.size());
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
        {
            uint32_t iClusterGroup = aiClusterGroups[iCluster];

            //assert(aaClusterVertexPositions[iCluster].size() == aaClusterVertexNormals[iCluster].size());
            //assert(aaClusterVertexPositions[iCluster].size() == aaClusterVertexUVs[iCluster].size());

            aaClusterGroupVertexPositions[iClusterGroup].insert(
                aaClusterGroupVertexPositions[iClusterGroup].end(),
                aaClusterVertexPositions[iCluster].begin(),
                aaClusterVertexPositions[iCluster].end());

            aaClusterGroupVertexNormals[iClusterGroup].insert(
                aaClusterGroupVertexNormals[iClusterGroup].end(),
                aaClusterVertexNormals[iCluster].begin(),
                aaClusterVertexNormals[iCluster].end());

            aaClusterGroupVertexUVs[iClusterGroup].insert(
                aaClusterGroupVertexUVs[iClusterGroup].end(),
                aaClusterVertexUVs[iCluster].begin(),
                aaClusterVertexUVs[iCluster].end());

            aiClusterGroupMap[iCluster] = iClusterGroup;
        }
    }
    else
    {
        for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
        {
            aaClusterGroupVertexPositions[0].insert(
                aaClusterGroupVertexPositions[0].end(),
                aaClusterVertexPositions[iCluster].begin(),
                aaClusterVertexPositions[iCluster].end());

            aaClusterGroupVertexNormals[0].insert(
                aaClusterGroupVertexNormals[0].end(),
                aaClusterVertexNormals[iCluster].begin(),
                aaClusterVertexNormals[iCluster].end());

            aaClusterGroupVertexUVs[0].insert(
                aaClusterGroupVertexUVs[0].end(),
                aaClusterVertexUVs[iCluster].begin(),
                aaClusterVertexUVs[iCluster].end());

            aiClusterGroupMap[iCluster] = 0;
        }
    }

    for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
    {
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
    }

    static float const kfEqualityThreshold = 1.0e-8f;

    // POSITIONS: remap cluster triangle indices into cluster group's own indices
    std::vector<uint32_t> aiDiscardTris;
    for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
    {
        uint32_t iClusterGroup = aiClusterGroupMap[iCluster];
        for(int32_t iTri = 0; iTri < static_cast<int32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); iTri += 3)
        {
            // position
            uint32_t iOrigPos0 = aaiClusterTrianglePositionIndices[iCluster][iTri];
            uint32_t iOrigPos1 = aaiClusterTrianglePositionIndices[iCluster][iTri + 1];
            uint32_t iOrigPos2 = aaiClusterTrianglePositionIndices[iCluster][iTri + 2];

            float3 const& position0 = aaClusterVertexPositions[iCluster][iOrigPos0];
            float3 const& position1 = aaClusterVertexPositions[iCluster][iOrigPos1];
            float3 const& position2 = aaClusterVertexPositions[iCluster][iOrigPos2];

            auto posIter0 = std::find_if(
                aaClusterGroupVertexPositions[iClusterGroup].begin(),
                aaClusterGroupVertexPositions[iClusterGroup].end(),
                [position0](float3 const& checkPosition)
                {
                    return (length(checkPosition - position0) <= kfEqualityThreshold);
                });
            assert(posIter0 != aaClusterGroupVertexPositions[iClusterGroup].end());
            uint32_t iRemapPos0 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexPositions[iClusterGroup].begin(), posIter0));

            auto posIter1 = std::find_if(
                aaClusterGroupVertexPositions[iClusterGroup].begin(),
                aaClusterGroupVertexPositions[iClusterGroup].end(),
                [position1](float3 const& checkPosition)
                {
                    return (length(checkPosition - position1) <= kfEqualityThreshold);
                });
            assert(posIter1 != aaClusterGroupVertexPositions[iClusterGroup].end());
            uint32_t iRemapPos1 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexPositions[iClusterGroup].begin(), posIter1));

            auto posIter2 = std::find_if(
                aaClusterGroupVertexPositions[iClusterGroup].begin(),
                aaClusterGroupVertexPositions[iClusterGroup].end(),
                [position2](float3 const& checkPosition)
                {
                    return (length(checkPosition - position2) <= kfEqualityThreshold);
                });
            assert(posIter2 != aaClusterGroupVertexPositions[iClusterGroup].end());
            uint32_t iRemapPos2 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexPositions[iClusterGroup].begin(), posIter2));

            // normal
            uint32_t iOrigNormal0 = aaiClusterTriangleNormalIndices[iCluster][iTri];
            uint32_t iOrigNormal1 = aaiClusterTriangleNormalIndices[iCluster][iTri + 1];
            uint32_t iOrigNormal2 = aaiClusterTriangleNormalIndices[iCluster][iTri + 2];

            float3 const& normal0 = aaClusterVertexNormals[iCluster][iOrigNormal0];
            float3 const& normal1 = aaClusterVertexNormals[iCluster][iOrigNormal1];
            float3 const& normal2 = aaClusterVertexNormals[iCluster][iOrigNormal2];

            auto normalIter0 = std::find_if(
                aaClusterGroupVertexNormals[iClusterGroup].begin(),
                aaClusterGroupVertexNormals[iClusterGroup].end(),
                [normal0](float3 const& checkNormal)
                {
                    return (length(checkNormal - normal0) <= kfEqualityThreshold);
                });
            assert(normalIter0 != aaClusterGroupVertexNormals[iClusterGroup].end());
            uint32_t iRemapNormal0 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexNormals[iClusterGroup].begin(), normalIter0));

            auto normalIter1 = std::find_if(
                aaClusterGroupVertexNormals[iClusterGroup].begin(),
                aaClusterGroupVertexNormals[iClusterGroup].end(),
                [normal1](float3 const& checkNormal)
                {
                    return (length(checkNormal - normal1) <= kfEqualityThreshold);
                });
            assert(normalIter1 != aaClusterGroupVertexNormals[iClusterGroup].end());
            uint32_t iRemapNormal1 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexNormals[iClusterGroup].begin(), normalIter1));

            auto normalIter2 = std::find_if(
                aaClusterGroupVertexNormals[iClusterGroup].begin(),
                aaClusterGroupVertexNormals[iClusterGroup].end(),
                [normal2](float3 const& checkNormal)
                {
                    return (length(checkNormal - normal2) <= kfEqualityThreshold);
                });
            assert(normalIter2 != aaClusterGroupVertexNormals[iClusterGroup].end());
            uint32_t iRemapNormal2 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexNormals[iClusterGroup].begin(), normalIter2));
            
            // uv
            uint32_t iOrigUV0 = aaiClusterTriangleUVIndices[iCluster][iTri];
            uint32_t iOrigUV1 = aaiClusterTriangleUVIndices[iCluster][iTri + 1];
            uint32_t iOrigUV2 = aaiClusterTriangleUVIndices[iCluster][iTri + 2];

            float2 const& uv0 = aaClusterVertexUVs[iCluster][iOrigUV0];
            float2 const& uv1 = aaClusterVertexUVs[iCluster][iOrigUV1];
            float2 const& uv2 = aaClusterVertexUVs[iCluster][iOrigUV2];

            auto uvIter0 = std::find_if(
                aaClusterGroupVertexUVs[iClusterGroup].begin(),
                aaClusterGroupVertexUVs[iClusterGroup].end(),
                [uv0](float2 const& checkUV)
                {
                    return (length(checkUV - uv0) <= kfEqualityThreshold);
                });
            assert(uvIter0 != aaClusterGroupVertexUVs[iClusterGroup].end());
            uint32_t iRemapUV0 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexUVs[iClusterGroup].begin(), uvIter0));

            // uv
            auto uvIter1 = std::find_if(
                aaClusterGroupVertexUVs[iClusterGroup].begin(),
                aaClusterGroupVertexUVs[iClusterGroup].end(),
                [uv1](float2 const& checkUV)
                {
                    return (length(checkUV - uv1) <= kfEqualityThreshold);
                });
            assert(uvIter1 != aaClusterGroupVertexUVs[iClusterGroup].end());
            uint32_t iRemapUV1 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexUVs[iClusterGroup].begin(), uvIter1));

            auto uvIter2 = std::find_if(
                aaClusterGroupVertexUVs[iClusterGroup].begin(),
                aaClusterGroupVertexUVs[iClusterGroup].end(),
                [uv2](float2 const& checkUV)
                {
                    return (length(checkUV - uv2) <= kfEqualityThreshold);
                });
            assert(uvIter2 != aaClusterGroupVertexUVs[iClusterGroup].end());
            uint32_t iRemapUV2 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexUVs[iClusterGroup].begin(), uvIter2));

            if(iRemapPos0 != iRemapPos1 && iRemapPos0 != iRemapPos2 && iRemapPos1 != iRemapPos2)
            {
                aaiClusterGroupTrianglePositionIndices[iClusterGroup].push_back(iRemapPos0);
                aaiClusterGroupTrianglePositionIndices[iClusterGroup].push_back(iRemapPos1);
                aaiClusterGroupTrianglePositionIndices[iClusterGroup].push_back(iRemapPos2);

                aaiClusterGroupTriangleNormalIndices[iClusterGroup].push_back(iRemapNormal0);
                aaiClusterGroupTriangleNormalIndices[iClusterGroup].push_back(iRemapNormal1);
                aaiClusterGroupTriangleNormalIndices[iClusterGroup].push_back(iRemapNormal2);

                aaiClusterGroupTriangleUVIndices[iClusterGroup].push_back(iRemapUV0);
                aaiClusterGroupTriangleUVIndices[iClusterGroup].push_back(iRemapUV1);
                aaiClusterGroupTriangleUVIndices[iClusterGroup].push_back(iRemapUV2);

                //if(iClusterGroup == 1)
                //{
                //    DEBUG_PRINTF("cluster group %d add tri %d position remap (%d, %d, %d)\n", iClusterGroup, iTri, iRemap0, iRemap1, iRemap2);
                //}
            }
            else
            {
                aiDiscardTris.push_back(iTri);
            }
            
        }   // for tri in cluster

    }   // for cluster = 0 to num clusters

    //DEBUG_PRINTF("\n****\n");

#if 0
    // NORMALS: remap cluster triangle indices into cluster group's own indices
    {
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexNormals.size()); iCluster++)
        {
            uint32_t iClusterGroup = aiClusterGroupMap[iCluster];
            for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTriangleNormalIndices[iCluster].size()); iTri += 3)
            {
                uint32_t iOrigNormal0 = aaiClusterTriangleNormalIndices[iCluster][iTri];
                uint32_t iOrigNormal1 = aaiClusterTriangleNormalIndices[iCluster][iTri + 1];
                uint32_t iOrigNormal2 = aaiClusterTriangleNormalIndices[iCluster][iTri + 2];

                float3 const& normal0 = aaClusterVertexNormals[iCluster][iOrigNormal0];
                float3 const& normal1 = aaClusterVertexNormals[iCluster][iOrigNormal1];
                float3 const& normal2 = aaClusterVertexNormals[iCluster][iOrigNormal2];

                auto normalIter0 = std::find_if(
                    aaClusterGroupVertexNormals[iClusterGroup].begin(),
                    aaClusterGroupVertexNormals[iClusterGroup].end(),
                    [normal0](float3 const& checkNormal)
                    {
                        return (length(checkNormal - normal0) <= kfEqualityThreshold);
                    });
                assert(normalIter0 != aaClusterGroupVertexNormals[iClusterGroup].end());
                uint32_t iRemapNormal0 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexNormals[iClusterGroup].begin(), normalIter0));

                auto normalIter1 = std::find_if(
                    aaClusterGroupVertexNormals[iClusterGroup].begin(),
                    aaClusterGroupVertexNormals[iClusterGroup].end(),
                    [normal1](float3 const& checkNormal)
                    {
                        return (length(checkNormal - normal1) <= kfEqualityThreshold);
                    });
                assert(normalIter1 != aaClusterGroupVertexNormals[iClusterGroup].end());
                uint32_t iRemapNormal1 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexNormals[iClusterGroup].begin(), normalIter1));

                auto normalIter2 = std::find_if(
                    aaClusterGroupVertexNormals[iClusterGroup].begin(),
                    aaClusterGroupVertexNormals[iClusterGroup].end(),
                    [normal2](float3 const& checkNormal)
                    {
                        return (length(checkNormal - normal2) <= kfEqualityThreshold);
                    });
                assert(normalIter2 != aaClusterGroupVertexNormals[iClusterGroup].end());
                uint32_t iRemapNormal2 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexNormals[iClusterGroup].begin(), normalIter2));

                auto discardIter = std::find(aiDiscardTris.begin(), aiDiscardTris.end(), iTri);
                if(discardIter == aiDiscardTris.end())
                {
                    aaiClusterGroupTriangleNormalIndices[iClusterGroup].push_back(iRemap0);
                    aaiClusterGroupTriangleNormalIndices[iClusterGroup].push_back(iRemap1);
                    aaiClusterGroupTriangleNormalIndices[iClusterGroup].push_back(iRemap2);

                    //if(iClusterGroup == 1)
                    //{
                    //    DEBUG_PRINTF("cluster group %d add tri %d normal remap (%d, %d, %d)\n", iClusterGroup, iTri, iRemap0, iRemap1, iRemap2);
                    //}
                }

                assert(aaiClusterGroupTriangleNormalIndices[iClusterGroup].size() <= aaiClusterGroupTrianglePositionIndices[iClusterGroup].size());

            }   // for tri in cluster

        }   // for cluster = 0 to num clusters

    }   // build new normal list and normal indices

    //DEBUG_PRINTF("\n****\n");

    // UVs: remap cluster triangle indices into cluster group's own indices
    {
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexUVs.size()); iCluster++)
        {
            uint32_t iClusterGroup = aiClusterGroupMap[iCluster];
            for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTriangleUVIndices[iCluster].size()); iTri += 3)
            {
                uint32_t iOrigUV0 = aaiClusterTriangleUVIndices[iCluster][iTri];
                uint32_t iOrigUV1 = aaiClusterTriangleUVIndices[iCluster][iTri + 1];
                uint32_t iOrigUV2 = aaiClusterTriangleUVIndices[iCluster][iTri + 2];

                float2 const& uv0 = aaClusterVertexUVs[iCluster][iOrig0];
                float2 const& uv1 = aaClusterVertexUVs[iCluster][iOrig1];
                float2 const& uv2 = aaClusterVertexUVs[iCluster][iOrig2];

                auto uvIter0 = std::find_if(
                    aaClusterGroupVertexUVs[iClusterGroup].begin(),
                    aaClusterGroupVertexUVs[iClusterGroup].end(),
                    [uv0](float2 const& checkUV)
                    {
                        return (length(checkUV - uv0) <= kfEqualityThreshold);
                    });
                assert(uvIter0 != aaClusterGroupVertexUVs[iClusterGroup].end());
                uint32_t iRemapUV0 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexUVs[iClusterGroup].begin(), uvIter0));

                auto uvIter1 = std::find_if(
                    aaClusterGroupVertexUVs[iClusterGroup].begin(),
                    aaClusterGroupVertexUVs[iClusterGroup].end(),
                    [uv1](float2 const& checkUV)
                    {
                        return (length(checkUV - uv1) <= kfEqualityThreshold);
                    });
                assert(uvIter1 != aaClusterGroupVertexUVs[iClusterGroup].end());
                uint32_t iRemapUV1 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexUVs[iClusterGroup].begin(), uvIter1));

                auto uvIter2 = std::find_if(
                    aaClusterGroupVertexUVs[iClusterGroup].begin(),
                    aaClusterGroupVertexUVs[iClusterGroup].end(),
                    [uv2](float2 const& checkUV)
                    {
                        return (length(checkUV - uv2) <= kfEqualityThreshold);
                    });
                assert(uvIter2 != aaClusterGroupVertexUVs[iClusterGroup].end());
                uint32_t iRemapUV2 = static_cast<uint32_t>(std::distance(aaClusterGroupVertexUVs[iClusterGroup].begin(), uvIter2));

                auto discardIter = std::find(aiDiscardTris.begin(), aiDiscardTris.end(), iTri);
                if(discardIter == aiDiscardTris.end())
                {
                    aaiClusterGroupTriangleUVIndices[iClusterGroup].push_back(iRemap0);
                    aaiClusterGroupTriangleUVIndices[iClusterGroup].push_back(iRemap1);
                    aaiClusterGroupTriangleUVIndices[iClusterGroup].push_back(iRemap2);
                }

            }   // for tri in cluster

        }   // for cluster = 0 to num clusters

    }   // build new normal list and normal indices
#endif // #if 0


    // output cluster group obj
    {
        for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
        {
            assert(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size() == aaiClusterGroupTriangleNormalIndices[iClusterGroup].size());
            assert(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size() == aaiClusterGroupTriangleUVIndices[iClusterGroup].size());

            std::ostringstream clusterGroupName;
            clusterGroupName << "cluster-group-lod" << iLODLevel << "-group" << iClusterGroup;

            std::ostringstream clusterGroupFilePath;
            clusterGroupFilePath << "c:\\Users\\Dingwings\\demo-models\\cluster-groups\\";
            clusterGroupFilePath << clusterGroupName.str() << ".obj";
            FILE* fp = fopen(clusterGroupFilePath.str().c_str(), "wb");
            fprintf(fp, "o %s\n", clusterGroupName.str().c_str());
            fprintf(fp, "usemtl %s\n", clusterGroupName.str().c_str());
            for(uint32_t iV = 0; iV < static_cast<uint32_t>(aaClusterGroupVertexPositions[iClusterGroup].size()); iV++)
            {
                fprintf(fp, "v %.4f %.4f %.4f\n",
                    aaClusterGroupVertexPositions[iClusterGroup][iV].x,
                    aaClusterGroupVertexPositions[iClusterGroup][iV].y,
                    aaClusterGroupVertexPositions[iClusterGroup][iV].z);
            }

            for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size()); iTri += 3)
            {
                fprintf(fp, "f %d// %d// %d//\n",
                    aaiClusterGroupTrianglePositionIndices[iClusterGroup][iTri] + 1,
                    aaiClusterGroupTrianglePositionIndices[iClusterGroup][iTri + 1] + 1,
                    aaiClusterGroupTrianglePositionIndices[iClusterGroup][iTri + 2] + 1);
            }
            fclose(fp);

            float fRand0 = float(rand() % 255) / 255.0f;
            float fRand1 = float(rand() % 255) / 255.0f;
            float fRand2 = float(rand() % 255) / 255.0f;
            std::ostringstream clusterGroupMaterialFilePath;
            clusterGroupMaterialFilePath << "c:\\Users\\Dingwings\\demo-models\\cluster-groups\\";
            clusterGroupMaterialFilePath << clusterGroupName.str() << ".mtl";
            fp = fopen(clusterGroupMaterialFilePath.str().c_str(), "wb");
            fprintf(fp, "newmtl %s\n", clusterGroupName.str().c_str());
            fprintf(fp, "Kd %.4f %.4f %.4f\n",
                fRand0,
                fRand1,
                fRand2);
            fclose(fp);

        }   // for cluster group = 0 to num cluster groups

    }   // output obj files for cluster groups
}

/*
**
*/
void getClusterGroupBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupNonBoundaryVertices,
    std::vector<std::vector<float3>> const& aaClusterGroupVertexPositions,
    uint32_t const& iNumClusterGroups)
{
    aaiClusterGroupBoundaryVertices.resize(iNumClusterGroups);

#if 0
    std::mutex threadMutex;

    uint32_t const kiMaxThreads = 8;
    std::unique_ptr<std::thread> apThreads[kiMaxThreads];
    std::atomic<uint32_t> iCurrClusterGroup{ 0 };
    for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
    {
        apThreads[iThread] = std::make_unique<std::thread>(
            [&iCurrClusterGroup,
            &aaiClusterGroupBoundaryVertices,
            aaClusterGroupVertexPositions,
            iNumClusterGroups,
            iThread,
            &threadMutex]()
            {
                for(;;)
                {
                    uint32_t iThreadClusterGroup = iCurrClusterGroup.fetch_add(1);
                    if(iThreadClusterGroup >= iNumClusterGroups)
                    {
                        break;
                    }

auto start = std::chrono::high_resolution_clock::now();

                    auto const& aClusterGroupVertexPositions = aaClusterGroupVertexPositions[iThreadClusterGroup];
                    auto const* paClusterGroupVertexPositions = aaClusterGroupVertexPositions[iThreadClusterGroup].data();
                    uint32_t iNumClusterGroupVertexPositions = static_cast<uint32_t>(aClusterGroupVertexPositions.size());

                    for(uint32_t iCheckClusterGroup = iThreadClusterGroup + 1; iCheckClusterGroup < iNumClusterGroups; iCheckClusterGroup++)
                    {
                        auto const& aCheckClusterGroupVertexPositions = aaClusterGroupVertexPositions[iCheckClusterGroup];
                        auto const* paCheckClusterGroupVertexPositions = aaClusterGroupVertexPositions[iCheckClusterGroup].data();
                        uint32_t iNumCheckClusterGroupVertexPositions = static_cast<uint32_t>(aCheckClusterGroupVertexPositions.size());

                        for(uint32_t iVert = 0; iVert < iNumClusterGroupVertexPositions; iVert++)
                        {
                            float3 const& vertPosition = paClusterGroupVertexPositions[iVert];
                            for(uint32_t iCheckVert = 0; iCheckVert < iNumCheckClusterGroupVertexPositions; iCheckVert++)
                            {
                                float3 diff = vertPosition - paCheckClusterGroupVertexPositions[iCheckVert];
                                float fLength = lengthSquared(diff);
                                if(fLength <= 1.0e-8f)
                                {
                                    {
                                        std::lock_guard<std::mutex> lock(threadMutex);
                                        aaiClusterGroupBoundaryVertices[iThreadClusterGroup].push_back(iVert);
                                        aaiClusterGroupBoundaryVertices[iCheckClusterGroup].push_back(iCheckVert);
                                    }
                                }

                            }   // for check vert = 0 to num vertices

                        }   // for vert = 0 to num vertices

                    }   // for cluster group = 0 to num cluster groups

auto end = std::chrono::high_resolution_clock::now();
uint64_t iMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
if(iThreadClusterGroup % 100 == 0)
{
    DEBUG_PRINTF("thread %d took %d milliseconds to complete getting boundary vertices for cluster %d (%d)\n", iThread, iMilliseconds, iThreadClusterGroup, iNumClusterGroups);
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
#endif // #if 0

    {
        //std::vector<std::vector<uint32_t>> aaiTestClusterGroupBoundaryVertices;
        if(aaClusterGroupVertexPositions.size() > 1)
        {
            getClusterGroupBoundaryVerticesCUDA(
                aaiClusterGroupBoundaryVertices, //aaiTestClusterGroupBoundaryVertices,
                aaClusterGroupVertexPositions);

            //for(uint32_t i = 0; i < aaiTestClusterGroupBoundaryVertices[0].size(); i++)
            //{
            //    auto iter = std::find(
            //        aaiClusterGroupBoundaryVertices[0].begin(),
            //        aaiClusterGroupBoundaryVertices[0].end(),
            //        aaiTestClusterGroupBoundaryVertices[0][i]);
            //
            //    if(iter == aaiClusterGroupBoundaryVertices[0].end())
            //    {
            //        int iDebug = 1;
            //    }
            //}
        }
    }

    // get non-boundary vertices by checking non-adjacent vertices from cluster groups
    aaiClusterGroupNonBoundaryVertices.resize(iNumClusterGroups);
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        auto const& aClusterGroupVertexPositions = aaClusterGroupVertexPositions[iClusterGroup];
        for(uint32_t iVert = 0; iVert < static_cast<uint32_t>(aClusterGroupVertexPositions.size()); iVert++)
        {
            auto iter = std::find(aaiClusterGroupBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupBoundaryVertices[iClusterGroup].end(), iVert);
            if(iter == aaiClusterGroupBoundaryVertices[iClusterGroup].end())
            {
                aaiClusterGroupNonBoundaryVertices[iClusterGroup].push_back(iVert);
            }
        }
    }
}

/*
**
*/
void getInnerEdgesAndVertices(
    std::vector<std::vector<uint32_t>>& aaiValidClusterGroupEdges,
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaValidClusterGroupEdgePairs,
    std::vector<std::map<uint32_t, uint32_t>>& aaValidVertices,
    std::vector<std::vector<uint32_t>>& aaiClusterGroupTriWithEdges,
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& aaClusterGroupEdges,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTriangles,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupNonBoundaryVertices,
    uint32_t const& iNumClusterGroups)
{
    for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
    {
        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterGroupTriangles[iClusterGroup].size()); iTri += 3)
        {
            uint32_t iV0 = aaiClusterGroupTriangles[iClusterGroup][iTri];
            uint32_t iV1 = aaiClusterGroupTriangles[iClusterGroup][iTri + 1];
            uint32_t iV2 = aaiClusterGroupTriangles[iClusterGroup][iTri + 2];

            //if(iV0 == iV1 || iV0 == iV2 || iV1 == iV2)
            //{
            //    continue;
            //}

            assert(iV0 != iV1);
            assert(iV0 != iV2);
            assert(iV1 != iV2);

            bool bValid0 = (std::find(aaiClusterGroupNonBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupNonBoundaryVertices[iClusterGroup].end(), iV0) != aaiClusterGroupNonBoundaryVertices[iClusterGroup].end());
            bool bValid1 = (std::find(aaiClusterGroupNonBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupNonBoundaryVertices[iClusterGroup].end(), iV1) != aaiClusterGroupNonBoundaryVertices[iClusterGroup].end());
            bool bValid2 = (std::find(aaiClusterGroupNonBoundaryVertices[iClusterGroup].begin(), aaiClusterGroupNonBoundaryVertices[iClusterGroup].end(), iV2) != aaiClusterGroupNonBoundaryVertices[iClusterGroup].end());

            aaClusterGroupEdges[iClusterGroup].push_back(std::make_pair(iV0, iV1));
            aaClusterGroupEdges[iClusterGroup].push_back(std::make_pair(iV0, iV2));
            aaClusterGroupEdges[iClusterGroup].push_back(std::make_pair(iV1, iV2));

            if(bValid0)
            {
                auto pair0 = std::make_pair(iV0, iV1);
                auto iter0 = std::find_if(
                    aaValidClusterGroupEdgePairs[iClusterGroup].begin(),
                    aaValidClusterGroupEdgePairs[iClusterGroup].end(),
                    [pair0](std::pair<uint32_t, uint32_t> const& checkPair)
                    {
                        return (checkPair.first == pair0.first && checkPair.second == pair0.second) || (checkPair.first == pair0.second && checkPair.second == pair0.first);
                    });
                if(iter0 == aaValidClusterGroupEdgePairs[iClusterGroup].end())
                {
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV0);
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV1);

                    aaValidClusterGroupEdgePairs[iClusterGroup].push_back(pair0);

                    assert(pair0.first != pair0.second);

                    // save the triangle index containing this edge
                    aaiClusterGroupTriWithEdges[iClusterGroup].push_back(iTri);
                }

                auto pair1 = std::make_pair(iV0, iV2);
                auto iter1 = std::find_if(
                    aaValidClusterGroupEdgePairs[iClusterGroup].begin(),
                    aaValidClusterGroupEdgePairs[iClusterGroup].end(),
                    [pair1](std::pair<uint32_t, uint32_t> const& checkPair)
                    {
                        return (checkPair.first == pair1.first && checkPair.second == pair1.second) || (checkPair.first == pair1.second && checkPair.second == pair1.first);
                    });
                if(iter1 == aaValidClusterGroupEdgePairs[iClusterGroup].end())
                {
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV0);
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV2);

                    aaValidClusterGroupEdgePairs[iClusterGroup].push_back(pair1);
                
                    assert(pair1.first != pair1.second);

                    // save the triangle index containing this edge
                    aaiClusterGroupTriWithEdges[iClusterGroup].push_back(iTri);
                }
            }

            if(bValid1 || bValid2)
            {
                auto pair2 = std::make_pair(iV1, iV2);
                auto iter = std::find_if(
                    aaValidClusterGroupEdgePairs[iClusterGroup].begin(),
                    aaValidClusterGroupEdgePairs[iClusterGroup].end(),
                    [pair2](std::pair<uint32_t, uint32_t> const& checkPair)
                    {
                        return (checkPair.first == pair2.first && checkPair.second == pair2.second) || (checkPair.first == pair2.second && checkPair.second == pair2.first);
                    });
                if(iter == aaValidClusterGroupEdgePairs[iClusterGroup].end())
                {
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV1);
                    aaiValidClusterGroupEdges[iClusterGroup].push_back(iV2);

                    aaValidClusterGroupEdgePairs[iClusterGroup].push_back(pair2);

                    assert(pair2.first != pair2.second);
                }

                // save the triangle index containing this edge
                aaiClusterGroupTriWithEdges[iClusterGroup].push_back(iTri);
            }

            aaValidVertices[iClusterGroup][iV0] = (bValid0 == true) ? 1 : 0;
            aaValidVertices[iClusterGroup][iV1] = (bValid1 == true) ? 1 : 0;
            aaValidVertices[iClusterGroup][iV2] = (bValid2 == true) ? 1 : 0;
            
        }
    }
}

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

            aTriPositions[iTri] =       aClusterGroupVertexPositions[iPos0];
            aTriPositions[iTri + 1] =   aClusterGroupVertexPositions[iPos1];
            aTriPositions[iTri + 2] =   aClusterGroupVertexPositions[iPos2];
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
struct EdgeCollapseInfo
{
    float3      mOptimalVertexPosition;
    float3      mOptimalNormal;
    float2      mOptimalUV;
    float       mfCost;
};

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
    std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices,
    uint32_t iClusterGroup,
    std::vector<std::pair<uint32_t, uint32_t>> const& aClusterGroupEdgePositions,
    std::vector<std::pair<uint32_t, uint32_t>> const& aClusterGroupEdgeNormals,
    std::vector<std::pair<uint32_t, uint32_t>> const& aClusterGroupEdgeUVs)
{
    std::vector<EdgeCollapseInfo> aEdgeCollapseCosts;
    std::vector<std::pair<uint32_t, uint32_t>> aEdges;

#if 0
    // test test test
    {
        std::lock_guard<std::mutex> lock(gMutex);
        
        std::vector<float> afCollapseCosts;
        std::vector<float3> aOptimalVertexPositions;
        std::vector<float3> aOptimalVertexNormals;
        std::vector<float2> aOptimalUVs;
        std::vector<std::pair<uint32_t, uint32_t>> aRetEdges;

        computeEdgeCollapseInfoCUDA(
            afCollapseCosts,
            aOptimalVertexPositions,
            aOptimalVertexNormals,
            aOptimalUVs,
            aRetEdges,
            aClusterGroupVertexPositions,
            aClusterGroupVertexNormals,
            aClusterGroupVertexUVs,
            aValidClusterGroupEdgePairs,
            aiClusterGroupNonBoundaryVertices,
            aiClusterGroupTrianglePositionIndices,
            aiClusterGroupTriangleNormalIndices,
            aiClusterGroupTriangleUVIndices,
            aBoundaryVertices);

        for(uint32_t i = 0; i < static_cast<uint32_t>(aRetEdges.size()); i++)
        {
            EdgeCollapseInfo edgeCollapseInfo;
            edgeCollapseInfo.mfCost = afCollapseCosts[i];
            edgeCollapseInfo.mOptimalVertexPosition = aOptimalVertexPositions[i];
            edgeCollapseInfo.mOptimalNormal = aOptimalVertexNormals[i];
            edgeCollapseInfo.mOptimalUV = aOptimalUVs[i];
            
            aEdgeCollapseCosts.push_back(edgeCollapseInfo);
            aEdges.push_back(aRetEdges[i]);
        }
    }
#endif // #if 0

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

#if 0
        // look for the triangle containing the given edge and get the normal and uv indices for the edge
        auto clusterGroupEdgePositionIter = std::find(aClusterGroupEdgePositions.begin(), aClusterGroupEdgePositions.end(), edge);
        if(clusterGroupEdgePositionIter == aClusterGroupEdgePositions.end())
        {
            // didn't find a matching triangle for this edge, continue to the next edge
            continue;
        }

        uint32_t iTriPositionIndex = static_cast<uint32_t>(std::distance(aClusterGroupEdgePositions.begin(), clusterGroupEdgePositionIter));
        uint32_t iNorm0 = aClusterGroupEdgeNormals[iTriPositionIndex].first;
        uint32_t iNorm1 = aClusterGroupEdgeNormals[iTriPositionIndex].second;
        uint32_t iUV0 = aClusterGroupEdgeUVs[iTriPositionIndex].first;
        uint32_t iUV1 = aClusterGroupEdgeUVs[iTriPositionIndex].second;

        assert(iNorm0 < aClusterGroupVertexNormals.size());
        assert(iNorm1 < aClusterGroupVertexNormals.size());
        assert(iUV0 < aClusterGroupVertexUVs.size());
        assert(iUV1 < aClusterGroupVertexUVs.size());
#endif // #if 0

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

        // see which vertex is on the boundary, need this to set contraction position to the boundary vertex
        //auto iter0 = std::find(
        //    aiClusterGroupNonBoundaryVertices.begin(),
        //    aiClusterGroupNonBoundaryVertices.end(),
        //    edge.first);
        //auto iter1 = std::find(
        //    aiClusterGroupNonBoundaryVertices.begin(),
        //    aiClusterGroupNonBoundaryVertices.end(),
        //    edge.second);
        //bool bValid0 = (iter0 != aiClusterGroupNonBoundaryVertices.end());
        //bool bValid1 = (iter1 != aiClusterGroupNonBoundaryVertices.end());
        
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
        //else
        //{
        //    quadric0 = aQuadrics[edge.first];
        //}

        //mat4 quadric1;
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
        //else
        //{
        //    quadric1 = aQuadrics[edge.second];
        //}

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

                //float fLength = length(paClusterGroupVertexPositions[edge.first] - paClusterGroupVertexPositions[edge.second]);
                //fLength = (fLength <= 0.0f) ? 1.0e-3f : fLength;
                //assert(length(edgeCollapseInfo.mOptimalVertexPosition - aClusterGroupVertexPositions[edge.first]) <= fLength ||
                //    length(edgeCollapseInfo.mOptimalVertexPosition - aClusterGroupVertexPositions[edge.second]) <= fLength);
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
            edgeCollapseInfo.mfCost *= 1.0e+10f;
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
    std::vector<std::pair<uint32_t, uint32_t>> const& aBoundaryVertices,
    uint32_t iMaxTriangles,
    uint32_t iClusterGroup,
    uint32_t iLODLevel)
{
    uint32_t const kiNumEdgesToCollapse = 10;

    fTotalError = 0.0f;
    while(aiClusterGroupTrianglePositions.size() >= iMaxTriangles)
    {
auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<uint32_t, uint32_t>> aClusterGroupTriEdgePositions;
        std::vector<std::pair<uint32_t, uint32_t>> aClusterGroupTriEdgeNormals;
        std::vector<std::pair<uint32_t, uint32_t>> aClusterGroupTriEdgeUVs;
#if 0
        {
            for(uint32_t iTri = 0; iTri < aiClusterGroupTrianglePositions.size(); iTri += 3)
            {
                uint32_t const& iPos0 = aiClusterGroupTrianglePositions[iTri];
                uint32_t const& iPos1 = aiClusterGroupTrianglePositions[iTri + 1];
                uint32_t const& iPos2 = aiClusterGroupTrianglePositions[iTri + 2];

                uint32_t const& iNorm0 = aiClusterGroupTriangleNormals[iTri];
                uint32_t const& iNorm1 = aiClusterGroupTriangleNormals[iTri + 1];
                uint32_t const& iNorm2 = aiClusterGroupTriangleNormals[iTri + 2];

                uint32_t const& iUV0 = aiClusterGroupTriangleUVs[iTri];
                uint32_t const& iUV1 = aiClusterGroupTriangleUVs[iTri + 1];
                uint32_t const& iUV2 = aiClusterGroupTriangleUVs[iTri + 2];

                aClusterGroupTriEdgePositions.push_back(std::make_pair(iPos0, iPos1));
                aClusterGroupTriEdgePositions.push_back(std::make_pair(iPos1, iPos0));
                aClusterGroupTriEdgePositions.push_back(std::make_pair(iPos0, iPos2));
                aClusterGroupTriEdgePositions.push_back(std::make_pair(iPos2, iPos0));
                aClusterGroupTriEdgePositions.push_back(std::make_pair(iPos1, iPos2));
                aClusterGroupTriEdgePositions.push_back(std::make_pair(iPos2, iPos1));

                aClusterGroupTriEdgeNormals.push_back(std::make_pair(iNorm0, iNorm1));
                aClusterGroupTriEdgeNormals.push_back(std::make_pair(iNorm1, iNorm0));
                aClusterGroupTriEdgeNormals.push_back(std::make_pair(iNorm0, iNorm2));
                aClusterGroupTriEdgeNormals.push_back(std::make_pair(iNorm2, iNorm0));
                aClusterGroupTriEdgeNormals.push_back(std::make_pair(iNorm1, iNorm2));
                aClusterGroupTriEdgeNormals.push_back(std::make_pair(iNorm2, iNorm1));

                aClusterGroupTriEdgeUVs.push_back(std::make_pair(iUV0, iUV1));
                aClusterGroupTriEdgeUVs.push_back(std::make_pair(iUV1, iUV0));
                aClusterGroupTriEdgeUVs.push_back(std::make_pair(iUV0, iUV2));
                aClusterGroupTriEdgeUVs.push_back(std::make_pair(iUV2, iUV0));
                aClusterGroupTriEdgeUVs.push_back(std::make_pair(iUV1, iUV2));
                aClusterGroupTriEdgeUVs.push_back(std::make_pair(iUV2, iUV1));
            }
        }
#endif // #if 0

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
            aBoundaryVertices,
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
        std::ostringstream clusterGroupName;
        clusterGroupName << "simplified-cluster-group-lod";
        clusterGroupName << iLODLevel << "-group";
        clusterGroupName << iClusterGroup;

        std::ostringstream outputFilePath;
        outputFilePath << "c:\\Users\\Dingwings\\demo-models\\simplified-cluster-groups\\";
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
        clusterGroupMaterialFilePath << "c:\\Users\\Dingwings\\demo-models\\simplified-cluster-groups\\";
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
    uint32_t iLODLevel)
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
        outputFilePath << "c:\\Users\\Dingwings\\demo-models\\simplified\\";
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
        outputMaterialFilePath << "c:\\Users\\Dingwings\\demo-models\\simplified\\";
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

#include <array>
#include <memory>

/*
**
*/
std::string execCommand(std::string const& command, bool bEchoCommand)
{
    if(bEchoCommand)
    {
        DEBUG_PRINTF("%s\n", command.c_str());
    }

    std::array<char, 256> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    if(!pipe)
    {
        return "ERROR";
    }

    while(fgets(buffer.data(), static_cast<uint32_t>(buffer.size()), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    //DEBUG_PRINTF("%s\n", result.c_str());

    return result;
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
void splitClusterGroups(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    uint32_t& iTotalClusterIndex,
    std::vector<float3> const& aClusterGroupVertexPositions,
    std::vector<float3> const& aClusterGroupVertexNormals,
    std::vector<float2> const& aClusterGroupVertexUVs,
    std::vector<uint32_t> const& aiClusterTrianglePositionIndices,
    std::vector<uint32_t> const& aiClusterTriangleNormalIndices,
    std::vector<uint32_t> const& aiClusterTriangleUVIndices,
    uint32_t iMaxTrianglesPerCluster,
    uint32_t iNumSplitClusters,
    uint32_t iLODLevel,
    uint32_t iClusterGroup)
{

    // update cluster group vertex positions and triangles, generate metis mesh file to partition, and output the max partitions of 2 into clusters

    // triangle positions
    std::vector<float3> aClusterGroupTrianglePositions(aiClusterTrianglePositionIndices.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTrianglePositionIndices.size()); iTri += 3)
    {
        uint32_t iPos0 = aiClusterTrianglePositionIndices[iTri];
        uint32_t iPos1 = aiClusterTrianglePositionIndices[iTri + 1];
        uint32_t iPos2 = aiClusterTrianglePositionIndices[iTri + 2];

        aClusterGroupTrianglePositions[iTri] =      aClusterGroupVertexPositions[iPos0];
        aClusterGroupTrianglePositions[iTri + 1] =  aClusterGroupVertexPositions[iPos1];
        aClusterGroupTrianglePositions[iTri + 2] =  aClusterGroupVertexPositions[iPos2];
    }

    // triangle normals
    std::vector<float3> aClusterGroupTriangleNormals(aiClusterTriangleNormalIndices.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTriangleNormalIndices.size()); iTri += 3)
    {
        uint32_t iNormal0 = aiClusterTriangleNormalIndices[iTri];
        uint32_t iNormal1 = aiClusterTriangleNormalIndices[iTri + 1];
        uint32_t iNormal2 = aiClusterTriangleNormalIndices[iTri + 2];

        aClusterGroupTriangleNormals[iTri] = aClusterGroupVertexNormals[iNormal0];
        aClusterGroupTriangleNormals[iTri + 1] = aClusterGroupVertexNormals[iNormal1];
        aClusterGroupTriangleNormals[iTri + 2] = aClusterGroupVertexNormals[iNormal2];
    }

    // triangle uvs
    std::vector<float2> aClusterGroupTriangleUVs(aiClusterTriangleUVIndices.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiClusterTriangleUVIndices.size()); iTri += 3)
    {
        uint32_t iUV0 = aiClusterTriangleUVIndices[iTri];
        uint32_t iUV1 = aiClusterTriangleUVIndices[iTri + 1];
        uint32_t iUV2 = aiClusterTriangleUVIndices[iTri + 2];

        aClusterGroupTriangleUVs[iTri] =     aClusterGroupVertexUVs[iUV0];
        aClusterGroupTriangleUVs[iTri + 1] = aClusterGroupVertexUVs[iUV1];
        aClusterGroupTriangleUVs[iTri + 2] = aClusterGroupVertexUVs[iUV2];
    }


    // re-build triangle indices
    std::vector<float3> aTrimmedTotalVertexPositions;
    std::vector<uint32_t> aiTrianglePositionIndices(aClusterGroupTrianglePositions.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTrianglePositions.size()); i++)
    {
        auto const& vertexPosition = aClusterGroupTrianglePositions[i];
        auto iter = std::find_if(
            aTrimmedTotalVertexPositions.begin(),
            aTrimmedTotalVertexPositions.end(),
            [vertexPosition](float3 const& checkPos)
            {
                return length(checkPos - vertexPosition) < 1.0e-5f;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aTrimmedTotalVertexPositions.size());
        if(iter == aTrimmedTotalVertexPositions.end())
        {
            aTrimmedTotalVertexPositions.push_back(vertexPosition);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aTrimmedTotalVertexPositions.begin(), iter));
        }

        aiTrianglePositionIndices[i] = iVertexIndex;
    }

    // re-build triangle normals
    std::vector<float3> aTrimmedTotalVertexNormals;
    std::vector<uint32_t> aiTriangleNormalIndices(aClusterGroupTriangleNormals.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTriangleNormals.size()); i++)
    {
        auto const& vertexNormal = aClusterGroupTriangleNormals[i];
        auto iter = std::find_if(
            aTrimmedTotalVertexNormals.begin(),
            aTrimmedTotalVertexNormals.end(),
            [vertexNormal](float3 const& checkPos)
            {
                return length(checkPos - vertexNormal) < 1.0e-5f;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aTrimmedTotalVertexNormals.size());
        if(iter == aTrimmedTotalVertexNormals.end())
        {
            aTrimmedTotalVertexNormals.push_back(vertexNormal);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aTrimmedTotalVertexNormals.begin(), iter));
        }

        aiTriangleNormalIndices[i] = iVertexIndex;
    }

    // re-build triangle uvs
    std::vector<float2> aTrimmedTotalVertexUVs;
    std::vector<uint32_t> aiTriangleUVIndices(aClusterGroupTriangleUVs.size());
    for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupTriangleUVs.size()); i++)
    {
        auto const& vertexUV = aClusterGroupTriangleUVs[i];
        auto iter = std::find_if(
            aTrimmedTotalVertexUVs.begin(),
            aTrimmedTotalVertexUVs.end(),
            [vertexUV](float2 const& checkUV)
            {
                return length(checkUV - vertexUV) < 1.0e-5f;
            }
        );

        uint32_t iVertexIndex = static_cast<uint32_t>(aTrimmedTotalVertexUVs.size());
        if(iter == aTrimmedTotalVertexUVs.end())
        {
            aTrimmedTotalVertexUVs.push_back(vertexUV);
        }
        else
        {
            iVertexIndex = static_cast<uint32_t>(std::distance(aTrimmedTotalVertexUVs.begin(), iter));
        }

        aiTriangleUVIndices[i] = iVertexIndex;
    }

    std::ostringstream outputMetisMeshFilePath;
    std::ostringstream outputPartitionFilePath;
    if(iNumSplitClusters > 1)
    {
        outputMetisMeshFilePath << "c:\\Users\\Dingwings\\demo-models\\metis\\split-cluster-group";
        outputMetisMeshFilePath << "-lod" << iLODLevel << "-group" << iClusterGroup << ".mesh";

        buildMETISMeshFile2(
            outputMetisMeshFilePath.str(),
            aiTrianglePositionIndices);

        // generate cluster groups for all the initial clusters
        std::ostringstream clusterGroupCommand;
        clusterGroupCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
        clusterGroupCommand << outputMetisMeshFilePath.str() << " ";
        clusterGroupCommand << "-gtype=dual ";
        clusterGroupCommand << "-ncommon=2 ";
        clusterGroupCommand << "-objtype=vol ";
        clusterGroupCommand << "-ufactor=20 ";
        clusterGroupCommand << "-contig ";
        clusterGroupCommand << iNumSplitClusters;
        std::string result = execCommand(clusterGroupCommand.str(), false);
        if(result.find("Metis returned with an error.") != std::string::npos)
        {
            clusterGroupCommand = std::ostringstream();
            clusterGroupCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
            clusterGroupCommand << outputMetisMeshFilePath.str() << " ";
            clusterGroupCommand << "-gtype=dual ";
            clusterGroupCommand << "-ncommon=2 ";
            clusterGroupCommand << "-objtype=vol ";
            clusterGroupCommand << "-ufactor=20 ";
            clusterGroupCommand << iNumSplitClusters;
            std::string result = execCommand(clusterGroupCommand.str(), false);

            DEBUG_PRINTF("*** metis reported non-contiguous cluster (%d, %d) of total %d in group %d ***\n", 
                iTotalClusterIndex, 
                iTotalClusterIndex + 1, 
                aaClusterVertexPositions.size(),
                iClusterGroup);
            assert(result.find("Metis returned with an error.") == std::string::npos);
        }

        outputPartitionFilePath << outputMetisMeshFilePath.str() << ".epart.";
        outputPartitionFilePath << iNumSplitClusters;
    }

    // map of element indices in cluster
    std::map<uint32_t, std::vector<uint32_t>> aClusterMap;
    {
        std::vector<uint32_t> aiNewClusters;
        readMetisClusterFile(aiNewClusters, outputPartitionFilePath.str());
        for(uint32_t i = 0; i < static_cast<uint32_t>(aiNewClusters.size()); i++)
        {
            uint32_t const& iCluster = aiNewClusters[i];
            aClusterMap[iCluster].push_back(i);
        }
    }

    uint32_t iPrevNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());

    std::vector<std::vector<float3>> aaTempClusterVertexPositions;
    std::vector<std::vector<float3>> aaTempClusterVertexNormals;
    std::vector<std::vector<float2>> aaTempClusterVertexUVs;
    std::vector<std::vector<uint32_t>> aaiTempClusterTrianglePositionIndices;
    std::vector<std::vector<uint32_t>> aaiTempClusterTriangleNormalIndices;
    std::vector<std::vector<uint32_t>> aaiTempClusterTriangleUVIndices;

#if 0
    splitCluster(
        aaTempClusterVertexPositions,
        aaTempClusterVertexNormals,
        aaTempClusterVertexUVs,
        aaiTempClusterTrianglePositionIndices,
        aaiTempClusterTriangleNormalIndices,
        aaiTempClusterTriangleUVIndices,
        aTrimmedTotalVertexPositions,
        aTrimmedTotalVertexNormals,
        aTrimmedTotalVertexUVs,
        aiTrianglePositionIndices,
        aiTriangleNormalIndices,
        aiTriangleUVIndices,
        128 * 3);

    for(uint32_t i = 0; i < static_cast<uint32_t>(aaTempClusterVertexPositions.size()); i++)
    {
        if(aaTempClusterVertexPositions[i].size() > 0)
        {
            aaClusterVertexPositions.push_back(aaTempClusterVertexPositions[i]);
            aaClusterVertexNormals.push_back(aaTempClusterVertexNormals[i]);
            aaClusterVertexUVs.push_back(aaTempClusterVertexUVs[i]);

            aaiClusterTrianglePositionIndices.push_back(aaiTempClusterTrianglePositionIndices[i]);
            aaiClusterTriangleNormalIndices.push_back(aaiTempClusterTriangleNormalIndices[i]);
            aaiClusterTriangleUVIndices.push_back(aaiTempClusterTriangleUVIndices[i]);
        }

        std::ostringstream clusterName;
        clusterName << "cluster-lod" << iLODLevel << "-" << aaClusterVertexPositions.size() - 1;

        std::ostringstream outputFilePath;
        outputFilePath << "c:\\Users\\Dingwings\\demo-models\\clusters\\" << clusterName.str() << ".obj";

        writeOBJFile(
            aaTempClusterVertexPositions[i],
            aaTempClusterVertexNormals[i],
            aaTempClusterVertexUVs[i],
            aaiTempClusterTrianglePositionIndices[i],
            aaiTempClusterTriangleNormalIndices[i],
            aaiTempClusterTriangleUVIndices[i],
            outputFilePath.str(),
            clusterName.str());
    }

    iTotalClusterIndex += static_cast<uint32_t>(aaTempClusterVertexPositions.size());
#endif // #if 0

    // create clusters based on the partition files from the above metis command
    for(uint32_t iCluster = 0; iCluster < iNumSplitClusters; iCluster++)
    {
        std::vector<float3> aNewClusterTrianglePositions;
        std::vector<float3> aNewClusterTriangleNormals;
        std::vector<float2> aNewClusterTriangleUVs;
        std::vector<uint32_t> aiNewClusterTriangleIndices;
        std::vector<uint32_t> aiNewClusterTriangleNormalIndices;
        std::vector<uint32_t> aiNewClusterTriangleUVIndices;
        outputMeshClusters2(
            aNewClusterTrianglePositions,
            aNewClusterTriangleNormals,
            aNewClusterTriangleUVs,
            aiNewClusterTriangleIndices,
            aiNewClusterTriangleNormalIndices,
            aiNewClusterTriangleUVIndices,
            aClusterMap[iCluster],
            aTrimmedTotalVertexPositions,
            aTrimmedTotalVertexNormals,
            aTrimmedTotalVertexUVs,
            aiTrianglePositionIndices,
            aiTriangleNormalIndices,
            aiTriangleUVIndices,
            iLODLevel + 1,
            iTotalClusterIndex);

        DEBUG_PRINTF("cluster %d num triangle indices: %d\n",
            iCluster,
            aiNewClusterTriangleIndices.size());

        // check for dis-jointed cluster
        {
            std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleIndices;
            checkClusterAdjacency(
                aaiSplitClusterTriangleIndices,
                aiNewClusterTriangleIndices);

            if(aaiSplitClusterTriangleIndices.size() > 0)
            {
                // dis-jointed cluster, split it
                std::vector<std::vector<float3>> aaSplitClusterVertexPositions;
                std::vector<std::vector<float3>> aaSplitClusterVertexNormals;
                std::vector<std::vector<float2>> aaSplitClusterVertexUVs;
                std::vector<std::vector<uint32_t>> aaiSplitClusterTrianglePositionIndices;
                std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleNormalIndices;
                std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleUVIndices;
                std::vector<uint32_t> aiDeleteClusters(aaClusterVertexPositions.size());
                createSplitClusters2(
                    aaSplitClusterVertexPositions,
                    aaSplitClusterVertexNormals,
                    aaSplitClusterVertexUVs,
                    aaiSplitClusterTrianglePositionIndices,
                    aaiSplitClusterTriangleNormalIndices,
                    aaiSplitClusterTriangleUVIndices,
                    aiDeleteClusters,
                    aNewClusterTrianglePositions,
                    aNewClusterTriangleNormals,
                    aNewClusterTriangleUVs,
                    aiNewClusterTriangleIndices,
                    aiNewClusterTriangleNormalIndices,
                    aiNewClusterTriangleUVIndices,
                    aaiSplitClusterTriangleIndices);

                // add split clusters to total clusters
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaSplitClusterVertexPositions.size()); i++)
                {
                    aaTempClusterVertexPositions.push_back(aaSplitClusterVertexPositions[i]);
                    aaTempClusterVertexNormals.push_back(aaSplitClusterVertexNormals[i]);
                    aaTempClusterVertexUVs.push_back(aaSplitClusterVertexUVs[i]);

                    aaiTempClusterTrianglePositionIndices.push_back(aaiSplitClusterTrianglePositionIndices[i]);
                    aaiTempClusterTriangleNormalIndices.push_back(aaiSplitClusterTriangleNormalIndices[i]);
                    aaiTempClusterTriangleUVIndices.push_back(aaiSplitClusterTriangleUVIndices[i]);
                }
                
                //DEBUG_PRINTF("!!! split cluster %d into %d separate clusters\n", iCluster, static_cast<uint32_t>(aaiSplitClusterTrianglePositionIndices.size()));
            }
            else
            {
                // add to total clusters
                aaTempClusterVertexPositions.push_back(aNewClusterTrianglePositions);
                aaiTempClusterTrianglePositionIndices.push_back(aiNewClusterTriangleIndices);

                aaTempClusterVertexNormals.push_back(aNewClusterTriangleNormals);
                aaiTempClusterTriangleNormalIndices.push_back(aiNewClusterTriangleNormalIndices);

                aaTempClusterVertexUVs.push_back(aNewClusterTriangleUVs);
                aaiTempClusterTriangleUVIndices.push_back(aiNewClusterTriangleUVIndices);
            }
        }

    }   // for cluster = 0 to num split clusters

    // move large cluster triangles to smaller ones or merge small cluster triangles to larger clusters
    {
        bool bResetLoop = false;
        for(uint32_t iCheckCluster = 0; iCheckCluster < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); iCheckCluster++)
        {
            if(bResetLoop)
            {
                iCheckCluster = 0;
                bResetLoop = false;
            }

            if(aaiTempClusterTrianglePositionIndices[iCheckCluster].size() > 128 * 3)
            {
                for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                {
                    DEBUG_PRINTF("BEFORE moveTriangle cluster %d size: %d\n",
                        j,
                        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                }

                bResetLoop = moveTriangles(
                    aaTempClusterVertexPositions,
                    aaTempClusterVertexNormals,
                    aaTempClusterVertexUVs,
                    aaiTempClusterTrianglePositionIndices,
                    aaiTempClusterTriangleNormalIndices,
                    aaiTempClusterTriangleUVIndices,
                    iCheckCluster,
                    128 * 3);

                for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                {
                    DEBUG_PRINTF("AFTER moveTriangles cluster %d size: %d\n",
                        j,
                        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                }
                int iDebug = 1;
            }
            else if(aaiTempClusterTrianglePositionIndices[iCheckCluster].size() <= 12)
            {
                DEBUG_PRINTF("\n");
                for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                {
                    DEBUG_PRINTF("BEFORE mergeTriangles cluster %d size: %d\n",
                        j,
                        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                }

                bResetLoop = mergeTriangles(
                    aaTempClusterVertexPositions,
                    aaTempClusterVertexNormals,
                    aaTempClusterVertexUVs,
                    aaiTempClusterTrianglePositionIndices,
                    aaiTempClusterTriangleNormalIndices,
                    aaiTempClusterTriangleUVIndices,
                    iCheckCluster);

                for(uint32_t j = 0; j < static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices.size()); j++)
                {
                    DEBUG_PRINTF("AFTER mergeTriangles cluster %d size: %d\n",
                        j,
                        static_cast<uint32_t>(aaiTempClusterTrianglePositionIndices[j].size()));
                }
                DEBUG_PRINTF("\n");
                int iDebug = 1;
            }
        }
    }

    for(uint32_t i = 0; i < static_cast<uint32_t>(aaTempClusterVertexPositions.size()); i++)
    {
        if(aaTempClusterVertexPositions[i].size() > 0)
        {
            aaClusterVertexPositions.push_back(aaTempClusterVertexPositions[i]);
            aaClusterVertexNormals.push_back(aaTempClusterVertexNormals[i]);
            aaClusterVertexUVs.push_back(aaTempClusterVertexUVs[i]);

            aaiClusterTrianglePositionIndices.push_back(aaiTempClusterTrianglePositionIndices[i]);
            aaiClusterTriangleNormalIndices.push_back(aaiTempClusterTriangleNormalIndices[i]);
            aaiClusterTriangleUVIndices.push_back(aaiTempClusterTriangleUVIndices[i]);
        }
    }

#if 0
    // join any dis-jointed clusters if possible and add to total list
    if(aaTempClusterVertexPositions.size() <= 2)
    {
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaTempClusterVertexPositions.size()); i++)
        {
            if(aaTempClusterVertexPositions[i].size() > 0)
            {
                aaClusterVertexPositions.push_back(aaTempClusterVertexPositions[i]);
                aaClusterVertexNormals.push_back(aaTempClusterVertexNormals[i]);
                aaClusterVertexUVs.push_back(aaTempClusterVertexUVs[i]);

                aaiClusterTrianglePositionIndices.push_back(aaiTempClusterTrianglePositionIndices[i]);
                aaiClusterTriangleNormalIndices.push_back(aaiTempClusterTriangleNormalIndices[i]);
                aaiClusterTriangleUVIndices.push_back(aaiTempClusterTriangleUVIndices[i]);
            }
        }
    }
    else
    {
        uint32_t const kiSmallClusterVertexCountThreshold = 12;

        std::vector<uint32_t> aiJoined(aaTempClusterVertexPositions.size());
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaTempClusterVertexPositions.size()); i++)
        {
            if(aiJoined[i] > 0)
            {
                continue;
            }

            for(uint32_t j = i + 1; j < static_cast<uint32_t>(aaTempClusterVertexPositions.size()); j++)
            {
                if(aiJoined[j] > 0)
                {
                    continue;
                }
                
                if((aaiTempClusterTrianglePositionIndices[i].size() <= kiSmallClusterVertexCountThreshold || aaiTempClusterTrianglePositionIndices[j].size() <= kiSmallClusterVertexCountThreshold) &&
                   canJoinClusters(
                    aaTempClusterVertexPositions[i],
                    aaTempClusterVertexPositions[j],
                    aaiTempClusterTrianglePositionIndices[i],
                    aaiTempClusterTrianglePositionIndices[j]))
                {
                    std::vector<float3> aJoinedVertexPositions;
                    std::vector<float3> aJoinedVertexNormals;
                    std::vector<float2> aJoinedVertexUVs;

                    std::vector<uint32_t> aiJoinedTrianglePositionIndices;
                    std::vector<uint32_t> aiJoinedTriangleNormalIndices;
                    std::vector<uint32_t> aiJoinedTriangleUVIndices;
                    joinSharedClusters(
                        aJoinedVertexPositions,
                        aJoinedVertexNormals,
                        aJoinedVertexUVs,
                        aiJoinedTrianglePositionIndices,
                        aiJoinedTriangleNormalIndices,
                        aiJoinedTriangleUVIndices,
                        aaTempClusterVertexPositions[i],
                        aaTempClusterVertexPositions[j],
                        aaTempClusterVertexNormals[i],
                        aaTempClusterVertexNormals[j],
                        aaTempClusterVertexUVs[i],
                        aaTempClusterVertexUVs[j],
                        aaiTempClusterTrianglePositionIndices[i],
                        aaiTempClusterTrianglePositionIndices[j],
                        aaiTempClusterTriangleNormalIndices[i],
                        aaiTempClusterTriangleNormalIndices[j],
                        aaiTempClusterTriangleUVIndices[i],
                        aaiTempClusterTriangleUVIndices[j]);

                    DEBUG_PRINTF("total cluster index: %d joined cluster %d (indices #: %d) and %d (indices #: %d)\n",
                        iTotalClusterIndex,
                        i,
                        aaiTempClusterTrianglePositionIndices[i].size(),
                        j,
                        aaiTempClusterTrianglePositionIndices[j].size());

                    aiJoined[i] = 1; aiJoined[j] = 1;

                    if(aJoinedVertexPositions.size() > 0)
                    {
                        aaClusterVertexPositions.push_back(aJoinedVertexPositions);
                        aaClusterVertexNormals.push_back(aJoinedVertexNormals);
                        aaClusterVertexUVs.push_back(aJoinedVertexUVs);

                        aaiClusterTrianglePositionIndices.push_back(aiJoinedTrianglePositionIndices);
                        aaiClusterTriangleNormalIndices.push_back(aiJoinedTriangleNormalIndices);
                        aaiClusterTriangleUVIndices.push_back(aiJoinedTriangleUVIndices);
                    }

                    break;
                
                }   // if can join clusters
            
            }   // for j = i + 1 to num clusters

        }   // for i = 0 to num clusters
        
        // add any remain non-joined clusters
        for(uint32_t i = 0; i < static_cast<uint32_t>(aiJoined.size()); i++)
        {
            if(aiJoined[i] <= 0)
            {
                if(aaTempClusterVertexPositions[i].size() > 0)
                {
                    aaClusterVertexPositions.push_back(aaTempClusterVertexPositions[i]);
                    aaClusterVertexNormals.push_back(aaTempClusterVertexNormals[i]);
                    aaClusterVertexUVs.push_back(aaTempClusterVertexUVs[i]);

                    aaiClusterTrianglePositionIndices.push_back(aaiTempClusterTrianglePositionIndices[i]);
                    aaiClusterTriangleNormalIndices.push_back(aaiTempClusterTriangleNormalIndices[i]);
                    aaiClusterTriangleUVIndices.push_back(aaiTempClusterTriangleUVIndices[i]);
                }
            }
        }

    }   // if temp cluster size > 2
#endif // #if 0

    uint32_t iCurrNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
    iTotalClusterIndex += (iCurrNumClusters - iPrevNumClusters);

    assert(aaClusterVertexPositions.size() == aaClusterVertexNormals.size());
    assert(aaClusterVertexPositions.size() == aaClusterVertexUVs.size());

    assert(aaiClusterTrianglePositionIndices.size() == aaiClusterTriangleNormalIndices.size());
    assert(aaiClusterTrianglePositionIndices.size() == aaiClusterTriangleUVIndices.size());

    for(uint32_t i = iPrevNumClusters; i < iCurrNumClusters; i++)
    {
        std::ostringstream clusterName;
        clusterName << "cluster-lod" << iLODLevel + 1 << "-" << i;

        std::ostringstream outputFilePath;
        outputFilePath << "c:\\Users\\Dingwings\\demo-models\\clusters\\" << clusterName.str() << ".obj";

        writeOBJFile(
            aaClusterVertexPositions[i],
            aaClusterVertexNormals[i],
            aaClusterVertexUVs[i],
            aaiClusterTrianglePositionIndices[i],
            aaiClusterTriangleNormalIndices[i],
            aaiClusterTriangleUVIndices[i],
            outputFilePath.str(),
            clusterName.str());
    }

}

/*
**
*/
struct VertexMappingInfo
{
    float3              mPosition;
    uint32_t            miClusterVertexID;
    uint32_t            miCluster;
    uint32_t            miClusterGroup;
    uint32_t            miLODLevel;
    float               mfDistance;
    uint32_t            miMIPLevel;
};

/*
**
*/
void getVertexMappingAndMaxDistances(
    std::vector<float>& afMaxClusterDistances,
    std::map<std::pair<uint32_t, uint32_t>, VertexMappingInfo>& aVertexMapping,
    std::vector<std::pair<float3, float3>>& aMaxErrorPositions,
    std::vector<std::vector<MeshCluster>> const& aaMeshClusters,
    std::vector<std::vector<MeshClusterGroup>> const& aaMeshClusterGroups,
    std::vector<MeshCluster*> const& apTotalMeshClusters,
    std::vector<MeshClusterGroup*> const& apTotalMeshClusterGroups,
    uint32_t iLODLevel,
    uint32_t iUpperLODLevel)
{

auto start = std::chrono::high_resolution_clock::now();

    auto const& aMeshClusters = aaMeshClusters[iLODLevel];
    uint32_t iNumMeshClusters = static_cast<uint32_t>(aMeshClusters.size());
    for(uint32_t iMeshCluster = 0; iMeshCluster < iNumMeshClusters; iMeshCluster++)
    {
        float fMaxVertexPositionDistance = 0.0f;

        auto const& meshCluster = aMeshClusters[iMeshCluster];
        std::pair<float3, float3> maxErrorPositions;

        uint32_t const kiMIPLevel = 0;

        // get closest vertex positions from the upper clusters (given upper LOD)
        for(uint32_t iVertex = 0; iVertex < static_cast<uint32_t>(meshCluster.miNumVertexPositions); iVertex++)
        {
            uint32_t iBestUpperMeshCluster = UINT32_MAX;
            uint32_t iBestUpperMeshClusterGroup = UINT32_MAX;
            uint32_t iBestUpperClusterVertexID = UINT32_MAX;
            float3 bestUpperClusterVertexPosition = float3(FLT_MAX, FLT_MAX, FLT_MAX);
            float fShortestDistance = FLT_MAX;

            float3 const* aClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshCluster.miVertexPositionStartAddress * sizeof(float3));
            auto const& vertexPosition = aClusterVertexPositions[iVertex];

            if(iLODLevel == 0)
            {
                fShortestDistance = 0.0f;
                bestUpperClusterVertexPosition = aClusterVertexPositions[iVertex];
                iBestUpperClusterVertexID = iMeshCluster;
                iBestUpperMeshClusterGroup = 0;
                iBestUpperMeshCluster = iMeshCluster;

                continue;
            }

            // mesh cluster group of given upper LOD
            uint32_t iNumUpperMeshClusterGroups = static_cast<uint32_t>(aaMeshClusterGroups[iUpperLODLevel].size());
            for(uint32_t iUpperMeshClusterGroup = 0; iUpperMeshClusterGroup < iNumUpperMeshClusterGroups; iUpperMeshClusterGroup++)
            {
                auto const& meshClusterGroup = aaMeshClusterGroups[iUpperLODLevel][iUpperMeshClusterGroup];
                
                // mesh clusters of given upper LOD associated with this mesh cluster group
                for(uint32_t iUpperMeshClusterIndex = 0; iUpperMeshClusterIndex < static_cast<uint32_t>(meshClusterGroup.maiNumClusters[kiMIPLevel]); iUpperMeshClusterIndex++)
                {
                    uint32_t iMeshClusterID = meshClusterGroup.maiClusters[kiMIPLevel][iUpperMeshClusterIndex];
                    auto const& upperMeshCluster = *apTotalMeshClusters[iMeshClusterID];

                    float3 const* aUpperMeshClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + upperMeshCluster.miVertexPositionStartAddress * sizeof(float3));
                    for(uint32_t iUpperClusterVertex = 0; iUpperClusterVertex < static_cast<uint32_t>(upperMeshCluster.miNumVertexPositions); iUpperClusterVertex++)
                    {
                        auto const& upperMeshClusterVertexPosition = aUpperMeshClusterVertexPositions[iUpperClusterVertex];
                        float fDistance = lengthSquared(vertexPosition - upperMeshClusterVertexPosition);
                        if(fDistance < fShortestDistance)
                        {
                            fShortestDistance = fDistance;
                            bestUpperClusterVertexPosition = upperMeshClusterVertexPosition;
                            iBestUpperClusterVertexID = iUpperClusterVertex;
                            iBestUpperMeshClusterGroup = iUpperMeshClusterGroup;
                            iBestUpperMeshCluster = iMeshClusterID;
                        }
                    }

                }   // for cluster = 0 to num upper cluster
                     
            }   // for cluster group = 0 to num upper cluster group

            // verify
            float3 const* aCheckClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + apTotalMeshClusters[iBestUpperMeshCluster]->miVertexPositionStartAddress * sizeof(float3));
            auto const& checkVertexPosition = aCheckClusterVertexPositions[iBestUpperClusterVertexID];
            float fCheckDistance = length(checkVertexPosition - bestUpperClusterVertexPosition);
            assert(fCheckDistance <= 1.0e-5f);
            //DEBUG_PRINTF("found cluster %d of lod %d with position (%.4f, %.4f, %.4f) compared to\n cluster %d of lod %d with position (%.4f, %.4f, %.4f)\ndistance: %.4f\n",
            //    iBestUpperMeshCluster,
            //    iUpperLODLevel,
            //    checkVertexPosition.x,
            //    checkVertexPosition.y,
            //    checkVertexPosition.z,
            //    iMeshCluster,
            //    iLODLevel,
            //    vertexPosition.x,
            //    vertexPosition.y,
            //    vertexPosition.z,
            //    fShortestDistance);

            VertexMappingInfo mappingInfo;
            mappingInfo.miCluster = iBestUpperMeshCluster;
            mappingInfo.miClusterGroup = iBestUpperMeshClusterGroup;
            mappingInfo.miLODLevel = iUpperLODLevel;
            mappingInfo.mPosition = bestUpperClusterVertexPosition;
            mappingInfo.miClusterVertexID = iBestUpperClusterVertexID;
            mappingInfo.mfDistance = fShortestDistance;

            aVertexMapping[std::make_pair(iMeshCluster, iVertex)] = mappingInfo;

            if(fShortestDistance > fMaxVertexPositionDistance)
            {
                fMaxVertexPositionDistance = fShortestDistance;

                maxErrorPositions = std::make_pair(mappingInfo.mPosition, vertexPosition);
            }

        }   // for vertex = 0 to num vertices in cluster

        {
            std::lock_guard<std::mutex> lock(gMutex);
            afMaxClusterDistances.push_back(fMaxVertexPositionDistance);
            aMaxErrorPositions.push_back(maxErrorPositions);
        }

    }   // for cluster = 0 to num clusters

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took total %d seconds to get lod distance error between lod %d and lod %d\n", iSeconds, iLODLevel, iUpperLODLevel);
}

/*
**
*/
void writeClusters(
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices,
    std::string const& outputDirectory,
    uint32_t iLODLevel)
{
    uint32_t iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        std::ostringstream clusterName;
        clusterName << "cluster-lod" << iLODLevel << "-" << iCluster;

        std::ostringstream outputFilePath;
        outputFilePath << outputDirectory;
        outputFilePath << "//" << clusterName.str() << ".obj";
        
        FILE* fp = fopen(outputFilePath.str().c_str(), "wb");
        fprintf(fp, "o %s\n", clusterName.str().c_str());
        fprintf(fp, "usemtl %s\n", clusterName.str().c_str());
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size()); i++)
        {
            fprintf(fp, "v %.4f %.4f %.4f\n", aaClusterVertexPositions[iCluster][i].x, aaClusterVertexPositions[iCluster][i].y, aaClusterVertexPositions[iCluster][i].z);
        }
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexNormals[iCluster].size()); i++)
        {
            fprintf(fp, "vn %.4f %.4f %.4f\n", aaClusterVertexNormals[iCluster][i].x, aaClusterVertexNormals[iCluster][i].y, aaClusterVertexNormals[iCluster][i].z);
        }
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexUVs[iCluster].size()); i++)
        {
            fprintf(fp, "vt %.4f %.4f\n", aaClusterVertexUVs[iCluster][i].x, aaClusterVertexUVs[iCluster][i].y);
        }
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); i += 3)
        {
            fprintf(fp, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                aaiClusterTrianglePositionIndices[iCluster][i] + 1,
                aaiClusterTriangleUVIndices[iCluster][i] + 1,
                aaiClusterTriangleNormalIndices[iCluster][i] + 1,
                aaiClusterTrianglePositionIndices[iCluster][i + 1] + 1,
                aaiClusterTriangleUVIndices[iCluster][i + 1] + 1,
                aaiClusterTriangleNormalIndices[iCluster][i + 1] + 1,
                aaiClusterTrianglePositionIndices[iCluster][i + 2] + 1,
                aaiClusterTriangleUVIndices[iCluster][i + 2] + 1,
                aaiClusterTriangleNormalIndices[iCluster][i + 2] + 1);
        }
        fclose(fp);

        float fRand0 = static_cast<float>(rand() % 255) / 255.0f;
        float fRand1 = static_cast<float>(rand() % 255) / 255.0f;
        float fRand2 = static_cast<float>(rand() % 255) / 255.0f;

        std::ostringstream outputMaterialfilePath;
        outputMaterialfilePath << outputDirectory << "//" << clusterName.str() << ".mtl";
        fp = fopen(outputMaterialfilePath.str().c_str(), "wb");
        fprintf(fp, "newmtl %s\n", clusterName.str().c_str());
        fprintf(fp, "Kd %.4f %.4f %.4f\n", fRand0, fRand1, fRand2);
        fclose(fp);
    }
    
}

/*
**
*/
void splitLargeClusters(
    std::vector<std::vector<float3>>& aaClusterVertexPositions,
    std::vector<std::vector<float3>>& aaClusterVertexNormals,
    std::vector<std::vector<float2>>& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>>& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>>& aaiClusterTriangleUVIndices,
    uint32_t& iNumClusters,
    uint32_t& iNumClusterGroups,
    uint32_t iMaxTrianglesPerCluster)
{
    uint32_t iNumTotalSplitCluster = iNumClusters;
    for(int32_t iCluster = 0; iCluster < static_cast<int32_t>(aaClusterVertexPositions.size()); iCluster++)
    {
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
        assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());

        if(aaiClusterTrianglePositionIndices[iCluster].size() >= iMaxTrianglesPerCluster)
        {
            std::ostringstream outputMetisSplitMeshFilePath;
            outputMetisSplitMeshFilePath << "c:\\Users\\Dingwings\\demo-models\\metis\\split-cluster";
            outputMetisSplitMeshFilePath << iCluster << "-lod" << 0 << ".mesh";

            buildMETISMeshFile2(
                outputMetisSplitMeshFilePath.str(),
                aaiClusterTrianglePositionIndices[iCluster]);

            uint32_t iNumSplitClusters = static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()) / (128 * 3);
            iNumSplitClusters = std::max(iNumSplitClusters, 2u);
            // exec the mpmetis to generate the initial clusters
            std::ostringstream metisCommand;
            metisCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
            metisCommand << outputMetisSplitMeshFilePath.str() << " ";
            metisCommand << "-gtype=dual ";
            metisCommand << "-ncommon=2 ";
            metisCommand << "-objtype=cut ";
            metisCommand << "-ufactor=60 ";
            metisCommand << "-contig ";
            //metisCommand << "-minconn ";
            //metisCommand << "-niter=20 ";
            metisCommand << iNumSplitClusters;
            std::string result = execCommand(metisCommand.str(), false);
            if(result.find("Metis returned with an error.") != std::string::npos)
            {
                metisCommand = std::ostringstream();
                metisCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
                metisCommand << outputMetisSplitMeshFilePath.str() << " ";
                metisCommand << "-gtype=dual ";
                metisCommand << "-ncommon=2 ";
                metisCommand << "-objtype=cut ";
                //metisCommand << "-ufactor=600 ";
                metisCommand << iNumSplitClusters;
                result = execCommand(metisCommand.str(), false);
                assert(result.find("Metis returned with an error.") == std::string::npos);
            }

            std::ostringstream outputSplitPartitionFilePath;
            outputSplitPartitionFilePath << outputMetisSplitMeshFilePath.str() << ".epart.";
            outputSplitPartitionFilePath << iNumSplitClusters;

            std::vector<uint32_t> aiClusters;
            readMetisClusterFile(aiClusters, outputSplitPartitionFilePath.str());

            // map of element index (triangle index) to cluster
            std::map<uint32_t, std::vector<uint32_t>> aSplitClusterMap;
            {
                std::vector<uint32_t> aiSplitClusters;
                readMetisClusterFile(aiSplitClusters, outputSplitPartitionFilePath.str());
                for(uint32_t i = 0; i < static_cast<uint32_t>(aiSplitClusters.size()); i++)
                {
                    uint32_t const& iSplitCluster = aiSplitClusters[i];
                    aSplitClusterMap[iSplitCluster].push_back(i);
                }
            }

            std::vector<std::vector<float3>> aaSplitClusterVertexPositions(iNumSplitClusters);
            std::vector<std::vector<float3>> aaSplitClusterVertexNormals(iNumSplitClusters);
            std::vector<std::vector<float2>> aaSplitClusterVertexUVs(iNumSplitClusters);
            std::vector<std::vector<uint32_t>> aaiSplitClusterTrianglePositionIndices(iNumSplitClusters);
            std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleNormalIndices(iNumSplitClusters);
            std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleUVIndices(iNumSplitClusters);
            for(uint32_t iSplitCluster = 0; iSplitCluster < iNumSplitClusters; iSplitCluster++)
            {
                outputMeshClusters2(
                    aaSplitClusterVertexPositions[iSplitCluster],
                    aaSplitClusterVertexNormals[iSplitCluster],
                    aaSplitClusterVertexUVs[iSplitCluster],
                    aaiSplitClusterTrianglePositionIndices[iSplitCluster],
                    aaiSplitClusterTriangleNormalIndices[iSplitCluster],
                    aaiSplitClusterTriangleUVIndices[iSplitCluster],
                    aSplitClusterMap[iSplitCluster],
                    aaClusterVertexPositions[iCluster],
                    aaClusterVertexNormals[iCluster],
                    aaClusterVertexUVs[iCluster],
                    aaiClusterTrianglePositionIndices[iCluster],
                    aaiClusterTriangleNormalIndices[iCluster],
                    aaiClusterTriangleUVIndices[iCluster],
                    0,
                    iNumTotalSplitCluster);
                ++iNumTotalSplitCluster;
            }

            //{
            //    FILE* fp = fopen("C:\\Users\\Dingwings\\demo-models\\clusters\\large-cluster.obj", "wb");
            //    fprintf(fp, "o large-cluster\n");
            //    for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size()); i++)
            //    {
            //        fprintf(fp, "v %.4f %.4f %.4f\n", aaClusterVertexPositions[iCluster][i].x, aaClusterVertexPositions[iCluster][i].y, aaClusterVertexPositions[iCluster][i].z);
            //    }
            //    for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); i += 3)
            //    {
            //        fprintf(fp, "f %d// %d// %d//\n",
            //            aaiClusterTrianglePositionIndices[iCluster][i] + 1,
            //            aaiClusterTrianglePositionIndices[iCluster][i + 1] + 1,
            //            aaiClusterTrianglePositionIndices[iCluster][i + 2] + 1);
            //    }
            //    fclose(fp);
            //}

            bool bReset = false;
            for(uint32_t iSplitCluster = 0; iSplitCluster < iNumSplitClusters; iSplitCluster++)
            {
                if(bReset)
                {
                    iSplitCluster = 0;
                    bReset = false;
                }
                for(int32_t iTri = 0; iTri < static_cast<int32_t>(aaiSplitClusterTrianglePositionIndices[iSplitCluster].size()); iTri += 3)
                {
                    uint32_t iPos0 = aaiSplitClusterTrianglePositionIndices[iSplitCluster][iTri];
                    uint32_t iPos1 = aaiSplitClusterTrianglePositionIndices[iSplitCluster][iTri + 1];
                    uint32_t iPos2 = aaiSplitClusterTrianglePositionIndices[iSplitCluster][iTri + 2];

                    if(iPos0 == iPos1 || iPos0 == iPos2 || iPos1 == iPos2)
                    {
                        aaiSplitClusterTrianglePositionIndices[iSplitCluster].erase(
                            aaiSplitClusterTrianglePositionIndices[iSplitCluster].begin() + iTri,
                            aaiSplitClusterTrianglePositionIndices[iSplitCluster].begin() + iTri + 3);

                        aaiSplitClusterTriangleNormalIndices[iSplitCluster].erase(
                            aaiSplitClusterTriangleNormalIndices[iSplitCluster].begin() + iTri,
                            aaiSplitClusterTriangleNormalIndices[iSplitCluster].begin() + iTri + 3);

                        aaiSplitClusterTriangleUVIndices[iSplitCluster].erase(
                            aaiSplitClusterTriangleUVIndices[iSplitCluster].begin() + iTri,
                            aaiSplitClusterTriangleUVIndices[iSplitCluster].begin() + iTri + 3);

                        bReset = true;

                        DEBUG_PRINTF("!!! Remove degenerate triangle %d from split cluster %d\n", iTri, iSplitCluster);

                        break;
                    }
                }
            }

            aaClusterVertexPositions.erase(aaClusterVertexPositions.begin() + iCluster);
            aaClusterVertexNormals.erase(aaClusterVertexNormals.begin() + iCluster);
            aaClusterVertexUVs.erase(aaClusterVertexUVs.begin() + iCluster);

            aaiClusterTrianglePositionIndices.erase(aaiClusterTrianglePositionIndices.begin() + iCluster);
            aaiClusterTriangleNormalIndices.erase(aaiClusterTriangleNormalIndices.begin() + iCluster);
            aaiClusterTriangleUVIndices.erase(aaiClusterTriangleUVIndices.begin() + iCluster);

            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());

            for(uint32_t iSplitCluster = 0; iSplitCluster < iNumSplitClusters; iSplitCluster++)
            {
                aaClusterVertexPositions.push_back(aaSplitClusterVertexPositions[iSplitCluster]);
                aaClusterVertexNormals.push_back(aaSplitClusterVertexNormals[iSplitCluster]);
                aaClusterVertexUVs.push_back(aaSplitClusterVertexUVs[iSplitCluster]);

                aaiClusterTrianglePositionIndices.push_back(aaiSplitClusterTrianglePositionIndices[iSplitCluster]);
                aaiClusterTriangleNormalIndices.push_back(aaiSplitClusterTriangleNormalIndices[iSplitCluster]);
                aaiClusterTriangleUVIndices.push_back(aaiSplitClusterTriangleUVIndices[iSplitCluster]);
            }

            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());

            iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
            iNumClusterGroups = iNumClusters / 4;

            iCluster = -1;

        }   // if cluster size > threshold

    }   // for cluster = 0 to num clusters
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
void checkClusterGroupBoundaryVertices(
    std::vector<std::vector<uint32_t>>& aaiClusterGroupBoundaryVertices,
    uint32_t iClusterGroup,
    uint32_t iNumClusterGroups,
    std::vector<std::vector<float3>> const& aaClusterGroupVertexPositions,
    std::vector<std::vector<uint32_t>> const& aaiClusterGroupTrianglePositionIndices)
{
    uint32_t iNumClusterGroupTriangleIndices = static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size());
    uint32_t const* paiClusterGroupTriangles = aaiClusterGroupTrianglePositionIndices[iClusterGroup].data();
    float3 const* paClusterGroupVertexPositions = aaClusterGroupVertexPositions[iClusterGroup].data();

    std::vector<std::vector<TriAdjacentInfo>> aaAdjacentTris(iNumClusterGroupTriangleIndices);
    for(uint32_t iTri = 0; iTri < iNumClusterGroupTriangleIndices; iTri += 3)
    {
        for(uint32_t iCheckClusterGroup = 0; iCheckClusterGroup < iNumClusterGroups; iCheckClusterGroup++)
        {
            uint32_t iNumCheckClusterGroupTriangleIndices = static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices[iCheckClusterGroup].size());
            uint32_t const* paiCheckClusterGroupTriangles = aaiClusterGroupTrianglePositionIndices[iCheckClusterGroup].data();
            float3 const* paCheckClusterGroupVertexPositions = aaClusterGroupVertexPositions[iCheckClusterGroup].data();
            for(uint32_t iCheckTri = 0; iCheckTri < iNumCheckClusterGroupTriangleIndices; iCheckTri += 3)
            {
                if(iClusterGroup == iCheckClusterGroup && iTri == iCheckTri)
                {
                    continue;
                }

                // check the number of same vertex positions
                uint32_t aiSamePositionIndices[3] = { 0, 0, 0 };
                uint32_t iNumSamePositions = 0;
                for(uint32_t i = 0; i < 3; i++)
                {
                    uint32_t iPos = paiClusterGroupTriangles[iTri + i];
                    float3 const& pos = paClusterGroupVertexPositions[iPos];
                    for(uint32_t j = 0; j < 3; j++)
                    {
                        uint32_t iCheckPos = paiCheckClusterGroupTriangles[iCheckTri + j];
                        float3 const& checkPos = paCheckClusterGroupVertexPositions[iCheckPos];
                        float3 diff = checkPos - pos;
                        if(length(diff) < 1.0e-6f)
                        {
                            aiSamePositionIndices[i] = 1;
                            ++iNumSamePositions;
                            break;
                        }
                    }
                }

                if(iNumSamePositions >= 2)
                {
                    // edge index based on the same vertex positions
                    uint32_t iEdge = UINT32_MAX;
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

                    TriAdjacentInfo triAdjacentInfo;
                    triAdjacentInfo.miClusterGroup = iClusterGroup;
                    triAdjacentInfo.miEdge = iEdge;
                    triAdjacentInfo.miTriangle = iCheckTri / 3;
                    if(iEdge == 0)
                    {
                        triAdjacentInfo.mEdgePosition0 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri]];
                        triAdjacentInfo.mEdgePosition1 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 1]];
                    }
                    else if(iEdge == 1)
                    {
                        triAdjacentInfo.mEdgePosition0 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri]];
                        triAdjacentInfo.mEdgePosition1 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 2]];
                    }
                    else if(iEdge == 2)
                    {
                        triAdjacentInfo.mEdgePosition0 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 1]];
                        triAdjacentInfo.mEdgePosition1 = paClusterGroupVertexPositions[paiClusterGroupTriangles[iTri + 2]];
                    }

                    aaAdjacentTris[iTri / 3].push_back(triAdjacentInfo);
                }

            }   // for check tri = 0 to num tris in check cluster group

        }   // for check cluster group = 0 to num cluster groups

        uint32_t aiNumAdjacentEdges[3] = { 0, 0, 0 };
        for(auto const& adjacentTri : aaAdjacentTris[iTri / 3])
        {
            if(adjacentTri.miEdge != UINT32_MAX)
            {
                aiNumAdjacentEdges[adjacentTri.miEdge] += 1;
            }
        }

        uint32_t iPos0 = paiClusterGroupTriangles[iTri];
        uint32_t iPos1 = paiClusterGroupTriangles[iTri + 1];
        uint32_t iPos2 = paiClusterGroupTriangles[iTri + 2];

        if(aiNumAdjacentEdges[0] <= 0)
        {
            // mark edge 0 vertices as boundary
            {
                std::lock_guard<std::mutex> lock(gMutex);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos0);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos1);
            }

            DEBUG_PRINTF("\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                paClusterGroupVertexPositions[iPos0].x,
                paClusterGroupVertexPositions[iPos0].y,
                paClusterGroupVertexPositions[iPos0].z,
                paClusterGroupVertexPositions[iPos1].x,
                paClusterGroupVertexPositions[iPos1].y,
                paClusterGroupVertexPositions[iPos1].z);

        }
        else if(aiNumAdjacentEdges[1] <= 0)
        {
            // mark edge 1 vertices as boundary
            {
                std::lock_guard<std::mutex> lock(gMutex);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos0);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos2);
            }

            DEBUG_PRINTF("\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                paClusterGroupVertexPositions[iPos0].x,
                paClusterGroupVertexPositions[iPos0].y,
                paClusterGroupVertexPositions[iPos0].z,
                paClusterGroupVertexPositions[iPos2].x,
                paClusterGroupVertexPositions[iPos2].y,
                paClusterGroupVertexPositions[iPos2].z);
        }
        else if(aiNumAdjacentEdges[2] <= 0)
        {
            // mark edge 1 vertices as boundary
            {
                std::lock_guard<std::mutex> lock(gMutex);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos1);
                aaiClusterGroupBoundaryVertices[iClusterGroup].push_back(iPos2);
            }

            DEBUG_PRINTF("\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n\tdraw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0)\n",
                paClusterGroupVertexPositions[iPos1].x,
                paClusterGroupVertexPositions[iPos1].y,
                paClusterGroupVertexPositions[iPos1].z,
                paClusterGroupVertexPositions[iPos2].x,
                paClusterGroupVertexPositions[iPos2].y,
                paClusterGroupVertexPositions[iPos2].z);
        }

    }   // for tri = 0 to num triangles in cluster group
}

#include <filesystem>
/*
**
*/
void writeTotalClusterOBJ(
    std::string const& outputTotalClusterFilePath,
    std::string const& objectName,
    std::vector<std::vector<float3>> const& aaClusterVertexPositions,
    std::vector<std::vector<float3>> const& aaClusterVertexNormals,
    std::vector<std::vector<float2>> const& aaClusterVertexUVs,
    std::vector<std::vector<uint32_t>> const& aaiClusterTrianglePositionIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleNormalIndices,
    std::vector<std::vector<uint32_t>> const& aaiClusterTriangleUVIndices)
{
    uint32_t iNumTotalVertexPositions = 0;
    uint32_t iNumTotalVertexNormals = 0;
    uint32_t iNumTotalVertexUVs = 0;

    auto directoryEnd = outputTotalClusterFilePath.rfind("\\");
    if(directoryEnd == std::string::npos)
    {
        directoryEnd = outputTotalClusterFilePath.rfind("/");
    }
    assert(directoryEnd != std::string::npos);
    std::string directory = outputTotalClusterFilePath.substr(0, directoryEnd);
    if(!std::filesystem::exists(directory))
    {
        std::filesystem::create_directory(directory);
    }
    
    FILE* fp = fopen(outputTotalClusterFilePath.c_str(), "wb");
    fprintf(fp, "o %s\n", objectName.c_str());
    
    uint32_t iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
    for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
    {
        std::ostringstream objectClusterName;
        objectClusterName << objectName << "-cluster-" << iCluster;
        
        std::ostringstream objectClusterMaterialName;
        objectClusterMaterialName << objectClusterName.str();

        fprintf(fp, "g %s\n", objectClusterName.str().c_str());
        
        {
            std::string materialOutputFullPath = directory + "\\" + objectClusterMaterialName.str() + ".mtl";
            FILE * materialFP = fopen(materialOutputFullPath.c_str(), "wb");
            float fRand0 = static_cast<float>(rand() % 255) / 255.0f;
            float fRand1 = static_cast<float>(rand() % 255) / 255.0f;
            float fRand2 = static_cast<float>(rand() % 255) / 255.0f;
            fprintf(materialFP, "newmtl %s\n", objectClusterMaterialName.str().c_str());
            fprintf(materialFP, "Kd %.4f %.4f %4f\n", fRand0, fRand1, fRand2);
            fprintf(materialFP, "Ka %.4f %.4f %4f\n", fRand0, fRand1, fRand2);
            fprintf(materialFP, "Ks %.4f %.4f %4f\n", fRand0, fRand1, fRand2);
            fprintf(materialFP, "Ke %.4f %.4f %4f\n", fRand0, fRand1, fRand2);
            fclose(materialFP);

            fprintf(fp, "mtllib %s\n", materialOutputFullPath.c_str());
        }

        fprintf(fp, "usemtl %s\n", objectClusterMaterialName.str().c_str());
        fprintf(fp, "# num positions: %d\n", static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size()));
        for(uint32_t iPos = 0; iPos < static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size()); iPos++)
        {
            fprintf(fp, "v %.4f %.4f %.4f\n",
                aaClusterVertexPositions[iCluster][iPos].x,
                aaClusterVertexPositions[iCluster][iPos].y,
                aaClusterVertexPositions[iCluster][iPos].z);
        }

        fprintf(fp, "# num normals: %d\n", static_cast<uint32_t>(aaClusterVertexNormals[iCluster].size()));
        for(uint32_t iNorm = 0; iNorm < static_cast<uint32_t>(aaClusterVertexNormals[iCluster].size()); iNorm++)
        {
            fprintf(fp, "vn %.4f %.4f %.4f\n",
                aaClusterVertexNormals[iCluster][iNorm].x,
                aaClusterVertexNormals[iCluster][iNorm].y,
                aaClusterVertexNormals[iCluster][iNorm].z);
        }

        fprintf(fp, "# num uvs: %d\n", static_cast<uint32_t>(aaClusterVertexUVs[iCluster].size()));
        for(uint32_t iUV = 0; iUV < static_cast<uint32_t>(aaClusterVertexUVs[iCluster].size()); iUV++)
        {
            fprintf(fp, "vt %.4f %.4f\n",
                aaClusterVertexUVs[iCluster][iUV].x,
                aaClusterVertexUVs[iCluster][iUV].y);
        }

        for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); iTri += 3)
        {
            fprintf(fp, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                aaiClusterTrianglePositionIndices[iCluster][iTri] + iNumTotalVertexPositions + 1,
                aaiClusterTriangleUVIndices[iCluster][iTri] + iNumTotalVertexUVs + 1,
                aaiClusterTriangleNormalIndices[iCluster][iTri] + iNumTotalVertexNormals + 1,
                
                aaiClusterTrianglePositionIndices[iCluster][iTri + 1] + iNumTotalVertexPositions + 1,
                aaiClusterTriangleUVIndices[iCluster][iTri + 1] + iNumTotalVertexUVs + 1,
                aaiClusterTriangleNormalIndices[iCluster][iTri + 1] + iNumTotalVertexNormals + 1,
                
                aaiClusterTrianglePositionIndices[iCluster][iTri + 2] + iNumTotalVertexPositions + 1,
                aaiClusterTriangleUVIndices[iCluster][iTri + 2] + iNumTotalVertexUVs + 1,
                aaiClusterTriangleNormalIndices[iCluster][iTri + 2] + iNumTotalVertexNormals + 1);
        }

        iNumTotalVertexPositions += static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size());
        iNumTotalVertexNormals += static_cast<uint32_t>(aaClusterVertexNormals[iCluster].size());
        iNumTotalVertexUVs += static_cast<uint32_t>(aaClusterVertexUVs[iCluster].size());
    }

    fclose(fp);
}



/*
**
*/
int main(char* argv[], int argc)
{
    float result = 0.0f;
    
    // TODO: add support for non-closed manifold meshes
    //       marking bound vertices and not collapse them?

    srand(static_cast<uint32_t>(time(nullptr)));

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> aShapes;
    std::vector<tinyobj::material_t> aMaterials;
    std::vector<std::vector<uint32_t>> aaiAdjacencyList;

    // load initial mesh file
    //std::string fullOBJFilePath = "c:\\Users\\Dingwings\\demo-models\\face-meshlet-test.obj";
    std::string fullOBJFilePath = "c:\\Users\\Dingwings\\demo-models\\guan-yu-4-meshlet-test.obj";
    std::string warnings;
    std::string errors;
    bool bRet = tinyobj::LoadObj(
        &attrib,
        &aShapes,
        &aMaterials,
        &warnings,
        &errors,
        fullOBJFilePath.c_str());

    assert(aShapes.size() == 1);

    std::vector<uint32_t> aiTrianglePositionIndices;
    std::vector<uint32_t> aiTriangleNormalIndices;
    std::vector<uint32_t> aiTriangleTexCoordIndices;
    for(auto const& index : aShapes[0].mesh.indices)
    {
        aiTrianglePositionIndices.push_back(index.vertex_index);
        aiTriangleNormalIndices.push_back(index.normal_index);
        aiTriangleTexCoordIndices.push_back(index.texcoord_index);
    }

    std::vector<float3> aVertexPositions;
    for(uint32_t iV = 0; iV < static_cast<uint32_t>(attrib.vertices.size()); iV += 3)
    {
        float3 position = float3(attrib.vertices[iV], attrib.vertices[iV + 1], attrib.vertices[iV + 2]);
        aVertexPositions.push_back(position);
    }

    std::vector<float3> aVertexNormals;
    for(uint32_t iV = 0; iV < static_cast<uint32_t>(attrib.normals.size()); iV += 3)
    {
        float3 normal = float3(attrib.normals[iV], attrib.normals[iV + 1], attrib.normals[iV + 2]);
        aVertexNormals.push_back(normal);
    }
    std::vector<float2> aVertexTexCoords;
    for(uint32_t iV = 0; iV < static_cast<uint32_t>(attrib.texcoords.size()); iV += 2)
    {
        float2 texCoord = float2(attrib.texcoords[iV], attrib.texcoords[iV + 1]);
        aVertexTexCoords.push_back(texCoord);
    }

    // build metis mesh file 
    //buildMETISMeshFile(
    //    "c:\\Users\\Dingwings\\demo-models\\metis\\output.mesh",
    //    aShapes,
    //    attrib);

    uint32_t const kiMaxTrianglesPerCluster = 128;

    uint32_t iNumLODLevels = 0;
    uint32_t iNumTris = static_cast<uint32_t>(aShapes[0].mesh.indices.size() / 3);
    for(iNumLODLevels = 0;; iNumLODLevels++)
    {
        iNumTris = iNumTris >> 1;
        if(iNumTris <= kiMaxTrianglesPerCluster)
        {
            break;
        }
    }

    iNumLODLevels += 1;

    std::vector<std::vector<MeshCluster>> aaMeshClusters(iNumLODLevels);
    std::vector<std::vector<MeshClusterGroup>> aaMeshClusterGroups(iNumLODLevels);

    // generate initial clusters
    uint32_t iNumClusters = uint32_t(ceilf(float(aiTrianglePositionIndices.size()) / 3.0f) / kiMaxTrianglesPerCluster);
    uint32_t iNumClusterGroups = iNumClusters / 4;
    std::vector<std::vector<float3>> aaClusterVertexPositions(iNumClusters);
    std::vector<std::vector<float3>> aaClusterVertexNormals(iNumClusters);
    std::vector<std::vector<float2>> aaClusterVertexUVs(iNumClusters);
    std::vector<std::vector<uint32_t>> aaiClusterTrianglePositionIndices(iNumClusters);
    std::vector<std::vector<uint32_t>> aaiClusterTriangleNormalIndices(iNumClusters);
    std::vector<std::vector<uint32_t>> aaiClusterTriangleUVIndices(iNumClusters);
    {
        std::ostringstream outputMetisMeshFilePath;
        outputMetisMeshFilePath << "c:\\Users\\Dingwings\\demo-models\\metis\\output";
        outputMetisMeshFilePath << "-lod" << 0 << ".mesh";

auto start = std::chrono::high_resolution_clock::now();

        buildMETISMeshFile2(
            outputMetisMeshFilePath.str(),
            aiTrianglePositionIndices);

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to build cluster Metis mesh file\n", iSeconds);


        assert(iNumClusters > 0);

        // exec the mpmetis to generate the initial clusters
        std::ostringstream metisCommand;
        metisCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
        metisCommand << outputMetisMeshFilePath.str() << " ";
        metisCommand << "-gtype=dual ";
        metisCommand << "-ncommon=2 ";
        metisCommand << "-objtype=vol ";
        //metisCommand << "-ufactor=100 ";
        metisCommand << "-contig ";
        //metisCommand << "-minconn ";
        //metisCommand << "-niter=20 ";
        metisCommand << iNumClusters;
        std::string result = execCommand(metisCommand.str(), false);
        if(result.find("Metis returned with an error.") != std::string::npos)
        {
            metisCommand = std::ostringstream();
            metisCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\mpmetis.exe ";
            metisCommand << outputMetisMeshFilePath.str() << " ";
            metisCommand << "-gtype=dual ";
            metisCommand << "-ncommon=2 ";
            metisCommand << "-objtype=vol ";
            metisCommand << iNumClusters;
            std::string result = execCommand(metisCommand.str(), false);

            assert(result.find("Metis returned with an error.") == std::string::npos);
        }

        // create clusters based on the partition files from the above metis command
        std::ostringstream outputPartitionFilePath;
        //outputPartitionFilePath << "c:\\Users\\Dingwings\\demo-models\\metis\\output.mesh.epart.";
        outputPartitionFilePath << outputMetisMeshFilePath.str() << ".epart.";
        outputPartitionFilePath << iNumClusters;
        
start = std::chrono::high_resolution_clock::now();

        // map of element index to cluster
        std::map<uint32_t, std::vector<uint32_t>> aClusterMap;
        {
            std::vector<uint32_t> aiClusters;
            readMetisClusterFile(aiClusters, outputPartitionFilePath.str());
            for(uint32_t i = 0; i < static_cast<uint32_t>(aiClusters.size()); i++)
            {
                uint32_t const& iCluster = aiClusters[i];
                aClusterMap[iCluster].push_back(i);
            }
        }
        for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
        {
            outputMeshClusters2(
                aaClusterVertexPositions[iCluster],
                aaClusterVertexNormals[iCluster],
                aaClusterVertexUVs[iCluster],
                aaiClusterTrianglePositionIndices[iCluster],
                aaiClusterTriangleNormalIndices[iCluster],
                aaiClusterTriangleUVIndices[iCluster],
                aClusterMap[iCluster],
                aVertexPositions,
                aVertexNormals,
                aVertexTexCoords,
                aiTrianglePositionIndices,
                aiTriangleNormalIndices,
                aiTriangleTexCoordIndices,
                0,
                iCluster);
            assert(aaiClusterTrianglePositionIndices[iCluster].size() % 3 == 0);
            assert(aaiClusterTriangleNormalIndices[iCluster].size() % 3 == 0);
            assert(aaiClusterTriangleUVIndices[iCluster].size() % 3 == 0);

            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
        }
end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to build clusters\n", iSeconds);
    }

    // check cluster validity
    {
        for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
        {
            bool bRestart = false;
            int32_t iNumClusterTri = static_cast<int32_t>(aaiClusterTrianglePositionIndices[iCluster].size());
            for(int32_t iTri = 0; iTri < iNumClusterTri; iTri += 3)
            {
                if(bRestart)
                {
                    iTri = 0;
                    bRestart = false;
                }

                uint32_t iPos0 = aaiClusterTrianglePositionIndices[iCluster][iTri];
                uint32_t iPos1 = aaiClusterTrianglePositionIndices[iCluster][iTri + 1];
                uint32_t iPos2 = aaiClusterTrianglePositionIndices[iCluster][iTri + 2];

                if(iPos0 == iPos1 || iPos0 == iPos2 || iPos1 == iPos2)
                {
                    aaiClusterTrianglePositionIndices[iCluster].erase(aaiClusterTrianglePositionIndices[iCluster].begin() + iTri, aaiClusterTrianglePositionIndices[iCluster].begin() + iTri + 3);
                    aaiClusterTriangleNormalIndices[iCluster].erase(aaiClusterTriangleNormalIndices[iCluster].begin() + iTri, aaiClusterTriangleNormalIndices[iCluster].begin() + iTri + 3);
                    aaiClusterTriangleUVIndices[iCluster].erase(aaiClusterTriangleUVIndices[iCluster].begin() + iTri, aaiClusterTriangleUVIndices[iCluster].begin() + iTri + 3);
                    iNumClusterTri = static_cast<int32_t>(aaiClusterTrianglePositionIndices[iCluster].size());
                    bRestart = true;
                }
            }

            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
        }
    }

    uint32_t const kiMaxTrianglesToSplit = 384;

    DEBUG_PRINTF("start split large clusters\n");
    {
        auto start = std::chrono::high_resolution_clock::now();
        splitLargeClusters(
            aaClusterVertexPositions,
            aaClusterVertexNormals,
            aaClusterVertexUVs,
            aaiClusterTrianglePositionIndices,
            aaiClusterTriangleNormalIndices,
            aaiClusterTriangleUVIndices,
            iNumClusters,
            iNumClusterGroups,
            kiMaxTrianglesToSplit);

        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
        {
            std::ostringstream clusterName;
            clusterName << "cluster-lod0-" << iCluster;

            std::ostringstream outputFilePath;
            outputFilePath << "c:\\Users\\Dingwings\\demo-models\\clusters\\" << clusterName.str() << ".obj";
            writeOBJFile(
                aaClusterVertexPositions[iCluster],
                aaClusterVertexNormals[iCluster],
                aaClusterVertexUVs[iCluster],
                aaiClusterTrianglePositionIndices[iCluster],
                aaiClusterTriangleNormalIndices[iCluster],
                aaiClusterTriangleUVIndices[iCluster],
                outputFilePath.str(),
                clusterName.str());
        }

        uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        DEBUG_PRINTF("took %d seconds to split large clusters\n", iSeconds);
    }

    std::vector<std::vector<uint32_t>> aaiClusterGroupMap;
    std::vector<std::vector<float>> aafClusterGroupErrors;
    std::vector<std::vector<float>> aafClusterAverageTriangleArea(iNumLODLevels);
    std::vector<std::vector<float4>> aaClusterNormalCones(iNumLODLevels);
    std::vector<uint32_t> aiStartClusterGroupIndices;
    
    aiStartClusterGroupIndices.push_back(0);

    uint32_t iTotalMeshClusters = 0;
    uint32_t iTotalMeshClusterGroups = 0;

auto start = std::chrono::high_resolution_clock::now();

    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
    {
auto totalLODStart = std::chrono::high_resolution_clock::now();

        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
        {
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
        }

        {
            std::ostringstream objectName;
            objectName << "total-cluster-lod" << iLODLevel;

            std::ostringstream outputTotalClusterFilePath;
            outputTotalClusterFilePath << "c:\\Users\\Dingwings\\demo-models\\total-clusters\\" << objectName.str();
            outputTotalClusterFilePath << ".obj";
            writeTotalClusterOBJ(
                outputTotalClusterFilePath.str(),
                objectName.str(),
                aaClusterVertexPositions,
                aaClusterVertexNormals,
                aaClusterVertexUVs,
                aaiClusterTrianglePositionIndices,
                aaiClusterTriangleNormalIndices,
                aaiClusterTriangleUVIndices);
        }


start = std::chrono::high_resolution_clock::now();
DEBUG_PRINTF("start average triangle surface area\n");
        // average triangle surface area of individual clusters
        {
            iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
            for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
            {
                aafClusterAverageTriangleArea[iLODLevel].push_back(0.0f);
                bool bRestart = false;
                for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); iTri += 3)
                {
                    if(bRestart)
                    {
                        iTri = 0;
                        bRestart = false;
                    }

                    uint32_t iPos0 = aaiClusterTrianglePositionIndices[iCluster][iTri];
                    uint32_t iPos1 = aaiClusterTrianglePositionIndices[iCluster][iTri + 1];
                    uint32_t iPos2 = aaiClusterTrianglePositionIndices[iCluster][iTri + 2];

                    float3 const& pos0 = aaClusterVertexPositions[iCluster][iPos0];
                    float3 const& pos1 = aaClusterVertexPositions[iCluster][iPos1];
                    float3 const& pos2 = aaClusterVertexPositions[iCluster][iPos2];

                    float3 diff0 = pos1 - pos0;
                    float3 diff1 = pos2 - pos0;
                    float3 diff2 = pos1 - pos2;

                    float const kfDiffThreshold = 1.0e-8f;
                    if((length(diff0) < kfDiffThreshold || length(diff1) < kfDiffThreshold || length(diff2) < kfDiffThreshold) && (iPos0 != iPos1 && iPos0 != iPos2 && iPos1 != iPos2) ||
                        (iPos0 == iPos1 || iPos0 == iPos2 || iPos1 == iPos2))
                    {
                        aaiClusterTrianglePositionIndices[iCluster].erase(
                            aaiClusterTrianglePositionIndices[iCluster].begin() + iTri,
                            aaiClusterTrianglePositionIndices[iCluster].begin() + iTri + 3);
                        
                        aaiClusterTriangleNormalIndices[iCluster].erase(
                            aaiClusterTriangleNormalIndices[iCluster].begin() + iTri,
                            aaiClusterTriangleNormalIndices[iCluster].begin() + iTri + 3);

                        aaiClusterTriangleUVIndices[iCluster].erase(
                            aaiClusterTriangleUVIndices[iCluster].begin() + iTri,
                            aaiClusterTriangleUVIndices[iCluster].begin() + iTri + 3);
                        
                        bRestart = true;
                    }

                    float fArea = length(cross(diff0, diff1)) * 0.5f;
                    aafClusterAverageTriangleArea[iLODLevel][iCluster] += fArea;
                }

                aafClusterAverageTriangleArea[iLODLevel][iCluster] /= static_cast<float>(aaiClusterTrianglePositionIndices[iCluster].size());

                // compute cluster's average normal
                float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
                std::vector<float3> aFaceNormals(aaiClusterTrianglePositionIndices[iCluster].size() / 3);
                for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); iTri += 3)
                {
                    uint32_t iPos0 = aaiClusterTrianglePositionIndices[iCluster][iTri];
                    uint32_t iPos1 = aaiClusterTrianglePositionIndices[iCluster][iTri + 1];
                    uint32_t iPos2 = aaiClusterTrianglePositionIndices[iCluster][iTri + 2];

                    float3 const& pos0 = aaClusterVertexPositions[iCluster][iPos0];
                    float3 const& pos1 = aaClusterVertexPositions[iCluster][iPos1];
                    float3 const& pos2 = aaClusterVertexPositions[iCluster][iPos2];

                    float3 diff0 = pos1 - pos0;
                    float3 diff1 = pos2 - pos0;
                    float3 diff2 = pos1 - pos2;

                    aFaceNormals[iTri / 3] = cross(normalize(diff1), normalize(diff0));
                    avgNormal += aFaceNormals[iTri / 3];
                }
                avgNormal /= static_cast<float>(aaiClusterTrianglePositionIndices[iCluster].size() / 3);
                
                // get the cone radius (min dot product of average normal with triangle normal, ie. greatest cosine angle)
                float fMinDP = FLT_MAX;
                for(auto const& faceNormal : aFaceNormals)
                { 
                    fMinDP = minf(fMinDP, dot(avgNormal, faceNormal));
                }
                aaClusterNormalCones[iLODLevel].push_back(float4(avgNormal, fMinDP));
            }

            for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
            {
                assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
                assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
            }
        }


auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to compute cluster average triangle area and normal cones\n", iSeconds);

        std::ostringstream outputClusterMeshFilePath;
        outputClusterMeshFilePath << "c:\\Users\\Dingwings\\demo-models\\metis\\clusters-lod" << iLODLevel << ".mesh";

        // build a metis graph file with the number of shared vertices as edge weights between clusters
        if(iNumClusterGroups > 1)
        {
start = std::chrono::high_resolution_clock::now();

            buildMETISGraphFile(
                outputClusterMeshFilePath.str(),
                aaClusterVertexPositions);

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to build metis graph file for generating cluster groups\n", iSeconds);

            // generate cluster groups for all the initial clusters
            std::ostringstream clusterGroupCommand;
            clusterGroupCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\gpmetis.exe ";
            clusterGroupCommand << outputClusterMeshFilePath.str() << " ";
            clusterGroupCommand << iNumClusterGroups;
            std::string result = execCommand(clusterGroupCommand.str(), false);
            if(result.find("Metis returned with an error.") != std::string::npos)
            {
                int iDebug = 1;
            }
        }

auto start = std::chrono::high_resolution_clock::now();

        // build cluster groups
        std::vector<std::vector<float3>> aaClusterGroupVertexPositions;
        std::vector<std::vector<float3>> aaClusterGroupVertexNormals;
        std::vector<std::vector<float2>> aaClusterGroupVertexUVs;
        std::vector<std::vector<uint32_t>> aaiClusterGroupTrianglePositionIndices;
        std::vector<std::vector<uint32_t>> aaiClusterGroupTriangleNormalIndices;
        std::vector<std::vector<uint32_t>> aaiClusterGroupTriangleUVIndices;
        std::vector<uint32_t> aiClusterGroupMap;
        buildClusterGroups(
            aaClusterGroupVertexPositions,
            aaClusterGroupVertexNormals,
            aaClusterGroupVertexUVs,
            aaiClusterGroupTrianglePositionIndices,
            aaiClusterGroupTriangleNormalIndices,
            aaiClusterGroupTriangleUVIndices,
            aiClusterGroupMap,
            aaClusterVertexPositions,
            aaClusterVertexNormals,
            aaClusterVertexUVs,
            aaiClusterTrianglePositionIndices,
            aaiClusterTriangleNormalIndices,
            aaiClusterTriangleUVIndices,
            "c:\\Users\\Dingwings\\demo-models\\metis",
            iNumClusterGroups,
            iNumClusters,
            iLODLevel,
            outputClusterMeshFilePath.str());

        uint32_t iLastClusterGroupIndex = aiStartClusterGroupIndices.back();
        aiStartClusterGroupIndices.push_back(static_cast<uint32_t>(iNumClusterGroups + iLastClusterGroupIndex));

        std::map<uint32_t, uint32_t> aClusterGroupMapCount;
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aiClusterGroupMap.size()); iCluster++)
        {
            uint32_t iClusterGroup = aiClusterGroupMap[iCluster];
            if(aClusterGroupMapCount.find(iClusterGroup) == aClusterGroupMapCount.end())
            {
                aClusterGroupMapCount[iClusterGroup] = 0;
            }

            aClusterGroupMapCount[iClusterGroup] += 1;
        }

        aaiClusterGroupMap.push_back(aiClusterGroupMap);

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to build cluster groups\n", iSeconds);

start = std::chrono::high_resolution_clock::now();
        // build mesh clusters
        uint32_t iLastTotalMeshClusters = iTotalMeshClusters;
        std::map<uint32_t, std::vector<uint32_t>> aClusterGroupToClusterMap;
        for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
        {
            // parent mesh cluster will be cluster group index * 2
            aaMeshClusters[iLODLevel].emplace_back(
                giTotalVertexPositionDataOffset,
                giTotalVertexNormalDataOffset,
                giTotalVertexUVDataOffset,
                giTotalTrianglePositionIndexDataOffset,
                giTotalTriangleNormalIndexDataOffset,
                giTotalTriangleUVIndexDataOffset,
                static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size()),
                static_cast<uint32_t>(aaClusterVertexNormals[iCluster].size()),
                static_cast<uint32_t>(aaClusterVertexUVs[iCluster].size()),
                static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()),
                aiClusterGroupMap[iCluster],
                iLODLevel,
                iTotalMeshClusters,
                iTotalMeshClusterGroups);

            // copy vertex positions, normals, uvs, and triangle indices their respective buffers
            // position
            uint64_t iDataSize = aaClusterVertexPositions[iCluster].size();
            if(vertexPositionBuffer.size() <= giTotalVertexPositionDataOffset * sizeof(float3) + iDataSize * sizeof(float3))
            {
                vertexPositionBuffer.resize(vertexPositionBuffer.size() * 2);
                assert(vertexPositionBuffer.size() > giTotalVertexPositionDataOffset * sizeof(float3) + iDataSize * sizeof(float3));
            }
            memcpy(
                vertexPositionBuffer.data() + giTotalVertexPositionDataOffset * sizeof(float3),
                aaClusterVertexPositions[iCluster].data(),
                iDataSize * sizeof(float3));
            giTotalVertexPositionDataOffset += iDataSize;

            // normal
            iDataSize = aaClusterVertexNormals[iCluster].size();
            if(vertexNormalBuffer.size() <= giTotalVertexPositionDataOffset * sizeof(float3) + iDataSize * sizeof(float3))
            {
                vertexNormalBuffer.resize(vertexNormalBuffer.size() * 2);
                assert(vertexNormalBuffer.size() > giTotalVertexPositionDataOffset * sizeof(float3) + iDataSize * sizeof(float3));
            }
            memcpy(
                vertexNormalBuffer.data() + giTotalVertexNormalDataOffset * sizeof(float3),
                aaClusterVertexNormals[iCluster].data(),
                iDataSize * sizeof(float3));
            giTotalVertexNormalDataOffset += iDataSize;

            // uv
            iDataSize = aaClusterVertexUVs[iCluster].size();
            if(vertexUVBuffer.size() <= giTotalVertexUVDataOffset * sizeof(float2) + iDataSize * sizeof(float2))
            {
                vertexUVBuffer.resize(vertexUVBuffer.size() * 2);
                assert(vertexUVBuffer.size() > giTotalVertexUVDataOffset * sizeof(float2) + iDataSize * sizeof(float2));
            }
            memcpy(
                vertexUVBuffer.data() + giTotalVertexUVDataOffset * sizeof(float2),
                aaClusterVertexUVs[iCluster].data(),
                iDataSize * sizeof(float2));
            giTotalVertexUVDataOffset += iDataSize;

            // position indices
            iDataSize = aaiClusterTrianglePositionIndices[iCluster].size();
            if(trianglePositionIndexBuffer.size() <= giTotalTrianglePositionIndexDataOffset * sizeof(uint32_t) + iDataSize * sizeof(uint32_t))
            {
                trianglePositionIndexBuffer.resize(trianglePositionIndexBuffer.size() * 2);
                assert(trianglePositionIndexBuffer.size() > giTotalTrianglePositionIndexDataOffset * sizeof(uint32_t) + iDataSize * sizeof(uint32_t));
            }
            memcpy(
                trianglePositionIndexBuffer.data() + giTotalTrianglePositionIndexDataOffset * sizeof(uint32_t),
                aaiClusterTrianglePositionIndices[iCluster].data(),
                iDataSize * sizeof(uint32_t));
            giTotalTrianglePositionIndexDataOffset += iDataSize;

            // normal indices
            iDataSize = aaiClusterTriangleNormalIndices[iCluster].size();
            if(triangleNormalIndexBuffer.size() <= giTotalTriangleNormalIndexDataOffset * sizeof(uint32_t) + iDataSize * sizeof(uint32_t))
            {
                triangleNormalIndexBuffer.resize(triangleNormalIndexBuffer.size() * 2);
                assert(triangleNormalIndexBuffer.size() > giTotalTriangleNormalIndexDataOffset * sizeof(uint32_t) + iDataSize * sizeof(uint32_t));
            }
            memcpy(
                triangleNormalIndexBuffer.data() + giTotalTriangleNormalIndexDataOffset * sizeof(uint32_t),
                aaiClusterTriangleNormalIndices[iCluster].data(),
                iDataSize * sizeof(uint32_t));
            giTotalTriangleNormalIndexDataOffset += iDataSize;

            // uv indices
            iDataSize = aaiClusterTriangleUVIndices[iCluster].size();
            if(triangleUVIndexBuffer.size() <= giTotalTriangleUVIndexDataOffset * sizeof(uint32_t) + iDataSize * sizeof(uint32_t))
            {
                triangleUVIndexBuffer.resize(triangleUVIndexBuffer.size() * 2);
                assert(triangleUVIndexBuffer.size() > giTotalTriangleUVIndexDataOffset * sizeof(uint32_t) + iDataSize * sizeof(uint32_t));
            }
            memcpy(
                triangleUVIndexBuffer.data() + giTotalTriangleUVIndexDataOffset * sizeof(uint32_t),
                aaiClusterTriangleUVIndices[iCluster].data(),
                iDataSize * sizeof(uint32_t));
            giTotalTriangleUVIndexDataOffset += iDataSize;

            aClusterGroupToClusterMap[aiClusterGroupMap[iCluster]].push_back(iCluster);
            ++iTotalMeshClusters;
        }

        // verify vertex positions
        for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
        {
            for(uint32_t iV = 0; iV < aaMeshClusters[iLODLevel][iCluster].miNumVertexPositions; iV++)
            {
                auto const& pos = aaClusterVertexPositions[iCluster][iV];

                uint64_t iAbsoluteAddress = aaMeshClusters[iLODLevel][iCluster].miVertexPositionStartAddress * sizeof(float3);
                auto const& checkPos = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + iAbsoluteAddress)[iV];

                float3 diff = pos - checkPos;
                assert(length(diff) < 1.0e-6f);
            }
        }

        // set as the MIP 0 of the new cluster groups
        for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
        {
            aaMeshClusterGroups[iLODLevel].emplace_back(
                aClusterGroupToClusterMap[iClusterGroup],
                iLODLevel,
                0,
                iTotalMeshClusterGroups,
                iLastTotalMeshClusters);

            ++iTotalMeshClusterGroups;
        }

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to pack cluster group data\n", iSeconds);

start = std::chrono::high_resolution_clock::now();

        // get cluster boundary vertices
        std::vector<std::vector<uint32_t>> aaiClusterGroupBoundaryVertices;
        std::vector<std::vector<uint32_t>> aaiClusterGroupNonBoundaryVertices;
        getClusterGroupBoundaryVertices(
            aaiClusterGroupBoundaryVertices,
            aaiClusterGroupNonBoundaryVertices,
            aaClusterGroupVertexPositions,
            iNumClusterGroups);

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %d seconds to get cluster group boundary vertices\n", iSeconds);
   
#if 0
        {
            PrintOptions printOption;
            printOption.mbDisplayTime = false;
            setPrintOptions(printOption);

            uint32_t const kiMaxThreads = 8;
            std::unique_ptr<std::thread> apThreads[kiMaxThreads];
            std::atomic<uint32_t> iCurrClusterGroup{ 0 };
            for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
            {
                apThreads[iThread] = std::make_unique<std::thread>(
                    [&iCurrClusterGroup,
                     &aaiClusterGroupBoundaryVertices,
                    iNumClusterGroups,
                    aaClusterGroupVertexPositions,
                    aaiClusterGroupTrianglePositionIndices,
                     iThread]()
                    {
                        for(;;)
                        {
                            uint32_t iThreadClusterGroup = iCurrClusterGroup.fetch_add(1);
                            if(iThreadClusterGroup >= iNumClusterGroups)
                            {
                                return;
                            }

                            checkClusterGroupBoundaryVertices(
                                aaiClusterGroupBoundaryVertices,
                                iThreadClusterGroup,
                                iNumClusterGroups,
                                aaClusterGroupVertexPositions,
                                aaiClusterGroupTrianglePositionIndices);
                        }
                    });
            }

            for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
            {
                if(apThreads[iThread]->joinable())
                {
                    apThreads[iThread]->join();
                }
            }

            printOption.mbDisplayTime = true;
            setPrintOptions(printOption);
        }
#endif // #if 0


start = std::chrono::high_resolution_clock::now();

        // get inner edges of all the cluster groups
        std::vector<std::vector<uint32_t>> aaiValidClusterGroupEdges(iNumClusterGroups);
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> aaValidClusterGroupEdgePairs(iNumClusterGroups);
        std::vector<std::map<uint32_t, uint32_t>> aaValidVerticesFlags(iNumClusterGroups);
        std::vector<std::vector<uint32_t>> aaiClusterGroupTrisWithEdges(iNumClusterGroups);
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> aaClusterGroupEdges(iNumClusterGroups);
        getInnerEdgesAndVertices(
            aaiValidClusterGroupEdges,
            aaValidClusterGroupEdgePairs,
            aaValidVerticesFlags,
            aaiClusterGroupTrisWithEdges,
            aaClusterGroupEdges,
            aaiClusterGroupTrianglePositionIndices,
            aaiClusterGroupNonBoundaryVertices,
            iNumClusterGroups);

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %d seconds to get inner edges and vertices\n", iSeconds);

        // check if edges are not adjacent to anything
        std::vector<BoundaryEdgeInfo> aBoundaryEdges;


        
start = std::chrono::high_resolution_clock::now();

        // delete boundary edges from collapse candidates 
        std::vector<std::pair<uint32_t, uint32_t>> aBoundaryVertices;
        for(auto const& boundaryEdgeInfo : aBoundaryEdges)
        {
            auto boundaryEdgeIter = std::find_if(
                aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].begin(),
                aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].end(),
                [boundaryEdgeInfo](std::pair<uint32_t, uint32_t> const& checkEdge)
                {
                    return (boundaryEdgeInfo.miPos0 == checkEdge.first && boundaryEdgeInfo.miPos1 == checkEdge.second) || (boundaryEdgeInfo.miPos1 == checkEdge.first && boundaryEdgeInfo.miPos0 == checkEdge.second);
                }
            );

            std::pair<uint32_t, uint32_t> pair0 = std::make_pair(boundaryEdgeInfo.miClusterGroup, boundaryEdgeInfo.miPos0);
            std::pair<uint32_t, uint32_t> pair1 = std::make_pair(boundaryEdgeInfo.miClusterGroup, boundaryEdgeInfo.miPos1);

            auto checkIter0 = std::find(
                aBoundaryVertices.begin(),
                aBoundaryVertices.end(),
                pair0);
            if(checkIter0 == aBoundaryVertices.end())
            {
                aBoundaryVertices.push_back(pair0);
            }

            auto checkIter1 = std::find(
                aBoundaryVertices.begin(),
                aBoundaryVertices.end(),
                pair1);
            if(checkIter1 == aBoundaryVertices.end())
            {
                aBoundaryVertices.push_back(pair1);
            }

            if(boundaryEdgeIter != aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].end())
            {
                DEBUG_PRINTF("delete from cluster group %d edge (%d, %d)\n",
                    boundaryEdgeInfo.miClusterGroup,
                    boundaryEdgeInfo.miPos0,
                    boundaryEdgeInfo.miPos1);
                aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].erase(boundaryEdgeIter);
            }
        }

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%d seconds to delete boundary edges\n", iSeconds);

#if 0
        // cuda simplify cluster groups
        std::vector<float> afErrors(iNumClusterGroups);
        {
            auto start0 = std::chrono::high_resolution_clock::now();
            std::vector<std::map<uint32_t, mat4>> aaQuadrics(iNumClusterGroups);
            float fTotalError = 0.0f;
            for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
            {
                uint32_t iMaxTriangles = static_cast<uint32_t>(static_cast<float>(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size()) * 0.5f);
                simplifyClusterGroup(
                    aaQuadrics[iClusterGroup],
                    aaClusterGroupVertexPositions[iClusterGroup],
                    aaClusterGroupVertexNormals[iClusterGroup],
                    aaClusterGroupVertexUVs[iClusterGroup],
                    aaiClusterGroupNonBoundaryVertices[iClusterGroup],
                    aaiClusterGroupBoundaryVertices[iClusterGroup],
                    aaiClusterGroupTrianglePositionIndices[iClusterGroup],
                    aaiClusterGroupTriangleNormalIndices[iClusterGroup],
                    aaiClusterGroupTriangleUVIndices[iClusterGroup],
                    aaValidClusterGroupEdgePairs[iClusterGroup],
                    fTotalError,
                    aBoundaryVertices,
                    iMaxTriangles,
                    iClusterGroup,
                    iLODLevel);
                auto end0 = std::chrono::high_resolution_clock::now();
                uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end0 - start0).count();
                if(iClusterGroup % 10 == 0)
                {
                    DEBUG_PRINTF("took %d seconds to simplify cluster group %d of %d\n",
                        iSeconds,
                        iClusterGroup,
                        iNumClusterGroups);
                }
            }

        }   //   cuda simplify cluster group
#endif // #if 0

        std::vector<std::map<uint32_t, mat4>> aaQuadrics(iNumClusterGroups);
        std::vector<float> afErrors(iNumClusterGroups);
        float fTotalError = 0.0f;
        
start = std::chrono::high_resolution_clock::now();

        uint32_t const kiMaxThreads = 8;
        std::unique_ptr<std::thread> apThreads[kiMaxThreads];
        std::atomic<uint32_t> iCurrClusterGroup{ 0 };
        for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
        {
            apThreads[iThread] = std::make_unique<std::thread>(
                [&iCurrClusterGroup,
                &aaQuadrics,
                &aaClusterGroupVertexPositions,
                &aaClusterGroupVertexNormals,
                &aaClusterGroupVertexUVs,
                &aaiClusterGroupNonBoundaryVertices,
                &aaiClusterGroupBoundaryVertices,
                &aaiClusterGroupTrianglePositionIndices,
                &aaiClusterGroupTriangleNormalIndices,
                &aaiClusterGroupTriangleUVIndices,
                &aaValidClusterGroupEdgePairs,
                &fTotalError,
                &afErrors,
                aBoundaryVertices,
                iLODLevel,
                iNumClusterGroups,
                start,
                iThread]()
                {
                    for(;;)
                    {
auto clusterGroupStart = std::chrono::high_resolution_clock::now();
                        uint32_t iThreadClusterGroup = iCurrClusterGroup.fetch_add(1);
                        if(iThreadClusterGroup >= iNumClusterGroups)
                        {
                            break;
                        }

                        assert(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size() == aaiClusterGroupTriangleNormalIndices[iThreadClusterGroup].size());
                        assert(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size() == aaiClusterGroupTriangleUVIndices[iThreadClusterGroup].size());

                        uint32_t iMaxTriangles = static_cast<uint32_t>(static_cast<float>(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size()) * 0.5f);
                        simplifyClusterGroup(
                            aaQuadrics[iThreadClusterGroup],
                            aaClusterGroupVertexPositions[iThreadClusterGroup],
                            aaClusterGroupVertexNormals[iThreadClusterGroup],
                            aaClusterGroupVertexUVs[iThreadClusterGroup],
                            aaiClusterGroupNonBoundaryVertices[iThreadClusterGroup],
                            aaiClusterGroupBoundaryVertices[iThreadClusterGroup],
                            aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup],
                            aaiClusterGroupTriangleNormalIndices[iThreadClusterGroup],
                            aaiClusterGroupTriangleUVIndices[iThreadClusterGroup],
                            aaValidClusterGroupEdgePairs[iThreadClusterGroup],
                            fTotalError,
                            aBoundaryVertices,
                            iMaxTriangles,
                            iThreadClusterGroup,
                            iLODLevel);
                        afErrors[iThreadClusterGroup] = fTotalError;

                        assert(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size() == aaiClusterGroupTriangleNormalIndices[iThreadClusterGroup].size());
                        assert(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size() == aaiClusterGroupTriangleUVIndices[iThreadClusterGroup].size());

auto clusterGroupEnd = std::chrono::high_resolution_clock::now();
uint64_t iClusterGroupMilliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(clusterGroupEnd - clusterGroupStart).count();
uint64_t iTotalSeconds = std::chrono::duration_cast<std::chrono::seconds>(clusterGroupEnd - start).count();
if(iThreadClusterGroup % 10 == 0)
{
    DEBUG_PRINTF("took %d milliseconds (total: %d secs) to simplify cluster group %d of cluster groups %d\n", 
        iClusterGroupMilliSeconds, 
        iTotalSeconds,
        iThreadClusterGroup, 
        iNumClusterGroups);
}
                    }
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

        aafClusterGroupErrors.push_back(afErrors);

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %d seconds to simplify cluster groups\n", iSeconds);

        // set the error for each clusters
        if(iLODLevel > 0)
        {
            for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
            {
                uint32_t iClusterGroup = iCluster / 2;
                if(iClusterGroup < aafClusterGroupErrors[iLODLevel - 1].size())
                {
                    aaMeshClusters[iLODLevel][iCluster].mfError = aafClusterGroupErrors[iLODLevel - 1][iClusterGroup];
                }
            }
        }

        // clear old cluster data
        {
            for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexPositions.size()); i++)
            {
                aaClusterVertexPositions[i].clear();
            }
            aaClusterVertexPositions.clear();

            for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterTrianglePositionIndices.size()); i++)
            {
                aaiClusterTrianglePositionIndices[i].clear();
            }
            aaiClusterTrianglePositionIndices.clear();

            for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexNormals.size()); i++)
            {
                aaClusterVertexNormals[i].clear();
            }
            aaClusterVertexNormals.clear();

            for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterTriangleNormalIndices.size()); i++)
            {
                aaiClusterTriangleNormalIndices[i].clear();
            }
            aaiClusterTriangleNormalIndices.clear();

            for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexUVs.size()); i++)
            {
                aaClusterVertexUVs[i].clear();
            }
            aaClusterVertexUVs.clear();

            for(uint32_t i = 0; i < static_cast<uint32_t>(aaiClusterTriangleUVIndices.size()); i++)
            {
                aaiClusterTriangleUVIndices[i].clear();
            }
            aaiClusterTriangleUVIndices.clear();

        }   // clear old cluster data

        // split cluster groups
        {
auto start = std::chrono::high_resolution_clock::now();

            uint32_t iNumClusters = 0;
            uint32_t iTotalClusterIndex = 0;
            bool bResetLoop = false;
            uint32_t iLastClusterSize = 0;
            for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices.size()); iClusterGroup++)
            {
                if(bResetLoop)
                {
                    iClusterGroup = 0;
                    bResetLoop = false;
                }

                auto const& aClusterGroupVertexPositions = aaClusterGroupVertexPositions[iClusterGroup];
                auto const& aiClusterGroupTrianglePositionIndices = aaiClusterGroupTrianglePositionIndices[iClusterGroup];

                auto const& aClusterGroupVertexNormals = aaClusterGroupVertexNormals[iClusterGroup];
                auto const& aiClusterGroupTriangleNormalIndices = aaiClusterGroupTriangleNormalIndices[iClusterGroup];

                auto const& aClusterGroupVertexUVs = aaClusterGroupVertexUVs[iClusterGroup];
                auto const& aiClusterGroupTriangleUVIndices = aaiClusterGroupTriangleUVIndices[iClusterGroup];

                if(aiClusterGroupTrianglePositionIndices.size() <= 0)
                {
                    aaClusterGroupVertexPositions.erase(aaClusterGroupVertexPositions.begin() + iClusterGroup);
                    aaClusterGroupVertexNormals.erase(aaClusterGroupVertexNormals.begin() + iClusterGroup);
                    aaClusterGroupVertexUVs.erase(aaClusterGroupVertexUVs.begin() + iClusterGroup);

                    aaiClusterGroupTrianglePositionIndices.erase(aaiClusterGroupTrianglePositionIndices.begin() + iClusterGroup);
                    aaiClusterGroupTriangleNormalIndices.erase(aaiClusterGroupTriangleNormalIndices.begin() + iClusterGroup);
                    aaiClusterGroupTriangleUVIndices.erase(aaiClusterGroupTriangleUVIndices.begin() + iClusterGroup);

                    bResetLoop = true;
                    continue;
                }

                uint32_t iNumSplitClusters = (aaiClusterGroupTrianglePositionIndices.size() <= 1 && aaiClusterGroupTrianglePositionIndices[0].size() / 3 <= kiMaxTrianglesPerCluster) ? 1 : 2;
                uint32_t iPrevNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
                splitClusterGroups(
                    aaClusterVertexPositions,
                    aaClusterVertexNormals,
                    aaClusterVertexUVs,
                    aaiClusterTrianglePositionIndices,
                    aaiClusterTriangleNormalIndices,
                    aaiClusterTriangleUVIndices,
                    iTotalClusterIndex,
                    aClusterGroupVertexPositions,
                    aClusterGroupVertexNormals,
                    aClusterGroupVertexUVs,
                    aiClusterGroupTrianglePositionIndices,
                    aiClusterGroupTriangleNormalIndices,
                    aiClusterGroupTriangleUVIndices,
                    kiMaxTrianglesPerCluster,
                    iNumSplitClusters,
                    iLODLevel,
                    iClusterGroup);
                uint32_t iCurrNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());

                // add the split clusters into the cluster group
                uint32_t iNumNewlySplitClusters = iCurrNumClusters - iPrevNumClusters;
                assert(iNumNewlySplitClusters < MAX_CLUSTERS_IN_GROUP);
                aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] = (aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] > 20) ? 0 : aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1];
                for(uint32_t i = 0; i < iNumNewlySplitClusters; i++)
                {
                    aaMeshClusterGroups[iLODLevel][iClusterGroup].maiClusters[1][i] = iTotalMeshClusters + iNumClusters + i;
                    assert(aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] < MAX_CLUSTERS_IN_GROUP);
                    aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] += 1;
                }

                DEBUG_PRINTF("*** add cluster %d to %d to cluster group %d lod %d\n",
                    iPrevNumClusters,
                    iCurrNumClusters,
                    iClusterGroup,
                    iLODLevel);

                // set the parents of the previous clusters from the split clusters
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[0]); i++)
                {
                    uint32_t iClusterID = aaMeshClusterGroups[iLODLevel][iClusterGroup].maiClusters[0][i];
                    auto iter = std::find_if(
                        aaMeshClusters[iLODLevel].begin(),
                        aaMeshClusters[iLODLevel].end(),
                        [iClusterID](MeshCluster const& checkCluster)
                        {
                            return checkCluster.miIndex == iClusterID;
                        }
                    );
                    assert(iter != aaMeshClusters[iLODLevel].end());
                    for(uint32_t i = 0; i < iNumNewlySplitClusters; i++)
                    {
                        assert(iter->miNumParentClusters < MAX_PARENT_CLUSTERS);
                        iter->maiParentClusters[iter->miNumParentClusters + i] = iTotalMeshClusters + iNumClusters + i;
                        ++iter->miNumParentClusters;

                        DEBUG_PRINTF("set cluster %d as parent for cluster %d (%d)\n",
                            iTotalMeshClusters + iNumClusters + i,
                            iter->miIndex,
                            iter->miNumParentClusters - 1);
                    }
                }

                ++aaMeshClusterGroups[iLODLevel][iClusterGroup].miNumMIPS;
                iNumClusters = iTotalClusterIndex;

                iLastClusterSize = static_cast<uint32_t>(aaClusterGroupVertexPositions.size());

            }   // for cluster group = 0 to num cluster groups

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %d seconds to split cluster groups\n", iSeconds);

        }   // split cluster groups

        // delete de-generate cluster with no vertex positions
        bool bReset = false;
        for(uint32_t i = 0; i < static_cast<uint32_t>(aaClusterVertexPositions.size()); i++)
        {
            if(bReset)
            {
                i = 0;
                bReset = false;
            }

            if(aaClusterVertexPositions[i].size() <= 0)
            {
                DEBUG_PRINTF("!!! delete cluster %d with %d vertex positions !!!\n",
                    i,
                    static_cast<uint32_t>(aaClusterVertexPositions[i].size()));

                aaClusterVertexPositions.erase(aaClusterVertexPositions.begin() + i);
                aaClusterVertexNormals.erase(aaClusterVertexNormals.begin() + i);
                aaClusterVertexUVs.erase(aaClusterVertexUVs.begin() + i);

                aaiClusterTrianglePositionIndices.erase(aaiClusterTrianglePositionIndices.begin() + i);
                aaiClusterTriangleNormalIndices.erase(aaiClusterTriangleNormalIndices.begin() + i);
                aaiClusterTriangleUVIndices.erase(aaiClusterTriangleUVIndices.begin() + i);

                bReset = true;
            }
        }

        iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());
        iNumClusterGroups = static_cast<uint32_t>(ceilf(float(iNumClusters) / 4.0f));

auto totalLODEnd = std::chrono::high_resolution_clock::now();
uint64_t iTotalLODSeconds = std::chrono::duration_cast<std::chrono::seconds>(totalLODEnd - totalLODStart).count();
DEBUG_PRINTF("took total %d seconds for lod %d\n", iTotalLODSeconds, iLODLevel);
int iDebug = 1;

    }   // for lod = 0 to num lod levels

    // last mesh cluster
    {
        uint32_t iLastTotalMeshClusters = iTotalMeshClusters;
        std::map<uint32_t, std::vector<uint32_t>> aClusterGroupToClusterMap;
        for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
        {
            std::vector<uint32_t> aiClusterGroupMap(iNumClusters);
            aiClusterGroupMap[0] = 0;

            // parent mesh cluster will be cluster group index * 2
            aaMeshClusters[iNumLODLevels - 1].emplace_back(
                giTotalVertexPositionDataOffset,
                giTotalVertexNormalDataOffset,
                giTotalVertexUVDataOffset,
                giTotalTrianglePositionIndexDataOffset,
                giTotalTriangleNormalIndexDataOffset,
                giTotalTriangleUVIndexDataOffset,
                static_cast<uint32_t>(aaClusterVertexPositions[iCluster].size()),
                static_cast<uint32_t>(aaClusterVertexNormals[iCluster].size()),
                static_cast<uint32_t>(aaClusterVertexUVs[iCluster].size()),
                static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()),
                aiClusterGroupMap[iCluster],
                iNumLODLevels - 1,
                iTotalMeshClusters,
                iTotalMeshClusterGroups - 1);

            // compute cluster's average normal
            float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
            std::vector<float3> aFaceNormals(aaiClusterTrianglePositionIndices[iCluster].size() / 3);
            for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTrianglePositionIndices[iCluster].size()); iTri += 3)
            {
                uint32_t iPos0 = aaiClusterTrianglePositionIndices[iCluster][iTri];
                uint32_t iPos1 = aaiClusterTrianglePositionIndices[iCluster][iTri + 1];
                uint32_t iPos2 = aaiClusterTrianglePositionIndices[iCluster][iTri + 2];

                float3 const& pos0 = aaClusterVertexPositions[iCluster][iPos0];
                float3 const& pos1 = aaClusterVertexPositions[iCluster][iPos1];
                float3 const& pos2 = aaClusterVertexPositions[iCluster][iPos2];

                float3 diff0 = pos1 - pos0;
                float3 diff1 = pos2 - pos0;

                aFaceNormals[iTri / 3] = cross(normalize(diff1), normalize(diff0));
                avgNormal += aFaceNormals[iTri / 3];
            }
            avgNormal /= static_cast<float>(aaiClusterTrianglePositionIndices[iCluster].size() / 3);

            // get the cone radius (min dot product of average normal with triangle normal, ie. greatest cosine angle)
            float fMinDP = FLT_MAX;
            for(auto const& faceNormal : aFaceNormals)
            {
                fMinDP = minf(fMinDP, dot(avgNormal, faceNormal));
            }
            aaClusterNormalCones[iNumLODLevels - 1].push_back(float4(avgNormal, fMinDP));


            // copy vertex positions and triangle indices their respective buffers
            uint64_t iDataSize = aaClusterVertexPositions[iCluster].size();
            memcpy(
                vertexPositionBuffer.data() + giTotalVertexPositionDataOffset * sizeof(float3),
                aaClusterVertexPositions[iCluster].data(),
                iDataSize * sizeof(float3));
            giTotalVertexPositionDataOffset += iDataSize;

            // normal
            iDataSize = aaClusterVertexNormals[iCluster].size();
            memcpy(
                vertexNormalBuffer.data() + giTotalVertexNormalDataOffset * sizeof(float3),
                aaClusterVertexNormals[iCluster].data(),
                iDataSize * sizeof(float3));
            giTotalVertexNormalDataOffset += iDataSize;

            // uv
            iDataSize = aaClusterVertexUVs[iCluster].size();
            memcpy(
                vertexUVBuffer.data() + giTotalVertexUVDataOffset * sizeof(float3),
                aaClusterVertexUVs[iCluster].data(),
                iDataSize * sizeof(float2));
            giTotalVertexUVDataOffset += iDataSize;

            // position indices
            iDataSize = aaiClusterTrianglePositionIndices[iCluster].size();
            memcpy(
                trianglePositionIndexBuffer.data() + giTotalTrianglePositionIndexDataOffset * sizeof(uint32_t),
                aaiClusterTrianglePositionIndices[iCluster].data(),
                iDataSize * sizeof(uint32_t));
            giTotalTrianglePositionIndexDataOffset += iDataSize;

            // normal indices
            iDataSize = aaiClusterTriangleNormalIndices[iCluster].size();
            memcpy(
                triangleNormalIndexBuffer.data() + giTotalTriangleNormalIndexDataOffset * sizeof(uint32_t),
                aaiClusterTriangleNormalIndices[iCluster].data(),
                iDataSize * sizeof(uint32_t));
            giTotalTriangleNormalIndexDataOffset += iDataSize;

            // uv indices
            iDataSize = aaiClusterTriangleUVIndices[iCluster].size();
            memcpy(
                triangleUVIndexBuffer.data() + giTotalTriangleUVIndexDataOffset * sizeof(uint32_t),
                aaiClusterTriangleUVIndices[iCluster].data(),
                iDataSize * sizeof(uint32_t));
            giTotalTriangleUVIndexDataOffset += iDataSize;

            aClusterGroupToClusterMap[aiClusterGroupMap[iCluster]].push_back(iCluster);
            ++iTotalMeshClusters;
        }
    }

    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
    {
        for(uint32_t iMeshCluster = 0; iMeshCluster < static_cast<uint32_t>(aaMeshClusters[iLODLevel].size()); iMeshCluster++)
        {
            aaMeshClusters[iLODLevel][iMeshCluster].mNormalCone = aaClusterNormalCones[iLODLevel][iMeshCluster];

            // get min and max bounds
            MeshCluster& meshCluster = aaMeshClusters[iLODLevel][iMeshCluster];
            float3 const* pClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshCluster.miVertexPositionStartAddress * sizeof(float3));
            float3 minBounds(FLT_MAX, FLT_MAX, FLT_MAX);
            float3 maxBounds(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for(uint32_t iPos = 0; iPos < meshCluster.miNumVertexPositions; iPos++)
            {
                minBounds = fminf(minBounds, pClusterVertexPositions[iPos]);
                maxBounds = fmaxf(maxBounds, pClusterVertexPositions[iPos]);
            }

            meshCluster.mMinBounds = minBounds;
            meshCluster.mMaxBounds = maxBounds;
            meshCluster.mCenter = (meshCluster.mMaxBounds + meshCluster.mMinBounds) * 0.5f;
        }
    }

    // copy mesh clusters and mesh cluster groups into contiguous list
    std::vector<MeshCluster*> apTotalMeshClusters;
    std::vector<MeshClusterGroup*> apTotalMeshClusterGroups;
    {
        for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
        {
            for(uint32_t iMeshCluster = 0; iMeshCluster < static_cast<uint32_t>(aaMeshClusters[iLODLevel].size()); iMeshCluster++)
            {
                apTotalMeshClusters.push_back(&aaMeshClusters[iLODLevel][iMeshCluster]);
            }

            for(uint32_t iMeshClusterGroup = 0; iMeshClusterGroup < static_cast<uint32_t>(aaMeshClusterGroups[iLODLevel].size()); iMeshClusterGroup++)
            {
                apTotalMeshClusterGroups.push_back(&aaMeshClusterGroups[iLODLevel][iMeshClusterGroup]);
            }
        }
    }

    // final check for checking cluster groups to cluster
    {
        for(auto* pCluster : apTotalMeshClusters)
        {
            assert(pCluster->miNumClusterGroups == 0);
            for(auto const* pClusterGroup : apTotalMeshClusterGroups)
            {
                for(uint32_t iMIP = 0; iMIP < 2; iMIP++)
                {
                    for(uint32_t i = 0; i < pClusterGroup->maiNumClusters[iMIP]; i++)
                    {
                        if(pClusterGroup->maiClusters[iMIP][i] == pCluster->miIndex)
                        {
                            assert(pCluster->miNumClusterGroups < MAX_ASSOCIATED_GROUPS);
                            pCluster->maiClusterGroups[pCluster->miNumClusterGroups] = pClusterGroup->miIndex;
                            DEBUG_PRINTF("cluster %d lod %d group: %d\n", pCluster->miIndex, pCluster->miLODLevel, pClusterGroup->miIndex);
                            ++pCluster->miNumClusterGroups;
                        }
                    }
                }
            }

        }
    }

    // cuda version of cluster vertex mapping and error terms
    std::vector<std::vector<float>> aafClusterDistancesFromLOD0(iNumLODLevels);
    std::vector<std::vector<std::pair<float3, float3>>> aaMaxErrorPositionsFromLOD0(iNumLODLevels);
    std::vector<std::vector<float>> aafClusterAverageDistanceFromLOD0(iNumLODLevels);
    {
        struct MeshClusterDistanceInfo
        {
            uint32_t        miClusterLOD0;
            float           mfDistance;
        };

        // get cluster distances from the LOD 0 clusters
        std::vector<std::vector<std::vector<MeshClusterDistanceInfo>>> aaaMeshClusterDistanceInfo(iNumLODLevels);
        for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
        {
            uint32_t iNumClusters = static_cast<uint32_t>(aaMeshClusters[iLODLevel].size());
            aaaMeshClusterDistanceInfo[iLODLevel].resize(iNumClusters);
            for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
            {
                aaaMeshClusterDistanceInfo[iLODLevel][iCluster].resize(aaMeshClusters[0].size());
                auto& meshCluster = aaMeshClusters[iLODLevel][iCluster];
                for(uint32_t iUpperCluster = 0; iUpperCluster < static_cast<uint32_t>(aaMeshClusters[0].size()); iUpperCluster++)
                {
                    auto const& upperMeshCluster = aaMeshClusters[0][iUpperCluster];
                    float fDistance = length(upperMeshCluster.mCenter - meshCluster.mCenter);
                    aaaMeshClusterDistanceInfo[iLODLevel][iCluster][iUpperCluster].miClusterLOD0 = iUpperCluster;
                    aaaMeshClusterDistanceInfo[iLODLevel][iCluster][iUpperCluster].mfDistance = fDistance;
                }

                std::sort(
                    aaaMeshClusterDistanceInfo[iLODLevel][iCluster].begin(),
                    aaaMeshClusterDistanceInfo[iLODLevel][iCluster].end(),
                    [](MeshClusterDistanceInfo& info0, MeshClusterDistanceInfo& info1)
                    {
                        return info0.mfDistance < info1.mfDistance;
                    }
                );
            }
        }

        uint32_t const kiNumUpperClustersToCheck = 3;

        struct ClosestVertexInfo
        {
            uint32_t            miVertexIndex;
            uint32_t            miVertexIndexLOD0;
            uint32_t            miClusterIndexLOD0;
            float               mfClosestDistance;
            float3              mVertexPositionLOD0;
        };

        auto start = std::chrono::high_resolution_clock::now();
        for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
        {
            aafClusterDistancesFromLOD0[iLODLevel].resize(aaMeshClusters[iLODLevel].size());
            aaMaxErrorPositionsFromLOD0[iLODLevel].resize(aaMeshClusters[iLODLevel].size());
            aafClusterAverageDistanceFromLOD0[iLODLevel].resize(aaMeshClusters[iLODLevel].size());
            uint32_t iNumClusters = static_cast<uint32_t>(aaMeshClusters[iLODLevel].size());
            for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
            {
                if(iLODLevel == 0)
                {
                    aafClusterDistancesFromLOD0[iLODLevel][iCluster] = 0.0f;
                    aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster] = std::make_pair(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
                    continue;
                }

                auto start0 = std::chrono::high_resolution_clock::now();

                auto const& meshCluster = aaMeshClusters[iLODLevel][iCluster];
                float3 const* pClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshCluster.miVertexPositionStartAddress * sizeof(float3));
                std::vector<float3> aClusterVertexPositions(meshCluster.miNumVertexPositions);
                memcpy(
                    aClusterVertexPositions.data(),
                    pClusterVertexPositions,
                    aClusterVertexPositions.size() * sizeof(float3));

                std::vector<ClosestVertexInfo> aClosestClusterVertexPositions(meshCluster.miNumVertexPositions);
                uint32_t iIndex = 0;
                for(auto& closestVertexInfo : aClosestClusterVertexPositions)
                {
                    closestVertexInfo.miVertexIndex = iIndex;
                    closestVertexInfo.mfClosestDistance = 1.0e+10f;
                    ++iIndex;
                }

                // use the top given number of closest LOD 0 clusters 
                for(uint32_t iUpperCluster = 0; iUpperCluster < kiNumUpperClustersToCheck; iUpperCluster++)
                {
                    uint32_t iUpperClusterIndex = aaaMeshClusterDistanceInfo[iLODLevel][iCluster][iUpperCluster].miClusterLOD0;

                    auto const& meshClusterLOD0 = aaMeshClusters[0][iUpperClusterIndex];

                    // check distance
                    float fDistance = length(meshCluster.mCenter - meshClusterLOD0.mCenter);

                    std::vector<float3> aClusterVertexPositionsLOD0(meshClusterLOD0.miNumVertexPositions);
                    float3 const* pClusterVertexPositionsLOD0 = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshClusterLOD0.miVertexPositionStartAddress * sizeof(float3));
                    memcpy(
                        aClusterVertexPositionsLOD0.data(),
                        pClusterVertexPositionsLOD0,
                        aClusterVertexPositionsLOD0.size() * sizeof(float));

                    // get the vertex distances from LOD 0
                    std::vector<float> afClosestDistances(meshCluster.miNumVertexPositions);
                    std::vector<uint32_t> aiClosestVertexPositions(meshCluster.miNumVertexPositions);
                    getShortestVertexDistancesCUDA(
                        afClosestDistances,
                        aiClosestVertexPositions,
                        aClusterVertexPositions,
                        aClusterVertexPositionsLOD0);

                    // store the closest vertex position info
                    for(uint32_t i = 0; i < meshCluster.miNumVertexPositions; i++)
                    {
                        if(afClosestDistances[i] < aClosestClusterVertexPositions[i].mfClosestDistance)
                        {
                            aClosestClusterVertexPositions[i].mfClosestDistance = afClosestDistances[i];
                            aClosestClusterVertexPositions[i].miClusterIndexLOD0 = iUpperClusterIndex;
                            aClosestClusterVertexPositions[i].miVertexIndexLOD0 = aiClosestVertexPositions[i];
                            aClosestClusterVertexPositions[i].mVertexPositionLOD0 = aClusterVertexPositionsLOD0[aiClosestVertexPositions[i]];
                        }
                    }

                }   // for cluster = 0 to num clusters in LOD 0

                std::sort(
                    aClosestClusterVertexPositions.begin(),
                    aClosestClusterVertexPositions.end(),
                    [](ClosestVertexInfo& info0, ClosestVertexInfo& info1)
                    {
                        return info0.mfClosestDistance > info1.mfClosestDistance;
                    }
                );

                // average position deviation distance from LOD 0
                float fNumVertices = 0.0f;
                float fTotalDistances = 0.0f;
                for(auto const& vertexPositionInfo : aClosestClusterVertexPositions)
                {
                    fTotalDistances = (vertexPositionInfo.mfClosestDistance >= 1.0e-3f) ? fTotalDistances + vertexPositionInfo.mfClosestDistance : fTotalDistances;
                    fNumVertices += 1.0f;
                }
                float fAverageDistance = (fNumVertices > 0.0f) ? fTotalDistances / fNumVertices : 0.0f;
                aafClusterAverageDistanceFromLOD0[iLODLevel][iCluster] = fAverageDistance;

                ClosestVertexInfo const& largestDistanceInfo = aClosestClusterVertexPositions.front();
                
                aafClusterDistancesFromLOD0[iLODLevel][iCluster] = largestDistanceInfo.mfClosestDistance;
                aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster].first = aClusterVertexPositions[largestDistanceInfo.miVertexIndex];
                aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster].second = largestDistanceInfo.mVertexPositionLOD0;

                auto end = std::chrono::high_resolution_clock::now();
                uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                uint64_t iMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count();
                if(iCluster % 10 == 0)
                {
                    DEBUG_PRINTF("took %d ms (total: %d sec) seconds to compute the shortest distance to LOD 0 for cluster %d (%d) of lod %d\n",
                        iMilliseconds,
                        iSeconds,
                        iCluster,
                        iNumClusters,
                        iLODLevel);
                }


            }   // for cluster = 0 to num clusters

        }   // for lod = 0 to num lod levels

    }   // cuda version of cluster vertex mapping and error terms

    // make sure LOD always have larger error value than LOD - 1
    for(uint32_t iLODLevel = 1; iLODLevel < iNumLODLevels; iLODLevel++)
    {
        float fPrevLevelLargestErrorDistance = 0.0f;
        float fPrevLevelLargestAverageErrorDistance = 0.0f;
        for(uint32_t i = 0; i < static_cast<uint32_t>(aafClusterDistancesFromLOD0[iLODLevel - 1].size()); i++)
        {
            float fPrevLevelErrorDistance = aafClusterDistancesFromLOD0[iLODLevel - 1][i];
            if(fPrevLevelErrorDistance > fPrevLevelLargestErrorDistance)
            {
                fPrevLevelLargestErrorDistance = fPrevLevelErrorDistance;
            }

            fPrevLevelLargestAverageErrorDistance = maxf(fPrevLevelLargestAverageErrorDistance, aafClusterAverageDistanceFromLOD0[iLODLevel - 1][i]);
        }

        for(uint32_t i = 0; i < static_cast<uint32_t>(aafClusterDistancesFromLOD0[iLODLevel].size()); i++)
        {
            float fCurrLevelErrorDistance = aafClusterDistancesFromLOD0[iLODLevel][i];
            if(fPrevLevelLargestErrorDistance > fCurrLevelErrorDistance)
            {
                float fPct = fPrevLevelLargestErrorDistance / maxf(fCurrLevelErrorDistance, 0.01f);
                fCurrLevelErrorDistance = fPrevLevelLargestErrorDistance;
                aaMaxErrorPositionsFromLOD0[iLODLevel][i].first = aaMaxErrorPositionsFromLOD0[iLODLevel][i].first * fPct;
                aaMaxErrorPositionsFromLOD0[iLODLevel][i].second = aaMaxErrorPositionsFromLOD0[iLODLevel][i].second * fPct;

                float fOldDistance = aafClusterDistancesFromLOD0[iLODLevel][i];
                aafClusterDistancesFromLOD0[iLODLevel][i] = length(aaMaxErrorPositionsFromLOD0[iLODLevel][i].second - aaMaxErrorPositionsFromLOD0[iLODLevel][i].first);
            }

            aafClusterAverageDistanceFromLOD0[iLODLevel][i] = maxf(aafClusterAverageDistanceFromLOD0[iLODLevel][i], fPrevLevelLargestAverageErrorDistance);
        }
    }

    // one big list for cluster distances from LOD 0
    std::vector<float> afTotalClusterDistanceFromLOD0;
    std::vector<std::pair<float3, float3>> aTotalMaxClusterDistancePositionFromLOD0;
    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
    {
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaMeshClusters[iLODLevel].size()); iCluster++)
        {
            afTotalClusterDistanceFromLOD0.push_back(aafClusterDistancesFromLOD0[iLODLevel][iCluster]);
            aTotalMaxClusterDistancePositionFromLOD0.push_back(aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster]);
        }
    }

    // set error terms for cluster groups and clusters
    std::vector<std::pair<float3, float3>> aLODMaxErrorPositions(iNumLODLevels);
    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
    {
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaMeshClusters[iLODLevel].size()); iCluster++)
        {
            auto& cluster = aaMeshClusters[iLODLevel][iCluster];
            if(iLODLevel > 0)
            {
                cluster.mfError = aafClusterDistancesFromLOD0[iLODLevel][iCluster];
            }

            cluster.mfAverageDistanceFromLOD0 = aafClusterAverageDistanceFromLOD0[iLODLevel][iCluster];

            cluster.mMinBounds = float3(FLT_MAX, FLT_MAX, FLT_MAX);
            cluster.mMaxBounds = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for(uint32_t i = 0; i < cluster.miNumVertexPositions; i++)
            {
                float3 const& pos = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + cluster.miVertexPositionStartAddress * sizeof(float3))[i];

                cluster.mMinBounds = fminf(cluster.mMinBounds, pos);
                cluster.mMaxBounds = fmaxf(cluster.mMaxBounds, pos);
            }

            float fBounds = length(cluster.mMaxBounds - cluster.mMinBounds);
            cluster.mfPctError = cluster.mfError / fBounds;

            if(iCluster < aaMaxErrorPositionsFromLOD0[iLODLevel].size())
            {
                cluster.mMaxErrorPosition0 = aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster].first;
                cluster.mMaxErrorPosition1 = aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster].second;
            }
            else
            {
                cluster.mMaxErrorPosition0 = float3(0.0f, 0.0f, 0.0f);
                cluster.mMaxErrorPosition1 = float3(0.0f, 0.0f, 0.0f);
            }
        }

        for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaMeshClusterGroups[iLODLevel].size()); iClusterGroup++)
        {
            auto& clusterGroup = aaMeshClusterGroups[iLODLevel][iClusterGroup];

            clusterGroup.mfMinError = FLT_MAX;
            clusterGroup.mfMaxError = 0.0f;
            for(uint32_t iMIP = 0; iMIP < clusterGroup.miNumMIPS; iMIP++)
            {
                clusterGroup.mafMaxErrors[iMIP] = 0.0f;
                clusterGroup.mafMinErrors[iMIP] = FLT_MAX;
                for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(clusterGroup.maiNumClusters[iMIP]); iCluster++)
                {
                    uint32_t iClusterID = clusterGroup.maiClusters[iMIP][iCluster];
                    {
                        float fClusterDistanceFromLOD0 = afTotalClusterDistanceFromLOD0[iClusterID];
                        
                        if(clusterGroup.mafMaxErrors[iMIP] < fClusterDistanceFromLOD0)
                        {
                            clusterGroup.maMaxErrorPositions[iMIP][0] = aTotalMaxClusterDistancePositionFromLOD0[iClusterID].first;
                            clusterGroup.maMaxErrorPositions[iMIP][1] = aTotalMaxClusterDistancePositionFromLOD0[iClusterID].second;
                        }
                        
                        clusterGroup.mafMaxErrors[iMIP] = maxf(fClusterDistanceFromLOD0, clusterGroup.mafMaxErrors[iMIP]);
                        clusterGroup.mafMinErrors[iMIP] = minf(fClusterDistanceFromLOD0, clusterGroup.mafMinErrors[iMIP]);

                        clusterGroup.mfMaxError = maxf(clusterGroup.mfMaxError, clusterGroup.mafMaxErrors[iMIP]);
                        clusterGroup.mfMinError = minf(clusterGroup.mfMinError, clusterGroup.mafMaxErrors[iMIP]);
                    }

                    clusterGroup.mMinBounds = fminf(clusterGroup.mMinBounds, aaMeshClusters[iLODLevel][iCluster].mMinBounds);
                    clusterGroup.mMaxBounds = fmaxf(clusterGroup.mMaxBounds, aaMeshClusters[iLODLevel][iCluster].mMaxBounds);
                }
            }
        }
    }

    // error values for the last cluster
    for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
    {
        uint32_t iClusterGroup = iCluster / 2;
        if(iCluster < aafClusterGroupErrors[iNumLODLevels - 1].size())
        {
            aaMeshClusters[iNumLODLevels - 1][iCluster].mfError = aafClusterGroupErrors[iNumLODLevels - 1][iClusterGroup];
        }
    }

    // mesh cluster group buffer
    uint64_t iDataOffset = 0;
    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
    {
        for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaMeshClusterGroups[iLODLevel].size()); iClusterGroup++)
        {
            memcpy(gMeshClusterGroupBuffer.data() + iDataOffset, &aaMeshClusterGroups[iLODLevel][iClusterGroup], sizeof(MeshClusterGroup));
            iDataOffset += sizeof(MeshClusterGroup);
        }
    }

    // mesh cluster buffer
    iDataOffset = 0;
    for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
    {
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaMeshClusters[iLODLevel].size()); iCluster++)
        {
            memcpy(gMeshClusterBuffer.data() + iDataOffset, &aaMeshClusters[iLODLevel][iCluster], sizeof(MeshCluster));
            iDataOffset += sizeof(MeshCluster);
        }
    }

    {
        for(uint32_t i = 0; i < static_cast<uint32_t>(apTotalMeshClusters.size()); i++)
        {
            std::vector<float3> aVertexPositions(apTotalMeshClusters[i]->miNumVertexPositions);
            uint8_t* pAddress = vertexPositionBuffer.data() + apTotalMeshClusters[i]->miVertexPositionStartAddress * sizeof(float3);
            memcpy(
                aVertexPositions.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumVertexPositions * sizeof(float3));

            std::vector<float3> aVertexNormals(apTotalMeshClusters[i]->miNumVertexNormals);
            pAddress = vertexNormalBuffer.data() + apTotalMeshClusters[i]->miVertexNormalStartAddress * sizeof(float3);
            memcpy(
                aVertexNormals.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumVertexNormals * sizeof(float3));

            std::vector<float2> aVertexUVs(apTotalMeshClusters[i]->miNumVertexUVs);
            pAddress = vertexUVBuffer.data() + apTotalMeshClusters[i]->miVertexUVStartAddress * sizeof(float2);
            memcpy(
                aVertexUVs.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumVertexUVs * sizeof(float2));

            std::vector<uint32_t> aiVertexPositionIndices(apTotalMeshClusters[i]->miNumTrianglePositionIndices);
            pAddress = trianglePositionIndexBuffer.data() + apTotalMeshClusters[i]->miTrianglePositionIndexAddress * sizeof(uint32_t);
            memcpy(
                aiVertexPositionIndices.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumTrianglePositionIndices * sizeof(uint32_t));

            std::vector<uint32_t> aiVertexNormalIndices(apTotalMeshClusters[i]->miNumTriangleNormalIndices);
            pAddress = triangleNormalIndexBuffer.data() + apTotalMeshClusters[i]->miTriangleNormalIndexAddress * sizeof(uint32_t);
            memcpy(
                aiVertexNormalIndices.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumTriangleNormalIndices * sizeof(uint32_t));

            std::vector<uint32_t> aiVertexUVIndices(apTotalMeshClusters[i]->miNumTriangleUVIndices);
            pAddress = triangleUVIndexBuffer.data() + apTotalMeshClusters[i]->miTriangleUVIndexAddress * sizeof(uint32_t);
            memcpy(
                aiVertexUVIndices.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumTriangleUVIndices * sizeof(uint32_t));

            std::ostringstream clusterName;
            clusterName << "cluster-lod" << apTotalMeshClusters[i]->miLODLevel << "-group" << apTotalMeshClusters[i]->miClusterGroup << "-" << apTotalMeshClusters[i]->miIndex;

            std::ostringstream outputFilePath;
            outputFilePath << "c:\\Users\\Dingwings\\demo-models\\separated-clusters\\" << clusterName.str() << ".obj";

            writeOBJFile(
                aVertexPositions,
                aVertexNormals,
                aVertexUVs,
                aiVertexPositionIndices,
                aiVertexNormalIndices,
                aiVertexUVIndices,
                outputFilePath.str(),
                clusterName.str());
        }
    }

    // save out cluster groups
    {
        for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(apTotalMeshClusterGroups.size()); iClusterGroup++)
        {
            MeshClusterGroup const* pClusterGroup = apTotalMeshClusterGroups[iClusterGroup];
            for(uint32_t iMIP = 0; iMIP < MAX_MIP_LEVELS; iMIP++)
            {
                for(uint32_t iCluster = 0; iCluster < MAX_CLUSTERS_IN_GROUP; iCluster++)
                {
                    uint32_t iClusterID = pClusterGroup->maiClusters[iMIP][iCluster];
                    if(iClusterID < static_cast<uint32_t>(apTotalMeshClusters.size()))
                    {
                        MeshCluster const* pCluster = apTotalMeshClusters[iClusterID];

                        std::vector<float3> aVertexPositions(pCluster->miNumVertexPositions);
                        uint8_t* pAddress = vertexPositionBuffer.data() + pCluster->miVertexPositionStartAddress * sizeof(float3);
                        memcpy(
                            aVertexPositions.data(),
                            pAddress,
                            pCluster->miNumVertexPositions * sizeof(float3));

                        std::vector<float3> aVertexNormals(pCluster->miNumVertexNormals);
                        pAddress = vertexNormalBuffer.data() + pCluster->miVertexNormalStartAddress * sizeof(float3);
                        memcpy(
                            aVertexNormals.data(),
                            pAddress,
                            pCluster->miNumVertexNormals * sizeof(float3));

                        std::vector<float2> aVertexUVs(pCluster->miNumVertexUVs);
                        pAddress = vertexUVBuffer.data() + pCluster->miVertexUVStartAddress * sizeof(float2);
                        memcpy(
                            aVertexUVs.data(),
                            pAddress,
                            pCluster->miNumVertexUVs * sizeof(float2));

                        std::vector<uint32_t> aiVertexPositionIndices(pCluster->miNumTrianglePositionIndices);
                        pAddress = trianglePositionIndexBuffer.data() + pCluster->miTrianglePositionIndexAddress * sizeof(uint32_t);
                        memcpy(
                            aiVertexPositionIndices.data(),
                            pAddress,
                            pCluster->miNumTrianglePositionIndices * sizeof(uint32_t));

                        std::vector<uint32_t> aiVertexNormalIndices(pCluster->miNumTriangleNormalIndices);
                        pAddress = triangleNormalIndexBuffer.data() + pCluster->miTriangleNormalIndexAddress * sizeof(uint32_t);
                        memcpy(
                            aiVertexNormalIndices.data(),
                            pAddress,
                            pCluster->miNumTriangleNormalIndices * sizeof(uint32_t));

                        std::vector<uint32_t> aiVertexUVIndices(pCluster->miNumTriangleUVIndices);
                        pAddress = triangleUVIndexBuffer.data() + pCluster->miTriangleUVIndexAddress * sizeof(uint32_t);
                        memcpy(
                            aiVertexUVIndices.data(),
                            pAddress,
                            pCluster->miNumTriangleUVIndices * sizeof(uint32_t));

                        std::ostringstream clusterName;
                        clusterName << "cluster-lod" << pClusterGroup->miLODLevel << "-group" << pClusterGroup->miIndex << "-mip" << iMIP << "-" << pCluster->miIndex;

                        std::ostringstream outputFilePath;
                        outputFilePath << "c:\\Users\\Dingwings\\demo-models\\separated-cluster-groups\\" << clusterName.str() << ".obj";

                        writeOBJFile(
                            aVertexPositions,
                            aVertexNormals,
                            aVertexUVs,
                            aiVertexPositionIndices,
                            aiVertexNormalIndices,
                            aiVertexUVIndices,
                            outputFilePath.str(),
                            clusterName.str());
                    

                        {
                            std::ostringstream testClusterName;
                            testClusterName << "cluster-" << pCluster->miIndex;

                            std::ostringstream testOutputFilePath;
                            testOutputFilePath << "c:\\Users\\Dingwings\\demo-models\\test-render-clusters\\" << testClusterName.str() << ".obj";
                            writeOBJFile(
                                aVertexPositions,
                                aVertexNormals,
                                aVertexUVs,
                                aiVertexPositionIndices,
                                aiVertexNormalIndices,
                                aiVertexUVIndices,
                                testOutputFilePath.str(),
                                testClusterName.str());
                        }

                    }   // if cluster id < max clusters in group
                
                }   // for cluster = 0 to num clusters in group
            
            }   // for mip = 0 to max mips 
              
        }   // for cluste group = 0 to num cluster groups
    }

    {
        std::vector<ClusterTreeNode> aNodes;
        createTreeNodes2(
            aNodes,
            iNumLODLevels,          
            gMeshClusterBuffer,
            gMeshClusterGroupBuffer,
            aaMeshClusterGroups,
            aaMeshClusters,
            aTotalMaxClusterDistancePositionFromLOD0);

        std::vector<ClusterGroupTreeNode> aClusterGroupNodes;
        for(auto const& node : aNodes)
        {
            uint32_t iClusterGroupAddress = node.miClusterGroupAddress;
            uint32_t iLODLevel = node.miLevel;
            auto iter = std::find_if(
                aClusterGroupNodes.begin(),
                aClusterGroupNodes.end(),
                [iClusterGroupAddress,
                iLODLevel](ClusterGroupTreeNode const& checkNode)
                {
                    return checkNode.miClusterGroupAddress == iClusterGroupAddress && checkNode.miLevel == iLODLevel;
                });

            if(iter == aClusterGroupNodes.end())
            {
                ClusterGroupTreeNode groupNode;
                groupNode.miClusterGroupAddress = node.miClusterGroupAddress;
                groupNode.maiClusterAddress[groupNode.miNumChildClusters] = node.miClusterAddress;
                groupNode.mMaxDistanceCurrClusterPosition = node.mMaxDistanceCurrLODClusterPosition;
                groupNode.mMaxDistanceLOD0ClusterPosition = node.mMaxDistanceLOD0ClusterPosition;
                groupNode.miLevel = node.miLevel;
                ++groupNode.miNumChildClusters;

                aClusterGroupNodes.push_back(groupNode);
            }
            else
            {
                float fLength0 = lengthSquared(iter->mMaxDistanceCurrClusterPosition - iter->mMaxDistanceLOD0ClusterPosition);
                float fLength1 = lengthSquared(node.mMaxDistanceCurrLODClusterPosition - node.mMaxDistanceLOD0ClusterPosition);
                if(fLength1 > fLength0)
                {
                    iter->mMaxDistanceCurrClusterPosition = node.mMaxDistanceCurrLODClusterPosition;
                    iter->mMaxDistanceLOD0ClusterPosition = node.mMaxDistanceLOD0ClusterPosition;
                }

                iter->maiClusterAddress[iter->miNumChildClusters] = node.miClusterAddress;
                ++iter->miNumChildClusters;
            }
        }

        std::sort(
            aClusterGroupNodes.begin(),
            aClusterGroupNodes.end(),
            [](ClusterGroupTreeNode const& left, ClusterGroupTreeNode const& right)
            {
                return left.miLevel > right.miLevel;
            }
        );

        std::vector<uint32_t> aiLevelStartIndex(iNumLODLevels + 1);         // + 1 for MIP 1 of last LOD
        std::vector<uint32_t> aiNumLevelNodes(iNumLODLevels + 1);
        memset(aiLevelStartIndex.data(), 0xff, sizeof(uint32_t) * iNumLODLevels);
        memset(aiNumLevelNodes.data(), 0, sizeof(uint32_t)* iNumLODLevels);

        for(uint32_t i = 0; i < static_cast<uint32_t>(aClusterGroupNodes.size()); i++)
        {
            auto const& groupNode = aClusterGroupNodes[i];
            aiNumLevelNodes[groupNode.miLevel] += 1;
            aiLevelStartIndex[groupNode.miLevel] = std::min(aiLevelStartIndex[groupNode.miLevel], i);
        }

#if 0
        float const kfPixelErrorThreshold = 3.0f;
        float fZ = 90.0f;
        testClusterLOD2(
            aNodes,
            aClusterGroupNodes,
            aiLevelStartIndex,
            aiNumLevelNodes,
            float3(0.0f, 0.0f, fZ),
            float3(0.0f, 0.0f, 0.0f),
            1000,
            1000,
            kfPixelErrorThreshold
        );
#endif // #if 0

        for(float fZ = 2.0f; fZ <= 50.0f; fZ += 1.0f)
        {
            float const kfPixelErrorThreshold = 3.0f;
            std::vector<uint32_t> aiDrawClusters;
            testClusterLOD3(
                aiDrawClusters,
                aNodes,
                aClusterGroupNodes,
                aiLevelStartIndex,
                aiNumLevelNodes,
                float3(0.0f, 0.0f, fZ),
                float3(0.0f, 0.0f, 0.0f),
                1000,
                1000,
                kfPixelErrorThreshold
            );

            //std::ostringstream imageName;
            //imageName << "output-" << uint32_t(fZ) << ".png";
            //drawMeshClusterImage(
            //    aiDrawClusters,
            //    apTotalMeshClusters,
            //    vertexPositionBuffer,
            //    trianglePositionIndexBuffer,
            //    float3(0.0f, 0.0f, fZ),
            //    float3(0.0f, 0.0f, 0.0f),
            //    1024,
            //    1024,
            //    "c:\\Users\\Dingwings\\demo-models\\cluster-images",
            //    imageName.str());

            int iDebug = 1;
        }

        int iDebug = 1;
    }

#if 0
    std::vector<ClusterTreeNode> aNodes;
    createTreeNodes(
        aNodes,
        iNumLODLevels,
        gMeshClusterBuffer,
        gMeshClusterGroupBuffer,
        aaMeshClusterGroups,
        aaMeshClusters);

    float const kfPixelErrorThreshold = 10.0f;
    for(float fZ = 15.0f; fZ <= 20.0f; fZ += 0.5f)
    {
        DEBUG_PRINTF("\n*************\n");
        DEBUG_PRINTF("CAMERA Z: %.4f\n", fZ);
        testClusterLOD(
            gMeshClusterBuffer,
            gMeshClusterGroupBuffer,
            vertexPositionBuffer,
            trianglePositionIndexBuffer,
            float3(0.0f, 0.0f, fZ),
            float3(0.0f, 0.0f, 0.0f),
            1000,
            1000,
            aNodes[0],
            gMeshClusterGroupBuffer,
            gMeshClusterBuffer,
            kfPixelErrorThreshold);
        int iDebug = 1;
    }
#endif // #if 0

    return 0;
}


