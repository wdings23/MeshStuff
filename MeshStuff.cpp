#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <sstream>
#include <map>
#include <filesystem>

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


#include <atomic>
#include <mutex>



#include <map>

#include <time.h>

#include "rasterizer.h"
#include "rasterizerCUDA.h"

#include "Camera.h"

#include "test.h"
#include "split_operations.h"
#include "join_operations.h"
#include "move_operations.h"

#include "utils.h"

#include "obj_helper.h"

#include "mesh_cluster.h"
#include "test_raster.h"
#include "test_cluster_streaming.h"

#include "boundary_operations.h"
#include "simplify_operations.h"
#include "metis_operations.h"
#include "system_command.h"
#include "cleanup_operations.h"

#include "obj_helper.h"

#include "adjacency_operations_cuda.h"

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
    uint32_t iNumClusterGroups,
    uint32_t iNumClusters,
    uint32_t iLODLevel,
    std::string const& meshClusterOutputName,
    std::string const& homeDirectory,
    std::string const& meshModelName);

/*
**
*/
int main(int argc, char* argv[])
{
    float result = 0.0f;

    // TODO: add support for non-closed manifold meshes
    //       marking bound vertices and not collapse them?

    srand(static_cast<uint32_t>(time(nullptr)));

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> aShapes;
    std::vector<tinyobj::material_t> aMaterials;
    std::vector<std::vector<uint32_t>> aaiAdjacencyList;

    std::string homeDirectory = getenv("HOME");
    homeDirectory += "\\demo-models\\";

    std::string objMeshModelName = argv[1];

    // load initial mesh file
    //std::string fullOBJFilePath = homeDirectory + "face-meshlet-test.obj";
    //std::string fullOBJFilePath = homeDirectory + "guan-yu-5-meshlet-test.obj";
    //std::string fullOBJFilePath = homeDirectory + "ritual-bell-meshlet-test.obj";
    //std::string fullOBJFilePath = homeDirectory + "ritual-bell-trimmed.obj";
    //std::string fullOBJFilePath = homeDirectory + "dragon.obj";
    //std::string fullOBJFilePath = homeDirectory + "dragon-trimmed.obj";
    //std::string fullOBJFilePath = homeDirectory + "guan-yu-full.obj";
    std::string fullOBJFilePath = homeDirectory + objMeshModelName;
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

    auto iStartObjectName = fullOBJFilePath.find_last_of("\\");
    auto iEndObjectName = fullOBJFilePath.find_last_of(".obj") - strlen(".obj");
    std::string meshModelName = fullOBJFilePath.substr(iStartObjectName + 1, iEndObjectName - iStartObjectName);

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
        std::ostringstream outputMetisMeshFolderPath;
        {
            outputMetisMeshFolderPath << homeDirectory << "metis\\" << meshModelName << "\\";

            std::filesystem::path outputMetisMeshFolderFileSystemPath(outputMetisMeshFolderPath.str());
            if(!std::filesystem::exists(outputMetisMeshFolderFileSystemPath))
            {
                std::filesystem::create_directory(outputMetisMeshFolderFileSystemPath);
            }
        }

        std::ostringstream outputMetisMeshFilePath;
        outputMetisMeshFilePath << outputMetisMeshFolderPath.str() << "output";
        outputMetisMeshFilePath << "-lod" << 0 << ".mesh";

auto start = std::chrono::high_resolution_clock::now();

        buildMETISMeshFile2(
            outputMetisMeshFilePath.str(),
            aiTrianglePositionIndices);

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%lld seconds to build cluster Metis mesh file\n", iSeconds);


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
DEBUG_PRINTF("%lld seconds to build clusters\n", iSeconds);
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
        {
            std::vector<std::vector<float3>> aaTempClusterVertexPositions;
            std::vector<std::vector<float3>> aaTempClusterVertexNormals;
            std::vector<std::vector<float2>> aaTempClusterVertexUVs;
            std::vector<std::vector<uint32_t>> aaiTempClusterTrianglePositionIndices;
            std::vector<std::vector<uint32_t>> aaiTempClusterTriangleNormalIndices;
            std::vector<std::vector<uint32_t>> aaiTempClusterTriangleUVIndices;

            auto start0 = std::chrono::high_resolution_clock::now();

            for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
            {
                std::vector<std::vector<float3>> aaSplitClusterVertexPositions;
                std::vector<std::vector<float3>> aaSplitClusterVertexNormals;
                std::vector<std::vector<float2>> aaSplitClusterVertexUVs;
                std::vector<std::vector<uint32_t>> aaiSplitClusterTrianglePositionIndices;
                std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleNormalIndices;
                std::vector<std::vector<uint32_t>> aaiSplitClusterTriangleUVIndices;

                splitCluster3(
                    aaSplitClusterVertexPositions,
                    aaSplitClusterVertexNormals,
                    aaSplitClusterVertexUVs,
                    aaiSplitClusterTrianglePositionIndices,
                    aaiSplitClusterTriangleNormalIndices,
                    aaiSplitClusterTriangleUVIndices,
                    aaClusterVertexPositions,
                    aaClusterVertexNormals,
                    aaClusterVertexUVs,
                    aaiClusterTrianglePositionIndices,
                    aaiClusterTriangleNormalIndices,
                    aaiClusterTriangleUVIndices,
                    iCluster,
                    kiMaxTrianglesToSplit);

                for(auto const& aSplitclusterVertexPositions : aaSplitClusterVertexPositions)
                {
                    aaTempClusterVertexPositions.push_back(aSplitclusterVertexPositions);
                }

                for(auto const& aSplitclusterVertexNormals : aaSplitClusterVertexNormals)
                {
                    aaTempClusterVertexNormals.push_back(aSplitclusterVertexNormals);
                }

                for(auto const& aSplitclusterVertexUVs : aaSplitClusterVertexUVs)
                {
                    aaTempClusterVertexUVs.push_back(aSplitclusterVertexUVs);
                }

                for(auto const& aiSplitClusterVertexPositionIndices : aaiSplitClusterTrianglePositionIndices)
                {
                    aaiTempClusterTrianglePositionIndices.push_back(aiSplitClusterVertexPositionIndices);
                }

                for(auto const& aiSplitClusterTriangleNormalIndices : aaiSplitClusterTriangleNormalIndices)
                {
                    aaiTempClusterTriangleNormalIndices.push_back(aiSplitClusterTriangleNormalIndices);
                }

                for(auto const& aiSplitClusterVertexUVIndices : aaiSplitClusterTriangleUVIndices)
                {
                    aaiTempClusterTriangleUVIndices.push_back(aiSplitClusterVertexUVIndices);
                }
            }
            uint64_t iSplitElapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start0).count();
            DEBUG_PRINTF("Took %lld seconds to split all clusters\n", iSplitElapsedSeconds);

            start0 = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<uint32_t>> aaiGroupClusterIndices;
            cleanupClusters2(
                aaTempClusterVertexPositions,
                aaTempClusterVertexNormals,
                aaTempClusterVertexUVs,
                aaiTempClusterTrianglePositionIndices,
                aaiTempClusterTriangleNormalIndices,
                aaiTempClusterTriangleUVIndices,
                aaiGroupClusterIndices);

            aaClusterVertexPositions = aaTempClusterVertexPositions;
            aaClusterVertexNormals = aaTempClusterVertexNormals;
            aaClusterVertexUVs = aaTempClusterVertexUVs;
            aaiClusterTrianglePositionIndices = aaiTempClusterTrianglePositionIndices;
            aaiClusterTriangleNormalIndices = aaiTempClusterTriangleNormalIndices;
            aaiClusterTriangleUVIndices = aaiTempClusterTriangleUVIndices;

            assert(aaClusterVertexPositions.size() == aaiClusterTrianglePositionIndices.size());
            assert(aaClusterVertexNormals.size() == aaiClusterTriangleNormalIndices.size());
            assert(aaClusterVertexUVs.size() == aaiClusterTriangleUVIndices.size());

            iNumClusters = static_cast<uint32_t>(aaTempClusterVertexPositions.size());
            iNumClusterGroups = iNumClusters / 4;

            uint64_t iCleanupElapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start0).count();
            DEBUG_PRINTF("Took %lld seconds to clean up all clusters\n", iCleanupElapsedSeconds);

            //std::ostringstream outputFolderPath;
            //{
            //    outputFolderPath << homeDirectory << "large-clusters\\" << meshModelName;
            //    std::filesystem::path folderFileSystemPath(outputFolderPath.str());
            //    if(!std::filesystem::exists(folderFileSystemPath))
            //    {
            //        std::filesystem::create_directories(folderFileSystemPath);
            //    }
            //}
            //
            //std::ostringstream clusterName;
            //clusterName << meshModelName << "-cluster-lod0";
            //outputFolderPath << "\\" << clusterName.str() << ".obj";
            //writeTotalClusterOBJ(
            //    outputFolderPath.str(),
            //    clusterName.str(),
            //    aaClusterVertexPositions,
            //    aaClusterVertexNormals,
            //    aaClusterVertexUVs,
            //    aaiClusterTrianglePositionIndices,
            //    aaiClusterTriangleNormalIndices,
            //    aaiClusterTriangleUVIndices);
        }

        uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        DEBUG_PRINTF("took %lld seconds to split large clusters\n", iSeconds);
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

        assert(aaClusterVertexPositions.size() == aaiClusterTrianglePositionIndices.size());
        for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
        {
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
            assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
        }

        start = std::chrono::high_resolution_clock::now();
        DEBUG_PRINTF("*** start saving total cluster obj ***\n");
        {
            std::ostringstream totalClusterFolderPath;
            totalClusterFolderPath << homeDirectory << "total-clusters\\" << meshModelName << "\\";
            std::filesystem::path folderPath(totalClusterFolderPath.str());
            if(!std::filesystem::exists(folderPath))
            {
                std::filesystem::create_directories(folderPath);
            }

            std::ostringstream objectName;
            objectName << "total-cluster-lod" << iLODLevel;

            std::ostringstream outputTotalClusterFilePath;
            outputTotalClusterFilePath << totalClusterFolderPath.str() << objectName.str();
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
auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%lld seconds save total cluster obj\n", iSeconds);


start = std::chrono::high_resolution_clock::now();
DEBUG_PRINTF("*** start average triangle surface area ***\n");
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
                for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTriangleNormalIndices[iCluster].size()); iTri += 3)
                {
                    avgNormal += normalize(aaClusterVertexNormals[iCluster][aaiClusterTriangleNormalIndices[iCluster][iTri]]);
                    avgNormal += normalize(aaClusterVertexNormals[iCluster][aaiClusterTriangleNormalIndices[iCluster][iTri + 1]]);
                    avgNormal += normalize(aaClusterVertexNormals[iCluster][aaiClusterTriangleNormalIndices[iCluster][iTri + 2]]);
                }
                avgNormal = normalize(avgNormal);
                
                // get the cone radius (min dot product of average normal with triangle normal, ie. greatest cosine angle)
                float fMinDP = FLT_MAX;
                for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aaiClusterTriangleNormalIndices[iCluster].size()); iTri += 3)
                {
                    fMinDP = minf(fMinDP, dot(avgNormal, normalize(aaClusterVertexNormals[iCluster][aaiClusterTriangleNormalIndices[iCluster][iTri]])));
                    fMinDP = minf(fMinDP, dot(avgNormal, normalize(aaClusterVertexNormals[iCluster][aaiClusterTriangleNormalIndices[iCluster][iTri + 1]])));
                    fMinDP = minf(fMinDP, dot(avgNormal, normalize(aaClusterVertexNormals[iCluster][aaiClusterTriangleNormalIndices[iCluster][iTri + 2]])));
                }

                float fSinAngle = (fMinDP < 0.0f) ? -1.0f : -sqrtf(1.0f - fMinDP * fMinDP);
                aaClusterNormalCones[iLODLevel].push_back(float4(avgNormal, fSinAngle));
            }

            for(uint32_t iCluster = 0; iCluster < static_cast<uint32_t>(aaClusterVertexPositions.size()); iCluster++)
            {
                assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleNormalIndices[iCluster].size());
                assert(aaiClusterTrianglePositionIndices[iCluster].size() == aaiClusterTriangleUVIndices[iCluster].size());
            }
        }


end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%lld seconds to compute cluster average triangle area and normal cones\n", iSeconds);
        
        std::ostringstream metisClusterLODFolderPath;
        {
            metisClusterLODFolderPath << homeDirectory << "metis\\" << meshModelName << "\\";
            std::filesystem::path metisClusterLODFolderFileSystemPath(metisClusterLODFolderPath.str());
            if(!std::filesystem::exists(metisClusterLODFolderFileSystemPath))
            {
                std::filesystem::create_directory(metisClusterLODFolderFileSystemPath);
            }
        }

        std::ostringstream outputClusterMeshFilePath;
        outputClusterMeshFilePath << metisClusterLODFolderPath.str() << "clusters-lod" << iLODLevel << ".mesh";

        // build a metis graph file with the number of shared vertices as edge weights between clusters
        if(iNumClusterGroups > 1)
        {
            {
                DEBUG_PRINTF("*** start getting boundary vertices ***\n");
                
                auto startMetisTime = std::chrono::high_resolution_clock::now();

                uint32_t iNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());

                std::vector<std::vector<uint32_t>> aaiClusterBoundaryVertices;
                std::vector<std::vector<uint32_t>> aaiClusterNonBoundaryVertices;
                getBoundaryAndNonBoundaryVertices(
                    aaiClusterBoundaryVertices,
                    aaiClusterNonBoundaryVertices,
                    aaClusterVertexPositions,
                    aaiClusterTrianglePositionIndices);

                std::vector<float3> aBoundaryMin(iNumClusters);
                std::vector<float3> aBoundaryMax(iNumClusters);
                for(uint32_t iCluster = 0; iCluster < iNumClusters; iCluster++)
                {
                    float3 minPos(FLT_MAX, FLT_MAX, FLT_MAX);
                    float3 maxPos(-FLT_MAX, -FLT_MAX, -FLT_MAX);
                    for(auto const& iIndex : aaiClusterBoundaryVertices[iCluster])
                    {
                        float3 const& pos = aaClusterVertexPositions[iCluster][iIndex];
                        minPos = fminf(minPos, pos);
                        maxPos = fmaxf(maxPos, pos);
                    }

                    aBoundaryMin[iCluster] = minPos;
                    aBoundaryMax[iCluster] = maxPos;
                }

                //std::vector<std::vector<uint32_t>> aaiNumAdjacentClusters;
                //buildClusterEdgeAdjacencyCUDA3(
                //    aaiNumAdjacentClusters,
                //    aaClusterVertexPositions,
                //    aaiClusterBoundaryVertices,
                //    aBoundaryMin,
                //    aBoundaryMax);


                std::vector<std::vector<uint32_t>> aaiNumAdjacentClusters(aaClusterVertexPositions.size());
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaiNumAdjacentClusters.size()); i++)
                {
                    aaiNumAdjacentClusters[i].resize(aaiNumAdjacentClusters.size());
                }

                uint32_t const kiMaxThreads = 12;
                std::vector<std::unique_ptr<std::thread>> apThreads(kiMaxThreads);
                
                static std::atomic<uint32_t> siCurrCluster;
                static std::mutex sAdjacencyMutex;

                siCurrCluster = 0;
                for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
                {
                    apThreads[iThread] = std::make_unique<std::thread>(
                        [&aaiNumAdjacentClusters,
                        aaiClusterBoundaryVertices,
                        aaClusterVertexPositions,
                        iNumClusters,
                        startMetisTime,
                        aBoundaryMin,
                        aBoundaryMax]()
                        {
                            for(;;)
                            {
                                uint32_t iCluster = siCurrCluster.fetch_add(1);
                                if(iCluster >= iNumClusters)
                                {
                                    break;
                                }

                                //if(iCluster > 0 && iCluster % 100 == 0)
                                //{
                                //    uint64_t iElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startMetisTime).count();
                                //    DEBUG_PRINTF("Took %d seconds to process cluster %d (%d) adjacency\n", iElapsed / 1000, iCluster, iNumClusters);
                                //}

                                float3 const& clusterMin = aBoundaryMin[iCluster];
                                float3 const& clusterMax = aBoundaryMax[iCluster];

                                //std::vector<uint32_t> aiAdjacencyFlags(iNumClusters);
                                auto const& aiClusterBoundaryVertices = aaiClusterBoundaryVertices[iCluster];
                                auto const& aClusterVertexPositions = aaClusterVertexPositions[iCluster];
                                for(uint32_t i = 0; i < static_cast<uint32_t>(aiClusterBoundaryVertices.size()); i++)
                                {
                                    float3 const& clusterVertexPosition = aClusterVertexPositions[aiClusterBoundaryVertices[i]];

                                    for(uint32_t iCheckCluster = iCluster + 1; iCheckCluster < iNumClusters; iCheckCluster++)
                                    {
                                        if(iCheckCluster == iCluster)
                                        {
                                            continue;
                                        }
                                        auto const& aiCheckClusterBoundaryVertices = aaiClusterBoundaryVertices[iCheckCluster];
                                        auto const& aCheckClusterVertexPositions = aaClusterVertexPositions[iCheckCluster];

                                        float3 const& checkClusterMin = aBoundaryMin[iCheckCluster];
                                        float3 const& checkClusterMax = aBoundaryMax[iCheckCluster];

                                        // intersection?
                                        bool bIntersection = (
                                            clusterMin.x <= checkClusterMax.x &&
                                            clusterMax.x >= checkClusterMin.x &&
                                            clusterMin.y <= checkClusterMax.y &&
                                            clusterMax.y >= checkClusterMin.y &&
                                            clusterMin.z <= checkClusterMax.z &&
                                            clusterMax.z >= checkClusterMin.z);
                                        if(!bIntersection)
                                        {
                                            continue;
                                        }

                                        bool bAdjacent = false;
                                        for(uint32_t j = 0; j < static_cast<uint32_t>(aiCheckClusterBoundaryVertices.size()); j++)
                                        {
                                            float3 const& checkClusterVertexPosition = aCheckClusterVertexPositions[aiCheckClusterBoundaryVertices[j]];
                                            float fDistance = lengthSquared(checkClusterVertexPosition - clusterVertexPosition);
                                            if(fDistance <= 1.0e-8f)
                                            {
                                                bAdjacent = true;
                                                break;
                                            }
                                        }

                                        if(bAdjacent)
                                        {
                                            //aiAdjacencyFlags[iCheckCluster] += 1;
                                            std::lock_guard<std::mutex> lock(sAdjacencyMutex);
                                            aaiNumAdjacentClusters[iCluster][iCheckCluster] += 1;
                                            aaiNumAdjacentClusters[iCheckCluster][iCluster] += 1;
                                        }

                                    } // for boundary vertex to num boundary vertices for check cluster

                                }   // for boundary vertex to num boundary vertices for cluster

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


                uint32_t iNumEdges = 0;
                for(uint32_t i = 0; i < iNumClusters; i++)
                {
                    for(uint32_t j = i + 1; j < static_cast<uint32_t>(aaiNumAdjacentClusters[i].size()); j++)
                    {
                        if(aaiNumAdjacentClusters[i][j] > 0)
                        {
                            ++iNumEdges;
                        }
                    }
                }
                assert(iNumEdges > 0);
                
                FILE* fp = fopen(outputClusterMeshFilePath.str().c_str(), "wb");
                fprintf(fp, "%d %d 001\n", iNumClusters, iNumEdges);
                for(uint32_t i = 0; i < iNumClusters; i++)
                {
                    for(uint32_t j = 0; j < iNumClusters; j++)
                    {
                        if(aaiNumAdjacentClusters[i][j] > 0)
                        {
                            fprintf(fp, "%d %d", j + 1, aaiNumAdjacentClusters[i][j]);
                            if(j < iNumClusters - 1)
                            {
                                fprintf(fp, " ");
                            }
                        }
                    }

                    fprintf(fp, "\n");
                }

                fclose(fp);

                // generate cluster groups for all the initial clusters
                std::ostringstream clusterGroupCommand;
                clusterGroupCommand << "D:\\test\\METIS\\build\\windows\\programs\\Debug\\gpmetis.exe ";
                clusterGroupCommand << outputClusterMeshFilePath.str() << " ";
                clusterGroupCommand << iNumClusterGroups;
                std::string result = execCommand(clusterGroupCommand.str(), false);
                if(result.find("Metis returned with an error.") != std::string::npos)
                {
                    assert(0);
                }

                uint64_t iElapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startMetisTime).count();
                DEBUG_PRINTF("Took %lld seconds to build metis file from cluster adjacency for cluster group\n", iElapsed);

            }   // new cluster adjacency

        }

DEBUG_PRINTF("*** start build mesh cluster groups ***\n");
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
            iNumClusterGroups,
            iNumClusters,
            iLODLevel,
            outputClusterMeshFilePath.str(),
            homeDirectory,
            meshModelName);

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
DEBUG_PRINTF("%lld seconds to build cluster groups\n", iSeconds);

DEBUG_PRINTF("*** start build mesh cluster data ***\n");
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

                uint64_t iAbsoluteAddress = aaMeshClusters[iLODLevel][iCluster].miVertexPositionStartArrayAddress * sizeof(float3);
                auto const& checkPos = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + iAbsoluteAddress)[iV];

                float3 diff = pos - checkPos;
                assert(length(diff) < 1.0e-8f);
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
DEBUG_PRINTF("%lld seconds to pack cluster group data\n", iSeconds);

DEBUG_PRINTF("*** start getting boundary and non-boundary edges ***\n");
start = std::chrono::high_resolution_clock::now();

        // get cluster boundary vertices
        std::vector<std::vector<uint32_t>> aaiClusterGroupBoundaryVertices;
        std::vector<std::vector<uint32_t>> aaiClusterGroupNonBoundaryVertices;
        getBoundaryAndNonBoundaryVertices(
            aaiClusterGroupBoundaryVertices,
            aaiClusterGroupNonBoundaryVertices,
            aaClusterGroupVertexPositions,
            aaiClusterGroupTrianglePositionIndices);
        
end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %lld seconds to get cluster group boundary vertices\n", iSeconds);
   
DEBUG_PRINTF("*** start getting inner edges and vertices ***\n");
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
DEBUG_PRINTF("took %lld seconds to get inner edges and vertices\n", iSeconds);

DEBUG_PRINTF("*** start getting boundary edges ***\n");
start = std::chrono::high_resolution_clock::now();

        // check if edges are not adjacent to anything
        std::vector<BoundaryEdgeInfo> aBoundaryEdges;
        getBoundaryEdges(
            aBoundaryEdges,
            aaClusterGroupVertexPositions,
            aaClusterGroupVertexNormals,
            aaClusterGroupVertexUVs,
            aaiClusterGroupTrianglePositionIndices,
            aaiClusterGroupTriangleNormalIndices,
            aaiClusterGroupTriangleUVIndices);

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %lld seconds to get boundary edges\n", iSeconds);

start = std::chrono::high_resolution_clock::now();

        // delete boundary edges from collapse candidates 
        uint32_t iEdge = 0;
        //std::vector<std::pair<uint32_t, uint32_t>> aBoundaryVertices;
        for(auto const& boundaryEdgeInfo : aBoundaryEdges)
        {
            // see if the edges are boundary edges
            auto boundaryEdgeIter = std::find_if(
                aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].begin(),
                aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].end(),
                [boundaryEdgeInfo](std::pair<uint32_t, uint32_t> const& checkEdge)
                {
                    return (boundaryEdgeInfo.miPos0 == checkEdge.first && boundaryEdgeInfo.miPos1 == checkEdge.second) || (boundaryEdgeInfo.miPos1 == checkEdge.first && boundaryEdgeInfo.miPos0 == checkEdge.second);
                }
            );

            if(boundaryEdgeIter != aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].end())
            {
                //DEBUG_PRINTF("delete from cluster group %d edge (%d, %d)\n",
                //    boundaryEdgeInfo.miClusterGroup,
                //    boundaryEdgeInfo.miPos0,
                //    boundaryEdgeInfo.miPos1);
                aaValidClusterGroupEdgePairs[boundaryEdgeInfo.miClusterGroup].erase(boundaryEdgeIter);
            }

            ++iEdge;
        }

end = std::chrono::high_resolution_clock::now();
iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("%lld seconds to delete boundary edges\n", iSeconds);

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
                iLODLevel,
                iNumClusterGroups,
                start,
                meshModelName,
                homeDirectory,
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
                            iMaxTriangles,
                            iThreadClusterGroup,
                            iLODLevel,
                            meshModelName,
                            homeDirectory);
                        afErrors[iThreadClusterGroup] = fTotalError;

                        assert(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size() == aaiClusterGroupTriangleNormalIndices[iThreadClusterGroup].size());
                        assert(aaiClusterGroupTrianglePositionIndices[iThreadClusterGroup].size() == aaiClusterGroupTriangleUVIndices[iThreadClusterGroup].size());

auto clusterGroupEnd = std::chrono::high_resolution_clock::now();
uint64_t iClusterGroupMilliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(clusterGroupEnd - clusterGroupStart).count();
uint64_t iTotalSeconds = std::chrono::duration_cast<std::chrono::seconds>(clusterGroupEnd - start).count();
if(iThreadClusterGroup % 10 == 0)
{
    DEBUG_PRINTF("took %lld milliseconds (total: %lld secs) to simplify cluster group %d of cluster groups %d\n", 
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
DEBUG_PRINTF("took %lld seconds to simplify cluster groups\n", iSeconds);

        // set the error for each clusters using cluster group errors
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
            std::vector<std::vector<uint32_t>> aaiGroupClustersIndices(aaiClusterGroupTrianglePositionIndices.size());
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

                // invalid cluster group, skip
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

                // split cluster group, this will be store in MIP 1 of the cluster group, also the children of MIP 0 as well as children of the next LOD level
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
                    iClusterGroup,
                    meshModelName,
                    homeDirectory);

                uint32_t iNumCreatedClusters = static_cast<uint32_t>(aaClusterVertexPositions.size() - iPrevNumClusters);
                for(uint32_t iSplitCluster = 0; iSplitCluster < iNumCreatedClusters; iSplitCluster++)
                {
                    uint32_t iCurrCluster = iSplitCluster + iPrevNumClusters;
                    assert(aaClusterVertexPositions[iCurrCluster].size() > 0 && aaiClusterTrianglePositionIndices[iCurrCluster].size() > 0);

                    aaiGroupClustersIndices[iClusterGroup].push_back(iCurrCluster);
                }

            }   // for cluster group = 0 to num cluster groups

            iTotalClusterIndex = 0;
            for(uint32_t iClusterGroup = 0; iClusterGroup < static_cast<uint32_t>(aaiClusterGroupTrianglePositionIndices.size()); iClusterGroup++)
            {
                iTotalClusterIndex += static_cast<uint32_t>(aaiGroupClustersIndices[iClusterGroup].size());
                uint32_t iCurrNumClusters = static_cast<uint32_t>(aaClusterVertexPositions.size());

                // add the split clusters into the cluster group for MIP 1
                //uint32_t iNumNewlySplitClusters = iCurrNumClusters - iPrevNumClusters;
                uint32_t iNumNewlySplitClusters = static_cast<uint32_t>(aaiGroupClustersIndices[iClusterGroup].size());
                assert(iNumNewlySplitClusters < MAX_CLUSTERS_IN_GROUP);
                aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] = (aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] > 20) ? 0 : aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1];
                for(uint32_t i = 0; i < iNumNewlySplitClusters; i++)
                {
                    aaMeshClusterGroups[iLODLevel][iClusterGroup].maiClusters[1][i] = iTotalMeshClusters + iNumClusters + i;
                    assert(aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] < MAX_CLUSTERS_IN_GROUP);
                    aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[1] += 1;
                }

                // set the parents of the previous clusters (MIP 0) from the split clusters (MIP 1)
                uint32_t const kiParentMIP = 0;
                for(uint32_t i = 0; i < static_cast<uint32_t>(aaMeshClusterGroups[iLODLevel][iClusterGroup].maiNumClusters[kiParentMIP]); i++)
                {
                    uint32_t iClusterID = aaMeshClusterGroups[iLODLevel][iClusterGroup].maiClusters[kiParentMIP][i];
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
                        // parent cluster from the total cluster index
                        assert(iter->miNumParentClusters < MAX_PARENT_CLUSTERS);
                        iter->maiParentClusters[iter->miNumParentClusters + i] = iTotalMeshClusters + iNumClusters + i;
                        ++iter->miNumParentClusters;

                        //DEBUG_PRINTF("set cluster %d as parent for cluster %d (%d)\n",
                        //    iTotalMeshClusters + iNumClusters + i,
                        //    iter->miIndex,
                        //    iter->miNumParentClusters - 1);
                    }
                }

                // has MIP 1
                ++aaMeshClusterGroups[iLODLevel][iClusterGroup].miNumMIPS;
                iNumClusters = iTotalClusterIndex;

                iLastClusterSize = static_cast<uint32_t>(aaClusterGroupVertexPositions.size());

            }   // for cluster group = 0 to num cluster groups

auto end = std::chrono::high_resolution_clock::now();
uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
DEBUG_PRINTF("took %lld seconds to split cluster groups\n", iSeconds);

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
DEBUG_PRINTF("\n************\n\ntook total %lld seconds for lod %d\n\n**************\n", iTotalLODSeconds, iLODLevel);

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
            float3 const* pClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshCluster.miVertexPositionStartArrayAddress * sizeof(float3));
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
    DEBUG_PRINTF("*** assign cluster to cluster group ***\n");
    start = std::chrono::high_resolution_clock::now();
    {
        uint32_t iClusterIndex = 0;
        for(auto* pCluster : apTotalMeshClusters)
        {
            assert(pCluster->miNumClusterGroups == 0);
            for(auto const* pClusterGroup : apTotalMeshClusterGroups)
            {
                for(uint32_t iMIP = 0; iMIP < 2; iMIP++)
                {
                    if(pClusterGroup->maiNumClusters[iMIP] >= MAX_CLUSTERS_IN_GROUP)
                    {
                        continue;
                    }
                    uint32_t iNumClustersInGroup = pClusterGroup->maiNumClusters[iMIP];
                    //printf("mip %d iNumClustersInGroup = %d cluster group %d\n", 
                    //    iMIP,
                    //    iNumClustersInGroup, 
                    //    pClusterGroup->miIndex);
                    for(uint32_t iClusterInGroup = 0; iClusterInGroup < iNumClustersInGroup; iClusterInGroup++)
                    {
                        //printf("cluster in group %d(%d) curr cluster index %d, id %d with num associated cluster groups %d\n", 
                        //    iClusterInGroup,
                        //    iNumClustersInGroup,
                        //    iClusterIndex, 
                        //    pCluster->miIndex, 
                        //    pCluster->miNumClusterGroups);
                        if(pClusterGroup->maiClusters[iMIP][iClusterInGroup] == pCluster->miIndex)
                        {
                            //printf("\tcluster group %d, cluster %d = %d\n", 
                            //    pClusterGroup->miIndex, 
                            //    iClusterInGroup, 
                            //    pCluster->miIndex);
                            if(pCluster->maiClusterGroups[0] != pClusterGroup->miIndex && pCluster->maiClusterGroups[1] != pClusterGroup->miIndex)
                            {
                                assert(pCluster->miNumClusterGroups < MAX_ASSOCIATED_GROUPS);
                                pCluster->maiClusterGroups[pCluster->miNumClusterGroups] = pClusterGroup->miIndex;

                                //DEBUG_PRINTF("cluster %d lod %d group: %d\n", pCluster->miIndex, pCluster->miLODLevel, pClusterGroup->miIndex);
                                ++pCluster->miNumClusterGroups;
                            }
                        }
                    }
                }
            }

            ++iClusterIndex;
        }
    }
    uint64_t iElapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
    DEBUG_PRINTF("Took %lld seconds to assign cluster to cluster group\n", iElapsedSeconds);

    DEBUG_PRINTF("*** start getting shortest distance from LOD 0 and cluster error terms ***\n");
    start = std::chrono::high_resolution_clock::now();

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

        DEBUG_PRINTF("*** start getting shortest distance from LOD 0\n");
        start = std::chrono::high_resolution_clock::now();

        // get cluster distances from the LOD 0 clusters
        std::vector<std::vector<std::vector<MeshClusterDistanceInfo>>> aaaMeshClusterDistanceInfo(iNumLODLevels);

        static std::atomic<uint32_t> siCurrCluster;
        uint32_t const kiMaxThreads = 12;
        for(uint32_t iLODLevel = 0; iLODLevel < iNumLODLevels; iLODLevel++)
        {
            uint32_t iNumProcessClusters = static_cast<uint32_t>(aaMeshClusters[iLODLevel].size());
            aaaMeshClusterDistanceInfo[iLODLevel].resize(iNumProcessClusters);
            
            // allocate distance info for cluster
            for(uint32_t iCluster = 0; iCluster < iNumProcessClusters; iCluster++)
            {
                aaaMeshClusterDistanceInfo[iLODLevel][iCluster].resize(aaMeshClusters[0].size());
            }

            siCurrCluster = 0;

            // compute distance info from LOD 0
            std::vector<std::unique_ptr<std::thread>> apThreads(kiMaxThreads);
            for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
            {
                apThreads[iThread] = std::make_unique<std::thread>(
                    [&aaaMeshClusterDistanceInfo,
                     aaMeshClusters,
                     iNumProcessClusters,
                     iLODLevel]()
                    {
                        for(;;)
                        {
                            uint32_t iCluster = siCurrCluster.fetch_add(1);
                            if(iCluster >= iNumProcessClusters)
                            {
                                return;
                            }

                            auto const& meshCluster = aaMeshClusters[iLODLevel][iCluster];
                            for(uint32_t iUpperCluster = 0; iUpperCluster < static_cast<uint32_t>(aaaMeshClusterDistanceInfo[iLODLevel][iCluster].size()); iUpperCluster++)
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
                    });
            }

            for(uint32_t iThread = 0; iThread < kiMaxThreads; iThread++)
            {
                if(apThreads[iThread]->joinable())
                {
                    apThreads[iThread]->join();
                }
            }
        }

        iElapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        DEBUG_PRINTF("Took %lld seconds to get the shortest distance from LOD 0\n", iElapsedSeconds);

        start = std::chrono::high_resolution_clock::now();

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

                // get vertex positions from total vertex buffer for this cluster
                auto const& meshCluster = aaMeshClusters[iLODLevel][iCluster];
                float3 const* pClusterVertexPositions = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshCluster.miVertexPositionStartArrayAddress * sizeof(float3));
                std::vector<float3> aClusterVertexPositions(meshCluster.miNumVertexPositions);
                memcpy(
                    aClusterVertexPositions.data(),
                    pClusterVertexPositions,
                    aClusterVertexPositions.size() * sizeof(float3));

                // get position indices from total index buffer for this cluster
                uint32_t const* pClusterPositionIndices = reinterpret_cast<uint32_t const*>(trianglePositionIndexBuffer.data() + meshCluster.miTrianglePositionIndexArrayAddress * sizeof(uint32_t));
                std::vector<uint32_t> aiClusterPositionIndices(meshCluster.miNumTrianglePositionIndices);
                memcpy(
                    aiClusterPositionIndices.data(),
                    pClusterPositionIndices,
                    aiClusterPositionIndices.size() * sizeof(uint32_t));

                std::vector<ClosestVertexInfo> aClosestClusterVertexPositions(meshCluster.miNumVertexPositions);
                uint32_t iIndex = 0;
                for(auto& closestVertexInfo : aClosestClusterVertexPositions)
                {
                    closestVertexInfo.miVertexIndex = iIndex;
                    closestVertexInfo.mfClosestDistance = 1.0e+10f;
                    ++iIndex;
                }

                // triangle positions of cluster
                std::vector<float3> aMeshClusterTriangleVertexPositions(meshCluster.miNumTrianglePositionIndices);
                for(uint32_t iTriVert = 0; iTriVert < meshCluster.miNumTrianglePositionIndices; iTriVert++)
                {
                    aMeshClusterTriangleVertexPositions[iTriVert] = aClusterVertexPositions[aiClusterPositionIndices[iTriVert]];
                }

                // use the top given number of closest LOD 0 clusters 
                for(uint32_t iUpperCluster = 0; iUpperCluster < kiNumUpperClustersToCheck; iUpperCluster++)
                {
                    uint32_t iUpperClusterIndex = aaaMeshClusterDistanceInfo[iLODLevel][iCluster][iUpperCluster].miClusterLOD0;

                    auto const& meshClusterLOD0 = aaMeshClusters[0][iUpperClusterIndex];

                    // check distance
                    float fDistance = length(meshCluster.mCenter - meshClusterLOD0.mCenter);

                    // vertex positions of LOD0
                    std::vector<float3> aClusterVertexPositionsLOD0(meshClusterLOD0.miNumVertexPositions);
                    float3 const* pClusterVertexPositionsLOD0 = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + meshClusterLOD0.miVertexPositionStartArrayAddress * sizeof(float3));
                    memcpy(
                        aClusterVertexPositionsLOD0.data(),
                        pClusterVertexPositionsLOD0,
                        aClusterVertexPositionsLOD0.size() * sizeof(float3));

                    // position indices of LOD0
                    std::vector<uint32_t> aiClusterVertexIndicesLOD0(meshClusterLOD0.miNumTrianglePositionIndices);
                    uint32_t const* piClusterPositionIndicesLOD0 = reinterpret_cast<uint32_t const*>(trianglePositionIndexBuffer.data() + meshClusterLOD0.miTrianglePositionIndexArrayAddress * sizeof(uint32_t));
                    memcpy(
                        aiClusterVertexIndicesLOD0.data(),
                        piClusterPositionIndicesLOD0,
                        aiClusterVertexIndicesLOD0.size() * sizeof(uint32_t));

                    // get the vertex distances from LOD 0
                    std::vector<float> afClosestDistances(meshClusterLOD0.miNumVertexPositions);
                    std::vector<uint32_t> aiClosestVertexPositions(meshClusterLOD0.miNumVertexPositions);
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

                aafClusterAverageDistanceFromLOD0[iLODLevel][iCluster] = aClosestClusterVertexPositions[0].mfClosestDistance;

                ClosestVertexInfo const& largestDistanceInfo = aClosestClusterVertexPositions.front();
                
                aafClusterDistancesFromLOD0[iLODLevel][iCluster] = largestDistanceInfo.mfClosestDistance;
                aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster].first = aClusterVertexPositions[largestDistanceInfo.miVertexIndex];
                aaMaxErrorPositionsFromLOD0[iLODLevel][iCluster].second = largestDistanceInfo.mVertexPositionLOD0;

                auto end = std::chrono::high_resolution_clock::now();
                uint64_t iSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                uint64_t iMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count();
                if(iCluster % 1000 == 0)
                {
                    DEBUG_PRINTF("took %lld ms (total: %lld sec) seconds to compute the shortest distance to LOD 0 for cluster %d (%d) of lod %d\n",
                        iMilliseconds,
                        iSeconds,
                        iCluster,
                        iNumClusters,
                        iLODLevel);
                }


            }   // for cluster = 0 to num clusters

        }   // for lod = 0 to num lod levels

    }   // cuda version of cluster vertex mapping and error terms

    iElapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
    DEBUG_PRINTF("Took %lld seconds to get distances from current LOD to LOD 0\n", iElapsedSeconds);

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

    DEBUG_PRINTF("\n\n *** compute cluster error terms *** \n\n");

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
                float3 const& pos = reinterpret_cast<float3 const*>(vertexPositionBuffer.data() + cluster.miVertexPositionStartArrayAddress * sizeof(float3))[i];

                cluster.mMinBounds = fminf(cluster.mMinBounds, pos);
                cluster.mMaxBounds = fmaxf(cluster.mMaxBounds, pos);
            }

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
                    if(iClusterID < MAX_CLUSTERS_IN_GROUP)
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

    DEBUG_PRINTF("\n\n*** set cluster and group data ***\n\n");

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
            uint8_t* pAddress = vertexPositionBuffer.data() + apTotalMeshClusters[i]->miVertexPositionStartArrayAddress * sizeof(float3);
            memcpy(
                aVertexPositions.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumVertexPositions * sizeof(float3));

            std::vector<float3> aVertexNormals(apTotalMeshClusters[i]->miNumVertexNormals);
            pAddress = vertexNormalBuffer.data() + apTotalMeshClusters[i]->miVertexNormalStartArrayAddress * sizeof(float3);
            memcpy(
                aVertexNormals.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumVertexNormals * sizeof(float3));

            std::vector<float2> aVertexUVs(apTotalMeshClusters[i]->miNumVertexUVs);
            pAddress = vertexUVBuffer.data() + apTotalMeshClusters[i]->miVertexUVStartArrayAddress * sizeof(float2);
            memcpy(
                aVertexUVs.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumVertexUVs * sizeof(float2));

            std::vector<uint32_t> aiVertexPositionIndices(apTotalMeshClusters[i]->miNumTrianglePositionIndices);
            pAddress = trianglePositionIndexBuffer.data() + apTotalMeshClusters[i]->miTrianglePositionIndexArrayAddress * sizeof(uint32_t);
            memcpy(
                aiVertexPositionIndices.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumTrianglePositionIndices * sizeof(uint32_t));

            std::vector<uint32_t> aiVertexNormalIndices(apTotalMeshClusters[i]->miNumTriangleNormalIndices);
            pAddress = triangleNormalIndexBuffer.data() + apTotalMeshClusters[i]->miTriangleNormalIndexArrayAddress * sizeof(uint32_t);
            memcpy(
                aiVertexNormalIndices.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumTriangleNormalIndices * sizeof(uint32_t));

            std::vector<uint32_t> aiVertexUVIndices(apTotalMeshClusters[i]->miNumTriangleUVIndices);
            pAddress = triangleUVIndexBuffer.data() + apTotalMeshClusters[i]->miTriangleUVIndexArrayAddress * sizeof(uint32_t);
            memcpy(
                aiVertexUVIndices.data(),
                pAddress,
                apTotalMeshClusters[i]->miNumTriangleUVIndices * sizeof(uint32_t));

        }
    }

    DEBUG_PRINTF("*** save out cluster nodes and cluster group nodes ***\n");
    start = std::chrono::high_resolution_clock::now();
    {
        std::vector<ClusterTreeNode> aClusterNodes;
        createTreeNodes2(
            aClusterNodes,
            iNumLODLevels,          
            gMeshClusterBuffer,
            gMeshClusterGroupBuffer,
            aaMeshClusterGroups,
            aaMeshClusters,
            aTotalMaxClusterDistancePositionFromLOD0);

        std::vector<ClusterGroupTreeNode> aClusterGroupNodes;
        for(auto const& node : aClusterNodes)
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
                groupNode.miLevel = node.miLevel;
                ++groupNode.miNumChildClusters;

                aClusterGroupNodes.push_back(groupNode);
            }
            else
            {
                iter->maiClusterAddress[iter->miNumChildClusters] = node.miClusterAddress;
                ++iter->miNumChildClusters;
            }

        }

        std::sort(
            aClusterGroupNodes.begin(),
            aClusterGroupNodes.end(),
            [](ClusterGroupTreeNode const& left, ClusterGroupTreeNode const& right)
            {
                return left.miClusterGroupAddress < right.miClusterGroupAddress;
            }
        );

        std::ostringstream binaryOutputFolderPath;
        {
            binaryOutputFolderPath << homeDirectory << "debug-output\\" << meshModelName << "\\";
            std::filesystem::path binaryOutputFolderFileSystemPath(binaryOutputFolderPath.str());
            if(!std::filesystem::exists(binaryOutputFolderFileSystemPath))
            {
                std::filesystem::create_directory(binaryOutputFolderFileSystemPath);
            }
        }

        // save cluster info
        {
            saveClusterGroupTreeNodes(
                binaryOutputFolderPath.str() + "cluster-group-tree.bin",
                aClusterGroupNodes);
        }

        // load cluster info
        std::vector<ClusterGroupTreeNode> aTestClusterGroupTreeNodes;
        {
            loadClusterGroupTreeNodes(
                aTestClusterGroupTreeNodes, 
                binaryOutputFolderPath.str() + "cluster-group-tree.bin");
        }

        {
            saveClusterTreeNodes(
                binaryOutputFolderPath.str() + "cluster-tree.bin",
                aClusterNodes);
        }

        std::vector<ClusterTreeNode> aTestClusterTreeNodes;
        {
            loadClusterTreeNodes(
                aTestClusterTreeNodes,
                binaryOutputFolderPath.str() + "cluster-tree.bin");
        }

        {
            saveMeshClusters(
                binaryOutputFolderPath.str() + "mesh-clusters.bin",
                apTotalMeshClusters);
        }

        std::vector<MeshCluster> aTestMeshClusters;
        {
            loadMeshClusters(
                aTestMeshClusters,
                binaryOutputFolderPath.str() + "mesh-clusters.bin");
        }

        saveMeshClusterData(
            vertexPositionBuffer,
            vertexNormalBuffer,
            vertexUVBuffer,
            trianglePositionIndexBuffer,
            triangleNormalIndexBuffer,
            triangleUVIndexBuffer,
            apTotalMeshClusters,
            binaryOutputFolderPath.str() + "mesh-cluster-data.bin");

        saveMeshClusterTriangleData(
            vertexPositionBuffer,
            vertexNormalBuffer,
            vertexUVBuffer,
            trianglePositionIndexBuffer,
            triangleNormalIndexBuffer,
            triangleUVIndexBuffer,
            apTotalMeshClusters,
            binaryOutputFolderPath.str() + "mesh-cluster-triangle-vertex-index-data.bin",
            binaryOutputFolderPath.str() + "mesh-cluster-triangle-vertex-data.bin",
            binaryOutputFolderPath.str() + "mesh-cluster-triangle-index-data.bin");
        
        uint64_t iElapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        DEBUG_PRINTF("Took %lld seconds to save out cluster files\n", iElapsedSeconds);

        {
            std::vector<std::vector<ConvertedMeshVertexFormat>> aaVertices;
            std::vector<std::vector<uint32_t>> aaiTriangleVertexIndices;
            
            std::vector<uint32_t> aiNumClusterVertices;
            std::vector<uint32_t> aiNumClusterIndices;
            std::vector<uint64_t> aiVertexBufferArrayOffsets;
            std::vector<uint64_t> aiIndexBufferArrayOffset;
            loadMeshClusterTriangleDataTableOfContent(
                aiNumClusterVertices,
                aiNumClusterIndices,
                aiVertexBufferArrayOffsets,
                aiIndexBufferArrayOffset,
                binaryOutputFolderPath.str() + "mesh-cluster-triangle-vertex-data.bin",
                binaryOutputFolderPath.str() + "mesh-cluster-triangle-index-data.bin");

            aaVertices.resize(aiNumClusterIndices.size());
            aaiTriangleVertexIndices.resize(aiNumClusterIndices.size());
            for(uint32_t i = 0; i < aiNumClusterIndices.size(); i++)
            {
                loadMeshClusterTriangleDataChunk(
                    aaVertices[i],
                    aaiTriangleVertexIndices[i],
                    binaryOutputFolderPath.str() + "mesh-cluster-triangle-vertex-data.bin",
                    binaryOutputFolderPath.str() + "mesh-cluster-triangle-index-data.bin",
                    aiNumClusterVertices,
                    aiNumClusterIndices,
                    aiVertexBufferArrayOffsets,
                    aiIndexBufferArrayOffset,
                    i);
            }

            // verify data
            {
                assert(apTotalMeshClusters.size() == aaVertices.size());
                for(uint32_t iMeshCluster = 0; iMeshCluster < static_cast<uint32_t>(apTotalMeshClusters.size()); iMeshCluster++)
                {
                    MeshCluster const* pMeshCluster = apTotalMeshClusters[iMeshCluster];
                    std::vector<float3> aPositions(pMeshCluster->miNumVertexPositions);
                    std::vector<float3> aNormals(pMeshCluster->miNumVertexNormals);
                    std::vector<float2> aUVs(pMeshCluster->miNumVertexUVs);

                    std::vector<uint32_t> aiPositionIndices(pMeshCluster->miNumTrianglePositionIndices);
                    std::vector<uint32_t> aiNormalIndices(pMeshCluster->miNumTriangleNormalIndices);
                    std::vector<uint32_t> aiUVIndices(pMeshCluster->miNumTriangleUVIndices);

                    memcpy(
                        aPositions.data(),
                        vertexPositionBuffer.data() + pMeshCluster->miVertexPositionStartArrayAddress * sizeof(float3),
                        pMeshCluster->miNumVertexPositions * sizeof(float3));
                    memcpy(
                        aNormals.data(),
                        vertexNormalBuffer.data() + pMeshCluster->miVertexNormalStartArrayAddress * sizeof(float3),
                        pMeshCluster->miNumVertexNormals * sizeof(float3));
                    memcpy(
                        aUVs.data(),
                        vertexUVBuffer.data() + pMeshCluster->miVertexUVStartArrayAddress * sizeof(float2),
                        pMeshCluster->miNumVertexUVs * sizeof(float2));

                    memcpy(
                        aiPositionIndices.data(),
                        trianglePositionIndexBuffer.data() + pMeshCluster->miTrianglePositionIndexArrayAddress * sizeof(uint32_t),
                        pMeshCluster->miNumTrianglePositionIndices * sizeof(uint32_t));
                    memcpy(
                        aiNormalIndices.data(),
                        triangleNormalIndexBuffer.data() + pMeshCluster->miTriangleNormalIndexArrayAddress * sizeof(uint32_t),
                        pMeshCluster->miNumTriangleNormalIndices * sizeof(uint32_t));
                    memcpy(
                        aiUVIndices.data(),
                        triangleUVIndexBuffer.data() + pMeshCluster->miTriangleUVIndexArrayAddress * sizeof(uint32_t),
                        pMeshCluster->miNumTriangleUVIndices * sizeof(uint32_t));

                    for(uint32_t iTri = 0; iTri < pMeshCluster->miNumTrianglePositionIndices; iTri += 3)
                    {
                        assert(aiPositionIndices[iTri] < aPositions.size());
                        assert(aiNormalIndices[iTri] < aNormals.size());
                        assert(aiUVIndices[iTri] < aUVs.size());

                        for(uint32_t i = 0; i < 3; i++)
                        {
                            ConvertedMeshVertexFormat const& vertex = aaVertices[iMeshCluster][aaiTriangleVertexIndices[iMeshCluster][iTri + i]];

                            float3 const& pos = aPositions[aiPositionIndices[iTri + i]];
                            float3 const& norm = aNormals[aiNormalIndices[iTri + i]];
                            float2 const& uv = aUVs[aiUVIndices[iTri + i]];

                            float fLength0 = lengthSquared(float3(vertex.mPosition) - pos);
                            float fLength1 = lengthSquared(float3(vertex.mNormal) - norm);
                            float fLength2 = lengthSquared(vertex.mUV - vertex.mUV);

                            assert(fLength0 <= 1.0e-8f);
                            assert(fLength1 <= 1.0e-8f);
                            assert(fLength2 <= 1.0e-8f);

                        }
                    }

                }
            }   // verify data
        }
    }

    return 0;
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
    uint32_t iNumClusterGroups,
    uint32_t iNumClusters,
    uint32_t iLODLevel,
    std::string const& meshClusterOutputName,
    std::string const& homeDirectory,
    std::string const& meshModelName)
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

    // output cluster group obj
    {
        for(uint32_t iClusterGroup = 0; iClusterGroup < iNumClusterGroups; iClusterGroup++)
        {
            assert(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size() == aaiClusterGroupTriangleNormalIndices[iClusterGroup].size());
            assert(aaiClusterGroupTrianglePositionIndices[iClusterGroup].size() == aaiClusterGroupTriangleUVIndices[iClusterGroup].size());

            std::ostringstream clusterGroupName;
            clusterGroupName << "cluster-group-lod" << iLODLevel << "-group" << iClusterGroup;

            std::ostringstream clusterGroupFilePath;
            clusterGroupFilePath << homeDirectory << "cluster-groups\\" << meshModelName << "\\";
            if(!std::filesystem::exists(clusterGroupFilePath.str()))
            {
                std::filesystem::create_directory(clusterGroupFilePath.str());
            }

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
            clusterGroupMaterialFilePath << homeDirectory << "cluster-groups\\" << meshModelName << "\\";
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


