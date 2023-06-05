#include "obj_helper.h"

#include <cassert>
#include <sstream>
#include <filesystem>

/*
**
*/
void writeOBJFile(
    std::vector<float3> const& aVertexPositions,
    std::vector<float3> const& aVertexNormals,
    std::vector<float2> const& aVertexUV,
    std::vector<uint32_t> const& aiPositionIndices,
    std::vector<uint32_t> const& aiNormalIndices,
    std::vector<uint32_t> const& aiUVIndices,
    std::string const& outputFilePath,
    std::string const& objectName)
{
    auto end = outputFilePath.find_last_of("\\");
    std::string directory = outputFilePath.substr(0, end);

    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fprintf(fp, "# num positions: %d num normals: %d num uvs: %d\n",
        static_cast<uint32_t>(aVertexPositions.size()),
        static_cast<uint32_t>(aVertexNormals.size()),
        static_cast<uint32_t>(aVertexUV.size()));
    fprintf(fp, "o %s\n", objectName.c_str());
    fprintf(fp, "usemtl %s\n", objectName.c_str());
    for(auto const& pos : aVertexPositions)
    {
        fprintf(fp, "v %.4f %.4f %.4f\n", pos.x, pos.y, pos.z);
    }

    for(auto const& norm : aVertexNormals)
    {
        fprintf(fp, "vn %.4f %.4f %.4f\n", norm.x, norm.y, norm.z);
    }

    for(auto const& uv : aVertexUV)
    {
        fprintf(fp, "vt %.4f %.4f\n", uv.x, uv.y);
    }

    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiPositionIndices.size()); iTri += 3)
    {
        fprintf(fp, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
            aiPositionIndices[iTri] + 1,
            aiUVIndices[iTri] + 1,
            aiNormalIndices[iTri] + 1,
            aiPositionIndices[iTri + 1] + 1,
            aiUVIndices[iTri + 1] + 1,
            aiNormalIndices[iTri + 1] + 1,
            aiPositionIndices[iTri + 2] + 1,
            aiUVIndices[iTri + 2] + 1,
            aiNormalIndices[iTri + 2] + 1);
    }

    fclose(fp);

    float fRand0 = static_cast<float>(rand() % 255) / 255.0f;
    float fRand1 = static_cast<float>(rand() % 255) / 255.0f;
    float fRand2 = static_cast<float>(rand() % 255) / 255.0f;

    std::ostringstream materialFilePath;
    materialFilePath << directory << "\\" << objectName << ".mtl";
    fp = fopen(materialFilePath.str().c_str(), "wb");
    fprintf(fp, "newmtl %s\n", objectName.c_str());
    fprintf(fp, "Kd %.4f %.4f %.4f\n", fRand0, fRand1, fRand2);
    fclose(fp);

}

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
            FILE* materialFP = fopen(materialOutputFullPath.c_str(), "wb");
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