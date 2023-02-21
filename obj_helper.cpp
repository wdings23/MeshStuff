#include "obj_helper.h"

#include <sstream>

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