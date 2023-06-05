#include "tiny_obj_loader.h"
#include "adjacency_operations.h"

#include "LogPrint.h"


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