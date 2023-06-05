#pragma once

#include <string>
#include <vector>

/*
**
*/
void buildAdjacencyList(
    tinyobj::attrib_t& attrib,
    std::vector<tinyobj::shape_t>& aShapes,
    std::vector<tinyobj::material_t>& aMaterials,
    std::vector<std::vector<uint32_t>>& aaiAdjacencyList,
    std::string const& outputFilePath,
    std::string const& fullOBJFilePath);