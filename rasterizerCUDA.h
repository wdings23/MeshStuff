#pragma once
void rasterizeMeshCUDA(
    std::vector<vec4>& retColorBuffer,
    std::vector<vec3>& retNormalBuffer,
    std::vector<float>& retDepthBuffer,
    std::vector<vec3> const& aVertexPositions,
    std::vector<vec3> const& aVertexNormals,
    std::vector<vec2> const& aVertexUVs,
    std::vector<uint32_t> const& aiVertexPositionIndices,
    std::vector<uint32_t> const& aiVertexNormalIndices,
    std::vector<uint32_t> const& aiVertexUVIndices,
    std::vector<vec4> const& inputColorBuffer,
    std::vector<float> const& inputDepthBuffer);


void rasterizeMeshCUDA2(
    std::vector<vec3>& retLightIntensityBuffer,
    std::vector<vec3>& retPositionBuffer,
    std::vector<vec3>& retNormalBuffer,
    std::vector<float>& retDepthBuffer,
    std::vector<vec3>& retColorBuffer,
    std::vector<vec3> const& aVertexPositions,
    std::vector<vec3> const& aVertexNormals,
    std::vector<vec3> const& aVertexColors,
    uint32_t iImageWidth,
    uint32_t iImageHeight,
    uint32_t iImageFormatSize);