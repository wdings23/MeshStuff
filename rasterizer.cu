#include <cuda_runtime.h>
#include <algorithm>

#define uint32_t unsigned int
#define int32_t int

inline __host__ __device__ float3 operator + (float3& a, float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator - (float3& a, float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator * (float3& a, float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator / (float3& a, float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 operator * (float3& a, float& b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ float dot(float3& a, float3& b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.y);
}

inline __device__ __host__ float3 barycentric(
    float3& p, 
    float3& a, 
    float3& b, 
    float3& c)
{
    float3 v0 = b - a, v1 = c - a, v2 = p - a;
    float fD00 = dot(v0, v0);
    float fD01 = dot(v0, v1);
    float fD11 = dot(v1, v1);
    float fD20 = dot(v2, v0);
    float fD21 = dot(v2, v1);
    float fDenom = fD00 * fD11 - fD01 * fD01;
    float fV = (fD11 * fD20 - fD01 * fD21) / fDenom;
    float fW = (fD00 * fD21 - fD01 * fD20) / fDenom;
    float fU = 1.0f - fV - fW;

    return make_float3(fU, fV, fW);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

/*
**
*/
__device__ 
void _rasterizeTriangle(
    float* pafPositionBuffer,
    float* pafNormalBuffer,
    float* pafDepthBuffer,
    float* pafColorBuffer,
    float3 pos0,
    float3 pos1,
    float3 pos2,
    float3 normal0,
    float3 normal1,
    float3 normal2,
    float3 color0,
    float3 color1,
    float3 color2,
    uint32_t iBufferWidth,
    uint32_t iBufferHeight,
    uint32_t iTriangle)
{
    float3 screenDimension = make_float3(float(iBufferWidth), float(iBufferHeight), 0.0f);

    float3 screenCoord0 = pos0 * screenDimension;
    float3 screenCoord1 = pos1 * screenDimension;
    float3 screenCoord2 = pos2 * screenDimension;

    float3 minScreenPos = fminf(screenCoord0, fminf(screenCoord1, screenCoord2));
    float3 maxScreenPos = fmaxf(screenCoord0, fmaxf(screenCoord1, screenCoord2));

    screenCoord0.x = (screenCoord0.x == minScreenPos.x) ? floorf(screenCoord0.x) : screenCoord0.x;
    screenCoord0.y = (screenCoord0.y == minScreenPos.y) ? floorf(screenCoord0.y) : screenCoord0.y;
    screenCoord0.x = (screenCoord0.x == maxScreenPos.x) ? ceilf(screenCoord0.x) : screenCoord0.x;
    screenCoord0.y = (screenCoord0.y == maxScreenPos.y) ? ceilf(screenCoord0.y) : screenCoord0.y;

    screenCoord1.x = (screenCoord1.x == minScreenPos.x) ? floorf(screenCoord1.x) : screenCoord1.x;
    screenCoord1.y = (screenCoord1.y == minScreenPos.y) ? floorf(screenCoord1.y) : screenCoord1.y;
    screenCoord1.x = (screenCoord1.x == maxScreenPos.x) ? ceilf(screenCoord1.x) : screenCoord1.x;
    screenCoord1.y = (screenCoord1.y == maxScreenPos.y) ? ceilf(screenCoord1.y) : screenCoord1.y;

    screenCoord2.x = (screenCoord2.x == minScreenPos.x) ? floorf(screenCoord2.x) : screenCoord2.x;
    screenCoord2.y = (screenCoord2.y == minScreenPos.y) ? floorf(screenCoord2.y) : screenCoord2.y;
    screenCoord2.x = (screenCoord2.x == maxScreenPos.x) ? ceilf(screenCoord2.x) : screenCoord2.x;
    screenCoord2.y = (screenCoord2.y == maxScreenPos.y) ? ceilf(screenCoord2.y) : screenCoord2.y;

    uint32_t iScreenX0 = clamp(static_cast<uint32_t>(floorf(minScreenPos.x)), 0, iBufferWidth - 1);
    uint32_t iScreenX1 = clamp(static_cast<uint32_t>(ceilf(maxScreenPos.x)), 0, iBufferWidth - 1);

    uint32_t iScreenY0 = clamp(static_cast<uint32_t>(floorf(minScreenPos.y)), 0, iBufferHeight - 1);
    uint32_t iScreenY1 = clamp(static_cast<uint32_t>(ceilf(maxScreenPos.y)), 0, iBufferHeight - 1);

    // compute barycentric coordinate within the face 2d boundary to fetch the clipspace position and normal
    for(uint32_t iY = iScreenY0; iY <= iScreenY1; iY++)
    {
        for(uint32_t iX = iScreenX0; iX <= iScreenX1; iX++)
        {
            float3 currPos = make_float3(float(iX), float(iY), 0.0f);
            float3 barycentricCoord = barycentric(
                currPos,
                screenCoord0,
                screenCoord1,
                screenCoord2);

            if(barycentricCoord.x >= 0.0f && barycentricCoord.y >= 0.0f && barycentricCoord.x + barycentricCoord.y <= 1.0f)
            {
                // check depth buffer (larger than incoming depth -> replace)
                float3 currPos = pos0 * barycentricCoord.x + pos1 * barycentricCoord.y + pos2 * barycentricCoord.z;
                float3 normal = normal0 * barycentricCoord.x + normal1 * barycentricCoord.y + normal2 * barycentricCoord.z;
                uint32_t iIndex = iY * iBufferWidth + iX;
                if(pafDepthBuffer[iIndex] > currPos.z)
                {
                    pafPositionBuffer[iIndex * 3] = currPos.x;
                    pafPositionBuffer[iIndex * 3 + 1] = currPos.y;
                    pafPositionBuffer[iIndex * 3 + 2] = currPos.z;

                    pafNormalBuffer[iIndex * 3] = normal.x;
                    pafNormalBuffer[iIndex * 3 + 1] = normal.y;
                    pafNormalBuffer[iIndex * 3 + 2] = normal.z;

                    pafDepthBuffer[iIndex] = currPos.z;

                    pafColorBuffer[iIndex * 3] = color0.x;
                    pafColorBuffer[iIndex * 3 + 1] = color0.y;
                    pafColorBuffer[iIndex * 3 + 2] = color0.z;
                }
            }
        }
    }
}

/*
**
*/
__global__
void _rasterizeMesh(
    float* paPositionBuffer,
    float* paNormalBuffer,
    float* paDepthBuffer,
    float* paColorBuffer,
    float* paVertexPositions,
    float* paVertexNormals,
    float* paVertexUVs,
    uint32_t* paiVertexPositionIndices,
    uint32_t* paiVertexNormalIndices,
    uint32_t* paiVertexUVIndices,
    uint32_t* paiCountBuffers,
    uint32_t iBufferWidth,
    uint32_t iBufferHeight)
{
    uint32_t iNumTriangleVertices = paiCountBuffers[0];
    uint32_t iNumTriangles = iNumTriangleVertices / 3;

    uint32_t iTriangle = blockIdx.x * 512 + threadIdx.x;
    if(iTriangle >= iNumTriangles)
    {
        return;
    }

    uint32_t iIndex = iTriangle * 3;
    uint32_t iPos0 = paiVertexPositionIndices[iIndex] * 3;
    uint32_t iPos1 = paiVertexPositionIndices[iIndex + 1] * 3;
    uint32_t iPos2 = paiVertexPositionIndices[iIndex + 2] * 3;

    float3 pos0 = make_float3(
        paVertexPositions[iPos0], 
        paVertexPositions[iPos0 + 1], 
        paVertexPositions[iPos0 + 2]);
    
    float3 pos1 = make_float3(
        paVertexPositions[iPos1],
        paVertexPositions[iPos1 + 1],
        paVertexPositions[iPos1 + 2]);

    float3 pos2 = make_float3(
        paVertexPositions[iPos2],
        paVertexPositions[iPos2 + 1],
        paVertexPositions[iPos2 + 2]);

    uint32_t iNorm0 = paiVertexNormalIndices[iIndex] * 3;
    uint32_t iNorm1 = paiVertexNormalIndices[iIndex + 1] * 3;
    uint32_t iNorm2 = paiVertexNormalIndices[iIndex + 2] * 3;

    float3 normal0 = make_float3(
        paVertexNormals[iNorm0],
        paVertexNormals[iNorm0 + 1],
        paVertexNormals[iNorm0 + 2]);

    float3 normal1 = make_float3(
        paVertexNormals[iNorm1],
        paVertexNormals[iNorm1 + 1],
        paVertexNormals[iNorm1 + 2]);

    float3 normal2 = make_float3(
        paVertexNormals[iNorm2],
        paVertexNormals[iNorm2 + 1],
        paVertexNormals[iNorm2 + 2]);

    float3 color0 = make_float3(1.0f, 1.0f, 1.0f);
    float3 color1 = make_float3(1.0f, 1.0f, 1.0f);
    float3 color2 = make_float3(1.0f, 1.0f, 1.0f);

    _rasterizeTriangle(
        paPositionBuffer,
        paNormalBuffer,
        paDepthBuffer,
        paColorBuffer,
        pos0,
        pos1,
        pos2,
        normal0,
        normal1,
        normal2,
        color0,
        color1,
        color2,
        iBufferWidth,
        iBufferHeight,
        iTriangle);

}

/*
**
*/
__global__
void _rasterizeMesh2(
    float* paPositionBuffer,
    float* paNormalBuffer,
    float* paDepthBuffer,
    float* paColorBuffer,
    float* paVertexPositions,
    float* paVertexNormals,
    float* paVertexColors,
    uint32_t* paiCountBuffers,
    uint32_t iBufferWidth,
    uint32_t iBufferHeight)
{
    uint32_t iNumTriangleVertices = paiCountBuffers[0];
    uint32_t iNumTriangles = iNumTriangleVertices / 3;

    uint32_t iTriangle = blockIdx.x * 512 + threadIdx.x;
    if(iTriangle >= iNumTriangles)
    {
        return;
    }

    uint32_t iPos0 = iTriangle * 9;
    uint32_t iPos1 = iPos0 + 3;
    uint32_t iPos2 = iPos1 + 3;

    float3 pos0 = make_float3(
        paVertexPositions[iPos0],
        paVertexPositions[iPos0 + 1],
        paVertexPositions[iPos0 + 2]);

    float3 pos1 = make_float3(
        paVertexPositions[iPos1],
        paVertexPositions[iPos1 + 1],
        paVertexPositions[iPos1 + 2]);

    float3 pos2 = make_float3(
        paVertexPositions[iPos2],
        paVertexPositions[iPos2 + 1],
        paVertexPositions[iPos2 + 2]);

    uint32_t iNorm0 = iPos0;
    uint32_t iNorm1 = iPos1;
    uint32_t iNorm2 = iPos2;

    float3 normal0 = make_float3(
        paVertexNormals[iNorm0],
        paVertexNormals[iNorm0 + 1],
        paVertexNormals[iNorm0 + 2]);

    float3 normal1 = make_float3(
        paVertexNormals[iNorm1],
        paVertexNormals[iNorm1 + 1],
        paVertexNormals[iNorm1 + 2]);

    float3 normal2 = make_float3(
        paVertexNormals[iNorm2],
        paVertexNormals[iNorm2 + 1],
        paVertexNormals[iNorm2 + 2]);

    float3 color0 = make_float3(
        paVertexColors[iPos0],
        paVertexColors[iPos0 + 1],
        paVertexColors[iPos0 + 2]);

    float3 color1 = make_float3(
        paVertexColors[iPos1],
        paVertexColors[iPos1 + 1],
        paVertexColors[iPos1 + 2]);

    float3 color2 = make_float3(
        paVertexColors[iPos2],
        paVertexColors[iPos2 + 1],
        paVertexColors[iPos2 + 2]);

    _rasterizeTriangle(
        paPositionBuffer,
        paNormalBuffer,
        paDepthBuffer,
        paColorBuffer,
        pos0,
        pos1,
        pos2,
        normal0,
        normal1,
        normal2,
        color0,
        color1,
        color2,
        iBufferWidth,
        iBufferHeight,
        iTriangle);

}

/*
**
*/
__global__
void compositeImage(
    float* paColorBuffer,
    float* paPositionBuffer,
    float* paNormalBuffer,
    float* paAlbedoBuffer,
    float* pLightDirection,
    uint32_t iImageWidth,
    uint32_t iImageHeight)
{
    uint32_t iPixel = blockIdx.x * 512 + threadIdx.x;
    if(iPixel >= iImageWidth * iImageHeight)
    {
        return;
    }

    uint32_t iPixelIndex = iPixel * 3;
    float3 position = make_float3(paPositionBuffer[iPixelIndex], paPositionBuffer[iPixelIndex + 1], paPositionBuffer[iPixelIndex + 2]);
    float3 normal = make_float3(paNormalBuffer[iPixelIndex], paNormalBuffer[iPixelIndex + 1], paNormalBuffer[iPixelIndex + 2]);
    float3 color = make_float3(paAlbedoBuffer[iPixelIndex], paAlbedoBuffer[iPixelIndex + 1], paAlbedoBuffer[iPixelIndex + 2]);
    if(normal.x == 0.0f && normal.y == 0.0f && normal.z == 0.0f)
    {
        return;
    }

    float3 lightDirection = make_float3(pLightDirection[0], pLightDirection[1], pLightDirection[2]);
    
    float fAmbient = 0.0f;

    float fIntensity = max(dot(lightDirection, normal), 0.0f);
    paColorBuffer[iPixelIndex] =        fIntensity * color.x + fAmbient;
    paColorBuffer[iPixelIndex + 1] =    fIntensity * color.y + fAmbient;
    paColorBuffer[iPixelIndex + 2] =    fIntensity * color.z + fAmbient;
}

#undef uint32_t
#undef int32_t


#include "vec.h"
#include <vector>

#if 0
/*
**
*/
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
    std::vector<vec3> const& inputColorBuffer,
    std::vector<vec3> const& inputNormalBuffer, 
    std::vector<float> const& inputDepthBuffer)
{
    float* paVertexPositions;
    cudaMalloc(
        &paVertexPositions,
        aVertexPositions.size() * 3 * sizeof(float));
    cudaMemcpy(
        paVertexPositions,
        aVertexPositions.data(),
        aVertexPositions.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paVertexNormals;
    cudaMalloc(
        &paVertexNormals,
        aVertexNormals.size() * 3 * sizeof(float));
    cudaMemcpy(
        paVertexNormals,
        aVertexNormals.data(),
        aVertexNormals.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paVertexUVs;
    cudaMalloc(
        &paVertexUVs,
        aVertexUVs.size() * 2 * sizeof(float));
    cudaMemcpy(
        paVertexUVs,
        aVertexUVs.data(),
        aVertexUVs.size() * 2 * sizeof(float),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexPositionIndices;
    cudaMalloc(
        &paiVertexPositionIndices,
        aiVertexPositionIndices.size() * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexPositionIndices,
        aiVertexPositionIndices.data(),
        aiVertexPositionIndices.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexNormalIndices;
    cudaMalloc(
        &paiVertexNormalIndices,
        aiVertexNormalIndices.size() * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexNormalIndices,
        aiVertexNormalIndices.data(),
        aiVertexNormalIndices.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    uint32_t* paiVertexUVIndices;
    cudaMalloc(
        &paiVertexUVIndices,
        aiVertexUVIndices.size() * sizeof(uint32_t));
    cudaMemcpy(
        paiVertexUVIndices,
        aiVertexUVIndices.data(),
        aiVertexUVIndices.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    uint32_t const kiImageWidth = 1024;
    uint32_t const kiImageHeight = 1024;
    uint32_t const kiFormatSize = 3;

    float* paPositionBuffer;
    cudaMalloc(
        &paPositionBuffer,
        kiImageWidth * kiImageHeight * kiFormatSize * sizeof(float));
    cudaMemcpy(
        paPositionBuffer,
        inputColorBuffer.data(),
        kiImageWidth * kiImageHeight * kiFormatSize * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paNormalBuffer;
    cudaMalloc(
        &paNormalBuffer,
        kiImageWidth * kiImageHeight * kiFormatSize * sizeof(float));
    cudaMemcpy(
        paNormalBuffer,
        inputNormalBuffer.data(),
        kiImageWidth * kiImageHeight * kiFormatSize * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paDepthBuffer;
    cudaMalloc(
        &paDepthBuffer,
        kiImageWidth * kiImageHeight * sizeof(float));
    cudaMemcpy(
        paDepthBuffer,
        inputDepthBuffer.data(),
        kiImageWidth * kiImageHeight * sizeof(float),
        cudaMemcpyHostToDevice);

    std::vector<uint32_t> aiCount(3);
    aiCount[0] = static_cast<uint32_t>(aiVertexPositionIndices.size());
    aiCount[1] = static_cast<uint32_t>(aiVertexNormalIndices.size());
    aiCount[2] = static_cast<uint32_t>(aiVertexUVIndices.size());

    uint32_t* paiCountBuffers;
    cudaMalloc(
        &paiCountBuffers,
        64 * sizeof(uint32_t));
    cudaMemcpy(
        paiCountBuffers,
        aiCount.data(),
        3 * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    uint32_t iNumBlocks = max((aiCount[0] / 3) / 512, 1);
    _rasterizeMesh<<<iNumBlocks, 512>>>(
        paPositionBuffer,
        paNormalBuffer,
        paDepthBuffer,
        paColorBuffer,
        paVertexPositions,
        paVertexNormals,
        paVertexUVs,
        paiVertexPositionIndices,
        paiVertexNormalIndices,
        paiVertexUVIndices,
        paiCountBuffers,
        kiImageWidth,
        kiImageHeight);

    cudaMemcpy(
        retColorBuffer.data(),
        paPositionBuffer,
        kiImageWidth * kiImageHeight * kiFormatSize * sizeof(float),
        cudaMemcpyDeviceToHost);
    
    cudaMemcpy(
        retNormalBuffer.data(),
        paNormalBuffer,
        kiImageWidth * kiImageHeight * kiFormatSize * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        retDepthBuffer.data(),
        paDepthBuffer,
        kiImageWidth * kiImageHeight * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaFree(paVertexPositions);
    cudaFree(paVertexNormals);
    cudaFree(paVertexUVs);
    cudaFree(paiVertexPositionIndices);
    cudaFree(paiVertexNormalIndices);
    cudaFree(paiVertexUVIndices);

    cudaFree(paiCountBuffers);
    cudaFree(paPositionBuffer);
    cudaFree(paNormalBuffer);
    cudaFree(paDepthBuffer);
}
#endif // #if 0

/*
**
*/
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
    uint32_t iImageFormatSize)
{
    float* paVertexPositions;
    cudaMalloc(
        &paVertexPositions,
        aVertexPositions.size() * 3 * sizeof(float));
    cudaMemcpy(
        paVertexPositions,
        aVertexPositions.data(),
        aVertexPositions.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paVertexNormals;
    cudaMalloc(
        &paVertexNormals,
        aVertexNormals.size() * 3 * sizeof(float));
    cudaMemcpy(
        paVertexNormals,
        aVertexNormals.data(),
        aVertexNormals.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paVertexColors;
    cudaMalloc(
        &paVertexColors,
        aVertexColors.size() * 3 * sizeof(float));
    cudaMemcpy(
        paVertexColors,
        aVertexColors.data(),
        aVertexColors.size() * 3 * sizeof(float),
        cudaMemcpyHostToDevice);
    

    float* paPositionBuffer;
    cudaMalloc(
        &paPositionBuffer,
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float));
    cudaMemcpy(
        paPositionBuffer,
        retPositionBuffer.data(),
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paNormalBuffer;
    cudaMalloc(
        &paNormalBuffer,
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float));
    cudaMemcpy(
        paNormalBuffer,
        retNormalBuffer.data(),
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paDepthBuffer;
    cudaMalloc(
        &paDepthBuffer,
        iImageWidth * iImageHeight * sizeof(float));
    cudaMemcpy(
        paDepthBuffer,
        retDepthBuffer.data(),
        iImageWidth * iImageHeight * sizeof(float),
        cudaMemcpyHostToDevice);

    float* paColorBuffer;
    cudaMalloc(
        &paColorBuffer,
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float));
    cudaMemcpy(
        paColorBuffer,
        retColorBuffer.data(),
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float),
        cudaMemcpyHostToDevice);
    
    std::vector<uint32_t> aiCount(3);
    aiCount[0] = static_cast<uint32_t>(aVertexPositions.size());
    aiCount[1] = static_cast<uint32_t>(aVertexPositions.size());
    aiCount[2] = static_cast<uint32_t>(aVertexPositions.size());

    uint32_t* paiCountBuffers;
    cudaMalloc(
        &paiCountBuffers,
        64 * sizeof(uint32_t));
    cudaMemcpy(
        paiCountBuffers,
        aiCount.data(),
        3 * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    uint32_t iNumBlocks = max(static_cast<uint32_t>(ceilf(float(aiCount[0] / 3) / 512.0f)), 1);
    _rasterizeMesh2<<<iNumBlocks, 512>>>(
        paPositionBuffer,
        paNormalBuffer,
        paDepthBuffer,
        paColorBuffer,
        paVertexPositions,
        paVertexNormals,
        paVertexColors,
        paiCountBuffers,
        iImageWidth,
        iImageHeight);

    
    vec3 lightDirection = normalize(vec3(1.0f, 1.0f, 1.0f));
    float* pLightDirection;
    cudaMalloc(
        &pLightDirection,
        sizeof(float) * 4);
    cudaMemcpy(
        pLightDirection,
        &lightDirection,
        sizeof(float) * 3,
        cudaMemcpyHostToDevice);
    
    float* pLightOutputBuffer;
    cudaMalloc(
        &pLightOutputBuffer,
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float));
    cudaMemset(
        pLightOutputBuffer,
        0,
        iImageWidth* iImageHeight* iImageFormatSize * sizeof(float));
    
    iNumBlocks = max(static_cast<uint32_t>(ceilf(float(iImageWidth * iImageHeight) / 512.0f)), 1);
    compositeImage<<<iNumBlocks, 512>>>(
        pLightOutputBuffer,
        paPositionBuffer,
        paNormalBuffer,
        paColorBuffer,
        pLightDirection,
        iImageWidth,
        iImageHeight);

    cudaMemcpy(
        retLightIntensityBuffer.data(),
        pLightOutputBuffer,
        iImageWidth* iImageHeight* iImageFormatSize * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        retPositionBuffer.data(),
        paPositionBuffer,
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        retNormalBuffer.data(),
        paNormalBuffer,
        iImageWidth * iImageHeight * iImageFormatSize * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        retDepthBuffer.data(),
        paDepthBuffer,
        iImageWidth * iImageHeight * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        retColorBuffer.data(),
        paColorBuffer,
        iImageWidth* iImageHeight* iImageFormatSize * sizeof(float),
        cudaMemcpyDeviceToHost);

    cudaFree(paVertexPositions);
    cudaFree(paVertexNormals);
    cudaFree(paVertexColors);

    cudaFree(paiCountBuffers);
    cudaFree(paPositionBuffer);
    cudaFree(paNormalBuffer);
    cudaFree(paDepthBuffer);
    cudaFree(paColorBuffer);
    cudaFree(pLightOutputBuffer);

    cudaFree(pLightDirection);
    
}