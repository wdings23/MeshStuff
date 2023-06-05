#include "vec.h"
#include "Camera.h"

#include "barycentric.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include <map>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include "stb_image_write.h"


#include "rasterizerCUDA.h"

void testCameraDistanceScales(
    std::map<uint32_t, float>& aCameraDistanceScales,
    std::vector<float3> const& aVertexPositions,
    std::vector<uint32_t> const& aiTriangles);

struct face
{
    uint32_t maiIndices[3];
    uint32_t miNumVerts;
};

/*
**
*/
void rasterizeTriangle(
    std::vector<float3>& aOutputBuffer,
    std::vector<float3>& aNormalBuffer,
    std::vector<float>& afDepthBuffer,
    float3 const& pos0,
    float3 const& pos1,
    float3 const& pos2,
    float3 const& normal0,
    float3 const& normal1,
    float3 const& normal2,
    uint32_t iBufferWidth,
    uint32_t iBufferHeight,
    uint32_t iTriangle)
{
    float3 screenCoord0 = pos0 * float3(float(iBufferWidth), float(iBufferHeight), 0.0f);
    float3 screenCoord1 = pos1 * float3(float(iBufferWidth), float(iBufferHeight), 0.0f);
    float3 screenCoord2 = pos2 * float3(float(iBufferWidth), float(iBufferHeight), 0.0f);

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

    float fDX = float(iScreenX1 - iScreenX0);
    float fDY = float(iScreenY1 - iScreenY0);

    // compute barycentric coordinate within the face 2d boundary to fetch the clipspace position and normal
    for(uint32_t iY = iScreenY0; iY <= iScreenY1; iY++)
    {
        for(uint32_t iX = iScreenX0; iX <= iScreenX1; iX++)
        {
            float3 currPos = float3(float(iX), float(iY), 0.0f);
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
                float fPrevDepth = afDepthBuffer[iIndex];
                if(fPrevDepth > barycentricCoord.z)
                {
                    aOutputBuffer[iIndex] = currPos;
                    aNormalBuffer[iIndex] = normal;
                    afDepthBuffer[iIndex] = currPos.z;
                }
            }
        }
    }
}

/*
**
*/
void outputMeshToImage(
    std::string const& outputDirectory,
    std::string const& outputName,
    std::vector<float3> const& aVertexPositions,
    std::vector<uint32_t> const& aiTriangles,
    CCamera const& camera,
    uint32_t iImageWidth,
    uint32_t iImageHeight)
{
    // bounds
    float3 minBounds(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 maxBounds(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for(auto const& pos : aVertexPositions)
    {
        minBounds = fminf(pos, minBounds);
        maxBounds = fmaxf(pos, maxBounds);
    }

    std::map<uint32_t, float3> aVertNormalMap;

    // get average normals
    std::vector<float3> aFaceNormals(static_cast<uint32_t>(aiTriangles.size() / 3));
    std::vector<float3> aTriangleVertexPositions(aiTriangles.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTriangles.size()); iTri += 3)
    {
        uint32_t iV0 = aiTriangles[iTri];
        uint32_t iV1 = aiTriangles[iTri + 1];
        uint32_t iV2 = aiTriangles[iTri + 2];

        float3 const& pos0 = aVertexPositions[iV0];
        float3 const& pos1 = aVertexPositions[iV1];
        float3 const& pos2 = aVertexPositions[iV2];

        float3 normal = normalize(cross(pos1 - pos0, pos2 - pos0));
        aFaceNormals[iTri / 3] = normal;

        aVertNormalMap[iV0] += aFaceNormals[iTri / 3];
        aVertNormalMap[iV1] += aFaceNormals[iTri / 3];
        aVertNormalMap[iV2] += aFaceNormals[iTri / 3];

        aTriangleVertexPositions[iTri] = pos0;
        aTriangleVertexPositions[iTri + 1] = pos1;
        aTriangleVertexPositions[iTri + 2] = pos2;
    }

    std::vector<float3> aVertNormals(aVertexPositions.size());
    for(auto& keyValue : aVertNormalMap)
    {
        keyValue.second = normalize(keyValue.second);
        aVertNormals[keyValue.first] = keyValue.second;
    }

    std::vector<float4> aTriangleVertexNormals(aiTriangles.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aiTriangles.size()); iTri += 3)
    {
        uint32_t iV0 = aiTriangles[iTri];
        uint32_t iV1 = aiTriangles[iTri + 1];
        uint32_t iV2 = aiTriangles[iTri + 2];

        float3 const& normal0 = aVertNormals[iV0];
        float3 const& normal1 = aVertNormals[iV1];
        float3 const& normal2 = aVertNormals[iV2];

        aTriangleVertexNormals[iTri] =      float4(normal0, 1.0f);
        aTriangleVertexNormals[iTri + 1] =  float4(normal1, 1.0f);
        aTriangleVertexNormals[iTri + 2] =  float4(normal2, 1.0f);
    }

    // average normals and up direction
    float3 avgNormal = float3(0.0f, 0.0f, 0.0f);
    for(auto const& normal : aFaceNormals)
    {
        avgNormal += normal;
    }
    avgNormal = normalize(avgNormal);
    float3 up = (fabsf(avgNormal.y) > fabsf(avgNormal.x) && fabsf(avgNormal.y) > fabsf(avgNormal.z)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    
    float const kfCameraNear = 1.0f;
    float const kfCameraFar = 100.0f;

    mat4 const& viewMatrix = camera.getViewMatrix();
    mat4 const& projectionMatrix = camera.getProjectionMatrix();
    mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

    //std::map<uint32_t, float> aCameraDistanceScales;
    //testCameraDistanceScales(
    //    aCameraDistanceScales,
    //    aVertexPositions, 
    //    aiTriangles);

    // transform into camera space (view projection)
    std::vector<float4> aXFormTriangleVertexPositions(aTriangleVertexPositions.size());
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aTriangleVertexPositions.size()); iTri += 3)
    {
        float4 vertexPos0 = float4(
            aTriangleVertexPositions[iTri].x,
            aTriangleVertexPositions[iTri].y,
            aTriangleVertexPositions[iTri].z,
            1.0f);

        float4 vertexPos1 = float4(
            aTriangleVertexPositions[iTri + 1].x,
            aTriangleVertexPositions[iTri + 1].y,
            aTriangleVertexPositions[iTri + 1].z,
            1.0f);

        float4 vertexPos2 = float4(
            aTriangleVertexPositions[iTri + 2].x,
            aTriangleVertexPositions[iTri + 2].y,
            aTriangleVertexPositions[iTri + 2].z,
            1.0f);

        aXFormTriangleVertexPositions[iTri]     = viewProjectionMatrix * vertexPos0;
        aXFormTriangleVertexPositions[iTri + 1] = viewProjectionMatrix * vertexPos1;
        aXFormTriangleVertexPositions[iTri + 2] = viewProjectionMatrix * vertexPos2;
    }

    // clip space positions
    float3 minClipSpacePosition = float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 maxClipSpacePosition = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    std::vector<float4> aClipSpaceTriangleVertexPositions(aXFormTriangleVertexPositions.size());
    for(uint32_t iV = 0; iV < static_cast<uint32_t>(aXFormTriangleVertexPositions.size()); iV++)
    {
        aClipSpaceTriangleVertexPositions[iV].x = aXFormTriangleVertexPositions[iV].x / aXFormTriangleVertexPositions[iV].w;
        aClipSpaceTriangleVertexPositions[iV].y = aXFormTriangleVertexPositions[iV].y / aXFormTriangleVertexPositions[iV].w;
        aClipSpaceTriangleVertexPositions[iV].z = aXFormTriangleVertexPositions[iV].z / aXFormTriangleVertexPositions[iV].w;

        aClipSpaceTriangleVertexPositions[iV].x = aClipSpaceTriangleVertexPositions[iV].x * 0.5f + 0.5f;
        aClipSpaceTriangleVertexPositions[iV].y = 1.0f - (aClipSpaceTriangleVertexPositions[iV].y * 0.5f + 0.5f);
        aClipSpaceTriangleVertexPositions[iV].z = aClipSpaceTriangleVertexPositions[iV].z * 0.5f + 0.5f;

        minClipSpacePosition = fminf(minClipSpacePosition, aClipSpaceTriangleVertexPositions[iV]);
        maxClipSpacePosition = fmaxf(maxClipSpacePosition, aClipSpaceTriangleVertexPositions[iV]);
    }

    minClipSpacePosition = clamp(minClipSpacePosition, 0.0f, 1.0f);
    maxClipSpacePosition = clamp(maxClipSpacePosition, 0.0f, 1.0f);

    // intialize depth to 1.0 (far)
    std::vector<float3> aColorBuffer(iImageWidth* iImageHeight);
    std::vector<float3> aPositionBuffer(iImageWidth * iImageHeight);
    std::vector<float3> aNormalBuffer(iImageWidth * iImageHeight);
    std::vector<float> afDepthBuffer(iImageWidth * iImageHeight);
    for(uint32_t i = 0; i < iImageWidth * iImageHeight; i++)
    {
        afDepthBuffer[i] = 1.0f;
    }

    std::vector<float4> aTriangleVertexColors(aiTriangles.size());
    {
        //std::vector<float3> aColorBuffer(iImageWidth* iImageHeight);
        //memset(aColorBuffer.data(), 0, iImageWidth* iImageHeight * 3 * sizeof(float));
        //
        //std::vector<float3> aNormalBuffer(iImageWidth * iImageHeight);
        //memset(aNormalBuffer.data(), 0, iImageWidth * iImageHeight * 3 * sizeof(float));
        //
        //std::vector<float> aDepthBuffer(iImageWidth* iImageHeight);
        //for(uint32_t i = 0; i < iImageWidth * iImageHeight; i++)
        //{
        //    aDepthBuffer[i] = 1.0f;
        //}

        //rasterizeMeshCUDA2(
        //    aColorBuffer,
        //    aPositionBuffer,
        //    aNormalBuffer,
        //    afDepthBuffer,
        //    aColorBuffer,
        //    aClipSpaceTriangleVertexPositions,
        //    aTriangleVertexNormals,
        //    aTriangleVertexColors,
        //    iImageWidth,
        //    iImageHeight,
        //    3);
        
#if 0
        char const* pError = nullptr;
        SaveEXR(
            reinterpret_cast<float*>(aColorBuffer.data()),
            iImageWidth,
            iImageHeight,
            3,
            0,
            "c:\\Users\\Dingwings\\demo-models\\debug-output\\cluster-color-output.exr",
            &pError
        );

        SaveEXR(
            reinterpret_cast<float*>(aNormalBuffer.data()),
            iImageWidth,
            iImageHeight,
            3,
            0,
            "c:\\Users\\Dingwings\\demo-models\\debug-output\\cluster-normal-output.exr",
            &pError
        );

        std::vector<float3> aDepthBuffer3(iImageWidth * iImageHeight * 3);
        for(uint32_t iY = 0; iY < iImageHeight; iY++)
        {
            for(uint32_t iX = 0; iX < iImageWidth; iX++)
            {
                uint32_t iIndex = iY * iImageWidth + iX;
                float fDepth = aDepthBuffer[iIndex];
                aDepthBuffer3[iIndex] = float3(fDepth, fDepth, fDepth);
            }
        }
        SaveEXR(
            reinterpret_cast<float*>(aDepthBuffer3.data()),
            iImageWidth,
            iImageHeight,
            3,
            0,
            "c:\\Users\\Dingwings\\demo-models\\debug-output\\cluster-depth-output.exr",
            &pError
        );

        int iDebug = 1;
#endif // #if 0
    }

#if 0
    // rasterize all the triangles
    for(uint32_t iTri = 0; iTri < static_cast<uint32_t>(aXFormTriangleVertexPositions.size()); iTri += 3)
    {
        rasterizeTriangle(
            aOutputBuffer,
            aNormalBuffer,
            afDepthBuffer,
            aClipSpaceTriangleVertexPositions[iTri],
            aClipSpaceTriangleVertexPositions[iTri + 1],
            aClipSpaceTriangleVertexPositions[iTri + 2],
            aTriangleVertexNormals[iTri],
            aTriangleVertexNormals[iTri + 1],
            aTriangleVertexNormals[iTri + 2],
            iImageWidth,
            iImageHeight,
            iTri);
    }
#endif // 3if 0

    // invert depth buffer for better clarity
    std::vector<float4> aDepthBuffer(iImageWidth * iImageWidth);
    for(uint32_t i = 0; i < iImageWidth * iImageHeight; i++)
    {
        aDepthBuffer[i] = float4(
            1.0f - afDepthBuffer[i],
            1.0f - afDepthBuffer[i],
            1.0f - afDepthBuffer[i],
            1.0f);
    }

    // save out depth buffer
    std::ostringstream outputDepthImageFilePath;
    outputDepthImageFilePath << outputDirectory << "//" << outputName << "-depth.exr";
    char const* szError = nullptr;
    SaveEXR(
        reinterpret_cast<float*>(aDepthBuffer.data()),
        iImageWidth,
        iImageHeight,
        4,
        0,
        outputDepthImageFilePath.str().c_str(),
        &szError);
    
    // compute simple lighting (n dot l + ambient)
    std::vector<float3> aLighting(iImageWidth * iImageHeight);
    float3 lightDirection = normalize(float3(1.0f, 1.0f, 0.0f));
    float3 ambientColor = float3(0.2f, 0.2f, 0.2f);
    for(uint32_t i = 0; i < iImageWidth * iImageHeight; i++)
    {
        if(aNormalBuffer[i].x != 0.0f && aNormalBuffer[i].y != 0.0f && aNormalBuffer[i].z != 0.0f)
        {
            float fDP = maxf(dot(aNormalBuffer[i], lightDirection), 0.0f);
            aLighting[i] = float3(fDP, fDP, fDP) + ambientColor;
        }
    }

    // filter
    float afWeights[] =
    {
        1.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 1.0f,
    };
    std::vector<float3> aFilteredLighting(iImageWidth * iImageHeight);
    std::vector<uint8_t> acFilteredLighting(iImageWidth* iImageHeight * 3);
    int32_t const kiFilterRadius = 1;
    for(int32_t iY = 0; iY < static_cast<int32_t>(iImageHeight); iY++)
    {
        for(int32_t iX = 0; iX < static_cast<int32_t>(iImageWidth); iX++)
        {
            int32_t iImageIndex = iY * iImageWidth + iX;
            aFilteredLighting[iImageIndex] = float3(0.0f, 0.0f, 0.0f);
            float fTotalWeights = 0.0f;
            for(int32_t iOffsetY = -kiFilterRadius; iOffsetY <= kiFilterRadius; iOffsetY++)
            {
                int32_t iCurrY = clamp(iY + iOffsetY, 0, iImageHeight - 1);
                for(int32_t iOffsetX = -kiFilterRadius; iOffsetX <= kiFilterRadius; iOffsetX++)
                {
                    float fWeight = afWeights[(iOffsetY + kiFilterRadius) * 3 + (iOffsetX + kiFilterRadius)];

                    int32_t iCurrX = clamp(iX + iOffsetX, 0, iImageWidth - 1);
                    int32_t iCurrImageIndex = iCurrY * iImageWidth + iCurrX;
                    aFilteredLighting[iImageIndex] += aLighting[iCurrImageIndex] * fWeight;
                    
                    fTotalWeights += fWeight;
                }
            }

            aFilteredLighting[iImageIndex] /= fTotalWeights;
            acFilteredLighting[iImageIndex * 3] =       static_cast<uint8_t>(clamp(aFilteredLighting[iImageIndex].x * 255.0f, 0.0f, 255.0f));
            acFilteredLighting[iImageIndex * 3 + 1] =   static_cast<uint8_t>(clamp(aFilteredLighting[iImageIndex].y * 255.0f, 0.0f, 255.0f));
            acFilteredLighting[iImageIndex * 3 + 2] =   static_cast<uint8_t>(clamp(aFilteredLighting[iImageIndex].z * 255.0f, 0.0f, 255.0f));
        }
    }

    // crop image
    int32_t iCroppedImageLeft = int32_t(float(iImageWidth) * minClipSpacePosition.x);
    int32_t iCroppedImageTop = int32_t(float(iImageHeight) * minClipSpacePosition.y);
    int32_t iCroppedImageRight = int32_t(float(iImageWidth) * maxClipSpacePosition.x);
    int32_t iCroppedImageBottom = int32_t(float(iImageHeight) * maxClipSpacePosition.y);
    int32_t iCroppedImageWidth = iCroppedImageRight - iCroppedImageLeft;
    int32_t iCroppedImageHeight = iCroppedImageBottom - iCroppedImageTop;
    std::vector<float3> aCroppedImage(iCroppedImageWidth* iCroppedImageHeight);
    std::vector<uint8_t> acCroppedLDRImageData(iCroppedImageWidth* iCroppedImageHeight * 3);
    for(int32_t iY = iCroppedImageTop; iY < iCroppedImageBottom; iY++)
    {
        for(int32_t iX = iCroppedImageLeft; iX < iCroppedImageRight; iX++)
        {
            int32_t iOrigImageIndex = iY * iImageWidth + iX;
            int32_t iImageIndex = (iY - iCroppedImageTop) * iCroppedImageWidth + (iX - iCroppedImageLeft);
            aCroppedImage[iImageIndex] = aFilteredLighting[iOrigImageIndex];

            acCroppedLDRImageData[iImageIndex * 3] =     static_cast<uint8_t>(clamp(static_cast<uint32_t>(aFilteredLighting[iOrigImageIndex].x * 255.0f), 0, 255));
            acCroppedLDRImageData[iImageIndex * 3 + 1] = static_cast<uint8_t>(clamp(static_cast<uint32_t>(aFilteredLighting[iOrigImageIndex].y * 255.0f), 0, 255));
            acCroppedLDRImageData[iImageIndex * 3 + 2] = static_cast<uint8_t>(clamp(static_cast<uint32_t>(aFilteredLighting[iOrigImageIndex].z * 255.0f), 0, 255));
        }
    }

    std::ostringstream outputLightingImageFilePath;
    outputLightingImageFilePath << outputDirectory << "//" << outputName << "-lighting.exr";
    SaveEXR(
        reinterpret_cast<float*>(aFilteredLighting.data()),
        iImageWidth,
        iImageHeight,
        3,
        0,
        outputLightingImageFilePath.str().c_str(),
        &szError);

    // cropping computes local error and can fluctuate
    //SaveEXR(
    //    reinterpret_cast<float*>(aCroppedImage.data()),
    //    iCroppedImageWidth,
    //    iCroppedImageHeight,
    //    3,
    //    0,
    //    outputLightingImageFilePath.str().c_str(),
    //    &szError);


    std::ostringstream outputLigtingLDRImageFilePath;
    outputLigtingLDRImageFilePath << outputDirectory << "//" << outputName << "-lighting-ldr.png";
    stbi_write_png(
        outputLigtingLDRImageFilePath.str().c_str(),
        iImageWidth,
        iImageHeight,
        3,
        acFilteredLighting.data(),
        sizeof(char) * 3 * iImageWidth);

    int iDebug = 1;
}

/*
**
*/
void testCameraDistanceScales(
    std::map<uint32_t, float>& aCameraDistanceScales,
    std::vector<float3> const& aVertexPositions,
    std::vector<uint32_t> const& aiTriangles)
{
    float const kfCameraFar = 100.0f;
    float const kfCameraNear = 1.0f;

    float3 const& v0 = aVertexPositions[aiTriangles[0]];
    float3 const& v1 = aVertexPositions[aiTriangles[1]];
    float3 const& v2 = aVertexPositions[aiTriangles[2]];

    float3 triNormal = normalize(cross(normalize(v2 - v0), normalize(v1 - v0)));

    float3 triangleCenter = (v0 + v1 + v2) / 3.0f;
    float3 testCameraPositon = triangleCenter - triNormal;
    float3 lookDirection = normalize(triNormal);

    float3 up = (fabsf(triNormal.y) > fabsf(triNormal.x) && fabsf(triNormal.y) > fabsf(triNormal.z)) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    vec3 tangent = normalize(cross(up, triNormal));
    vec3 binormal = normalize(cross(triNormal, tangent));
    float afValue[16] =
    {
        tangent.x, tangent.y, tangent.z, 0.0f,
        binormal.x, binormal.y, binormal.z, 0.0f,
        -triNormal.x, -triNormal.y, -triNormal.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    mat4 tangentMatrix(afValue);
    float4 xformV0 = tangentMatrix * float4(v0.x, v0.y, v0.z, 1.0f);
    float4 xformV1 = tangentMatrix * float4(v1.x, v1.y, v1.z, 1.0f);
    float4 xformV2 = tangentMatrix * float4(v2.x, v2.y, v2.z, 1.0f);

    float fOrigArea = length(cross(xformV1 - xformV0, xformV2 - xformV0)) * 0.5f;

    for(float fScale = 2.0f; fScale < 50.0f; fScale += 1.0f)
    {
        for(float fCameraZ = -2.0f; fCameraZ >= -fScale * 3.0f; fCameraZ += -0.05f)
        {
            testCameraPositon = triangleCenter + lookDirection * fCameraZ;

            CCamera testCamera;
            testCamera.setFar(kfCameraFar);
            testCamera.setNear(kfCameraNear);
            testCamera.setLookAt(triangleCenter);
            testCamera.setPosition(testCameraPositon);
            CameraUpdateInfo cameraUpdateInfo =
            {
                /* .mfViewWidth      */  100.0f,
                /* .mfViewHeight     */  100.0f,
                /* .mfFieldOfView    */  3.14159f * 0.5f,
                /* .mUp              */  up,
                /* .mfNear           */  kfCameraNear,
                /* .mfFar            */  kfCameraFar,
            };
            testCamera.update(cameraUpdateInfo);
            mat4 const& viewMatrix = testCamera.getViewMatrix();
            mat4 const& projectionMatrix = testCamera.getProjectionMatrix();
            mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

            float3 testV0 = v0 * fScale;
            float3 testV1 = v1 * fScale;
            float3 testV2 = v2 * fScale;

            float4 xformTest0 = viewProjectionMatrix * float4(testV0.x, testV0.y, testV0.z, 1.0f);
            float4 xformTest1 = viewProjectionMatrix * float4(testV1.x, testV1.y, testV1.z, 1.0f);
            float4 xformTest2 = viewProjectionMatrix * float4(testV2.x, testV2.y, testV2.z, 1.0f);

            xformTest0.x /= xformTest0.w; xformTest0.y /= xformTest0.w; xformTest0.z /= xformTest0.w;
            xformTest1.x /= xformTest1.w; xformTest1.y /= xformTest1.w; xformTest1.z /= xformTest1.w;
            xformTest2.x /= xformTest2.w; xformTest2.y /= xformTest2.w; xformTest2.z /= xformTest2.w;

            float fNewArea = length(cross(float3(xformTest1.x, xformTest1.y, xformTest1.z) - float3(xformTest0.x, xformTest0.y, xformTest0.z), float3(xformTest2.x, xformTest2.y, xformTest2.z) - float3(xformTest0.x, xformTest0.y, xformTest0.z))) * 0.5f;

            float2 screenTestPos0 = float2(xformTest0.x * 0.5f + 0.5f, 1.0f - (xformTest0.y * 0.5f + 0.5f)) * float2(128.0f, 128.0f);
            float2 screenTestPos1 = float2(xformTest1.x * 0.5f + 0.5f, 1.0f - (xformTest1.y * 0.5f + 0.5f)) * float2(128.0f, 128.0f);
            float2 screenTestPos2 = float2(xformTest2.x * 0.5f + 0.5f, 1.0f - (xformTest2.y * 0.5f + 0.5f)) * float2(128.0f, 128.0f);

            float fCameraDistance = length(testCameraPositon - triangleCenter);

            if(fabsf(fNewArea - fOrigArea) <= fOrigArea * 0.01f)
            {
                aCameraDistanceScales[uint32_t(fScale)] = fCameraDistance;
                int iDebug = 1;
            }
        }
    }

    int iDebug = 1;
}