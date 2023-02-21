#include "vec.h"
#include "tiny_obj_loader.h"
#include "Camera.h"
#include "rasterizer.h"

#include <stdint.h>
#include <sstream>
#include <vector>

#include <assert.h>

extern std::string execCommand(std::string const& command, bool bOutput);

/*
**
*/
void computeScreenErrorsFLIP()
{
    uint32_t iLODLevel = 0;

    std::ostringstream inputOrigClusterGroupDirectory;
    inputOrigClusterGroupDirectory << "c:\\Users\\Dingwings\\demo-models\\cluster-groups\\";
    for(uint32_t iClusterGroup = 0; iClusterGroup < 1; iClusterGroup++)
    {
        // original cluster group
        std::ostringstream origClusterGroupFilePath;
        origClusterGroupFilePath << inputOrigClusterGroupDirectory.str() << "cluster-group-lod" << iLODLevel << "-group" << iClusterGroup << ".obj";
        std::vector<float3> aClusterGroupVertexPositions;
        std::vector<uint32_t> aiClusterGroupTriangleIndices;
        {
            tinyobj::attrib_t origAttrib;
            std::vector<tinyobj::shape_t> aOrigShapes;
            std::vector<tinyobj::material_t> aOrigMaterials;

            // load initial mesh file
            std::string warnings;
            std::string errors;
            bool bRet = tinyobj::LoadObj(
                &origAttrib,
                &aOrigShapes,
                &aOrigMaterials,
                &warnings,
                &errors,
                origClusterGroupFilePath.str().c_str());

            for(uint32_t i = 0; i < static_cast<uint32_t>(origAttrib.vertices.size()); i += 3)
            {
                aClusterGroupVertexPositions.push_back(float3(origAttrib.vertices[i], origAttrib.vertices[i + 1], origAttrib.vertices[i + 2]));
            }

            for(uint32_t i = 0; i < static_cast<uint32_t>(aOrigShapes[0].mesh.indices.size()); i++)
            {
                aiClusterGroupTriangleIndices.push_back(aOrigShapes[0].mesh.indices[i].vertex_index);
            }

        }   // cluster group data

        // simplified cluster group
        std::ostringstream inputSimplifiedClusterGroupDirectory;
        inputSimplifiedClusterGroupDirectory << "c:\\Users\\Dingwings\\demo-models\\simplified-cluster-groups\\";
        std::ostringstream simplifiedClusterGroupFilePath;
        simplifiedClusterGroupFilePath << inputSimplifiedClusterGroupDirectory.str() << "simplified-cluster-group-lod" << iLODLevel << "-group" << iClusterGroup << ".obj";
        std::vector<float3> aSimplifiedClusterGroupVertexPositions;
        std::vector<uint32_t> aiSimplifiedClusterGroupTriangleIndices;
        {
            tinyobj::attrib_t origAttrib;
            std::vector<tinyobj::shape_t> aOrigShapes;
            std::vector<tinyobj::material_t> aOrigMaterials;

            // load initial mesh file
            std::string warnings;
            std::string errors;
            bool bRet = tinyobj::LoadObj(
                &origAttrib,
                &aOrigShapes,
                &aOrigMaterials,
                &warnings,
                &errors,
                simplifiedClusterGroupFilePath.str().c_str());

            assert(errors.length() <= 0);

            for(uint32_t i = 0; i < static_cast<uint32_t>(origAttrib.vertices.size()); i += 3)
            {
                aSimplifiedClusterGroupVertexPositions.push_back(float3(origAttrib.vertices[i], origAttrib.vertices[i + 1], origAttrib.vertices[i + 2]));
            }

            for(uint32_t i = 0; i < static_cast<uint32_t>(aOrigShapes[0].mesh.indices.size()); i++)
            {
                aiSimplifiedClusterGroupTriangleIndices.push_back(aOrigShapes[0].mesh.indices[i].vertex_index);
            }

        }   // simplified cluster group data

        // camera
        float3 center = float3(0.0f, 0.0f, 0.0f);
        float const kfCameraFar = 100.0f;
        float const kfCameraNear = 1.0f;
        CCamera camera;
        camera.setFar(kfCameraFar);
        camera.setNear(kfCameraNear);
        camera.setLookAt(center);
        camera.setPosition(center - float3(kfCameraNear, kfCameraNear, kfCameraNear));
        float3 up = float3(0.0f, 1.0f, 0.0f);

        CameraUpdateInfo cameraUpdateInfo =
        {
            /* .mfViewWidth      */  100.0f,
            /* .mfViewHeight     */  100.0f,
            /* .mfFieldOfView    */  3.14159f * 0.5f,
            /* .mUp              */  up,
            /* .mfNear           */  kfCameraNear,
            /* .mfFar            */  kfCameraFar,
        };
        camera.update(cameraUpdateInfo);

        std::string outputImageDirectory = "c:\\Users\\Dingwings\\demo-models\\cluster-images\\";
        std::map<uint32_t, float> weightedMedians;
        std::map<uint32_t, float> means;
        for(float fCameraDistance = 1.0f; fCameraDistance <= 5.0f; fCameraDistance += 1.0f)
        {
            std::ostringstream outputImageFilePath;
            outputImageFilePath << "cluster-group-" << iClusterGroup << "-camera-" << fCameraDistance;
            outputMeshToImage(
                "c:\\Users\\Dingwings\\demo-models\\cluster-images",
                outputImageFilePath.str(),
                aClusterGroupVertexPositions,
                aiClusterGroupTriangleIndices,
                camera,
                256,
                256);

            std::ostringstream outputSimplifiedImageFilePath;
            outputSimplifiedImageFilePath << "simplified-cluster-group-" << iClusterGroup << "-camera-" << fCameraDistance;
            outputMeshToImage(
                "c:\\Users\\Dingwings\\demo-models\\cluster-images",
                outputSimplifiedImageFilePath.str(),
                aSimplifiedClusterGroupVertexPositions,
                aiSimplifiedClusterGroupTriangleIndices,
                camera,
                256,
                256);

            std::ostringstream origClusterGroupImageFilePath;
            //origClusterGroupImageFilePath << outputImageDirectory << "cluster-group-" << iClusterGroup << "-camera-" << fCameraDistance << "-lighting.exr";
            origClusterGroupImageFilePath << outputImageDirectory << "cluster-group-" << iClusterGroup << "-camera-" << fCameraDistance << "-lighting-ldr.png";

            std::ostringstream simplifiedClusterGroupImageFilePath;
            //simplifiedClusterGroupImageFilePath << outputImageDirectory << "simplified-cluster-group-" << iClusterGroup << "-camera-" << fCameraDistance << "-lighting.exr";
            simplifiedClusterGroupImageFilePath << outputImageDirectory << "simplified-cluster-group-" << iClusterGroup << "-camera-" << fCameraDistance << "-lighting-ldr.png";

            uint32_t iNumExposures = 2;
            float fStopExposure = 1.5f;

            std::string imageDirectory = outputImageDirectory;
            std::ostringstream csvInfoFilePath;
            csvInfoFilePath << outputImageDirectory << "diff-result-cluster-group-" << iClusterGroup << ".csv";

            std::ostringstream flipCommand;
            flipCommand << "d:\\test\\MeshStuff\\externals\\flip\\flip.exe ";
            flipCommand << "--reference ";
            flipCommand << origClusterGroupImageFilePath.str() << " ";
            flipCommand << "--test " << simplifiedClusterGroupImageFilePath.str() << " ";
            flipCommand << "--num-exposures " << iNumExposures << " ";
            flipCommand << "--stop-exposure " << fStopExposure << " ";
            flipCommand << "--directory " << imageDirectory << " ";
            flipCommand << "--csv " << csvInfoFilePath.str() << " ";
            flipCommand << "--basename error-image";
            std::string output = execCommand(flipCommand.str(), false);

            auto weightedMedianStart = output.find("Weighted median: ");
            weightedMedianStart += strlen("Weighted median: ");
            auto weightedMedianEnd = output.find("\n", weightedMedianStart);
            std::string weightedMedianStr = output.substr(weightedMedianStart, weightedMedianEnd - weightedMedianStart);
            float fWeightedMedian = static_cast<float>(atof(weightedMedianStr.c_str()));
            weightedMedians[uint32_t(fCameraDistance)] = fWeightedMedian;

            auto meanStart = output.find("Mean: ");
            meanStart += strlen("Mean: ");
            auto meanEnd = output.find("\n", meanStart);
            std::string meanStr = output.substr(meanStart, meanEnd - meanStart);
            float fMean = static_cast<float>(atof(meanStr.c_str()));
            means[uint32_t(fCameraDistance)] = fMean;

        }   // for camera distance to max camera distance


    }   // for cluster group = 0 to num cluster groups

}   // compute cluster error using FLIP