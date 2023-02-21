#include <math.h>

#include "Camera.h"
#include "LogPrint.h"
#include "wtfassert.h"

#include "quaternion.h"

/*
**
*/
CCamera::CCamera() :
    mfFar(1000.0f),
    mfNear(1.0f),
    mProjectionType(PROJECTION_PERSPECTIVE)
{
    
}

/*
**
*/
CCamera::~CCamera()
{
    
}

/*
**
*/
void CCamera::update(CameraUpdateInfo& info)
{
//    quaternion qX, qY;
//    qX = qX.fromAngleAxis(vec3(1.0f, 0.0f, 0.0f), info.mfTotalPitch);
//    qY = qY.fromAngleAxis(vec3(0.0f, 1.0f, 0.0f), info.mfTotalYaw);
//
//    mat4 pitchMatrix = qX.matrix();
//    mat4 yawMatrix = qY.matrix();
//    mat4 yawPitchMatrix = yawMatrix * pitchMatrix;
//    vec3 newPos = yawPitchMatrix * info.mLookV;
//    mPosition = newPos + mLookAt;
//
//    vec3 up(0.0f, 1.0f, 0.0f);
//    if(fabs(info.mfTotalPitch) >= (float)M_PI * 0.5f && fabs(info.mfTotalPitch) <= (float)M_PI * 1.5f)
//    {
//        up.y = -1.0f;
//    }
    
    WTFASSERT(info.mfFar > info.mfNear, "Invalid near/far values, near = %.4f far = %.4f", info.mfNear, info.mfFar);
    mfFar = info.mfFar;
    mfNear = info.mfNear;

    mViewMatrix = makeViewMatrix(mPosition, mLookAt, info.mUp);
    
    if(mProjectionType == PROJECTION_PERSPECTIVE)
    {
        mProjectionMatrix = perspectiveProjection(info.mfFieldOfView, (uint32_t)info.mfViewWidth, (uint32_t)info.mfViewHeight, mfFar, mfNear);
    }
    else
    {
        float fFarMinusNear = mfFar - mfNear;
        
        float fLeft = (float)info.mfViewWidth * -0.5f;
        float fRight = (float)info.mfViewWidth * 0.5f;
        float fTop = (float)info.mfViewHeight * 0.5f;
        float fBottom = (float)info.mfViewHeight * -0.5f;
        mProjectionMatrix = orthographicProjection(fLeft, fRight, fTop, fBottom, fFarMinusNear * 0.5f, -fFarMinusNear * 0.5f);
    }

    float fAspectRatio = 1.0f;
    float fTan = (float)tan(info.mfFieldOfView * 0.25f);
    float fNearHeight = mfNear * fTan;
    float fNearWidth = fNearHeight * fAspectRatio;
    float fFarHeight = mfFar * fTan;
    float fFarWidth = fFarHeight * fAspectRatio;

    vec3 Z = normalize(mLookAt - mPosition);
    vec3 X = normalize(cross(info.mUp, Z));
    vec3 Y = normalize(cross(Z, X));

    vec3 nc = mPosition + (Z * mfNear);
    vec3 fc = mPosition + (Z * mfFar);

    vec3 ntl = nc + Y * fNearHeight - X * fNearWidth;
    vec3 ntr = nc + Y * fNearHeight + X * fNearWidth;
    vec3 nbl = nc - Y * fNearHeight - X * fNearWidth;
    vec3 nbr = nc - Y * fNearHeight + X * fNearWidth;

    vec3 ftl = fc + Y * fFarHeight - X * fFarWidth;
    vec3 ftr = fc + Y * fFarHeight + X * fFarWidth;
    vec3 fbl = fc - Y * fFarHeight - X * fFarWidth;
    vec3 fbr = fc - Y * fFarHeight + X * fFarWidth;
    
    {
        vec3 v0 = nbl - ntl;
        vec3 v1 = ntr - ntl;
        vec3 planeNormal = normalize(cross(v0, v1));
        float fD = -dot(planeNormal, nbl);
        maFrustomPlanes[FRUSTUM_PLANE_NEAR] = vec4(planeNormal.x, planeNormal.y, planeNormal.z, fD);
    }

    {
        vec3 v0 = fbl - ftl;
        vec3 v1 = ftr - ftl;
        vec3 planeNormal = normalize(cross(v1, v0));
        float fD = -dot(planeNormal, fbl);
        maFrustomPlanes[FRUSTUM_PLANE_FAR] = vec4(planeNormal.x, planeNormal.y, planeNormal.z, fD);
    }

    {
        vec3 v0 = ftl - ntl;
        vec3 v1 = nbl - ntl;
        vec3 planeNormal = normalize(cross(v0, v1));
        float fD = -dot(planeNormal, ntl);
        maFrustomPlanes[FRUSTUM_PLANE_LEFT] = vec4(planeNormal.x, planeNormal.y, planeNormal.z, fD);
    }

    {
        vec3 v0 = ftr - ntr;
        vec3 v1 = nbr - ntr;
        vec3 planeNormal = normalize(cross(v1, v0));
        float fD = -dot(planeNormal, ntr);
        maFrustomPlanes[FRUSTUM_PLANE_RIGHT] = vec4(planeNormal.x, planeNormal.y, planeNormal.z, fD);
    }

    {
        vec3 v0 = ftl - ntl;
        vec3 v1 = ntr - ntl;
        vec3 planeNormal = normalize(cross(v1, v0));
        float fD = -dot(planeNormal, ntl);
        maFrustomPlanes[FRUSTUM_PLANE_TOP] = vec4(planeNormal.x, planeNormal.y, planeNormal.z, fD);
    }

    {
        vec3 v0 = fbl - nbl;
        vec3 v1 = nbr - nbl;
        vec3 planeNormal = normalize(cross(v0, v1));
        float fD = -dot(planeNormal, fbl);
        maFrustomPlanes[FRUSTUM_PLANE_BOTTOM] = vec4(planeNormal.x, planeNormal.y, planeNormal.z, fD);
    }
}

/*
**
*/
bool CCamera::isBoxInFrustum(vec3 const& topLeftFront, vec3 const& bottomRightBack) const
{
    vec3 dimension(bottomRightBack.x - topLeftFront.x,
                   topLeftFront.y - bottomRightBack.y,
                   bottomRightBack.z - topLeftFront.z);
    vec3 center(topLeftFront.x + dimension.x * 0.5f,
                bottomRightBack.y + dimension.y * 0.5f,
                topLeftFront.z + dimension.z * 0.5f);
    
    float fRadius = dimension.x * 0.5f;
    if(dimension.y * 0.5f > fRadius)
    {
        fRadius = dimension.y * 0.5f;
    }
    else if(dimension.z * 0.5f > fRadius)
    {
        fRadius = dimension.z * 0.5f;
    }

    float fDP0 = dot(center, vec3(maFrustomPlanes[FRUSTUM_PLANE_LEFT].x, maFrustomPlanes[FRUSTUM_PLANE_LEFT].y, maFrustomPlanes[FRUSTUM_PLANE_LEFT].z)) + maFrustomPlanes[FRUSTUM_PLANE_LEFT].w;
    float fDP1 = dot(center, vec3(maFrustomPlanes[FRUSTUM_PLANE_RIGHT].x, maFrustomPlanes[FRUSTUM_PLANE_RIGHT].y, maFrustomPlanes[FRUSTUM_PLANE_RIGHT].z)) + maFrustomPlanes[FRUSTUM_PLANE_RIGHT].w;
    float fDP2 = dot(center, vec3(maFrustomPlanes[FRUSTUM_PLANE_TOP].x, maFrustomPlanes[FRUSTUM_PLANE_TOP].y, maFrustomPlanes[FRUSTUM_PLANE_TOP].z)) + maFrustomPlanes[FRUSTUM_PLANE_TOP].w;
    float fDP3 = dot(center, vec3(maFrustomPlanes[FRUSTUM_PLANE_BOTTOM].x, maFrustomPlanes[FRUSTUM_PLANE_BOTTOM].y, maFrustomPlanes[FRUSTUM_PLANE_BOTTOM].z)) + maFrustomPlanes[FRUSTUM_PLANE_BOTTOM].w;
    float fDP4 = dot(center, vec3(maFrustomPlanes[FRUSTUM_PLANE_NEAR].x, maFrustomPlanes[FRUSTUM_PLANE_NEAR].y, maFrustomPlanes[FRUSTUM_PLANE_NEAR].z)) + maFrustomPlanes[FRUSTUM_PLANE_NEAR].w;
    float fDP5 = dot(center, vec3(maFrustomPlanes[FRUSTUM_PLANE_FAR].x, maFrustomPlanes[FRUSTUM_PLANE_FAR].y, maFrustomPlanes[FRUSTUM_PLANE_FAR].z)) + maFrustomPlanes[FRUSTUM_PLANE_FAR].w;

    bool bRet = (fDP0 >= -fRadius && fDP1 >= -fRadius && fDP2 >= -fRadius && fDP3 >= -fRadius && fDP4 >= -fRadius && fDP5 >= -fRadius);
        
    return bRet;
    
}
