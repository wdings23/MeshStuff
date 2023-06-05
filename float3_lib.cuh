#pragma once

/*
**
*/
__device__
inline float3 operator + (float3 const& pt0, float3 const& pt1)
{
    return make_float3(
        pt0.x + pt1.x,
        pt0.y + pt1.y,
        pt0.z + pt1.z);
}

/*
**
*/
__device__
inline float3 operator - (float3 const& pt0, float3 const& pt1)
{
    return make_float3(
        pt0.x - pt1.x, 
        pt0.y - pt1.y, 
        pt0.z - pt1.z);
}

/*
**
*/
__device__
inline float3 operator * (float3 const& pt0, float3 const& pt1)
{
    return make_float3(
        pt0.x * pt1.x,
        pt0.y * pt1.y,
        pt0.z * pt1.z);
    
}

/*
**
*/
__device__
inline float3 operator / (float3 const& pt0, float3 const& pt1)
{
    return make_float3(
        pt0.x / pt1.x,
        pt0.y / pt1.y,
        pt0.z / pt1.z);
}

/*
**
*/
__device__
inline float3 operator * (float3 const& pt0, float fScalar)
{
    return make_float3(
        pt0.x * fScalar,
        pt0.y * fScalar,
        pt0.z * fScalar);
}

/*
**
*/
__device__
inline float3 operator / (float3 const& pt0, float fScalar)
{
    return make_float3(
        pt0.x / fScalar,
        pt0.y / fScalar,
        pt0.z / fScalar);
}


/*
**
*/
__device__
inline float dot(float3 const& pt0, float3 const& pt1)
{
    return pt0.x * pt1.x + pt0.y * pt1.y + pt0.z * pt1.z;
}

/*
**
*/
__device__
inline float3 cross(float3 const& v0, float3 const& v1)
{
   return make_float3(
        v0.y * v1.z - v1.y * v0.z,
        v0.z * v1.x - v1.z * v0.x,
        v0.x * v1.y - v1.x * v0.y);
}

/*
**
*/
__device__
inline float length(float3 const& v)
{
    return sqrt(dot(v, v));
}

/*
**
*/
__device__
inline float lengthSquared(float3 const& v)
{
    return dot(v, v);
}

/*
**
*/
__device__
inline float3 normalize(float3 const& v)
{
    float fLength = length(v);
    return make_float3(v.x / fLength, v.y / fLength, v.z / fLength);
}

/*
**
*/
__device__
inline float rayPlaneIntersection(
    float3 const& pt0,
    float3 const& pt1,
    float3 const& planeNormal,
    float fPlaneDistance)
{
    float fRet = FLT_MAX;
    float3 v = pt1 - pt0;

    float fDenom = dot(v, planeNormal);
    if(fabs(fDenom) > 0.00001f)
    {
        fRet = -(dot(pt0, planeNormal) + fPlaneDistance) / fDenom;
    }

    return fRet;
}

/*
**
*/
__device__
inline float3 barycentric(
    float3 p,
    float3 a,
    float3 b,
    float3 c)
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