#pragma once

#include <math.h>
#include <stdint.h>

struct vec4;

/*
**
*/
struct vec3
{
    vec3() {x = y = z = 0.0f;}
    vec3(float fX, float fY, float fZ) { x = fX; y = fY; z = fZ; }
    vec3(float fNum)
    {
        x = fNum;
        y = fNum;
        z = fNum;
    }
    vec3(vec4 const&);
    
    vec3 operator + (vec3 const& v) const
    {
        return vec3(x + v.x, y + v.y, z + v.z);
    }
    
    vec3 operator + (float fScalar) const
    {
        return vec3(x + fScalar, y + fScalar, z + fScalar);
    }

    vec3 operator - (vec3 const& v) const
    {
        return vec3(x - v.x, y - v.y, z - v.z);
    }

    vec3 operator - (float fScalar) const
    {
        return vec3(x - fScalar, y - fScalar, z - fScalar);
    }
    
    vec3 operator * (float fScalar) const
    {
        return vec3(x * fScalar, y * fScalar, z * fScalar);
    }
    
    vec3 operator / (float fScalar) const
    {
        return vec3(x / fScalar, y / fScalar, z / fScalar);
    }
    
    vec3 operator * (vec3 const& v) const
    {
        return vec3(x * v.x, y * v.y, z * v.z);
    }
    
    vec3 operator / (vec3 const& v) const
    {
        return vec3(x / v.x, y / v.y, z / v.z);
    }

    //vec3 operator / (vec3 const& v)
    //{
    //    return vec3(x / v.x, y / v.y, z / v.z);
    //}
    
    void operator += (vec3 const& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    void operator += (float fScalar)
    {
        x += fScalar;
        y += fScalar;
        z += fScalar;
    }
    
    void operator -= (vec3 const& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }
    
    void operator -= (float fScalar)
    {
        x -= fScalar;
        y -= fScalar;
        z -= fScalar;
    }

    void operator *= (float fScalar)
    {
        x *= fScalar;
        y *= fScalar;
        z *= fScalar;
    }
    
    void operator *= (vec3 const& v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
    }

    void operator /= (float fScalar)
    {
        x /= fScalar;
        y /= fScalar;
        z /= fScalar;
    }

    void operator /= (vec3 const& v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
    }
    
    bool operator == (vec3 const& v) const
    {
        bool bDiffX = (fabs(x - v.x) < 0.000001f);
        bool bDiffY = (fabs(y - v.y) < 0.000001f);
        bool bDiffZ = (fabs(z - v.z) < 0.000001f);
        
        return (bDiffX && bDiffY && bDiffZ);
    }
    
    bool operator != (vec3 const& v) const
    {
        bool bDiffX = (fabs(x - v.x) < 0.000001f);
        bool bDiffY = (fabs(y - v.y) < 0.000001f);
        bool bDiffZ = (fabs(z - v.z) < 0.000001f);
        
        return (!bDiffX || !bDiffY || !bDiffZ);
    }
    
    vec3 pow(float fNum)
    {
        return vec3(powf(x, fNum), powf(y, fNum), powf(z, fNum));
    }
    
    vec3 pow(vec3 const& v)
    {
        return vec3(powf(x, v.x), powf(y, v.y), powf(z, v.z));
    }
    
#ifndef _MSC_VER
    vec3 max(float fMax)
    {
        return vec3(fmax(x, fMax), fmax(y, fMax), fmax(z, fMax));
    }
    
    vec3 min(float fMin)
    {
        return vec3(fmin(x, fMin), fmin(y, fMin), fmin(z, fMin));
    }
#endif // WINDOWS
    
    float   x;
    float   y;
    float   z;
};

/*
 **
 */
struct vec4
{
    vec4() {x = y = z = 0.0f; w = 1.0f;}
    vec4(float fX, float fY, float fZ, float fW) { x = fX; y = fY; z = fZ; w = fW;}
    vec4(vec3 const& v, float fW) { x = v.x; y = v.y; z = v.z; w = fW;}
    
    vec4 operator + (vec4 const& v) const
    {
        return vec4(x + v.x, y + v.y, z + v.z, 1.0f);
    }

    vec4 operator + (float fScalar) const
    {
        return vec4(x + fScalar, y + fScalar, z + fScalar, w + fScalar);
    }
    
    vec4 operator - (vec4 const& v) const
    {
        return vec4(x - v.x, y - v.y, z - v.z, 1.0f);
    }

    vec4 operator - (float fScalar) const
    {
        return vec4(x - fScalar, y - fScalar, z - fScalar, w - fScalar);
    }
    
    vec4 operator * (float fScalar) const
    {
        return vec4(x * fScalar, y * fScalar, z * fScalar, 1.0f);
    }
    
    vec4 operator * (vec4 const& v) const
    {
        return vec4(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    vec4 operator / (float fScalar) const
    {
        return vec4(x / fScalar, y / fScalar, z / fScalar, 1.0f);
    }
    
    vec4 operator / (vec4 const& v) const
    {
        return vec4(x / v.x, y / v.y, z / v.z, 1.0f);
    }

    //vec4 operator = (vec3 const& v) const
    //{
    //    return vec4(v.x, v.y, v.z, 1.0f);
    //}
    
    void operator += (vec4 const& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    void operator += (float fScalar)
    {
        x += fScalar;
        y += fScalar;
        z += fScalar;
    }
    
    void operator -= (vec4 const& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    void operator -= (float fScalar)
    {
        x -= fScalar;
        y -= fScalar;
        z -= fScalar;
    }
    
    void operator *= (float fScalar)
    {
        x *= fScalar;
        y *= fScalar;
        z *= fScalar;
    }
    
    void operator /= (float fScalar)
    {
        x /= fScalar;
        y /= fScalar;
        z /= fScalar;
    }
    
    bool operator == (vec4 const& v) const
    {
        bool bDiffX = (fabsf(x - v.x) < 0.000001f);
        bool bDiffY = (fabsf(y - v.y) < 0.000001f);
        bool bDiffZ = (fabsf(z - v.z) < 0.000001f);
        bool bDiffW = (fabsf(w - v.w) < 0.000001f);
        
        return (bDiffX && bDiffY && bDiffZ && bDiffW);
    }
    
    bool operator != (vec4 const& v) const
    {
        bool bDiffX = (fabsf(x - v.x) < 0.000001f);
        bool bDiffY = (fabsf(y - v.y) < 0.000001f);
        bool bDiffZ = (fabsf(z - v.z) < 0.000001f);
        bool bDiffW = (fabsf(w - v.w) < 0.000001f);
        
        return (!bDiffX || !bDiffY || !bDiffZ || !bDiffW);
    }
    
    float x;
    float y;
    float z;
    float w;    
};

/*
 **
 */
struct vec2
{
    vec2() {x = y = 0.0f;}
    vec2(float fX, float fY) { x = fX; y = fY; }
    vec2(vec3 const& v) { x = v.x, y = v.y; }

    vec2 operator + (vec2 const& v) const
    {
        return vec2(x + v.x, y + v.y);
    }
    
    vec2 operator - (vec2 const& v) const
    {
        return vec2(x - v.x, y - v.y);
    }
    
    vec2 operator * (vec2 const& v) const
    {
        return vec2(x * v.x, y * v.y);
    }

    vec2 operator / (vec2 const& v) const
    {
        return vec2(x / v.x, y / v.y);
    }

    vec2 operator * (float fScalar) const
    {
        return vec2(x * fScalar, y * fScalar);
    }
    
    vec2 operator / (float fScalar) const
    {
        return vec2(x / fScalar, y / fScalar);
    }
    
    void operator += (vec2 const& v)
    {
        x += v.x;
        y += v.y;
    }
    
    void operator -= (vec2 const& v)
    {
        x -= v.x;
        y -= v.y;
    }
    
    void operator *= (float fScalar)
    {
        x *= fScalar;
        y *= fScalar;
    }
    
    void operator /= (float fScalar)
    {
        x /= fScalar;
        y /= fScalar;
    }
    
    float x;
    float y;
};

#if !defined(__CUDA_RUNTIME_H__)
struct int2
{
    int32_t x;
    int32_t y;

    int2()
    {
        x = y = 0;
    }

    int2(int32_t iX, int32_t iY)
    {
        x = iX; y = iY;
    }

    int2 operator + (int2 num)
    {
        return int2(x + num.x, y + num.y);
    }

    int2 operator += (int2 num)
    {
        return int2(x + num.x, y + num.y);
    }

    int2 operator -= (int2 num)
    {
        return int2(x - num.x, y - num.y);
    }

    int2 operator ^ (int2 num)
    {
        return int2(x ^ num.x, y ^ num.y);
    }
};

struct int3
{
    int32_t         x;
    int32_t         y;
    int32_t         z;

    int3()
    {
        x = y = z = 0;
    }

    int3(int32_t iX, int32_t iY, int32_t iZ)
    {
        x = iX; y = iY; z = iZ;
    }

    int3 operator + (int3 num)
    {
        return int3(x + num.x, y + num.y, z + num.z);
    }

    int3 operator += (int3 num)
    {
        return int3(x + num.x, y + num.y, z + num.z);
    }

    int3 operator -= (int3 num)
    {
        return int3(x - num.x, y - num.y, z - num.z);
    }

    int3 operator ^ (int3 num)
    {
        return int3(x ^ num.x, y ^ num.y, z ^ num.z);
    }
};

struct uint3
{
    uint32_t         x;
    uint32_t         y;
    uint32_t         z;

    uint3()
    {
        x = y = z = 0;
    }

    uint3(int32_t iX, int32_t iY, int32_t iZ)
    {
        x = iX; y = iY; z = iZ;
    }

    uint3 operator + (uint3 num)
    {
        return uint3(x + num.x, y + num.y, z + num.z);
    }

    uint3 operator += (uint3 num)
    {
        return uint3(x + num.x, y + num.y, z + num.z);
    }

    uint3 operator -= (uint3 num)
    {
        return uint3(x - num.x, y - num.y, z - num.z);
    }

    uint3 operator ^ (uint3 num)
    {
        return uint3(x ^ num.x, y ^ num.y, z ^ num.z);
    }
};
#endif // #if !defined(__CUDA_RUNTIME_H__)

float dot(vec2 const& v0, vec2 const& v1);
float dot(vec3 const& v0, vec3 const& v1);
float dot(vec4 const& v0, vec4 const& v1);

vec3 cross(vec3 const& v0, vec3 const& v1);

vec2 normalize(vec2 const& v);
vec3 normalize(vec3 const& v);
vec4 normalize(vec4 const& v);

float length(vec3 const& v);
float length(vec4 const& v);
float length(vec2 const& v);

float lengthSquared(vec3 const& v);
float lengthSquared(vec4 const& v);
float lengthSquared(vec2 const& v);

vec3 reflect(vec3 const& v, vec3 const& normal);

float minf(float fNum0, float fNum1);
float maxf(float fNum0, float fNum1);

vec3 fminf(vec3 const& v0, vec3 const& v1);
vec3 fmaxf(vec3 const& v0, vec3 const& v1);

uint8_t clamp(uint8_t v, uint8_t iMin, uint8_t iMax);
uint32_t clamp(uint32_t v, uint32_t iMin, uint32_t iMax);
float clamp(float v, float fMin, float fMax);
vec2 clamp(vec2 const& v, float fMin, float fMax);
vec3 clamp(vec3 const& v, float fMin, float fMax);
vec4 clamp(vec4 const& v, float fMin, float fMax);

vec2 lerp(vec2 const& v0, vec2 const& v1, float fStep);
vec3 lerp(vec3 const& v0, vec3 const& v1, float fStep);
vec4 lerp(vec4 const& v0, vec4 const& v1, float fStep);

vec3 maxf(vec3 const& v0, vec3 const& v1);

vec3 floor(vec3 const& v);
vec4 floor(vec4 const& v);

vec3 ceil(vec3 const& v);
vec4 ceil(vec4 const& v);

vec3 abs(vec3 const& v);
vec4 abs(vec4 const& v);

vec3 sign(vec3 const& v);
vec4 sign(vec4 const& v);

vec3 pow(vec3 const& v, float fVal);
vec4 pow(vec4 const& v, float fVal);

vec3 saturate(vec3 const& v);
vec4 saturate(vec4 const& v);

float step(float num0, float num1);
float smoothstep(float min, float max, float x);

vec3 frac(vec3 const& v);

#if !defined(__CUDA_RUNTIME_H__)
typedef vec3 float3;
typedef vec4 float4;
typedef vec2 float2;
#endif // __CUDA_RUNTIME_H__
