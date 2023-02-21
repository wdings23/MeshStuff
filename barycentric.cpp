#include "barycentric.h"

/*
**
*/
vec3 barycentric(vec3 const& p, vec3 const& a, vec3 const& b, vec3 const& c)
{
    vec3 v0 = b - a, v1 = c - a, v2 = p - a;
    float fD00 = dot(v0, v0);
    float fD01 = dot(v0, v1);
    float fD11 = dot(v1, v1);
    float fD20 = dot(v2, v0);
    float fD21 = dot(v2, v1);
    float fDenom = fD00 * fD11 - fD01 * fD01;
    float fV = (fD11 * fD20 - fD01 * fD21) / fDenom;
    float fW = (fD00 * fD21 - fD01 * fD20) / fDenom;
    float fU = 1.0f - fV - fW;

    return vec3(fU, fV, fW);
}