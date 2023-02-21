#include "utils.h"

/*
**
*/
void setBitFlag(uint32_t* aiFlags, uint32_t iIndex, uint32_t iValue)
{
    uint32_t iArrayIndex = iIndex / 32;
    uint32_t iBitIndex = iIndex % 32;
    if(iValue > 0)
    {
        uint32_t iRet = 1 << iBitIndex;
        aiFlags[iArrayIndex] |= iRet;
    }
    else
    {
        uint32_t iRet = ~(1 << iBitIndex);
        aiFlags[iArrayIndex] &= iRet;
    }
}

/*
**
*/
uint32_t getBitFlag(uint32_t* aiFlags, uint32_t iIndex)
{
    uint32_t iArrayIndex = iIndex / 32;
    uint32_t iBitIndex = iIndex % 32;
    uint32_t iRet = (aiFlags[iArrayIndex] & (1 << iBitIndex)) >> iBitIndex;
    return iRet;
}