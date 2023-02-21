#pragma once

#include <stdint.h>

#if defined(_DEBUG) || defined(DEBUG)
#define WTFASSERT(X, ...)	wtfAssert(__LINE__, __FUNCTION__, X, __VA_ARGS__)
#else
//#define DEBUG_PRINTF(...)	{}
#define WTFASSERT(X, ...)	
#endif // #if _DEBUG

void wtfAssert(uint32_t iLine, char const* szFunction, bool bStatement, char const* szFormat, ...);