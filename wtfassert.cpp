#include "wtfassert.h"

#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <string>
#include <sstream>
#include "LogPrint.h"
#include <assert.h>

#ifdef _MSC_VER
#include <Windows.h>
#include <DbgHelp.h>
#endif // _MSC_VER

#ifdef ANDROID
#include <android/log.h>
#endif // ANDROID

#if defined(SAVE_LOG_FILE)
static bool sbStarted = false;
extern "C" char const* getSaveDir();
#endif // SAVE_LOG_FILE

static char sacBuffer[65536];
static char sacPreStatement[1024];

void wtfAssert(
    uint32_t iLine, 
    char const* szFunction, 
    bool bStatement, 
    char const* szFormat, ...)
{
    if(!bStatement)
    {
        va_list args;
        va_start(args, szFormat);
        vsprintf(sacBuffer, szFormat, args);
        //perror(szBuffer);
        va_end(args);

        uint64_t iLast = strlen(sacBuffer);
        sacBuffer[iLast] = '\n';
        sacBuffer[iLast + 1] = '\0';

        sprintf(sacPreStatement, "\n!!!!!!!!!\n%s line %d\n", szFunction, iLine);

#ifdef _MSC_VER
        OutputDebugStringA(sacPreStatement);
        OutputDebugStringA(sacBuffer);
#else
#ifdef ANDROID
        __android_log_print(ANDROID_LOG_VERBOSE, "RenderWithMe", "%s", szBuffer);
#else 
        printf("%s", szBuffer);
#endif // ANDROID
#endif // _MSC_VER
        //#endif // _DEBUG
    
        if(!bStatement)
        {
            std::ostringstream oss;
            {
                typedef USHORT(WINAPI* CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
                CaptureStackBackTraceType func = (CaptureStackBackTraceType)(GetProcAddress(LoadLibrary(TEXT("kernel32.dll")), "RtlCaptureStackBackTrace"));

                if(func)
                {
                    // initialize symbols options
                    HANDLE process = GetCurrentProcess();
                    SymInitialize(process, nullptr, true);
                    SymSetOptions(SYMOPT_LOAD_LINES);

                    // Quote from Microsoft Documentation:
                    // ## Windows Server 2003 and Windows XP:  
                    // ## The sum of the FramesToSkip and FramesToCapture parameters must be less than 63.
                    const int kMaxCallers = 62;

                    std::vector<uint8_t> acSymbolBuffer(sizeof(SYMBOL_INFO) + 256);
                    SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(acSymbolBuffer.data());
                    symbol->MaxNameLen = 255;
                    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

                    void* callers[kMaxCallers];
                    int count = (func)(0, kMaxCallers, callers, NULL);
                    for(int32_t i = 1; i < count; i++)
                    {
                        
                        SymFromAddr(process, (DWORD64)(callers[i]), 0, symbol);

                        DWORD  dwDisplacement;
                        IMAGEHLP_LINE64 line;
                        SymGetLineFromAddr64(process, (DWORD64)(callers[i]), &dwDisplacement, &line);

                        oss << i << " ";
                        oss << symbol->Name << " : ";
                        oss << line.LineNumber << " ";
                        oss << line.FileName << "\n\n";
                        DEBUG_PRINTF("%d  %s %s : %d (0x%llX)\n", i, symbol->Name, line.FileName, line.LineNumber, line.Address);
                    }
                }
            }

            std::string totalString = std::string(sacPreStatement) + "\n" + sacBuffer + "\n" + oss.str();
            MessageBoxExA(nullptr, totalString.c_str(), "Assert Failed", MB_ICONERROR | MB_OK, 0);

            _CrtDbgBreak();
        }
    }
}