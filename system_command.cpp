#include "system_command.h"

#include <array>
#include <memory>

#include "LogPrint.h"

/*
**
*/
std::string execCommand(std::string const& command, bool bEchoCommand)
{
    if(bEchoCommand)
    {
        DEBUG_PRINTF("%s\n", command.c_str());
    }

    std::array<char, 256> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    if(!pipe)
    {
        return "ERROR";
    }

    while(fgets(buffer.data(), static_cast<uint32_t>(buffer.size()), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    //DEBUG_PRINTF("%s\n", result.c_str());

    return result;
}