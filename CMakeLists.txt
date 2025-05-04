cmake_minimum_required(VERSION 3.30 FATAL_ERROR) # v3.30 is required for C++23 `import std;` support
project(ga LANGUAGES CXX)

# ==================================================
# = Compilation Settings
# ==================================================

set(TERMINAL_BUILD_EXAMPLES ON CACHE BOOL "Build examples")

# Configure options
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdangling-gsl -Wreturn-stack-address -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
set(CMAKE_CXX_SCAN_FOR_MODULES 1)

# ==================================================
# = Dependencies Setup
# ==================================================
include(FetchContent)

# ------------------- spdlog -------------------
# FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.2.tar.gz)
# FetchContent_MakeAvailable(spdlog)

# ------------------- PaGMO -------------------
find_package(Pagmo REQUIRED)

# ==================================================
# = Executable Definitions
# ==================================================

# ------------------- Main Binary -------------------
add_executable(${PROJECT_NAME} src/main.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE Pagmo::pagmo)

# ==================================================
# = Target Configurations
# ==================================================

# ------------------- Link Time Optimizations -------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    # set_target_properties(${PROJECT_NAME}_tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO / LTO not supported: <${error}>")
endif ()
