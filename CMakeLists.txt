cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(ga LANGUAGES CXX)

# ==================================================
# = Compilation Settings
# ==================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdangling-gsl -Wreturn-stack-address -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")

# ==================================================
# = Dependencies Setup
# ==================================================
include(FetchContent)

# ------------------- spdlog -------------------
FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.2.tar.gz)
FetchContent_MakeAvailable(spdlog)

# ------------------- PaGMO -------------------
find_package(Pagmo REQUIRED)

# ==================================================
# = Executable Definitions
# ==================================================

# ------------------- Main Binary -------------------
add_executable(${PROJECT_NAME} 
    src/main.cc
    src/common.cc
    src/problems.cc
    src/operators.cc
    src/statistics.cc
    src/encoding.cc
    src/encoding_operator.cc
    src/initialization.cc
    src/mutation.cc
    src/termination.cc
    src/logging.cc
    src/experiment.cc
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE Pagmo::pagmo spdlog)

# ==================================================
# = Target Configurations
# ==================================================

# ------------------- Link Time Optimizations -------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO / LTO not supported: <${error}>")
endif ()
