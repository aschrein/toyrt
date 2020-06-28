cmake_minimum_required(VERSION 3.4.3)
project(toyrt)
add_subdirectory(3rdparty/assimp)
add_executable(toyrt
main.cpp
files.cpp
rt.hpp
utils.hpp
)
# -Wall
SET(CMAKE_CXX_FLAGS "-march=skylake-avx512 \
${CMAKE_CXX_FLAGS}")
add_library(TracyClient STATIC 3rdparty/tracy/TracyClient.cpp)
target_include_directories(TracyClient PRIVATE 3rdparty/tracy)
target_compile_definitions(TracyClient PRIVATE TRACY_ENABLE=1)

target_include_directories(toyrt
  PRIVATE
  3rdparty
  ${INCLUDES}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
target_link_libraries(toyrt
TracyClient
assimp
)
