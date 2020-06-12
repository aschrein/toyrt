cmake_minimum_required(VERSION 3.4.3)
project(toyrt)
add_subdirectory(3rdparty/assimp)
add_executable(toyrt
main.cpp
)
target_include_directories(toyrt
  PRIVATE
  3rdparty
  ${INCLUDES}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
target_link_libraries(toyrt
assimp
)
