# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(calpr VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CALPR_BUILD_LIBRARY "Build the ncnn-based library (needs ncnn)" ON)
option(CALPR_BUILD_APP "Build the command-line app (needs OpenCV)" ON)
option(CALPR_BUILD_TESTS "Build the unit tests (no dependencies)" ON)

# Header-only part: decoder, geometry, image ops, YOLO decoding.
add_library(calpr_headers INTERFACE)
target_include_directories(calpr_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CALPR_BUILD_LIBRARY)
  # If not found automatically: -Dncnn_DIR=<ncnn install>/lib/cmake/ncnn
  find_package(ncnn REQUIRED)
  add_library(calpr src/detector.cpp src/recognizer.cpp src/pipeline.cpp)
  target_link_libraries(calpr PUBLIC calpr_headers PRIVATE ncnn)

  if(CALPR_BUILD_APP)
    # If not found automatically: -DOpenCV_DIR=<opencv build or install dir>
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)
    add_executable(calpr_app src/app/main.cpp)
    target_link_libraries(calpr_app PRIVATE calpr ${OpenCV_LIBS})
  endif()
endif()

if(CALPR_BUILD_TESTS)
  enable_testing()
  foreach(name decoder geometry yolo_decode)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE calpr_headers)
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
endif()
