cmake_minimum_required(VERSION 3.16)
project(HumanActionRecognition CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -Wall")

#  Find OpenCV 
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs ml video)
message(STATUS "OpenCV  : ${OpenCV_VERSION}")
message(STATUS "Build   : ${CMAKE_BUILD_TYPE}")

# Include directories 

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Source files (src) 
set(SOURCES
    src/main.cpp
    src/dataset_loader.cpp
    src/detector.cpp
    src/feature_extractor.cpp
    src/action_classifier.cpp
    src/evaluator.cpp
)

#  Build executable 
add_executable(har ${SOURCES})
target_link_libraries(har ${OpenCV_LIBS})

# Copy binary to project root after build (convenience)
add_custom_command(TARGET har POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:har> ${CMAKE_SOURCE_DIR}/har
    COMMENT "→ Binary copied to project root: ./har")
