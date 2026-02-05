cmake_minimum_required(VERSION 3.16)

project(physis LANGUAGES CXX) 
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
find_package(OpenMP REQUIRED) 
# Collect only .cpp files for compilation 

file(GLOB_RECURSE PHYSIS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ) 
add_library(physis ${PHYSIS_SOURCES}) 
# Include directories for headers 
target_include_directories(physis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/bessel-library ) 

target_link_libraries(physis PUBLIC OpenMP::OpenMP_CXX ) 
#other flags : -Wall -Wextra

target_compile_options(physis PRIVATE -O3 -ffast-math -funroll-loops ) 

add_executable(main full_sim.cpp) 

target_link_libraries(main PRIVATE physis) 

target_compile_options(main PRIVATE -O3 -ffast-math -funroll-loops)