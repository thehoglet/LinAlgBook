cmake_minimum_required(VERSION 3.23)

project(cla-armadillo)

find_package(Armadillo CONFIG REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(Stb REQUIRED)

add_executable(cla-armadillo main.cpp rref.cpp)

target_include_directories(cla-armadillo PRIVATE ${Stb_INCLUDE_DIR})

target_link_libraries(cla-armadillo PRIVATE armadillo)
target_link_libraries(cla-armadillo PUBLIC Matplot++::matplot)

target_compile_features(cla-armadillo PUBLIC cxx_std_20)
