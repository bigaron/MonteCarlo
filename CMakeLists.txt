cmake_minimum_required(VERSION 3.20)
project(MonteCarlo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)

add_subdirectory(include/glfw/)
add_subdirectory(include/glew/build/cmake)

add_executable(MonteCarlo ./src/main.cpp ./src/Shader.h)

target_include_directories(MonteCarlo PUBLIC ./src/)


target_include_directories(MonteCarlo PUBLIC include/glfw/include/)
target_include_directories(MonteCarlo PUBLIC include/glew/include/GL/)
target_include_directories(MonteCarlo PUBLIC include/glm/)

target_link_libraries(MonteCarlo LINK_PUBLIC glfw)
target_link_libraries(MonteCarlo LINK_PUBLIC glew)
add_custom_command(TARGET MonteCarlo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ./bin/glew32.dll ./
    COMMAND_EXPAND_LISTS
)