

cmake_minimum_required(VERSION 3.20)

project(NeuralEmbedding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(NeuralEmbedding
    src/main.cpp
)

target_include_directories(NeuralEmbedding
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(NeuralEmbedding PRIVATE /W4)
else()
    target_compile_options(NeuralEmbedding PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()
