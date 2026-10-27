
cmake_minimum_required(VERSION 3.20)

project(NeuralEmbedding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Demo executable
add_executable(NeuralEmbedding
    src/main.cpp
)

# Matrix multiplication benchmark
add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
)

target_include_directories(benchmark_matmul PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Tensor unit tests
add_executable(test_tensor
    tests/test_tensor.cpp
)

# Tensor operation tests
add_executable(test_tensor_ops
    tests/test_tensor_ops.cpp
)

# Header files
target_include_directories(NeuralEmbedding PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_tensor PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_tensor_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Compiler warnings
if(MSVC)
    target_compile_options(NeuralEmbedding PRIVATE /W4)
    target_compile_options(test_tensor PRIVATE /W4)
    target_compile_options(test_tensor_ops PRIVATE /W4)
    target_compile_options(benchmark_matmul PRIVATE /W4)
else()
    target_compile_options(NeuralEmbedding PRIVATE
        -Wall -Wextra -Wpedantic
    )
    target_compile_options(test_tensor PRIVATE
        -Wall -Wextra -Wpedantic
    )
    target_compile_options(test_tensor_ops PRIVATE
        -Wall -Wextra -Wpedantic
    )
    target_compile_options(benchmark_matmul PRIVATE
    -Wall -Wextra -Wpedantic
)
endif()

# Automated testing with CTest
enable_testing()

add_test(
    NAME TensorTests
    COMMAND test_tensor
)

add_test(
    NAME TensorOperationTests
    COMMAND test_tensor_ops
)
