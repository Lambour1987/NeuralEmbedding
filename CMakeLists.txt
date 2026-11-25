
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

add_executable(
    test_linear tests/test_linear.cpp)
    target_include_directories(test_linear PRIVATE include
)

add_test(
    NAME LinearTests COMMAND test_linear
)


add_executable(test_activations tests/test_activations.cpp)
target_include_directories(test_activations PRIVATE include)
add_test(NAME ActivationTests COMMAND test_activations)

add_executable(test_linear_relu tests/test_linear_relu.cpp)
target_include_directories(test_linear_relu PRIVATE include)
add_test(NAME LinearReluTests COMMAND test_linear_relu)

add_executable(test_multi_layer tests/test_multi_layer.cpp)
target_include_directories(test_multi_layer PRIVATE include)
add_test(NAME MultiLayerTests COMMAND test_multi_layer)

add_executable(test_sequential tests/test_sequential.cpp)
target_include_directories(test_sequential PRIVATE include)
add_test(NAME SequentialTests COMMAND test_sequential)

add_executable(test_loss tests/test_loss.cpp)
target_include_directories(test_loss PRIVATE include)
add_test(NAME LossTests COMMAND test_loss)

add_executable(
    test_trainable_neuron
    tests/test_trainable_neuron.cpp
)

target_include_directories(test_trainable_neuron PRIVATE include)
add_test(NAME TrainableNeuronTests COMMAND test_trainable_neuron)
target_include_directories(
    test_trainable_neuron PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(test_trainable_neuron PRIVATE /W4)
else()
    target_compile_options(
        test_trainable_neuron PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()

#24-11-26: toegevoegd
add_executable(
    test_linear_backward
    tests/test_linear_backward.cpp
)

target_include_directories(
    test_linear_backward PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME LinearBackwardTests
    COMMAND test_linear_backward
)

if(MSVC)
    target_compile_options(test_linear_backward PRIVATE /W4)
else()
    target_compile_options(
        test_linear_backward PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()

add_executable(
    test_linear_relu_backward
    tests/test_linear_relu_backward.cpp
)

target_include_directories(
    test_linear_relu_backward PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME LinearReLUBackwardTests
    COMMAND test_linear_relu_backward
)

if(MSVC)
    target_compile_options(test_linear_relu_backward PRIVATE /W4)
else()
    target_compile_options(
        test_linear_relu_backward PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()


# Gradient Descent optimizer tests
add_executable(
    test_gradient_descent
    tests/test_gradient_descent.cpp
)

target_include_directories(
    test_gradient_descent PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME GradientDescentTests
    COMMAND test_gradient_descent
)

if(MSVC)
    target_compile_options(test_gradient_descent PRIVATE /W4)
else()
    target_compile_options(
        test_gradient_descent PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()


# Trainer tests
add_executable(
    test_trainer
    tests/test_trainer.cpp
)

target_include_directories(
    test_trainer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME TrainerTests
    COMMAND test_trainer
)

if(MSVC)
    target_compile_options(test_trainer PRIVATE /W4)
else()
    target_compile_options(
        test_trainer PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()


# Neural network training demo
add_executable(
    training_demo
    src/training_demo.cpp
)

target_include_directories(
    training_demo PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(training_demo PRIVATE /W4)
else()
    target_compile_options(
        training_demo PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()


# Linear Leaky ReLU backward tests
add_executable(
    test_linear_leaky_relu
    tests/test_linear_leaky_relu.cpp
)

target_include_directories(
    test_linear_leaky_relu PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME LinearLeakyReLUBackwardTests
    COMMAND test_linear_leaky_relu
)

if(MSVC)
    target_compile_options(test_linear_leaky_relu PRIVATE /W4)
else()
    target_compile_options(
        test_linear_leaky_relu PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()
