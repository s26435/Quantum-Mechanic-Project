cmake_minimum_required(VERSION 3.18)

project(Quantum_Physics_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(Quantum_Physics_Project main.cu)

set_target_properties(Quantum_Physics_Project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_property(TARGET Quantum_Physics_Project PROPERTY CUDA_ARCHITECTURES native)

target_compile_options(Quantum_Physics_Project PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
