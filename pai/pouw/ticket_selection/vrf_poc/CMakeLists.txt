cmake_minimum_required(VERSION 3.14)
project(vrf_poc)

set(CMAKE_C_STANDARD 99)

add_executable(vrf_poc main.cpp)
target_link_libraries(vrf_poc sodium)
target_include_directories(vrf_poc PRIVATE /usr/local/include /usr/local/include/sodium/)
target_link_directories(vrf_poc PRIVATE /usr/local/lib)