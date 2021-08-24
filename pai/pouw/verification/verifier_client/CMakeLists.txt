cmake_minimum_required(VERSION 3.20)

project(ClientVerifier C CXX)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  add_definitions(-D_WIN32_WINNT=0x600)
endif()

if(GRPC_AS_SUBMODULE)
  add_subdirectory(../../.. ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
  message(STATUS "Using gRPC via add_subdirectory.")
  
  # After using add_subdirectory, we can now use the grpc targets directly from
  # this build.
  set(_PROTOBUF_LIBPROTOBUF libprotobuf)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
  set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
else()
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${protobuf_VERSION}")

  set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")

  set(_GRPC_GRPCPP_UNSECURE gRPC::grpc++_unsecure)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

# Proto file for verification
get_filename_component(verification_proto "../verifier.proto" ABSOLUTE)
get_filename_component(verification_proto_path "${verification_proto}" PATH)

# Proto file for task info
get_filename_component(task_info_proto "../task_info.proto" ABSOLUTE)
get_filename_component(task_info_proto_path "${task_info_proto}" PATH)


# Generated sources for verification
set(verification_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/verifier.pb.cc")
set(verification_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/verifier.pb.h")
set(verification_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/verifier.grpc.pb.cc")
set(verification_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/verifier.grpc.pb.h")


# Generated sources for task info
set(task_info_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/task_info.pb.cc")
set(task_info_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/task_info.pb.h")
set(task_info_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/task_info.grpc.pb.cc")
set(task_info_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/task_info.grpc.pb.h")


add_custom_command(
      OUTPUT "${verification_proto_srcs}" "${verification_proto_hdrs}" "${verification_grpc_srcs}" "${verification_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${verification_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${verification_proto}"
      DEPENDS "${verification_proto}")


add_custom_command(
        OUTPUT "${task_info_proto_srcs}" "${task_info_proto_hdrs}" "${task_info_grpc_srcs}" "${task_info_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${task_info_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${task_info_proto}"
        DEPENDS "${task_info_proto}")


include_directories("${CMAKE_CURRENT_BINARY_DIR}")

foreach(_target
  verifier_client)
  add_executable(${_target} "${_target}.cc"
    ${verification_proto_srcs}
    ${verification_grpc_srcs}
          ${task_info_proto_srcs}
          ${task_info_grpc_srcs})
  target_link_libraries(${_target}
    ${_GRPC_GRPCPP_UNSECURE}
    ${_PROTOBUF_LIBPROTOBUF})
endforeach()
