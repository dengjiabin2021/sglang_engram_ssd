cmake_minimum_required(VERSION 3.18)
project(sglang_engram_ssd_query LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

# 同步读取核心保持为独立原生库，Python 模块按需构建。
add_library(engram_ssd_query STATIC batch_reader.cpp thread_pool.cpp io_uring_reader.cpp)
set_target_properties(engram_ssd_query PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(engram_ssd_query PUBLIC cxx_std_20)
target_compile_definitions(engram_ssd_query PRIVATE _FILE_OFFSET_BITS=64)

target_include_directories(
    engram_ssd_query
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(engram_ssd_query PUBLIC ${TORCH_LIBRARIES} Threads::Threads)

# 将 Torch 的 ABI 编译参数传递给后续链接此库的目标。
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(engram_ssd_query PUBLIC ${TORCH_CXX_FLAGS_LIST})

option(ENGRAM_SSD_ENABLE_IO_URING "Enable the optional liburing backend" OFF)
if(ENGRAM_SSD_ENABLE_IO_URING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The io_uring backend requires Linux")
  endif()
  find_path(LIBURING_INCLUDE_DIR NAMES liburing.h REQUIRED)
  find_library(LIBURING_LIBRARY NAMES uring REQUIRED)
  # 宏、头路径和链接依赖向绑定目标传播，保证类定义在所有目标中一致。
  target_compile_definitions(engram_ssd_query PUBLIC ENGRAM_SSD_ENABLE_IO_URING=1)
  target_include_directories(engram_ssd_query PUBLIC ${LIBURING_INCLUDE_DIR})
  target_link_libraries(engram_ssd_query PUBLIC ${LIBURING_LIBRARY})
else()
  target_compile_definitions(engram_ssd_query PUBLIC ENGRAM_SSD_ENABLE_IO_URING=0)
endif()

option(ENGRAM_SSD_BUILD_PYTHON "Build the optional Python binding" OFF)
if(ENGRAM_SSD_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  # Tensor 与 Python 对象的转换由 Torch 提供，不能只链接原生 Torch 库。
  find_library(
      TORCH_PYTHON_LIBRARY
      NAMES torch_python
      PATHS "${TORCH_INSTALL_PREFIX}/lib"
      NO_DEFAULT_PATH
      REQUIRED)

  Python3_add_library(_C MODULE WITH_SOABI bindings.cpp)
  target_link_libraries(_C PRIVATE engram_ssd_query ${TORCH_PYTHON_LIBRARY})
  # 放到当前源码包，现有 editable 安装即可通过 query._C 加载。
  set_target_properties(
      _C PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
