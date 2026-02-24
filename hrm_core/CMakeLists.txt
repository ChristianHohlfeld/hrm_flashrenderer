cmake_minimum_required(VERSION 3.16)
project(hrm_cpp_full_fledged_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HRM_BUILD_TESTS "Build tests" ON)

find_package(SQLite3 REQUIRED)

add_library(hrm_core STATIC
  src/hrm/signature.cpp
  src/hrm/router_index.cpp
  src/hrm/sqlite_store.cpp
  src/hrm/rank.cpp
  src/hrm/compose.cpp
  src/hrm/payload_jsonl.cpp
  src/hrm/engine.cpp
)

target_include_directories(hrm_core PUBLIC include)
target_link_libraries(hrm_core PUBLIC SQLite::SQLite3)
set_target_properties(hrm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C ABI shared library for production integration (no subprocess per query)
add_library(hrm_api SHARED
  src/api/hrm_api.cpp
)
target_include_directories(hrm_api PRIVATE include)
target_link_libraries(hrm_api PRIVATE hrm_core)

add_executable(hrm src/cli/main.cpp)
target_link_libraries(hrm PRIVATE hrm_core)

if(HRM_BUILD_TESTS)
  enable_testing()
  add_executable(hrm_tests
    tests/test_main.cpp
    tests/test_signature.cpp
    tests/test_payload.cpp
    tests/test_router_index.cpp
  )
  target_link_libraries(hrm_tests PRIVATE hrm_core)
  add_test(NAME hrm_tests COMMAND hrm_tests)
endif()
