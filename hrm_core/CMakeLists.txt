cmake_minimum_required(VERSION 3.16)
project(hrm_cpp_full_fledged_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HRM_BUILD_TESTS "Build tests" ON)

find_package(SQLite3 QUIET)

set(HRM_SQLITE_TARGET "")
if(SQLite3_FOUND)
  message(STATUS "Using SQLite3 from system development package")
  set(HRM_SQLITE_TARGET SQLite::SQLite3)
else()
  message(STATUS "SQLite3 dev package not found; falling back to runtime .so + bundled compat header")

  set(_sqlite_runtime_candidates
    /lib/x86_64-linux-gnu/libsqlite3.so.0
    /usr/lib/x86_64-linux-gnu/libsqlite3.so.0
    /lib64/libsqlite3.so.0
    /usr/lib64/libsqlite3.so.0
  )

  set(HRM_SQLITE_RUNTIME_LIB "")
  foreach(_cand IN LISTS _sqlite_runtime_candidates)
    if(EXISTS "${_cand}")
      set(HRM_SQLITE_RUNTIME_LIB "${_cand}")
      break()
    endif()
  endforeach()

  if(NOT HRM_SQLITE_RUNTIME_LIB)
    find_library(HRM_SQLITE_RUNTIME_LIB NAMES sqlite3 sqlite3.so.0)
  endif()

  if(NOT HRM_SQLITE_RUNTIME_LIB)
    message(FATAL_ERROR
      "Could not locate a SQLite runtime library. "
      "Install SQLite runtime (libsqlite3.so.0) or sqlite3 development package."
    )
  endif()

  set(HRM_SQLITE_COMPAT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/sqlite3.h")
  if(NOT EXISTS "${HRM_SQLITE_COMPAT_HEADER}")
    message(FATAL_ERROR
      "Missing bundled sqlite3 compat header at ${HRM_SQLITE_COMPAT_HEADER}. "
      "Repository is incomplete."
    )
  endif()

  add_library(hrm_sqlite3_runtime SHARED IMPORTED GLOBAL)
  set_target_properties(hrm_sqlite3_runtime PROPERTIES
    IMPORTED_LOCATION "${HRM_SQLITE_RUNTIME_LIB}"
  )

  add_library(hrm_sqlite3_headers INTERFACE)
  target_include_directories(hrm_sqlite3_headers INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

  add_library(hrm_sqlite3 INTERFACE)
  target_link_libraries(hrm_sqlite3 INTERFACE hrm_sqlite3_runtime hrm_sqlite3_headers)
  set(HRM_SQLITE_TARGET hrm_sqlite3)
endif()

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
target_link_libraries(hrm_core PUBLIC ${HRM_SQLITE_TARGET})
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
