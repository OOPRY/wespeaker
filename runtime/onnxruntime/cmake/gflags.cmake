set(BUILD_PACKAGING OFF CACHE BOOL "" FORCE)
set(INSTALL_HEADERS OFF CACHE BOOL "" FORCE)
set(INSTALL_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(INSTALL_STATIC_LIBS OFF CACHE BOOL "" FORCE)
set(INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.2.zip
  URL_HASH SHA256=19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)