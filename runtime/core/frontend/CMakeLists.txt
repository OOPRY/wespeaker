add_library(frontend STATIC
  feature_pipeline.cc
  fft.cc
)
target_link_libraries(frontend PUBLIC utils)
target_compile_options(frontend PRIVATE -Wno-pessimizing-move  -Wno-sign-compare)
