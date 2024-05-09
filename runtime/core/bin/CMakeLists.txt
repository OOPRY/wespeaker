add_executable(asv_main asv_main.cc)
target_link_libraries(asv_main PUBLIC speaker)

add_executable(extract_emb_main extract_emb_main.cc)
target_link_libraries(extract_emb_main PUBLIC speaker)
target_compile_options(asv_main PRIVATE -Wno-sign-compare)
target_compile_options(extract_emb_main PRIVATE -Wno-sign-compare)