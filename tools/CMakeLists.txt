# tools/CMakeLists.txt — CLI tools

add_executable(gguf_inspect gguf_inspect.cpp)
target_link_libraries(gguf_inspect PRIVATE jinf_core)

add_executable(prepare_model prepare_model.cpp)
target_link_libraries(prepare_model PRIVATE jinf_core)

add_executable(profile_activations profile_activations.cpp)
target_link_libraries(profile_activations PRIVATE jinf_core)

add_executable(train_predictor train_predictor.cpp)
target_link_libraries(train_predictor PRIVATE jinf_core)
