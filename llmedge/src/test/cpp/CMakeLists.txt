cmake_minimum_required(VERSION 3.22)
project(video_jni_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_package(Java REQUIRED COMPONENTS Development)

set(TEST_JAVA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/java/io/aatricks/llmedge/NativeTestProgressCallback.java)
set(TEST_JAVA_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java-classes)
set(TEST_JAVA_CLASS ${TEST_JAVA_OUT_DIR}/io/aatricks/llmedge/NativeTestProgressCallback.class)

add_custom_command(
    OUTPUT ${TEST_JAVA_CLASS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_JAVA_OUT_DIR}
    COMMAND ${Java_JAVAC_EXECUTABLE} -d ${TEST_JAVA_OUT_DIR} ${TEST_JAVA_SRC}
    DEPENDS ${TEST_JAVA_SRC}
    COMMENT "Compiling Java progress callback helper"
)

add_custom_target(java_test_classes ALL DEPENDS ${TEST_JAVA_CLASS})

add_library(sd_jni_under_test STATIC
    ../../main/cpp/sdcpp_jni.cpp
    sd_test_stubs.cpp
)

target_compile_definitions(sd_jni_under_test PRIVATE SD_JNI_TESTING)

target_include_directories(sd_jni_under_test PRIVATE
    ../../main/cpp
    ../../../../stable-diffusion.cpp
    ${JNI_INCLUDE_DIRS}
)

add_executable(video_jni_tests
    test_video_jni.cpp
)

target_include_directories(video_jni_tests PRIVATE
    ../../main/cpp
    ../../../../stable-diffusion.cpp
    ${JNI_INCLUDE_DIRS}
)

target_link_libraries(video_jni_tests PRIVATE
    sd_jni_under_test
    ${JNI_LIBRARIES}
)

target_compile_definitions(video_jni_tests PRIVATE
    SD_TEST_JAVA_CLASS_DIR="${TEST_JAVA_OUT_DIR}"
    SD_JNI_TESTING
)

add_dependencies(video_jni_tests java_test_classes)
