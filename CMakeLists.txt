# ---------------------------------------------------------------------------------
# Specify the minimum CMake version required, set project name and library
# ---------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.25)
project(cv_library)
add_library(${PROJECT_NAME})
message("Project name is: ${PROJECT_NAME}")

# ---------------------------------------------------------------------------------
# Find OpenMp
# ---------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------------
# Find packages installed
# ---------------------------------------------------------------------------------
# Find Protobuf
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${Protobuf_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Protobuf_LIBRARIES})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

# ---------------------------------------------------------------------------------
# Specify the directories containing header files
# ---------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME}
    PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
file(GLOB_RECURSE public_headers "${CMAKE_SOURCE_DIR}/include/**/*.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")
foreach(header ${public_headers})
    file(RELATIVE_PATH header_file_path "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}" "${header}")
    get_filename_component(header_directory_path "${header_file_path}" DIRECTORY)
    install(
        FILES ${header}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${header_directory_path}"
    )
endforeach()

# ---------------------------------------------------------------------------------
# Add source files to library
# ---------------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})

# ---------------------------------------------------------------------------------
# Add examples as executable files
# ---------------------------------------------------------------------------------
file(GLOB_RECURSE EXAMPLE_FILES "${CMAKE_SOURCE_DIR}/src/examples/*.cpp")
foreach(example_file ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_file} NAME_WE) # Extract name without extension
    add_executable(${example_name} ${example_file})              # Create executable
    target_link_libraries(${example_name} PRIVATE ${PROJECT_NAME}) # Link the library
    set_target_properties(${example_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Place executables in bin/
endforeach()
