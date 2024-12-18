# Define the static library from the source files
add_library(opencv_cv_lib STATIC
    image_processing.cpp
    filters.cpp
    feature_detection.cpp
    transformation.cpp
    utils.cpp
)

# Link OpenCV to the library
target_link_libraries(opencv_cv_lib ${OpenCV_LIBS})

# Include the necessary directories
target_include_directories(opencv_cv_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)