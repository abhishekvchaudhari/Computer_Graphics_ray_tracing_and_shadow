# Set minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set project name
project(raytracer1b)

# Add the source file
set(SOURCES
    main.cpp
)

# Add the executable target
add_executable(raytracer1b ${SOURCES})

# Add any necessary include directories
target_include_directories(raytracer1b
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ # Include the current source directory
        # Add any other include directories here
)

# Add any necessary libraries
target_link_libraries(raytracer1b
    # Add any necessary libraries here
)