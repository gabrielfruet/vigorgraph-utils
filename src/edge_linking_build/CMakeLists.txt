cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage edge_linking.cpp )
target_link_libraries( edge_linking ${OpenCV_LIBS} )
target_include_directories(edge_linking PRIVATE -I/home/gabrielfruet/.local/lib/python3.12/site-packages/pybind11/include) 

