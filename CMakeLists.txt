cmake_minimum_required  ( VERSION 2.8                                              )

project                 ( OpenVX                                          )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DIVX_USE_OPENCV)
find_package            ( OpenCV REQUIRED                                          )
include_directories     ( ${OpenCV_INCLUDE_DIRS}                                   )
include_directories     ( ${OpenVX_INCLUDE_DIRS}                                   )
link_directories        ( ${OpenVX_LIBS_DIR}                                       )


add_executable          ( openvx  opt_vx.cpp                       )
target_link_libraries   ( openvx ${OpenVX_LIBS} ${OpenCV_LIBRARIES} ${VISIONWORKS_LIBS}   ${visionworks}   )

