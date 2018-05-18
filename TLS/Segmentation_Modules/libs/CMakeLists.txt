cmake_minimum_required(VERSION 2.8.7)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
ADD_LIBRARY (helpers SHARED dicom_utilities) 

TARGET_LINK_LIBRARIES(helpers ${ITK_LIBRARIES})

