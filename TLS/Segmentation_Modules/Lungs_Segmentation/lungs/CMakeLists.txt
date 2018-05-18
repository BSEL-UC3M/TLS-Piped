#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
cmake_minimum_required(VERSION 2.8)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

include_directories(${PROJECT_SOURCE_DIR}/Segmentation_Modules/libs/)

ADD_EXECUTABLE(choose_lungs MACOSX_BUNDLE choose_lungs_main.cxx)
TARGET_LINK_LIBRARIES(choose_lungs ${ITK_LIBRARIES})

ADD_EXECUTABLE(choose_lungsV2 MACOSX_BUNDLE choose_lungsv2.cxx)
TARGET_LINK_LIBRARIES(choose_lungsV2 ${ITK_LIBRARIES})

ADD_EXECUTABLE(test_centroid MACOSX_BUNDLE test_centroid.cxx)
TARGET_LINK_LIBRARIES(test_centroid ${ITK_LIBRARIES})

ADD_EXECUTABLE(labelize MACOSX_BUNDLE labelize_main.cxx)
TARGET_LINK_LIBRARIES(labelize ${ITK_LIBRARIES})

ADD_EXECUTABLE(label_filter MACOSX_BUNDLE label_filter.cxx)
TARGET_LINK_LIBRARIES(label_filter ${ITK_LIBRARIES})

ADD_EXECUTABLE(hu_threshold MACOSX_BUNDLE itkHuThresholdImageFilter.txx hu_threshold_main.cxx)
TARGET_LINK_LIBRARIES(hu_threshold ${ITK_LIBRARIES} )

ADD_EXECUTABLE(trachea_center MACOSX_BUNDLE trachea_center.cxx)
TARGET_LINK_LIBRARIES(trachea_center ${ITK_LIBRARIES})
