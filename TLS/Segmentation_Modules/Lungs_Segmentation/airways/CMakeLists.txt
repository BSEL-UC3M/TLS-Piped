INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/Segmentation_Modules/libs/)
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)
find_package(Boost COMPONENTS filesystem REQUIRED)
message(from airways)

ADD_EXECUTABLE(air_ext_fm SegmentAirwayTree_FM.cxx)
TARGET_LINK_LIBRARIES(air_ext_fm ${ITK_LIBRARIES})

ADD_EXECUTABLE(air_ext_fm_ms SegmentAirwayTree_FM_multiseed.cxx)
TARGET_LINK_LIBRARIES(air_ext_fm_ms ${ITK_LIBRARIES})

ADD_EXECUTABLE(label_air label_air_main.cxx)
TARGET_LINK_LIBRARIES(label_air ${Boost_LIBRARIES})

ADD_EXECUTABLE(Tree_SP SecondPassFM.cxx)
TARGET_LINK_LIBRARIES(Tree_SP ${ITK_LIBRARIES})

