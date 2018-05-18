# Author: Eugenio Marinetto
# Date: 2015-09-23

set(EP_OPTION_NAME "USE_${EP_NAME}")
set(EP_REQUIRED_PROJECTS Slicer SlicerIGT SlicerPointSetProcessing PointRecorder)
set(EP_URL "https://github.com/HGGM-LIM/LumpNavLIM")

cma_envvar(SLICERPATH APPEND "@BINARY_DIR@/lib/Slicer-${${PROJECT_NAME}_Slicer_VERSION}/qt-scripted-modules")
cma_envvar(PYTHONPATH APPEND "@SOURCE_DIR@")


cma_end_definition()
# -----------------------------------------------------------------------------

set(EP_CMAKE_ARGS
  -DADDITIONAL_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  -DADDITIONAL_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DBUILD_TESTING:BOOL=OFF
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DPYTHON_EXECUTABLE:FILEPATH=${${PROJECT_NAME}_Python_EXECUTABLE}
  -DPYTHON_INCLUDE_DIR:PATH=${${PROJECT_NAME}_Python_INCLUDE_DIR}
  -DPYTHON_LIBRARY:FILEPATH=${${PROJECT_NAME}_Python_LIBRARY}
  -DSlicer_DIR:PATH=${${PROJECT_NAME}_Slicer_DIR})

list(APPEND EP_CMAKE_ARGS
)

ExternalProject_Add(${EP_NAME}
  DEPENDS ${EP_REQUIRED_PROJECTS}
  # download
  GIT_REPOSITORY ${EP_URL}
  #GIT_TAG master
  # patch
  # update
  UPDATE_COMMAND ""
  # configure
  SOURCE_DIR ${PROJECT_BINARY_DIR}/${EP_NAME}
  CMAKE_ARGS ${EP_CMAKE_ARGS}
  # build
  BINARY_DIR ${PROJECT_BINARY_DIR}/${EP_NAME}-build
  # install
  INSTALL_COMMAND ""
  # test
  )

set(${PROJECT_NAME}_${EP_NAME}_DIR "${PROJECT_BINARY_DIR}/${EP_NAME}-build/${EP_NAME}-build" CACHE INTERNAL "")