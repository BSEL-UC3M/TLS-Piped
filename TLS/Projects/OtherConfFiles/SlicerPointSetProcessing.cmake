# Author: Eugenio Marinetto
# Date: 2015-09-22

set(EP_OPTION_NAME "USE_${EP_NAME}")
set(EP_REQUIRED_PROJECTS Git Slicer Boost PointSetProcessing PoissonReconstruction)
set(EP_URL "https://github.com/nenetto/SlicerPointSetProcessing.git")
set(EP_GIT_TAG "master")
set(EP_OPTION_DESCRIPTION "${EP_NAME} Project")
set(EP_OPTION_DEFAULT OFF)

cma_list(APPEND EP_REQUIRED_PROJECTS Doxygen IF ${PROJECT_NAME}_BUILD_DOCUMENTATION)

cma_envvar(SLICERPATH APPEND "@BINARY_DIR@/SPSetProc-build/lib/Slicer-${${PROJECT_NAME}_Slicer_VERSION}/qt-loadable-modules/@INTDIR@")
cma_envvar(SLICERPATH APPEND "@BINARY_DIR@/SPSetProc-build/lib/Slicer-${${PROJECT_NAME}_Slicer_VERSION}/qt-scripted-modules")
cma_envvar(PYTHONPATH APPEND "@SOURCE_DIR@/SlicerPointSetProcessing")

cma_end_definition()
# -----------------------------------------------------------------------------

set(EP_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DCMAKE_INSTALL_PREFIX:PATH=${${PROJECT_NAME}_INSTALL_DIR}
  -D${EP_NAME}_INSTALL_DIR:BOOL=${${PROJECT_NAME}_INSTALL_DIR}
  -D${EP_NAME}_BUILD_DOCUMENTATION:BOOL=${${PROJECT_NAME}_BUILD_DOCUMENTATION}
  -D${EP_NAME}_CMAKE_DEBUG_FLAG:BOOL=${${PROJECT_NAME}_CMAKE_DEBUG_FLAG}
  -DUSE_External_Slicer_DIR:PATH=${${PROJECT_NAME}_Slicer_DIR}
  -DUSE_External_BOOST_ROOT:PATH=${${PROJECT_NAME}_BOOST_ROOT}
  -DUSE_External_PointSetProcessing_BIN_DIR:PATH==${${PROJECT_NAME}_PointSetProcessing_DIR}
  -DUSE_External_PointSetProcessing_SRC_DIR:PATH=${${PROJECT_NAME}_PointSetProcessing_SRC_DIR}
  -DUSE_External_PoissonReconstruction_BIN_DIR:PATH=${${PROJECT_NAME}_PoissonReconstruction_DIR}
  -DUSE_External_PoissonReconstruction_SRC_DIR:PATH=${${PROJECT_NAME}_PoissonReconstruction_SRC_DIR}
  )

if(${${PROJECT_NAME}_BUILD_DOCUMENTATION})
  list(APPEND EP_CMAKE_ARGS
    -DDOXYGEN_EXECUTABLE:PATH=${${PROJECT_NAME}_DOXYGEN_EXECUTABLE}
    )
endif()

ExternalProject_Add(${EP_NAME}
  DEPENDS ${EP_REQUIRED_PROJECTS}
  # download
  GIT_REPOSITORY ${EP_URL}
  GIT_TAG ${EP_GIT_TAG}
  # patch
  # update
  UPDATE_COMMAND ""
  # configure
  SOURCE_DIR ${PROJECT_BINARY_DIR}/${EP_NAME}
  CMAKE_ARGS ${EP_CMAKE_ARGS}
  # build
  BINARY_DIR ${PROJECT_BINARY_DIR}/SPSP-build
  # install
  INSTALL_COMMAND ""
  # test
  )
set(${PROJECT_NAME}_${EP_NAME}_DIR "${PROJECT_BINARY_DIR}/${EP_NAME}-build" CACHE INTERNAL "")