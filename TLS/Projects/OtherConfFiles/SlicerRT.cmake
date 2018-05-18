# Author: Ali Uneri
# Date: 2014-09-25

set(EP_OPTION_NAME "USE_${EP_NAME}")
set(EP_REQUIRED_PROJECTS Plastimatch Slicer)
set(EP_URL "https://subversion.assembla.com/svn/slicerrt/branches/Slicer-4.5/SlicerRt/src")
set(EP_PATCH "${CMAKE_CURRENT_LIST_DIR}/Patches/${EP_NAME}.patch")

cma_envvar(SLICERPATH APPEND "@BINARY_DIR@/inner-build/lib/Slicer-${${PROJECT_NAME}_Slicer_VERSION}/cli-modules/@INTDIR@")
cma_envvar(SLICERPATH APPEND "@BINARY_DIR@/inner-build/lib/Slicer-${${PROJECT_NAME}_Slicer_VERSION}/qt-loadable-modules/@INTDIR@")
cma_envvar(SLICERPATH APPEND "@BINARY_DIR@/inner-build/lib/Slicer-${${PROJECT_NAME}_Slicer_VERSION}/qt-scripted-modules")

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
  -DPlastimatch_DIR:PATH=${${PROJECT_NAME}_Plastimatch_DIR}
  -DSlicer_DIR:PATH=${${PROJECT_NAME}_Slicer_DIR})

ExternalProject_Add(${EP_NAME}
  DEPENDS ${EP_REQUIRED_PROJECTS}
  # download
  SVN_REPOSITORY ${EP_URL}
  #SVN_REVISION -r 2001
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