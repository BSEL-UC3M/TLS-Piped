# Author: Ali Uneri
# Date: 2012-10-28

set(EP_OPTION_NAME "USE_${EP_NAME}")
set(EP_REQUIRED_PROJECTS Git)
set(EP_URL "git://github.com/commontk/zlib.git")

cma_end_definition()
# -----------------------------------------------------------------------------

if(USE_Slicer)
  if(${PROJECT_NAME}_Slicer_VERSION VERSION_EQUAL "4.4")
    set(EP_VERSION "66a753054b356da85e1838a081aa94287226823e")
  elseif(${PROJECT_NAME}_Slicer_VERSION VERSION_EQUAL "4.5")
    set(EP_VERSION "66a753054b356da85e1838a081aa94287226823e")
  else()
    message(FATAL_ERROR "Slicer version ${${PROJECT_NAME}_Slicer_VERSION} is not supported")
  endif()
else()
  set(EP_VERSION "66a753054b356da85e1838a081aa94287226823e")
endif()

set(EP_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DZLIB_MANGLE_PREFIX:STRING=cma_
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)

ExternalProject_Add(${EP_NAME}
  DEPENDS ${EP_REQUIRED_PROJECTS}
  # download
  GIT_REPOSITORY ${EP_URL}
  GIT_TAG ${EP_VERSION}
  # patch
  # update
  UPDATE_COMMAND ""
  # configure
  SOURCE_DIR ${PROJECT_BINARY_DIR}/${EP_NAME}
  CMAKE_ARGS ${EP_CMAKE_ARGS}
  # build
  BINARY_DIR ${PROJECT_BINARY_DIR}/${EP_NAME}-build
  # install
  INSTALL_DIR ${PROJECT_BINARY_DIR}/${EP_NAME}-install
  # test
  )

set(${PROJECT_NAME}_ZLIB_ROOT "${PROJECT_BINARY_DIR}/${EP_NAME}-install" CACHE INTERNAL "")
set(${PROJECT_NAME}_ZLIB_INCLUDE_DIR "${${PROJECT_NAME}_ZLIB_ROOT}/include" CACHE INTERNAL "")
if(WIN32)
  set(${PROJECT_NAME}_ZLIB_LIBRARY "${${PROJECT_NAME}_ZLIB_ROOT}/lib/zlib.lib" CACHE INTERNAL "")
else()
  set(${PROJECT_NAME}_ZLIB_LIBRARY "${${PROJECT_NAME}_ZLIB_ROOT}/lib/libzlib.a" CACHE INTERNAL "")
endif()
