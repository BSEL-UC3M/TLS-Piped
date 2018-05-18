###########################################################################################################################
### DON'T FORGET TO INCLUDE CMAKE_BUILD_TYPE = DEBUG  IN CASE YOU WANT TO DEVELOP (at least is hot works for eclipse) #####
###########################################################################################################################

cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

set(PROJECT_NAME "TLS" CACHE STRING "")
project(${PROJECT_NAME})

#INCLUDE(CPack)
#INCLUDE(CTest)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
find_package(CMakeAll REQUIRED)

# options
option(TLS_BUILD_DOCUMENTATION "Support for Doxygen documentation" OFF)
option(TLS_BUILD_TESTING "Support for CTest testing" ON)
option(TLS_BUILD_PACKAGING "Support for CPack packaging" ON)
option(TLS_SECOND_STEP "Check to ON if yo have already compiled all the dependencies" OFF)


# version
set(TLS_VERSION_MAJOR 1)
set(TLS_VERSION_MINOR 0)
#cma_git_revision(${PROJECT_SOURCE_DIR} TLS_VERSION_PATCH)
#set(TLS "${TLS_VERSION_MAJOR}.${TLS_VERSION_MINOR}.${TLS_VERSION_PATCH}")
set(TLS_VERSION "${TLS_VERSION_MAJOR}.${TLS_VERSION_MINOR}")


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "/MP ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "/MP ${CMAKE_C_FLAGS}")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()


# configure projects
file(GLOB TLS_PROJECTS "${PROJECT_SOURCE_DIR}/Projects/*.cmake")
cma_add_projects(${TLS_PROJECTS})


cma_print_projects()

message(STATUS "${PROJECT_NAME} ${TLS_VERSION}")

if(TLS_BUILD_DOCUMENTATION AND NOT USE_Doxygen)
  message("TLS_BUILD_DOCUMENTATION is enabling Doxygen")
  set(USE_Doxygen ON CACHE BOOL "" FORCE)
endif()

cma_configure_projects()

if(TLS_BUILD_DOCUMENTATION)
  add_subdirectory(Doc)
endif()



# configure files
file(WRITE ${PROJECT_BINARY_DIR}/CMake.h "#ifndef __CMake_h\n#define __CMake_h\n\n")
file(WRITE ${PROJECT_BINARY_DIR}/CMake.py "")
file(WRITE ${PROJECT_BINARY_DIR}/CMake.m "")
get_cmake_property(VARIABLES VARIABLES)
foreach (KEY ${VARIABLES})
  if(KEY MATCHES "^TLS_")
    file(TO_NATIVE_PATH "${${KEY}}" VALUE)
    if(WIN32)
      string(REPLACE "\\" "\\\\" VALUE "${VALUE}")
    endif()
    file(APPEND ${PROJECT_BINARY_DIR}/CMake.h "#define ${KEY} \"${VALUE}\"\n")
    file(APPEND ${PROJECT_BINARY_DIR}/CMake.py "${KEY} = '${VALUE}'\n")
    file(APPEND ${PROJECT_BINARY_DIR}/CMake.m "${KEY} = '${VALUE}';\n")
  endif()
endforeach()
file(APPEND ${PROJECT_BINARY_DIR}/CMake.h "\n#endif  // __CMake_h\n")


# configure launcher
cma_configure_launcher("${PROJECT_SOURCE_DIR}/CMake/Launcher.cmake")

cma_launcher_target(Terminal
  NAME run_Terminal)
#cma_launcher_shortcut(Terminal NAME Terminal)


set(ITK_DIR ${PROJECT_BINARY_DIR}/ITK-build)
set(VTK_DIR ${PROJECT_BINARY_DIR}/VTK-build)
set(BOOST_ROOT ${PROJECT_BINARY_DIR}/Boost-build)
set(BOOST_LIBRARYDIR ${PROJECT_BINARY_DIR}/Boost-build)


if (TLS_SECOND_STEP)
ADD_SUBDIRECTORY(Segmentation_Modules/libs/)
ADD_SUBDIRECTORY(Segmentation_Modules/Lungs_Segmentation/)
ADD_SUBDIRECTORY(Segmentation_Modules/utilities)
endif()

if(USE_Qt)
  cma_launcher_target(designer
    NAME run_QtDesigner)
  cma_launcher_shortcut(designer
    NAME QtDesigner)
endif()

if(USE_Slicer)
  cma_launcher_target(Slicer
    NAME run_Slicer)
  cma_launcher_shortcut(Slicer
    NAME ${PROJECT_NAME}
    ICON "${PROJECT_SOURCE_DIR}/Resources/LogoIcon"
    COMMENT "Segmentation")
endif()


MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()
