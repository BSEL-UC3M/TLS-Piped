set(EP_OPTION_NAME "USE_${EP_NAME}")
set(EP_OPTION_DEFAULT OFF)

set(EP_REQUIRED_PROJECTS Git Mercurial Patch Qt Matlab zlib Boost Python Subversion VTK DCMTK ITK OpenIGTLink)

cma_end_definition()

ExternalProject_Add(${EP_NAME}
    DEPENDS ${EP_REQUIRED_PROJECTS}
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")



message("[External Testing] Running")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Git]    GIT_EXECUTABLE: ${GIT_EXECUTABLE}")
message(STATUS "[External Testing][Git]    ${PROJECT_NAME}_Git_EXECUTABLE: ${${PROJECT_NAME}_Git_EXECUTABLE}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Mercurial]    HG_EXECUTABLE: ${GIT_EXECUTABLE}")
message(STATUS "[External Testing][Mercurial]    ${PROJECT_NAME}_HG_EXECUTABLE: ${${PROJECT_NAME}_HG_EXECUTABLE}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Patch]    PATCH_EXECUTABLE: ${PATCH_EXECUTABLE}")
message(STATUS "[External Testing][Patch]    ${PROJECT_NAME}_PATCH_EXECUTABLE: ${${PROJECT_NAME}_PATCH_EXECUTABLE}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Qt]    ${PROJECT_NAME}_QT_QMAKE_EXECUTABLE: ${${PROJECT_NAME}_QT_QMAKE_EXECUTABLE}")
message(STATUS "[External Testing][Qt]    ${PROJECT_NAME}_QT_VERSION: ${${PROJECT_NAME}_QT_VERSION}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Matlab]    ${PROJECT_NAME}_Matlab_EXECUTABLE: ${${PROJECT_NAME}_Matlab_EXECUTABLE}")
message(STATUS "[External Testing][Matlab]    ${PROJECT_NAME}_Matlab_DIR: ${${PROJECT_NAME}_Matlab_DIR}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][zlib]    ${PROJECT_NAME}_zlib_ROOT: ${${PROJECT_NAME}_zlib_ROOT}")
message(STATUS "[External Testing][zlib]    ${PROJECT_NAME}_zlib_INCLUDE_DIR: ${${PROJECT_NAME}_zlib_INCLUDE_DIR}")
message(STATUS "[External Testing][zlib]    ${PROJECT_NAME}_zlib_LIBRARY: ${${PROJECT_NAME}_zlib_LIBRARY}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Boost]    ${PROJECT_NAME}_Boost_ROOT: ${${PROJECT_NAME}_Boost_ROOT}")
message(STATUS "[External Testing][Boost]    ${PROJECT_NAME}_Boost_LIB_DIR: ${${PROJECT_NAME}_Boost_LIB_DIR}")
message(STATUS "[External Testing][Boost]    ${PROJECT_NAME}_Boost_INCLUDE_DIR: ${${PROJECT_NAME}_Boost_INCLUDE_DIR}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Python]    Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "[External Testing][Python]    ${PROJECT_NAME}_Python_EXECUTABLE: ${${PROJECT_NAME}_Python_EXECUTABLE}")
message(STATUS "[External Testing][Python]    ${PROJECT_NAME}_Python_LIBRARY: ${${PROJECT_NAME}_Python_LIBRARY}")
message(STATUS "[External Testing][Python]    ${PROJECT_NAME}_Python_INCLUDE_DIR: ${${PROJECT_NAME}_Python_INCLUDE_DIR}")
message(STATUS "[External Testing][Python]    ${PROJECT_NAME}_Python_NUMPY_INCLUDE_DIR: ${${PROJECT_NAME}_Python_NUMPY_INCLUDE_DIR}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][Subversion]    Subversion_SVN_EXECUTABLE: ${Subversion_SVN_EXECUTABLE}")
message(STATUS "[External Testing][Subversion]    ${PROJECT_NAME}_SVN_EXECUTABLE: ${${PROJECT_NAME}_SVN_EXECUTABLE}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][VTK]    ${PROJECT_NAME}_VTK_DIR: ${${PROJECT_NAME}_VTK_DIR}")
message(STATUS "[External Testing][VTK]    ${PROJECT_NAME}_VTK_SRC_DIR: ${${PROJECT_NAME}_VTK_SRC_DIR}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][DCMTK]    ${PROJECT_NAME}_DCMTK_DIR: ${${PROJECT_NAME}_DCMTK_DIR}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][ITK]    ${PROJECT_NAME}_ITK_DIR: ${${PROJECT_NAME}_ITK_DIR}")
message(STATUS "[External Testing][ITK]    ${PROJECT_NAME}_ITK_SRC_DIR: ${${PROJECT_NAME}_ITK_SRC_DIR}")
message(STATUS "------------------------------------------------------------")
message(STATUS "[External Testing][OpenIGTLink] ${PROJECT_NAME}_OpenIGTLink_DIR: ${${PROJECT_NAME}_OpenIGTLink_DIR}")
message(STATUS "------------------------------------------------------------")

message("[External Testing] Finished!")

