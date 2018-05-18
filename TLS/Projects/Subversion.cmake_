# Author: Ali Uneri
# Date: 2013-05-02

set(EP_OPTION_NAME "USE_${EP_NAME}")

cma_end_definition()
# -----------------------------------------------------------------------------

find_package(Subversion REQUIRED)

ExternalProject_Add(${EP_NAME}
  DOWNLOAD_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND "")

set(Subversion_SVN_EXECUTABLE "${Subversion_SVN_EXECUTABLE}" CACHE INTERNAL "")
set(${PROJECT_NAME}_SVN_EXECUTABLE "${Subversion_SVN_EXECUTABLE}" CACHE INTERNAL "")
