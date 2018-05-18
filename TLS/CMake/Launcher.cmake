# command alias: Terminal
list(APPEND HELP "   Terminal - Open a terminal.\n")
if(ARGV MATCHES "^Terminal$")
  unset(ENV{PYTHONHOME})
  if(WIN32)
    set(ARGV ${CMD} start cmd.exe)
  elseif(APPLE)
    execute_process(COMMAND bash -c "env | pbcopy")
    set(ARGV osascript
      -e "tell application \"Terminal\" to activate"
      -e "tell application \"System Events\" to tell process \"Terminal\" to keystroke \"t\" using command down"
      -e "tell application \"Terminal\" to do script \"bash\" in selected tab of the front window"
      -e "tell application \"Terminal\" to do script \"for env in `pbpaste`\; do export $env\; done\; clear\" in selected tab of the front window"
      -e "tell application \"Terminal\" to do script \"clear\; $SHELL\" in selected tab of the front window"
    )
  elseif(UNIX)
    set(ARGV bash)
  else()
    message(FATAL_ERROR "Platform is not supported.")
  endif()
endif()

# command alias: Slicer
list(APPEND HELP "   Slicer - 3D Slicer.\n")
if(ARGV MATCHES "^Slicer$")
  string(REPLACE "${PATHSEP}" ";" SLICERPATH "$ENV{SLICERPATH}")
  set(ARGV ${CMD} $ENV{SLICER_EXECUTABLE} --additional-module-paths ${SLICERPATH})
endif()


# command alias: Slicelet
list(APPEND HELP "   Slicelet <path> - Slicer scripted module as a standalone application.\n")
if(ARGV MATCHES "^Slicelet")
  list(GET ARGV 1 MODULE)
  set(ARGV ${CMD} $ENV{SLICER_EXECUTABLE} --no-main-window --disable-cli-modules --disable-loadable-modules --disable-scripted-loadable-modules --python-script ${MODULE})
endif()


# command alias: MITK
list(APPEND HELP "   MITK - MITK workbench.\n")
if(ARGV MATCHES "^MITK$")
  set(ARGV ${CMD} $ENV{MITK_EXECUTABLE})
endif()


# command alias: BiiGTK
list(APPEND HELP "   BiiGTK - Experimental BiiGTK interface.\n")
if(ARGV MATCHES "^BiiGTK$")
  string(REPLACE "${PATHSEP}" ";" SLICERPATH "$ENV{SLICERPATH}")
  get_filename_component(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)
  set(ARGV ${CMD} $ENV{SLICER_EXECUTABLE} --no-main-window --python-script ${PROJECT_SOURCE_DIR}/SlicerBiiGTK/Main/Main.py --additional-module-paths ${SLICERPATH})
endif()