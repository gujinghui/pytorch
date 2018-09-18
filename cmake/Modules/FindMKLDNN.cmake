# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#  USE_IDEEP
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found
#  MKLDNN_INCLUDE_DIR
#  MKLDNN_LIBRARIES
#  CAFFE2_USE_IDEEP

IF (NOT MKLDNN_FOUND)

SET(MKLDNN_LIBRARIES)
SET(MKLDNN_INCLUDE_DIR)
SET(CAFFE2_USE_IDEEP)

IF (NOT USE_IDEEP)
  RETURN()
ENDIF(NOT USE_IDEEP)

SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
SET(MKLDNN_ROOT "${IDEEP_ROOT}/mkl-dnn")

FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
FIND_PATH(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
IF (NOT MKLDNN_INCLUDE_DIR)
  EXECUTE_PROCESS(COMMAND git submodule update --init mkl-dnn WORKING_DIRECTORY ${IDEEP_ROOT})
  FIND_PATH(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
ENDIF(NOT MKLDNN_INCLUDE_DIR)
IF (NOT IDEEP_INCLUDE_DIR OR NOT MKLDNN_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Can not find MKL-DNN files")
  RETURN()
ENDIF(NOT IDEEP_INCLUDE_DIR OR NOT MKLDNN_INCLUDE_DIR)
LIST(APPEND MKLDNN_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})

IF(NOT MKL_FOUND)
  FIND_PACKAGE(MKL)
ENDIF(NOT MKL_FOUND)

IF(MKL_FOUND)
  LIST(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
  LIST(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})

ELSE(MKL_FOUND)
  # If we cannot find MKL, we will use the Intel MKL Small library
  # comes with ${MKLDNN_ROOT}/external
  IF(NOT IS_DIRECTORY ${MKLDNN_ROOT}/external)
    IF(UNIX)
      EXECUTE_PROCESS(COMMAND "${MKLDNN_ROOT}/scripts/prepare_mkl.sh" RESULT_VARIABLE __result)
    ELSE(UNIX)
      EXECUTE_PROCESS(COMMAND "${MKLDNN_ROOT}/scripts/prepare_mkl.bat" RESULT_VARIABLE __result)
    ENDIF(UNIX)
  ENDIF(NOT IS_DIRECTORY ${MKLDNN_ROOT}/external)

  FILE(GLOB_RECURSE MKLML_INNER_INCLUDE_DIR ${MKLDNN_ROOT}/external/*/mkl.h)
  IF(MKLML_INNER_INCLUDE_DIR)
    # if user has multiple version under external/ then guess last
    # one alphabetically is "latest" and warn
    LIST(LENGTH MKLML_INNER_INCLUDE_DIR MKLINCLEN)
    IF(MKLINCLEN GREATER 1)
      LIST(SORT MKLML_INNER_INCLUDE_DIR)
      LIST(REVERSE MKLML_INNER_INCLUDE_DIR)
      LIST(GET MKLML_INNER_INCLUDE_DIR 0 MKLINCLST)
      SET(MKLML_INNER_INCLUDE_DIR "${MKLINCLST}")
    ENDIF()
    GET_FILENAME_COMPONENT(MKLML_INNER_INCLUDE_DIR ${MKLML_INNER_INCLUDE_DIR} DIRECTORY)
    LIST(APPEND MKLDNN_INCLUDE_DIR ${MKLML_INNER_INCLUDE_DIR})

    IF(APPLE)
      SET(__mklml_inner_libs mklml iomp5)
    ELSE(APPLE)
      SET(__mklml_inner_libs mklml_intel iomp5)
    ENDIF(APPLE)

    FOREACH(__mklml_inner_lib ${__mklml_inner_libs})
      STRING(TOUPPER ${__mklml_inner_lib} __mklml_inner_lib_upper)
      FIND_LIBRARY(${__mklml_inner_lib_upper}_LIBRARY
            NAMES ${__mklml_inner_lib}
            PATHS  "${MKLML_INNER_INCLUDE_DIR}/../lib"
            DOC "The path to Intel(R) MKLML ${__mklml_inner_lib} library")
      MARK_AS_ADVANCED(${__mklml_inner_lib_upper}_LIBRARY)
      LIST(APPEND MKLDNN_LIBRARIES ${${__mklml_inner_lib_upper}_LIBRARY})
    ENDFOREACH()
  ENDIF(MKLML_INNER_INCLUDE_DIR)
ENDIF(MKL_FOUND)

LIST(APPEND __mkldnn_looked_for MKLDNN_LIBRARIES)
LIST(APPEND __mkldnn_looked_for MKLDNN_INCLUDE_DIR)
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLDNN DEFAULT_MSG ${__mkldnn_looked_for})

if(MKLDNN_FOUND)
  SET(CAFFE2_USE_IDEEP 1)
  SET(MKLDNN_LIBRARY_TYPE STATIC CACHE STRING "Build mkl-dnn as static lib" FORCE)
  ADD_SUBDIRECTORY(${MKLDNN_ROOT})
  SET_PROPERTY(TARGET mkldnn PROPERTY POSITION_INDEPENDENT_CODE ON)
  LIST(APPEND MKLDNN_LIBRARIES mkldnn)
ELSE(MKLDNN_FOUND)
  MESSAGE(FATAL_ERROR "Did not find MKLDNN files!")
ENDIF(MKLDNN_FOUND)

caffe_clear_vars(__mkldnn_looked_for __mklml_inner_libs)

ENDIF(NOT MKLDNN_FOUND)
