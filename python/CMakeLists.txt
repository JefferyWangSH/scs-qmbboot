# Python package scs_qmbboot, built by scikit-build-core (pyproject.toml).
# The package loads the solver libraries with ctypes: one library per linear
# system backend, installed into the package directory, plus abi.c, which
# reports the struct sizes the ctypes mirror of scs.h is checked against.

if(DLONG OR SFLOAT)
  message(FATAL_ERROR "scs_qmbboot expects int indices and double floats: "
                      "build with DLONG=OFF and SFLOAT=OFF")
endif()
if(USE_SPECTRAL_CONES OR USE_SPECTRAL_TIMING_FLAG)
  message(FATAL_ERROR "scs_qmbboot does not support spectral cones: build "
                      "with USE_SPECTRAL_CONES=OFF and USE_SPECTRAL_TIMING_FLAG=OFF")
endif()

if(APPLE)
  set(SCS_PYTHON_TARGETS scsdir scsaccel)
  set(SCS_PYTHON_RPATH "@loader_path/../../..")
else()
  set(SCS_PYTHON_TARGETS scsdir scsmklgram)
  set(SCS_PYTHON_RPATH "$ORIGIN/../../..")
endif()

# -march=x86-64-v4 (AVX-512) when the build machine has it and CMAKE_C_FLAGS
# sets no -march of its own
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"
   AND NOT CMAKE_C_FLAGS MATCHES "-march=")
  file(READ /proc/cpuinfo _cpuinfo)
  set(SCS_PYTHON_MARCH -march=x86-64-v4)
  foreach(_flag avx512f avx512bw avx512cd avx512dq avx512vl)
    if(NOT _cpuinfo MATCHES " ${_flag}( |\n)")
      unset(SCS_PYTHON_MARCH)
    endif()
  endforeach()
endif()
message(STATUS "scs_qmbboot: extra C flags '${SCS_PYTHON_MARCH}'")

foreach(_target ${SCS_PYTHON_TARGETS})
  if(NOT TARGET ${_target})
    if(_target STREQUAL "scsmklgram")
      message(FATAL_ERROR "The MKL Gram backend needs MKL: set MKLROOT, in a "
                          "conda environment export MKLROOT=$CONDA_PREFIX")
    else()
      message(FATAL_ERROR "Target ${_target} is not configured: "
                          "build with USE_APPLE_ACCELERATE=ON")
    endif()
  endif()
  target_sources(${_target} PRIVATE abi.c)
  target_compile_options(${_target} PRIVATE ${SCS_PYTHON_MARCH})
  # Loaded by path, never linked against: no versioned file names, no
  # headers. MKL and libomp come from the environment the package is
  # installed in (site-packages/scs_qmbboot -> lib).
  set_property(TARGET ${_target} PROPERTY VERSION)
  set_property(TARGET ${_target} PROPERTY SOVERSION)
  set_property(TARGET ${_target} PROPERTY PUBLIC_HEADER)
  set_target_properties(${_target} PROPERTIES
    INSTALL_RPATH "${SCS_PYTHON_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  install(TARGETS ${_target}
          LIBRARY DESTINATION scs_qmbboot COMPONENT python
          RUNTIME DESTINATION scs_qmbboot COMPONENT python)
endforeach()
