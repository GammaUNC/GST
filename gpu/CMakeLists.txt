include_directories("${GLFW_SOURCE_DIR}/include")
include_directories("${GenTC_SOURCE_DIR}/gpu")

if ("${OPENGL_INCLUDE_DIR}")
  include_directories("${OPENGL_INCLUDE_DIR}")
endif()

FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

IF ( MSVC )
  INCLUDE_DIRECTORIES( ${GenTC_SOURCE_DIR}/lib/include )
ENDIF()

SET( HEADERS
  "gpu.h"
  "kernel_cache.h"
)  

SET( SOURCES
  "gpu.cpp"
  "kernel_cache.cpp"
)

ADD_LIBRARY(gentc_gpu ${HEADERS} ${SOURCES})

TARGET_LINK_LIBRARIES( gentc_gpu ${OPENCL_LIBRARIES} )
TARGET_LINK_LIBRARIES( gentc_gpu ${GLFW_LIBRARIES} )
