include_directories("${GLFW_SOURCE_DIR}/include")
include_directories("${GenTC_SOURCE_DIR}/demo")
include_directories("${GenTC_BINARY_DIR}/demo")

if ("${OPENGL_INCLUDE_DIR}")
  include_directories("${OPENGL_INCLUDE_DIR}")
endif()

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )
SET( OPENCL_KERNEL_PATH ${GenTC_SOURCE_DIR}/demo/kernel.cl )

CONFIGURE_FILE(
  "config.h.in"
  "config.h"
)

SET( HEADERS
  "gpu.h"
)  

SET( SOURCES
  "demo.cpp"
  "gpu.cpp"
)

ADD_EXECUTABLE(demo ${HEADERS} ${SOURCES})

# Add flag for link time code generation. This was used to build the libpng
# libraries, so we should probably also include it for this project as well...
IF( MSVC )
  SET_TARGET_PROPERTIES(demo PROPERTIES LINK_FLAGS "/LTCG")
ENDIF()

TARGET_LINK_LIBRARIES( demo glfw )
TARGET_LINK_LIBRARIES( demo ${GLFW_LIBRARIES} )
TARGET_LINK_LIBRARIES( demo ${OPENCL_LIBRARIES} )
TARGET_LINK_LIBRARIES( demo ${OPENGL_gl_LIBRARY} )
TARGET_LINK_LIBRARIES( demo ${OPENGL_glu_LIBRARY} )

INSTALL(TARGETS demo EXPORT GenTCTargets RUNTIME DESTINATION bin COMPONENT bin)
