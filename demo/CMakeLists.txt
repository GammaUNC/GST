include_directories("${GLFW_SOURCE_DIR}/include")
include_directories("${GenTC_SOURCE_DIR}/demo")
include_directories("${GenTC_BINARY_DIR}/demo")

if ("${OPENGL_INCLUDE_DIR}")
  include_directories("${OPENGL_INCLUDE_DIR}")
endif()

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )
SET( OPENCL_KERNEL_PATH ${GenTC_SOURCE_DIR}/demo/kernel.cl )

IF ( MSVC )
  INCLUDE_DIRECTORIES( ${GenTC_SOURCE_DIR}/lib/include )
ENDIF()

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
IF ( MSVC )
  IF( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
    TARGET_LINK_LIBRARIES( demo ${GenTC_SOURCE_DIR}/lib/x64/glew32.lib )
    add_custom_command(TARGET demo POST_BUILD          # Adds a post-build event to MyTest
      COMMAND ${CMAKE_COMMAND} -E copy_if_different    # which executes "cmake - E copy_if_different..."
      "${PROJECT_SOURCE_DIR}/lib/bin/x64/glew32.dll"   # <--this is in-file
      $<TARGET_FILE_DIR:demo>)                       # <--this is out-file path
  ELSE (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    TARGET_LINK_LIBRARIES( demo ${GenTC_SOURCE_DIR}/lib/Win32/glew32.lib )
    add_custom_command(TARGET demo POST_BUILD          # Adds a post-build event to MyTest
      COMMAND ${CMAKE_COMMAND} -E copy_if_different    # which executes "cmake - E copy_if_different..."
      "${PROJECT_SOURCE_DIR}/lib/bin/Win32/glew32.dll"   # <--this is in-file
      $<TARGET_FILE_DIR:demo>)                       # <--this is out-file path
  ENDIF( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
ENDIF()

INSTALL(TARGETS demo EXPORT GenTCTargets RUNTIME DESTINATION bin COMPONENT bin)
