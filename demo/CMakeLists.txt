include_directories("${GLFW_SOURCE_DIR}/include" "${GLFW_SOURCE_DIR}/deps")
include_directories("${GenTC_SOURCE_DIR}/demo")

if ("${OPENGL_INCLUDE_DIR}")
  include_directories("${OPENGL_INCLUDE_DIR}")
endif()

FIND_PACKAGE(OpenGL REQUIRED)

ADD_EXECUTABLE(
  demo
  "demo.cpp"
)

# Add flag for link time code generation. This was used to build the libpng
# libraries, so we should probably also include it for this project as well...
IF( MSVC )
  SET_TARGET_PROPERTIES(demo PROPERTIES LINK_FLAGS "/LTCG")
ENDIF()

TARGET_LINK_LIBRARIES( demo ${GLFW_LIBRARIES} )
TARGET_LINK_LIBRARIES( demo glfw )
TARGET_LINK_LIBRARIES( demo ${OPENGL_gl_LIBRARY} )

INSTALL(TARGETS demo EXPORT GenTCTargets RUNTIME DESTINATION bin COMPONENT bin)
