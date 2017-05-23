include_directories("${GLFW_SOURCE_DIR}/include")
include_directories("${GenTC_SOURCE_DIR}/demo")
include_directories("${GenTC_BINARY_DIR}/demo")

include_directories("${GenTC_SOURCE_DIR}/codec")
include_directories("${GenTC_SOURCE_DIR}/gpu")

if ("${OPENGL_INCLUDE_DIR}")
  include_directories("${OPENGL_INCLUDE_DIR}")
endif()

INCLUDE_DIRECTORIES( ${GenTC_SOURCE_DIR}/lib/include )

FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

ADD_EXECUTABLE(photos "photos.cpp")
ADD_EXECUTABLE(photos_sf "photos_sf.cpp")
ADD_EXECUTABLE(demo "demo.cpp")
ADD_EXECUTABLE(viewer "viewer.cpp")
ADD_EXECUTABLE(dxt_tester "dxt_tester.cpp")

FOREACH(prg photos photos_sf demo viewer dxt_tester)
  TARGET_LINK_LIBRARIES( ${prg} glfw )
  TARGET_LINK_LIBRARIES( ${prg} ${GLFW_LIBRARIES} )
  TARGET_LINK_LIBRARIES( ${prg} gentc_gpu )
  TARGET_LINK_LIBRARIES( ${prg} gentc_encoder )
  TARGET_LINK_LIBRARIES( ${prg} gentc_decoder )
  TARGET_LINK_LIBRARIES( ${prg} ${OPENCL_LIBRARIES} )
  TARGET_LINK_LIBRARIES( ${prg} ${OPENGL_gl_LIBRARY} )

  IF ( MSVC )
    IF( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
      TARGET_LINK_LIBRARIES( ${prg} ${GenTC_SOURCE_DIR}/lib/x64/glew32.lib )
      add_custom_command(TARGET ${prg} POST_BUILD          # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different    # which executes "cmake - E copy_if_different..."
        "${PROJECT_SOURCE_DIR}/lib/bin/x64/glew32.dll"   # <--this is in-file
        $<TARGET_FILE_DIR:${prg}>)                       # <--this is out-file path
    ELSE (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
      TARGET_LINK_LIBRARIES( ${prg} ${GenTC_SOURCE_DIR}/lib/Win32/glew32.lib )
      add_custom_command(TARGET ${prg} POST_BUILD          # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different    # which executes "cmake - E copy_if_different..."
        "${PROJECT_SOURCE_DIR}/lib/bin/Win32/glew32.dll"   # <--this is in-file
        $<TARGET_FILE_DIR:${prg}>)                       # <--this is out-file path
    ENDIF( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
  ENDIF()

  INSTALL(TARGETS ${prg} EXPORT GenTCTargets RUNTIME DESTINATION bin COMPONENT bin)
ENDFOREACH()
