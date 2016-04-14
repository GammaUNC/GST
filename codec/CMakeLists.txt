include_directories("${GenTC_SOURCE_DIR}/codec")
include_directories("${GenTC_BINARY_DIR}/codec")

include_directories("${GLFW_SOURCE_DIR}/include")
include_directories("${GenTC_SOURCE_DIR}/gpu")
include_directories("${GenTC_SOURCE_DIR}/ans")
include_directories("${GenTC_BINARY_DIR}/ans")
include_directories("${GenTC_SOURCE_DIR}/lib/include")
include_directories("${GenTC_SOURCE_DIR}/lib/vptree/include")
include_directories("${GenTC_SOURCE_DIR}/lib")
INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

if ("${OPENGL_INCLUDE_DIR}")
  include_directories("${OPENGL_INCLUDE_DIR}")
endif()

SET( INVERSE_WAVELET_KERNEL_PATH ${GenTC_SOURCE_DIR}/codec/inverse_wavelet.cl )
SET( ENDPOINT_KERNEL_PATH ${GenTC_SOURCE_DIR}/codec/collect_endpoints.cl )
SET( DECODE_INDICES_KERNEL_PATH ${GenTC_SOURCE_DIR}/codec/decode_indices.cl )

CONFIGURE_FILE(
  "codec_config.h.in"
  "codec_config.h"
)

SET( KERNELS
  ${INVERSE_WAVELET_KERNEL_PATH}
  ${ENDPOINT_KERNEL_PATH}
  ${DECODE_INDICES_KERNEL_PATH}
)

SET( HEADERS
  "codec.h"
  "codec_config.h.in"
  "data_stream.h"
  "dxt_image.h"
  "entropy.h"
  "fast_dct.h"
  "image.h"
  "image_utils.h"
  "image_processing.h"
  "pipeline.h"
  "pixel_traits.h"
  "wavelet.h"
)  

SET( SOURCES
  "codec.cpp"
  "data_stream.cpp"
  "dxt_image.cpp"
  "entropy.cpp"
  "image.cpp"
  "image_processing.cpp"
  "image_utils.cpp"
  "wavelet.cpp"
)

SOURCE_GROUP(OpenCL FILES ${KERNELS})
ADD_LIBRARY(gentc_codec ${HEADERS} ${SOURCES} ${KERNELS})
TARGET_LINK_LIBRARIES( gentc_codec ans)
TARGET_LINK_LIBRARIES( gentc_codec ${GLFW_LIBRARIES} )
TARGET_LINK_LIBRARIES( gentc_codec gentc_gpu )
TARGET_LINK_LIBRARIES( gentc_codec ${OPENCL_LIBRARIES} )
TARGET_LINK_LIBRARIES( gentc_codec ${OPENGL_gl_LIBRARY} )
TARGET_LINK_LIBRARIES( gentc_codec ${OPENGL_glu_LIBRARY} )
TARGET_LINK_LIBRARIES( gentc_codec vptree)
TARGET_LINK_LIBRARIES( gentc_codec squish)

IF ( MSVC )
  IF( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
    TARGET_LINK_LIBRARIES( gentc_codec ${GenTC_SOURCE_DIR}/lib/x64/glew32.lib )
    add_custom_command(TARGET gentc_codec POST_BUILD      # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different         # which executes "cmake - E copy_if_different..."
    "${PROJECT_SOURCE_DIR}/lib/bin/x64/glew32.dll"        # <--this is in-file
    $<TARGET_FILE_DIR:gentc_codec>)                       # <--this is out-file path
  ELSE (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    TARGET_LINK_LIBRARIES( gentc_codec ${GenTC_SOURCE_DIR}/lib/Win32/glew32.lib )
    add_custom_command(TARGET gentc_codec POST_BUILD      # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different         # which executes "cmake - E copy_if_different..."
    "${PROJECT_SOURCE_DIR}/lib/bin/Win32/glew32.dll"      # <--this is in-file
    $<TARGET_FILE_DIR:gentc_codec>)                       # <--this is out-file path
  ENDIF( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
ENDIF()

ADD_EXECUTABLE(gentenc encoder.cpp)
TARGET_LINK_LIBRARIES( gentenc gentc_codec )

INCLUDE_DIRECTORIES(${GenTC_SOURCE_DIR}/googletest/include)

# Summary of tuple support for Microsoft Visual Studio:
# Compiler    version(MS)  version(cmake)  Support
# ----------  -----------  --------------  -----------------------------
# <= VS 2010  <= 10        <= 1600         Use Google Tests's own tuple.
# VS 2012     11           1700            std::tr1::tuple + _VARIADIC_MAX=10
# VS 2013     12           1800            std::tr1::tuple
if (MSVC AND MSVC_VERSION EQUAL 1700)
  add_definitions(/D _VARIADIC_MAX=10)
endif()

if (MSVC AND MSVC_VERSION EQUAL 1900)
  add_definitions(/D _SCL_SECURE_NO_WARNINGS)
endif()

FOREACH(TEST image wavelet)
  ADD_EXECUTABLE(${TEST}_test ${TEST}_test.cpp)
  TARGET_LINK_LIBRARIES(${TEST}_test gentc_codec)

  IF ("${TEST}" STREQUAL "codec_ocl")
    TARGET_LINK_LIBRARIES(${TEST}_test gentc_gpu)
  ENDIF()
  TARGET_LINK_LIBRARIES(${TEST}_test gtest_main)

  ADD_TEST(Test_${TEST} ${TEST}_test)
ENDFOREACH()