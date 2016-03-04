include_directories("${GenTC_SOURCE_DIR}/codec")
include_directories("${GenTC_BINARY_DIR}/codec")

include_directories("${GenTC_SOURCE_DIR}/gpu")
include_directories("${GenTC_SOURCE_DIR}/ans")
include_directories("${GenTC_SOURCE_DIR}/lib/include")
include_directories("${GenTC_SOURCE_DIR}/lib/vptree/include")
include_directories("${GenTC_SOURCE_DIR}/lib")
INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

SET( HEADERS
  "codec.h"
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

ADD_LIBRARY(gentc_codec ${HEADERS} ${SOURCES})
TARGET_LINK_LIBRARIES(gentc_codec ans)
TARGET_LINK_LIBRARIES(gentc_codec gentc_gpu)
TARGET_LINK_LIBRARIES(gentc_codec vptree)
TARGET_LINK_LIBRARIES(gentc_codec squish)

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

FOREACH(TEST image wavelet)
  ADD_EXECUTABLE(${TEST}_test ${TEST}_test.cpp)
  TARGET_LINK_LIBRARIES(${TEST}_test gentc_codec)

  IF ("${TEST}" STREQUAL "codec_ocl")
    TARGET_LINK_LIBRARIES(${TEST}_test gentc_gpu)
  ENDIF()
  TARGET_LINK_LIBRARIES(${TEST}_test gtest_main)

  ADD_TEST(Test_${TEST} ${TEST}_test)
ENDFOREACH()