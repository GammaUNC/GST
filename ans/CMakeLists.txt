include_directories("${GenTC_SOURCE_DIR}/ans")
include_directories("${GenTC_BINARY_DIR}/ans")

include_directories("${GenTC_SOURCE_DIR}/gpu")
INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

SET( BUILD_TABLE_KERNEL_PATH ${GenTC_SOURCE_DIR}/ans/build_table.cl )
SET( ANS_DECODE_KERNEL_PATH ${GenTC_SOURCE_DIR}/ans/ans_decode.cl )

CONFIGURE_FILE(
  "ans_config.h.in"
  "ans_config.h"
)

SET( KERNELS
  "${GenTC_SOURCE_DIR}/ans/build_table.cl"
  "${GenTC_SOURCE_DIR}/ans/ans_decode.cl"
)

SET( HEADERS
  "ans_config.h"
  "ans_decode.h"
  "ans_encode.h"
  "bits.h"
  "histogram.h"
  "ans_ocl.h"
)  

SET( SOURCES
  "histogram.cpp"
  "ans_ocl.cpp"
)

SOURCE_GROUP(OpenCL FILES ${KERNELS})
ADD_LIBRARY(ans ${HEADERS} ${SOURCES} ${KERNELS})
TARGET_LINK_LIBRARIES(ans gentc_gpu)

# Add flag for link time code generation. This was used to build the libpng
# libraries, so we should probably also include it for this project as well...
IF( MSVC )
  SET_TARGET_PROPERTIES(ans PROPERTIES LINK_FLAGS "/LTCG")
ENDIF()

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

FOREACH(TEST "histogram" "bits" "ans" "ans_ocl")
  ADD_EXECUTABLE(${TEST}_test ${TEST}_test.cpp)
  TARGET_LINK_LIBRARIES(${TEST}_test ans)
  TARGET_LINK_LIBRARIES(${TEST}_test gtest)

  IF ("${TEST}" STREQUAL "ans_ocl")
    TARGET_LINK_LIBRARIES(${TEST}_test gentc_gpu)
  ELSE()
    TARGET_LINK_LIBRARIES(${TEST}_test gtest_main)
  ENDIF()

  ADD_TEST(Test_${TEST} ${TEST}_test)
ENDFOREACH()