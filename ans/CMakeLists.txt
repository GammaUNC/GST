include_directories("${GenTC_SOURCE_DIR}/ans")
include_directories("${GenTC_BINARY_DIR}/ans")

include_directories("${GenTC_SOURCE_DIR}/gpu")
INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

SET( HEADERS
  "ans.h"
  "ans_utils.h"
  "bits.h"
  "histogram.h"
)  

SET( SOURCES
  "decode.cpp"
  "ans_ocl_encode.cpp"
  "encode.cpp"
  "histogram.cpp"
)

IF( MSVC )
  ADD_DEFINITIONS(/D _SCL_SECURE_NO_WARNINGS)
ENDIF()

ADD_LIBRARY(ans ${HEADERS} ${SOURCES} ${KERNELS})

SET( BUILD_TABLE_KERNEL_PATH ${GenTC_SOURCE_DIR}/ans/build_table.cl )
SET( ANS_DECODE_KERNEL_PATH ${GenTC_SOURCE_DIR}/ans/ans_decode.cl )

CONFIGURE_FILE(
  "ans_config.h.in"
  "ans_config.h"
)

SET( KERNELS
  ${BUILD_TABLE_KERNEL_PATH}
  ${ANS_DECODE_KERNEL_PATH}
)

SET( HEADERS
  "ans_config.h"
  "ans_ocl.h"
)

SET( SOURCES
  "ans_ocl.cpp"
)

SOURCE_GROUP(OpenCL FILES ${KERNELS})
ADD_LIBRARY(ans_ocl ${HEADERS} ${SOURCES} ${KERNELS})
TARGET_LINK_LIBRARIES(ans_ocl ans)
TARGET_LINK_LIBRARIES(ans_ocl gentc_gpu)

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
    TARGET_LINK_LIBRARIES(${TEST}_test ans_ocl)
  ELSE()
    TARGET_LINK_LIBRARIES(${TEST}_test gtest_main)
  ENDIF()

  ADD_TEST(Test_${TEST} ${TEST}_test)
ENDFOREACH()