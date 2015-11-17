include_directories("${GenTC_SOURCE_DIR}/ans")
include_directories("${GenTC_BINARY_DIR}/ans")

SET( HEADERS
  "histogram.h"
)  

SET( SOURCES
  "histogram.cpp"
)

ADD_LIBRARY(ans ${HEADERS} ${SOURCES})

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

FOREACH(TEST "histogram" "bits" "ans")
  ADD_EXECUTABLE(${TEST}_test ${TEST}_test.cpp)
  TARGET_LINK_LIBRARIES(${TEST}_test ans)
  TARGET_LINK_LIBRARIES(${TEST}_test gtest)
  TARGET_LINK_LIBRARIES(${TEST}_test gtest_main)
  ADD_TEST(Test_${TEST} ${TEST}_test)
ENDFOREACH()
