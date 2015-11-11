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

FOREACH(TEST "histogram")
  ADD_EXECUTABLE(${TEST}_test ${TEST}_test.cpp)
  TARGET_LINK_LIBRARIES(${TEST}_test ans)
  TARGET_LINK_LIBRARIES(${TEST}_test gtest)
  TARGET_LINK_LIBRARIES(${TEST}_test gtest_main)
  ADD_TEST(Test_${TEST} ${TEST}_test)
ENDFOREACH()
