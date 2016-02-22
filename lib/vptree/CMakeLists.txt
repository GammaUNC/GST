include_directories("${GenTC_SOURCE_DIR}/lib/vptree/src")
include_directories("${GenTC_SOURCE_DIR}/lib/vptree/include")

SET( HEADERS
  "include/vptree/vptree.hh"
)

SET( SOURCES
  "src/pqueue.c"
  "src/vptree.c"
  "src/geom.c"
  "src/vptree_cpp.cc"
)

ADD_LIBRARY(vptree ${HEADERS} ${SOURCES})
