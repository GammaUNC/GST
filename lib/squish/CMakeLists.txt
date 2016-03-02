include_directories("${GenTC_SOURCE_DIR}/lib/squish")

SET( HEADERS
  "alpha.h"
  "colourfit.h"
  "maths.h"
  "simd_float.h"
  "singlecolourfit.h"
  "clusterfit.h"
  "colourset.h"
  "rangefit.h"
  "simd_sse.h"
  "singlecolourlookup.inl"
  "colourblock.h"
  "config.h"
  "simd.h"
  "simd_ve.h"
  "squish.h"
)

SET( SOURCES
  "alpha.cpp"
  "clusterfit.cpp"
  "colourblock.cpp"
  "colourfit.cpp"
  "colourset.cpp"
  "maths.cpp"
  "rangefit.cpp"
  "singlecolourfit.cpp"
  "squish.cpp"
)

ADD_LIBRARY(squish ${HEADERS} ${SOURCES})
