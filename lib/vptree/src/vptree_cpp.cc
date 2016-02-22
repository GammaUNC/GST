#include <cmath>

#include "vptree.hh"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

using namespace std;

VPTreeBase::VPTreeBase()
{
  vptree_options opts;

  opts = vptree_default_options;

  opts.user_data = this;
  opts.distance = &vptree_cpp_distance;

  vp = vptree_create(sizeof(opts), &opts);
}

VPTreeBase::~VPTreeBase()
{
  vptree_destroy(vp);
}

double vptree_cpp_distance(void *user_data, const void *p1, const void *p2)
{
  VPTreeBase *vptree = reinterpret_cast<VPTreeBase *>(user_data);
  return vptree->distance(p1, p2);
}

double EuclideanVPTree::distance(const std::vector<double> &p1, const std::vector<double> &p2)
{
  size_t n1 = p1.size(), n2 = p2.size();
  size_t n = MIN(n1, n2);

  size_t i = 0;
  double dist = 0;
  for(; i < n; ++i) {
    double diff = p1[i] - p2[i];
    dist += diff * diff;
  }

  // Implicitly zero-pad the shorter vector
  for(; i < n1; ++i) {
    dist += p1[i] * p1[i];
  }
  for(; i < n2; ++i) {
    dist += p2[i] * p2[i];
  }

  return sqrt(dist);
}
