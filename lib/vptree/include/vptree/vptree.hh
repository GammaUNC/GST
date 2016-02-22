#ifndef __VPTREE_HH__
#define __VPTREE_HH__

#include <vector>

#include "vptree.h"

extern "C" {
  double vptree_cpp_distance(void *user_data, const void *p1, const void *p2);
}

class VPTreeBase
{
public:
  VPTreeBase();
  virtual ~VPTreeBase();

protected:
  vptree *vp;

  friend double vptree_cpp_distance(void *user_data, const void *p1, const void *p2);
  virtual double distance(const void *p1, const void *p2) = 0;

private:
  VPTreeBase(const VPTreeBase &copy);
  const VPTreeBase &operator = (const VPTreeBase &assign);

};


template<class Point>
class VPTree;

/**
 * @todo bad craziness will happen if the vptree is destroyed while
 *       a IncrementalKNN still refers to it.  Does this need to be fixed?
 */
template<class Point>
class IncrementalKNN
{
  friend class VPTree<Point>;

protected:
  IncrementalKNN(const vptree *vp, const Point &query)
  {
    inc_ = vptree_incnn_begin(vp, &query);
    next();
  }

public:
  ~IncrementalKNN()
  {
    vptree_incnn_end(inc_);
  }

  void next()
  {
    p = reinterpret_cast<Point *>(vptree_incnn_next(inc_));
  }

  const Point *get()
  {
    return p;
  }

  const IncrementalKNN *operator ++ ()
  {
    next();
    return *this;
  };
    
private:
  vptree_incnn *inc_;
  const Point *p;
};

template<class Point>
class VPTree: public VPTreeBase
{
public:
  VPTree()
  {
    last_added = 0;
  }

  virtual ~VPTree()
  {
  }

  int size()
  {
    return static_cast<int>(points.size());
  }
  
  void add(const Point &p)
  {
    points.push_back(p);
  }

  template<class InputIterator>
  void addMany(InputIterator start, InputIterator end)
  {
    points.insert(points.end(), start, end);
  }

  std::vector<const Point *> nearestNeighbors(const Point &query, int k = 1)
  {
    update();
    if(k > vptree_npoints(vp)) {
      k = vptree_npoints(vp);
    }

    const void **nn_ptrs = new const void *[k];
    vptree_nearest_neighbor(vp, &query, k, nn_ptrs);

    std::vector<const Point *> return_nns = castPointers(k, nn_ptrs);

    delete [] nn_ptrs;
    return return_nns;
  }

  std::vector<const Point *> approxNearestNeighbors(const Point &query, int k = 1, int max_nodes = 1024)
  {
    update();
    if(k > vptree_npoints(vp)) {
      k = vptree_npoints(vp);
    }

    const void **nn_ptrs = new const void *[k];
    vptree_nearest_neighbor_approx(vp, &query, k, nn_ptrs, max_nodes);

    std::vector<const Point *> return_nns = castPointers(k, nn_ptrs);

    delete [] nn_ptrs;
    return return_nns;
  }

  std::vector<const Point *> neighborhood(const Point &query, double distance)
  {
    update();

    int k = 0;
    const void **ptrs = vptree_neighborhood(vp, &query, distance, &k);
    std::vector<const Point *> return_nns = castPointers(k, ptrs);
    free(ptrs);
    return return_nns;
  }

  IncrementalKNN<Point> incrementalNearestNeighbor(const Point &query)
  {
    update();

    return IncrementalKNN<Point>(vp, query);
  }

protected:
  virtual double distance(const Point &p1, const Point &p2) = 0;

  virtual double distance(const void *p1, const void *p2)
  {
    return distance(*reinterpret_cast<const Point *>(p1),
		    *reinterpret_cast<const Point *>(p2));
  }

private:
  std::vector<Point> points;
  size_t last_added;

  void update()
  {
    size_t num_to_add = points.size() - last_added;
    if(num_to_add == 0) {
      return;
    }

    const void **add_ptrs = new const void *[num_to_add];
    for(size_t i = 0; i < num_to_add; ++i) {
      add_ptrs[i] = reinterpret_cast<const void *>(&points[last_added + i]);
    }

    vptree_add_many(vp, (int)num_to_add, add_ptrs);

    delete [] add_ptrs;
    last_added = points.size();
  }

  std::vector<const Point *> castPointers(int k, const void * const *ptrs)
  {
    std::vector<const Point *> nns_vector;
    nns_vector.reserve(k);
    for(int i = 0; i < k; ++i) {
      nns_vector.push_back(reinterpret_cast<const Point *>(ptrs[i]));
    }
    return nns_vector;
  }  
};


class EuclideanVPTree: public VPTree<std::vector<double> >
{
protected:
  virtual double distance(const std::vector<double> &p1, const std::vector<double> &p2);
};

#endif // #ifndef __VPTREE_HH__
