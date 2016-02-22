#ifndef __VPTREE_STRUCT_H__
#define __VPTREE_STRUCT_H__

#include <stdint.h>
#include <stdbool.h>

typedef struct node node;

struct vptree
{
  vptree_options opts;

  node *root;

  /**
   * The number of points currently in the vp-tree
   */
  int n;
};

struct node
{
  vptree *vp;

  /**
   * Vantage point
   */
  const void *p;

  /**
   * Split distance.
   */
  double mu;

  /**
   * Parent node
   */
  node *parent;

  /**
   * Subnode for points at distance < mu
   */
  node *lt;

  /**
   * Subnode for points at distance >= mu
   */
  node *ge;
};

typedef struct incnode incnode;
struct incnode
{
  node *n;
  double d;
  bool exclude, exclude_tree;

  incnode *parent, *lt, *ge;
};

struct vptree_incnn
{
  /**
   * The vp-tree
   */
  const vptree *vp;

  /**
   * Query point for the k-nn search
   */
  const void *q;

  /**
   * Tree for marking nodes and caching distances between runs
   */
  incnode *marks;

  /**
   * Previous best point
   */
  incnode *prev;
};

/////////////////////////////// Utility Functions /////////////////////////

static void *allocate(const vptree *vp, size_t s)
{
  return vp->opts.allocate(vp->opts.user_data, s);
}

static void deallocate(const vptree *vp, void *data)
{
  vp->opts.deallocate(vp->opts.user_data, data);
}

static double distance(const vptree *vp, const void *p1, const void *p2)
{
  return vp->opts.distance(vp->opts.user_data, p1, p2);
}

#endif // #ifndef __VPTREE_STRUCT_H__
