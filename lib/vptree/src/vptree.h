#ifndef __VPTREE_H__
#define __VPTREE_H__

#ifdef __cplusplus

#include <cstdlib>

extern "C" {

#else

#include <stdlib.h>

#endif

typedef struct vptree vptree;

typedef struct {
  void *user_data;

  /* Distance closure */
  double (*distance)(void *user_data, const void *p1, const void *p2);

  /* Memory management closures */

  /**
   * Delete user data
   */
  void *(*allocate)(void *user_data, size_t s);
  void (*deallocate)(void *user_data, void *data);
  
} vptree_options;

extern const vptree_options vptree_default_options;

/**
 * Create a new vp-tree.
 */
vptree *vptree_create(size_t opts_size, const vptree_options *opts);

/**
 * Create a copy of a vp-tree.
 *
 * References to user data will be shallow-copied.
 *
 * @note The copy must independently be freed with vptree_destroy
 */
vptree *vptree_clone(const vptree *vp);

/**
 * Destroy vp-tree
 */
void vptree_destroy(vptree *vp);

/**
 * Get the options set for an existing vp-tree.
 */
const vptree_options *vptree_get_options(const vptree *vp);

/**
 * Get the number of points currently in the vp-tree.
 */
int vptree_npoints(const vptree *vp);

/**
 * Adds p at a leaf node in the vp-tree.
 *
 * @note Pointer @c p must be valid for the lifetime of the vp-tree
 * @returns 0 on success, nonzero on failure
 */
int vptree_add(vptree *vp, const void *p);

/**
 * Add multiple entries to a vp-tree simultaneously.
 *
 * Optimally splits at each level using all available data.
 *
 * @note Pointers @c p[0] to @c p[n-1] must be valid for the lifetime of the vp-tree
 * @returns 0 on success, nonzero on failure
 */
int vptree_add_many(vptree *vp, int n, const void * const *p);

/**
 * Add multiple entries to a vp-tree simultaneously.
 *
 * Optimally splits at each level using all available data.  Calls provided
 * function with each node added to the tree.
 *
 * @note Pointers @c p[0] to @c p[n-1] must be valid for the lifetime of the vp-tree
 */
int vptree_add_many_progress(
  vptree *vp, int n, const void * const *p,
  void *user_data, void (*callback)(void *user_data, int i, int n));

/**
 * Find k nearest neighbors.
 *
 * Returns neighbors sorted by distance in ascending order.
 *
 * @arg @c nn Output argument, must have space for @c k void pointers
 */
void vptree_nearest_neighbor(
  const vptree *vp, const void *p,
	int k, const void **nn);

/**
 * Find k nearest neighbors of multiple points.
 *
 * @see vptree_nearest_neighbor
 * @arg @c nn Output argument, each k points is for one n
 */
void vptree_nearest_neighbor_many(
  const vptree *vp, int n, const void * const *p,
  int k, const void **nn);

/**
 * Find all neighbors within a given ball of radius @c distance around p
 *
 * @note Caller must @c free returned pointer.
 * @arg @c n Output argument of the number of points in the neighborhood
 */
const void **vptree_neighborhood(
  const vptree *vp, const void *p, double distance, int *n);



typedef struct vptree_incnn vptree_incnn;

/**
 * Begin an incremental k-nearest neighbor search
 */
vptree_incnn *vptree_incnn_begin(const vptree *vp, const void *p);

/**
 * Get the next furthest neighbor of the point
 *
 * @note Will return NULL if all points have been exhausted
 */
const void *vptree_incnn_next(vptree_incnn *inc);

/**
 * Terminate an incremental k-nearest neighbor search
 */
void vptree_incnn_end(vptree_incnn *inc);

/**
 * Approximate search for k nearest neighbors.
 *
 * Returns neighbors sorted by distance in ascending order.
 *
 * @arg @c max_nodes The maximum number of nodes to visit
 * @arg @c nn Output argument, must have space for @c k void pointers
 */
void vptree_nearest_neighbor_approx(
  const vptree *vp, const void *p,
	int k, const void **nn, int max_nodes);

#ifdef __cplusplus
}
#endif

#endif /* #ifndef __VPTREE_H__ */
