#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <stdio.h>

#include "vptree.h"
#include "vptree_struct.h"
#include "pqueue.h"

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

/////////////////////////////// Default Options ////////////////////////////

static void *default_allocate(void *user_data, size_t s)
{
  return malloc(s);
}

static void default_deallocate(void *user_data, void *data)
{
  free(data);
}

const vptree_options vptree_default_options = {
  .user_data = NULL,
  .distance = NULL,
  .allocate = default_allocate,
  .deallocate = default_deallocate
};

/////////////////////////////// vp-tree Construction ////////////////////////

typedef struct distp distp;
static node *node_create(vptree *vp, node *parent, int n, distp *dp, int *alli, int alln, void *user_data, void (*callback)(void *user_data, int i, int n));
static void node_destroy(vptree *vp, node *nd);
static int node_add(vptree *vp, node *nd, int n, distp *dp, int *alli, int alln, void *user_data, void (*callback)(void *user_data, int i, int n));

vptree *vptree_create(size_t opts_size, const vptree_options *opts)
{
  vptree *vp;

  // Allocate vptree structure
  vp = (vptree *)opts->allocate(opts->user_data, sizeof(vptree));
  if(vp == NULL) {
    return NULL;
  }

  // Initialize options struct
  vp->opts = vptree_default_options;
  if(opts_size > sizeof(vp->opts)) {
    opts_size = sizeof(vp->opts);
  }
  memcpy(&vp->opts, opts, opts_size);

  // Empty tree
  vp->root = NULL;
  vp->n = 0;

  return vp;
}

static node *node_clone(vptree *vp, node *parent, const node *src)
{
  node *dst;

  if(src == NULL) {
    return NULL;
  }

  dst = (node *)allocate(vp, sizeof(node));
  dst->vp = vp;

  dst->p = src->p;
  dst->mu = src->mu;

  dst->parent = parent;
  dst->lt = node_clone(vp, dst, src->lt);
  dst->ge = node_clone(vp, dst, src->ge);

  return dst;
}

vptree *vptree_clone(const vptree *src)
{
  vptree *dst;

  // Copy vp-tree struct
  dst = (vptree *)src->opts.allocate(src->opts.user_data, sizeof(vptree));
  dst->opts = src->opts;
  dst->n = src->n;

  // Copy nodes
  dst->root = node_clone(dst, NULL, src->root);

  return dst;
}

static void node_destroy(vptree *vp, node *nd)
{
  if(nd == NULL) {
    return;
  }

  node_destroy(vp, nd->lt);
  node_destroy(vp, nd->ge);
  deallocate(vp, nd);
}

void vptree_destroy(vptree *vp)
{
  if(vp == NULL) {
    return;
  }

  node_destroy(vp, vp->root);
  deallocate(vp, vp);
}

const vptree_options *vptree_get_options(const vptree *vp)
{
  return &vp->opts;
}

int vptree_npoints(const vptree *vp)
{
  return vp->n;
}

///////////////////////////// vp-tree Addition /////////////////////////

typedef struct distp {
  double d;
  const void *p;
} distp;

static node *node_create(vptree *vp, node *parent, int n, distp *dp, int *alli, int alln, void *user_data, void (*callback)(void *user_data, int i, int n))
{
  node *nd;
  int v;
  const void *p, *swap;
  int stat;

  // Null node
  if(n == 0) {
    return NULL;
  }

  nd = allocate(vp, sizeof(node));
  if(nd == NULL) {
    return NULL;
  }
  nd->vp = vp;
  nd->parent = parent;

  // Update progress
  if(callback != NULL) {
    (*alli)++;
    callback(user_data, *alli, alln);
  }

  // Select reference node
  // TODO
  v = rand() % n;
  nd->p = p = dp[v].p;

  // Initalize as singleton node
  nd->mu = -1;
  nd->lt = nd->ge = NULL;

  if(n != 1) {
    // Add subnodes
    swap = dp[0].p;
    dp[0].p = dp[v].p;
    dp[v].p = swap;

    stat = node_add(vp, nd, n-1, dp+1, alli, alln, user_data, callback);
    if(stat == -1) {
      deallocate(vp, nd);
      return NULL;
    }
  }

  return nd;
}

static int compare_distp(const void *v1, const void *v2)
{
  double d1, d2;

  d1 = ((const distp *)v1)->d;
  d2 = ((const distp *)v2)->d;

  if(d1 < d2) {
    return -1;
  }
  else if(d1 == d2) {
    return 0;
  }
  else {
    return 1;
  }
}

static void sort_distp(int n, distp *dp)
{
  qsort(dp, (size_t)n, sizeof(distp), compare_distp);
}

static int node_add(vptree *vp, node *nd, int n, distp *dp, int *alli, int alln, void *user_data, void (*callback)(void *user_data, int i, int n))
{
  distp *lt, *ge;
  int i, m;
  int stat;

  // Calculate distances
  for(i = 0; i < n; i++) {
    dp[i].d = distance(vp, nd->p, dp[i].p);

    if(dp[i].d < 0) {
      return -1;
    }

    //if(i % (1024) == 0) {
    //  fprintf(stderr, "Distance %d of %d (%.2f%%)\n", i, n, (((float)i)*100)/n);
    //}
  }
  //fprintf(stderr, "Sorting...\n");
  sort_distp(n, dp);

  // Previously a leaf node, find median distance
  if(nd->mu < 0) {
    // Find median distance
    m = n/2;

    if(n % 2 == 0) {
      if(n == 2) {
        m = 0;
      }
      nd->mu = (dp[m].d + dp[m+1].d)/2;
    }
    else {
      nd->mu = dp[m].d;
    }
  }

  for(m = 0; m < n && dp[m].d < nd->mu; m++);
  //fprintf(stderr, "Split at m = %d (of %d), dist = %lg\n", m, n, nd->mu);

  // TODO: case with equal distances
  // Add lt nodes
  lt = dp;
  if(m > 0) {
    if(nd->lt == NULL) {
      nd->lt = node_create(vp, nd, m, lt, alli, alln, user_data, callback);
      if(nd->lt == NULL) {
        return -1;
      }
    }
    else {
      stat = node_add(vp, nd->lt, m, lt, alli, alln, user_data, callback);
      if(stat == -1) {
        return -1;
      }
    }
  }

  // Add ge nodes
  ge = dp + m;
  if(n - m > 0) {
    if(nd->ge == NULL) {
      nd->ge = node_create(vp, nd, n - m, ge, alli, alln, user_data, callback);
      if(nd->ge == NULL) {
        return -1;
      }
    }
    else {
      stat = node_add(vp, nd->ge, n - m, ge, alli, alln, user_data, callback);
      if(stat == -1) {
        return -1;
      }
    }
  }

  return 0;
}


int vptree_add_many_progress(
  vptree *vp, int n, const void * const *p,
  void *user_data, void (*callback)(void *user_data, int i, int n))
{
  int i, alli;
  distp *dp;
  int stat;

  stat = 0;
  alli = 0;


  // Create distance-comparison structure
  dp = allocate(vp, n * sizeof(distp));
  for(i = 0; i < n; i++) {
    dp[i].p = p[i];
  }

  // Add to tree
  if(vp->root == NULL) {
    vp->root = node_create(vp, NULL, n, dp, &alli, n, user_data, callback);
    if(vp->root == NULL) {
      stat = -1;
    }
  }
  else {
    stat = node_add(vp, vp->root, n, dp, &alli, n, user_data, callback);
  }

  deallocate(vp, dp);

  vp->n += n;

  return stat;
}

int vptree_add_many(vptree *vp, int n, const void * const *p)
{
  return vptree_add_many_progress(vp, n, p, NULL, NULL);
}

int vptree_add(vptree *vp, const void *p)
{
  return vptree_add_many(vp, 1, &p);
}

//////////////////////////////// k-NN Query ////////////////////////////

/**
 * Add @c ndp at distance @c d from the query point to the nearest neighbors.
 *
 * Will maintain the neighbor list in ascending order of distance from the
 * query point.
 *
 * @note If @c d is greater than the distance from the furthest of the @c k
 *       existing neighbors, this is a noop.
 */
static void add_knn(int k, const void **nn, double *nndist, const void *ndp, double d)
{
  int i, j;

  if(d >= nndist[k-1]) {
    return;
  }
  
  for(i = 0; i < k && nndist[i] < d; i++);
  for(j = k-1; j > i; j--) {
    nn[j] = nn[j-1];
    nndist[j] = nndist[j-1];
  }
  nn[i] = ndp;
  nndist[i] = d;
}


static void nn_query(
  const vptree *vp, node *nd,
  const void *p, int k,
  const void **nn, double *nndist)
{
  double d, mu;

  if(nd == NULL) {
    return;
  }

  // Calculate distance to current node
  d = distance(vp, p, nd->p);

  // Add to nearest neighbors (maintain sorted order)
  add_knn(k, nn, nndist, nd->p, d);

  // Recurse to children
  mu = nd->mu;
  if(mu < 0) {
    return;
  }
  
  if(d - nndist[k-1] < mu) {
    nn_query(vp, nd->lt, p, k, nn, nndist);
  }
  if(d + nndist[k-1] >= mu) {
    nn_query(vp, nd->ge, p, k, nn, nndist);
  }
}

void vptree_nearest_neighbor(
  const vptree *vp, const void *p,
	int k, const void **nn)
{
  int i;
  double *nndist;

  // Set up temporary array for storing distances to neighbors
  nndist = (double *)allocate(vp, sizeof(double) * k);
  for(i = 0; i < k; i++) {
    nn[i] = NULL;
    nndist[i] = INFINITY;
  }

  // Call real algorithm
  nn_query(vp, vp->root, p, k, nn, nndist);

  // Cleanup
  deallocate(vp, nndist);
}

////////////////////////////// Neighborhood Query ///////////////////////

static void add_nbr_point(int *n, const void ***nbr, const void *p)
{
  // TODO: using realloc here may not place nicely with user-provided
  // allocate/deallocate
  *n += 1;
  *nbr = realloc(*nbr, *n * sizeof(const void *));
  (*nbr)[*n - 1] = p;
}

static void epsilon_query(const vptree *vp, node *nd, const void *p, double epsilon,
                          int *nfound, const void ***nbr)
{
  double d, mu;

  if(nd == NULL) {
    return;
  }

  d = distance(vp, p, nd->p);
  if(d < epsilon) {
    add_nbr_point(nfound, nbr, nd->p);
  }

  mu = nd->mu;
  if(mu < 0) {
    return;
  }

  if(d - epsilon < mu) {
    epsilon_query(vp, nd->lt, p, epsilon, nfound, nbr);
  }
  if(d + epsilon >= mu) {
    epsilon_query(vp, nd->ge, p, epsilon, nfound, nbr);
  }
}


const void **vptree_neighborhood(
  const vptree *vp, const void *p, double distance,
  int *n)
{
  const void **nbr;

  *n = 0;
  nbr = NULL;

  epsilon_query(vp, vp->root, p, distance, n, &nbr);

  return nbr;
}

/////////////////////////////// Incremental knn /////////////////////////

static void destroy_inctree(const vptree *vp, incnode *n)
{
  if(n == NULL) {
    return;
  }

  destroy_inctree(vp, n->lt);
  destroy_inctree(vp, n->ge);
  deallocate(vp, n);
}

static incnode *make_incnode(const vptree *vp, incnode *parent, node *n, const void *q)
{
  incnode *incn;

  if(n == NULL) {
    return NULL;
  }

  incn = allocate(vp, sizeof(incnode));
  incn->n = n;
  incn->d = distance(vp, n->p, q);
  incn->exclude_tree = incn->exclude = false;
  
  incn->parent = parent;
  incn->ge = incn->lt = NULL;

  return incn;
}

static void prune_marks(const vptree *vp, incnode *n)
{
  bool lt_excluded, ge_excluded;

  // Check if entire tree below n is marked
  if(n->lt == NULL) {
    lt_excluded = (n->n->lt == NULL);
  }
  else {
    lt_excluded = n->lt->exclude_tree;
  }

  if(n->ge == NULL) {
    ge_excluded = (n->n->ge == NULL);
  }
  else {
    ge_excluded = n->ge->exclude_tree;
  }

  // Prune mark tree and mark parent
  if(lt_excluded && ge_excluded && n->exclude) {
    deallocate(vp, n->lt);
    deallocate(vp, n->ge);
    n->lt = n->ge = NULL;
    n->exclude_tree = true;
 
    if(n->parent != NULL) {
      prune_marks(vp, n->parent);
    }
  }
}

vptree_incnn *vptree_incnn_begin(const vptree *vp, const void *q)
{
  vptree_incnn *inc;

  inc = (vptree_incnn *)allocate(vp, sizeof(vptree_incnn));
  inc->vp = vp;
  inc->q = q;
  
  inc->prev = inc->marks = make_incnode(vp, NULL, vp->root, q);

  return inc;
}


static void incnn_query(vptree_incnn *inc, incnode *mark, incnode **nn, double *nnd, incnode *exclude)
{
  const vptree *vp;
  double d, mu;

  if(mark == NULL || mark->exclude_tree || mark == exclude) {
    return;
  }

  vp = inc->vp;
  d = mark->d;

  // Set as nearest neighbor
  if(d < *nnd && !mark->exclude) {
    *nn = mark;
    *nnd = d;
  }

  // Recurse to children
  mu = mark->n->mu;
  if(mu < 0) {
    return;
  }

  if(d - *nnd < mu) {
    if(mark->lt == NULL) {
      mark->lt = make_incnode(vp, mark, mark->n->lt, inc->q);
    }
    incnn_query(inc, mark->lt, nn, nnd, NULL);
  }
  if(d + *nnd >= mu) {
    if(mark->ge == NULL) {
      mark->ge = make_incnode(vp, mark, mark->n->ge, inc->q);
    }
    incnn_query(inc, mark->ge, nn, nnd, NULL);
  }
}

const void *vptree_incnn_next(vptree_incnn *inc)
{
  double nnd;
  incnode *nn, *query, *lastquery;
  const void *result;

  nn = NULL;
  nnd = INFINITY;

  // Walk up the tree to find more nodes
  lastquery = NULL;
  query = inc->prev;
  while(query != NULL) {
    incnn_query(inc, query, &nn, &nnd, lastquery);

    lastquery = query;
    query = query->parent;
  }

  // Return result
  if(nn == NULL) {
    return NULL;
  }
  else {
    // Mark node as returns
    result = nn->n->p;

    nn->exclude = true;
    prune_marks(inc->vp, nn);

    return result;
  }
}


void vptree_incnn_end(vptree_incnn *inc)
{
  if(inc == NULL) {
    return;
  }

  destroy_inctree(inc->vp, inc->marks);
  deallocate(inc->vp, inc);
}

////////////////////////////// Approximate k-NN /////////////////////////


typedef struct {
  node *nd;

  size_t pos;
  double prio;
} pqnode;

static int pqueue_cmp_prio(double next, double curr)
{
  if(curr < next) {
    return 1;
  }
  else {
    return 0;
  }
}

static double pqueue_get_prio(void *ptr)
{
  return ((pqnode *)ptr)->prio;
}

static void pqueue_set_prio(void *ptr, double prio)
{
  ((pqnode *)ptr)->prio = prio;
}

static size_t pqueue_get_pos(void *ptr)
{
  return ((pqnode *)ptr)->pos;
}

static void pqueue_set_pos(void *ptr, size_t pos)
{
  ((pqnode *)ptr)->pos = pos;
}

static void add_knn(int k, const void **nn, double *nndist, const void *ndp, double d);

void vptree_nearest_neighbor_approx(
  const vptree *vp, const void *p,
	int k, const void **nn, int max_nodes)
{
  int i, next_node, visited;
  double *nndist, d, mu;
  pqnode *pnodes, *pnd;
  node *nd;
  pqueue_t *pq;

  if(max_nodes > vp->n) {
    max_nodes = vp->n;
  }

  // Initialize nn state
  nndist = (double *)malloc(sizeof(double) * k);
  assert(nndist != NULL);
  for(i = 0; i < k; i++) {
    nndist[i] = INFINITY;
    nn[i] = NULL;
  }

  // Initialize priority queue
  pnodes = (pqnode *)malloc(sizeof(pqnode) * 2 * max_nodes);
  assert(pnodes != NULL);

  pnodes[0].nd = vp->root;
  pnodes[0].prio = 0;
  next_node = 1;

  pq = pqueue_init(2 * max_nodes, pqueue_cmp_prio, pqueue_get_prio, pqueue_set_prio, pqueue_get_pos, pqueue_set_pos);
  assert(pq != NULL);

#ifndef NDEBUG
  int r = pqueue_insert(pq, &pnodes[0]);
  assert(r == 0);
#else
  pqueue_insert(pq, &pnodes[0]);
#endif

  // Main loop
  for(visited = 0; visited < max_nodes; visited++) {
    pnd = pqueue_pop(pq);
    if(pnd == NULL) {
      break;
    }
    nd = pnd->nd;

    d = distance(vp, p, nd->p);
    add_knn(k, nn, nndist, nd->p, d);

    // Push children onto priority queue
    mu = nd->mu;
    if(mu < 0) {
      continue;
    }
  
    if(d - nndist[k-1] < mu && nd->lt != NULL) {
      pnodes[next_node].nd = nd->lt;
      pnodes[next_node].prio = distance(vp, p, nd->lt->p);

      pqueue_insert(pq, &pnodes[next_node]);

      next_node++;
    }
    if(d + nndist[k-1] >= mu && nd->ge != NULL) {
      pnodes[next_node].nd = nd->ge;
      pnodes[next_node].prio = distance(vp, p, nd->ge->p);

      pqueue_insert(pq, &pnodes[next_node]);

      next_node++;
    }
  }

  //fprintf(stderr, "Visited %d nodes\n", visited);

  // Cleanup
  pqueue_free(pq);
  free(pnodes);
  free(nndist);
}
