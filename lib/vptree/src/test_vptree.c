#include <stdlib.h>
#include <stdio.h>

#include "vptree.h"
#include "geom.h"
#include "timing.h"

#define N (1 << 18)
#define DIM (128)
#define TRIALS (128)
#define MAX_NODES (1024)

static double points[N * DIM];
static const void *ptr[N];

static double frand(unsigned *seed, double a, double b);
static void frandvec(unsigned *seed, int n, double *p, double a, double b);
static double distance(void *user_data, const void *p1, const void *p2);
static double exhaustive_search(const double *query);
static double avg_distance(const double *query);
static int numcloser(const double *query, int i);

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

int main(int argc, char **argv)
{
  int i, t;
  unsigned seed;
  double q[DIM];

  vptree_options vpopts;
  vptree *vp;

  const void *nn;
  double exhaustive_distance[TRIALS], vptree_distance[TRIALS], approx_distance[TRIALS], Erandom_distance[TRIALS];

  double err, avg_vptree_err, avg_approx_err, avg_approx_dist;

  struct timeval timer;
  double exhaustive_time, vptree_time, approx_time;

  double avg_random, avg_nearest;

  int ncloser[TRIALS];
  double avg_closer;

  // Create points
  seed = 0;
  frandvec(&seed, N * DIM, points, 0, 1);
  for(i = 0; i < N; i++) {
    ptr[i] = points + DIM * i;
  }

  // Construct VP-tree
  vpopts = vptree_default_options;
  vpopts.user_data = NULL;
  vpopts.distance = distance;

  vp = vptree_create(sizeof(vpopts), &vpopts);
  vptree_add_many(vp, N, ptr);

  // Test queries
  exhaustive_time = approx_time = vptree_time = 0;
  for(t = 0; t < TRIALS; t++) {
    frandvec(&seed, DIM, q, 0, 1); // random query point

    Erandom_distance[t] = avg_distance(q);

    // Exhaustive search
    timer_start(&timer);
    exhaustive_distance[t] = exhaustive_search(q);
    exhaustive_time += timer_interval(&timer);

    // VP-tree search
    timer_start(&timer);
    vptree_nearest_neighbor(vp, (const void *)q, 1, &nn);
    vptree_time += timer_interval(&timer);

    vptree_distance[t] = geom_distance(DIM, q, (const double *)nn);

    // Approximate VP-tree search
    timer_start(&timer);
    vptree_nearest_neighbor_approx(vp, (const void *)q, 1, &nn, MAX_NODES);
    approx_time += timer_interval(&timer);
    
    approx_distance[t] = geom_distance(DIM, q, (const double *)nn);
    ncloser[t] = numcloser(q, ((double *)nn - points)/DIM);
  }

  // Analyze results
  avg_vptree_err = 0;
  avg_approx_err = 0;
  avg_nearest = 0;
  avg_random = 0;
  avg_approx_dist = 0;
  avg_closer = 0;
  for(t = 0; t < TRIALS; t++) {
    err = vptree_distance[t] - exhaustive_distance[t];

    avg_nearest += exhaustive_distance[t];
    avg_random += Erandom_distance[t];
    avg_approx_dist += approx_distance[t];
    avg_closer += (double)ncloser[t];

    if(exhaustive_distance[t] == 0) {
      fprintf(stderr, "Bad craziness\n");
      continue;
    }
    else {
      avg_vptree_err += (vptree_distance[t] - exhaustive_distance[t]) / exhaustive_distance[t];
      avg_approx_err += (approx_distance[t] - exhaustive_distance[t]) / exhaustive_distance[t];
    }
  }
  avg_nearest /= TRIALS;
  avg_random /= TRIALS;
  avg_vptree_err /= TRIALS;
  avg_approx_err /= TRIALS;
  avg_approx_dist /= TRIALS;
  avg_closer /= TRIALS;

  printf("N = %d\nDIM = %d\n\n", N, DIM);

  printf("Average random distance: %lg\n\n", avg_random);

  printf("Linear search:\n");
  //printf("  mean time: "); time_print(stdout, exhaustive_time / TRIALS); printf("\n");
  printf("  mean time: %.5le\n", exhaustive_time / TRIALS);
  printf("  mean distance: %lg\n", avg_nearest); 
  printf("\n");

  printf("VP-tree:\n");
  printf("  mean error: %lg\n", avg_vptree_err);
  //printf("  mean time: "); time_print(stdout, vptree_time / TRIALS); printf("\n");
  printf("  mean time: %.5le\n", vptree_time / TRIALS);
  printf("\n");

  printf("Approx VP-tree:\n");
  printf("  mean error: %lg\n", avg_approx_err);
  printf("  mean distance: %lg\n", avg_approx_dist);
  printf("  num closer: %lg\n", avg_closer);
  printf("  mean time: %.5le\n", approx_time / TRIALS);
  //printf("  mean time: "); time_print(stdout, approx_time / TRIALS); printf("\n");
  printf("\n");

  // Cleanup
  vptree_destroy(vp);

  return 0;
}

static double frand(unsigned *seed, double a, double b)
{
  int r;

  r = rand_r(seed);

  return (r/(double)RAND_MAX) * (b-a) + a;
}

static void frandvec(unsigned *seed, int n, double *p, double a, double b)
{
  int i;

  for(i = 0; i < n; i++) {
    p[i] = frand(seed, a, b);
  }
}

static double distance(void *user_data, const void *p1, const void *p2)
{
  return geom_distance(DIM, (const double *)p1, (const double *)p2);
}

static double exhaustive_search(const double *query)
{
  int i;
  double d, bestd, *best;

  bestd = INFINITY;
  best = NULL;

  for(i = 0; i < N; i++) {
    d = geom_distance(DIM, query, points + DIM * i);
    if(d < bestd) {
      bestd = d;
      best = points + DIM * i;
    }
  }

  return bestd;
}

static double avg_distance(const double *query)
{
  int i;
  double avg;

  avg = 0;
  for(i = 0; i < N; i++) {
    avg += geom_distance(DIM, query, points + DIM * i);
  }
  avg /= N;

  return avg;
}

static int numcloser(const double *query, int i)
{
  int j, m;
  double di, dj;

  di = geom_distance(DIM, query, points + DIM * i);
  m = 0;

  for(j = 0; j < N; j++) {
    if(j == i) {
      continue;
    }

    dj = geom_distance(DIM, query, points + DIM * j);
    if(dj < di) {
      m++;
    }
  }

  return m;
}
