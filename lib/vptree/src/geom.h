#ifndef __BITS_GEOM_H__
#define __BITS_GEOM_H__

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

//////////////////////////// Distances /////////////////////

double geom_l1distance(size_t ndims, const double *p, const double *q);
double geom_l2distance(size_t ndims, const double *p, const double *q);
double geom_linftydistance(size_t ndims, const double *p, const double *q);
double geom_lpdistance(double p, size_t ndims, const double *q, const double *r);

#define geom_distance geom_l2distance

////////////////////////////// Norms //////////////////////////

double geom_l1norm(size_t ndims, const double *p);
double geom_l2norm(size_t ndims, const double *p);
double geom_linftynorm(size_t ndims, const double *p);
double geom_lpnorm(double p, size_t ndims, const double *q);

#define geom_norm geom_l2norm

////////////////////////// Unit Vectors ////////////////////////

void geom_l1_unit_vector(size_t ndims, double *u, const double *p);
void geom_l2_unit_vector(size_t ndims, double *u, const double *p);

#define geom_unit_vector geom_l2_unit_vector


/**
 * Find the minimum axis-aligned bounding box for a set of points.
 *
 * @arg @c ndims The dimensionality of each point.
 * @arg @c n The number of points.
 * @arg @c p The points, with each point taking @c ndims consecutive doubles.
 * @arg @c min Output argument, the minimum for each dimension of the box.
 *             Must have space for @c ndims elements.
 * @arg @c max Output argument, the maximum for each dimension of the box.
 *             Must have space for @c ndims elements.
 */
void geom_bounding_box(
  size_t ndims, size_t n, const double *p,
  double *min, double *max);

////////////////////////// Printing Vectors /////////////////////////////

void geom_vector_fprintf(FILE *fp, size_t ndims, const double *p);
void geom_vector_snprint(size_t n, char *buf, size_t ndims, const double *p);

#endif // #ifndef __BITS_GEOM_H__
