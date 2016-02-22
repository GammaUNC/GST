#include <math.h>

#include "geom.h"

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif


double geom_l1distance(size_t ndims, const double *p, const double *q)
{
  size_t i;
  double dist;

  dist = 0;
  for(i = 0; i < ndims; i++) {
    dist += fabs(p[i] - q[i]);
  }

  return dist;
}

double geom_l2distance(size_t ndims, const double *p, const double *q)
{
  size_t i;
  double dist, diff;

  dist = 0;
  for(i = 0; i < ndims; i++) {
    diff = p[i] - q[i];
    dist += diff * diff;
  }
  dist = sqrt(dist);

  return dist;
}

double geom_linftydistance(size_t ndims, const double *p, const double *q)
{
  size_t i;
  double dist, diff;

  dist = 0;
  for(i = 0; i < ndims; i++) {
    diff = fabs(p[i] - q[i]);
    dist = diff > dist ? diff : dist;
  }

  return dist;
}

double geom_lpdistance(double p, size_t ndims, const double *q, const double *r)
{
  size_t i;
  double dist;

  dist = 0;
  for(i = 0; i < ndims; i++) {
    dist += pow(fabs(q[i] - r[i]), p);
  }
  dist = pow(dist, 1.0/p);

  return dist;
}

double geom_l1norm(size_t ndims, const double *p)
{
  size_t i;
  double mag;

  mag = 0;
  for(i = 0; i < ndims; i++) {
    mag += fabs(p[i]);
  }

  return mag;
}

double geom_l2norm(size_t ndims, const double *p)
{
  size_t i;
  double mag;

  mag = 0;
  for(i = 0; i < ndims; i++) {
    mag += p[i] * p[i];
  }

  return sqrt(mag);
}

double geom_linftynorm(size_t ndims, const double *p)
{
  size_t i;
  double mag, abs;

  mag = 0;
  for(i = 0; i < ndims; i++) {
    abs = fabs(p[i]);
    mag = abs > mag ? abs : mag;
  }

  return mag;
}

double geom_lpnorm(double p, size_t ndims, const double *q)
{
  size_t i;
  double mag;

  mag = 0;
  for(i = 0; i < ndims; i++) {
    mag += pow(fabs(q[i]), p);
  }
  mag = pow(mag, 1.0/p);

  return mag;
}

void geom_bounding_box(
  size_t ndims, size_t n, const double *p,
  double *min, double *max)
{
  size_t i, j;
  double x;

  for(j = 0; j < ndims; j++) {
    min[j] = INFINITY;
    max[j] = -INFINITY;
  }

  for(i = 0; i < n; i++) {
    for(j = 0; j < ndims; j++) {
      x = p[ndims*i + j];

      if(x < min[j]) {
        min[j] = x;
      }
      if(x > max[j]) {
        max[j] = x;
      }
    }
  }
}

void geom_l1_unit_vector(size_t ndims, double *u, const double *p)
{
  double mag;
  size_t i;

  mag = geom_l1norm(ndims, p);
  for(i = 0; i < ndims; i++) {
    u[i] = p[i] / mag;
  }
}

void geom_l2_unit_vector(size_t ndims, double *u, const double *p)
{
  double mag;
  size_t i;

  mag = geom_l2norm(ndims, p);
  for(i = 0; i < ndims; i++) {
    u[i] = p[i] / mag;
  }
}

double geom_dot_product(size_t ndims, const double *u, const double *v)
{
  double dot;
  size_t i;

  dot = 0;
  for(i = 0; i < ndims; i++) {
    dot += u[i] * v[i];
  }

  return dot;
}

//////////////////////////// Vector I/O ///////////////////////

static void print(FILE *fp, size_t ndims, const double *p,
                  const char *left, const char *delim, const char *right,
                  const char *fmt)
{
  size_t i;

  fputs(left, fp);

  for(i = 0; i < ndims; i++) {
    if(i != 0) {
      fputs(delim, fp);
    }
    fprintf(fp, fmt, p[i]);
  }

  fputs(right, fp);
}

void geom_vector_fprintf(FILE *fp, size_t ndims, const double *p)
{
  print(fp, ndims, p , "(", ", ", ")", "%.2g");
}


static char *append(char *dst, const char *src, size_t *pn)
{
  size_t n;
  const char *s;

  n = *pn;
  for(s = src; *s != '\0' && n > 0; s++) {
    *dst = *s;

    dst++;
    n--;
  }
  *pn = n;

  return dst;
}

static void snprint(size_t n, char *buf,
                    size_t ndims, const double *p,
                    const char *left, const char *delim, const char *right,
                    const char *fmt)
{
  char numbuf[1024];
  size_t i;

  buf = append(buf, left, &n);

  for(i = 0; i < ndims; i++) {
    if(i != 0) {
      buf = append(buf, delim, &n);
    }

    snprintf(numbuf, 1024, fmt, p[i]);
    buf = append(buf, numbuf, &n);
  }

  buf = append(buf, right, &n);
  *buf = '\0';
}

void geom_vector_snprint(size_t n, char *buf, size_t ndims, const double *p)
{
  snprint(n, buf, ndims, p, "(", ", ", ")", "%.2g");
}
