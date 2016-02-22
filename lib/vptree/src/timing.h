#ifndef __TIMING_H__
#define __TIMING_H__

#include <sys/time.h>
#include <stdio.h>

// Finds the difference between two timeval's (end - diff)
// Returns as seconds in floating-point
double time_diff(struct timeval *end, struct timeval *start);

// Timers for (primitive, inaccurate) profiling
void timer_start(struct timeval *);
double timer_interval(struct timeval *start);

// Printing times
void time_str(int buflen, char *buf, double secs);
void time_print(FILE *, double secs);

#endif // #ifndef __TIMING_H__
