#include <stdlib.h>

#include "timing.h"

double time_diff(struct timeval *end, struct timeval *start)
{
	double secs;

	secs = (double)(end->tv_sec - start->tv_sec);
	secs += 1e-6 * (double)(end->tv_usec - start->tv_usec);

	return secs;
}

void timer_start(struct timeval *start)
{
  gettimeofday(start, NULL);
}

double timer_interval(struct timeval *start)
{
	struct timeval end;

	gettimeofday(&end, NULL);

	return time_diff(&end, start);
}

static const char *time_units(double *secs)
{
	if(*secs > 1e-1) {
		return "s";
	}
	else if(*secs >= 1e-3) {
		*secs *= 1e3;
		return "ms";
	}
	else if(*secs >= 1e-6) {
		*secs *= 1e6;
		return "us";
	}
	else {
		*secs *= 1e9;
		return "ns";
	}
}

void time_str(int buflen, char *buf, double secs)
{
	const char *suffix;

	suffix = time_units(&secs);

	snprintf(buf, buflen, "%.2f%s", secs, suffix);
}

void time_print(FILE *stream, double secs)
{
	const char *suffix;

	suffix = time_units(&secs);

	fprintf(stream, "%.2f%s", secs, suffix);
}
