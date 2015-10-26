#ifndef __STOP_WATCH_H__
#define __STOP_WATCH_H__

// Forward declare the private implementation of the class that will actually implement
// the timing features. This class is defined in each module depending on the platform...
class StopWatchImpl;

// A simple stopwatch class using Windows' high-resolution performance counters.
class StopWatch
{
 public:
  StopWatch();
  StopWatch(const StopWatch &);

  ~StopWatch();

  StopWatch &operator=(const StopWatch &);

  void Start();
  void Stop();
  void Reset();

  double TimeInSeconds() const;
  double TimeInMilliseconds() const;
  double TimeInMicroseconds() const;

 private:
  StopWatchImpl *impl;
};

#endif // __TEXCOMP_STOP_WATCH_H__
