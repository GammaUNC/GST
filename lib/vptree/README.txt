====Dependencies ====

  - scons (http://www.scons.org/)
  - cython (http://cython.org/), needed for Python bindings
  - Octave (http://www.gnu.org/software/octave/) or Matlab needed for Matlab
    bindings

==== Building ====

libvptree can be build by running the `scons` command in the root directory of
this distribution.

==== Running ====

A test program will be built in the 'bin/' directory.  It takes no arguments.

Example code to show how to use the package within each language is given in the
'examples/' directory.  Examples in C and C++ are compiled into the 'bin/'
directory.  Assuming all language bindings can be built, each example can be
run with the commands:

./bin/cities
./bin/cities_cpp
python examples/cities.py
octave -q --path examples --eval cities_octave
cd examples; matlab -r cities_matlab

TODO: Matlab script?

==== Linking ====

Static and shared libraries will be built in the 'lib/' directory.

Necessary include files will be placed in 'include/vptree/'
