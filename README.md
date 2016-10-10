# GST: GPU-decodable Supercompressed Textures
[![Build Status](https://travis-ci.org/Mokosha/GST.svg?branch=master)](https://travis-ci.org/Mokosha/GST)

This source code is intended to provide a reference encoding and GPU decoding
implementation for the paper
[GST: GPU-decodable Supercompressed Textures](http://gamma.cs.unc.edu/GST),
presented at SIGGRAPH ASIA 2016. Note that this code has been rebranded a
few times prior to submission, so there may be lingering references to "GTC",
or "GenTC", or even "TCAR".

## Licensing

GST is licensed under the Apache 2.0 license. The full license can be found
[online](http://www.apache.org/licenses/LICENSE-2.0). There are additional pieces
that have been redistributed and incorporated into this repository. They have been
made generously available by their authors. These pieces include:

1. GTest as a test framework, under
[Google's own](https://github.com/GammaUNC/FasTC/blob/master/GTest/LICENSE)
license

2. GLFW 3.1 for use with some of the demos and sample programs, provided under
[their own](https://github.com/glfw/glfw/blob/master/COPYING.txt) license.

3. Binary and header distribution of GLEW for Windows, released under
[their own](http://glew.sourceforge.net/glew.txt) license.

4. The vptree implementation from
[Maxwell Collins](http://pages.cs.wisc.edu/~mcollins/software/vptree.html)
at the University of Wisconsin, licensed under the Apache 2.0 license as well.

5. Vitaliy Vitsentiy's [threadpool library](https://github.com/vit-vit/CTPL),
also licensed under the Apache 2.0 license.

6. The [GLIML library](https://github.com/floooh/gliml) for loading DDS and KTX files,
distributed under the MIT license.

7. A [version of dirent](https://github.com/tronkko/dirent) that works on Windows,
also distributed under the MIT license.

## Building
The code has the following dependencies:

- OpenCL 1.2
- CMake 3.3
- Full C++11 support (GCC 4.8, Visual Studio 2015, etc.)

Additionally, the demos depend on available and up to date OpenGL libraries.
The code was tested on an AMD R9 Fury GPU using the latest available drivers.

Once the project is built, use your preferred CMake generator to run the tests.
- With makefiles, this is as simple as running "make test"
- With Visual Studio, build the "RUN_TESTS" project under CMakePredefinedTargets.

The tests that use the OpenCL runtime have _ocl in the name. If these do not pass,
then you may need to reconfigure your OpenCL implementation (or perhaps run the tests
as an administrator).

## Demos
The following applications are available for use:

- `codec/gentenc <input> [compressed] <output>`

  Generic encoder for processing images. **compressed** is expected to be a DDS or KTX
  file with an existing DXT encoding used instead of the stb_dxt implementation.

  Due to the limitations that we have set in the compression settings, all images
  must have dimensions (w x h) that satisfy the following constraints:
    1. w mod 128 == 0
    2. h mod 128 == 0
    3. (w * h) mod (16 * 256 * 32) == 0

  The most applicable dimensions that satisfy these constraints are 512x512.

- `demo/viewer <gst_file>`

  OpenGL program that loads and displays the images produced by the encoder

- `demo/photos[_sf] [-p] [-s] <directory>`

  An OpenGL program that batch loads all images in a folder. The "-s" flag
  denotes serial execution, and the "-p" flag is for profiling (timing information
  is reported to stdout). Testing has only been done using the same file types per
  folder. GST files have only been tested when all files use the same dimensions.
  The difference between photos and photos_sf is that photos_sf attempts to
  interleave the compressed streams for GST files in order to better batch load
  the images (and hence	increase the GPU parallelism of our method). One such dataset
  is the [Pixar dataset](https://community.renderman.pixar.com/article/114/library-pixar-one-twenty-eight.html).

- `demo/demo`

  A motion JPEG video player with hardcoded paths. Please refer to the source code for
  more information on how to use this demo. In short, the demo expects a series of
  frames produced by the encoder in a folder "../test/dump_gtc" relative to the
  working directory. It then loads and plays each file named "frameXXXX.gtc" in sequence
  where XXXX represents the frame number.
