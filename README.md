# GST
GPU-decodable Supercompressed Textures

This source code is intended to provide a reference encoding and GPU decoding implementation for
GST: GPU-decodable Supercompressed Textures, submitted to SIGGRAPH ASIA 2016. Note that this code
has been rebranded a few times prior to submission, so there may be lingering references to "gtc".

The code has the following dependencies:

- OpenCL 1.2
- CMake 3.3
- Full C++11 support (GCC 4.8, Visual Studio 2015, etc.)

Additionally, the demos depend on available and up to date OpenGL libraries. The code was tested on
an AMD R9 Fury GPU using the latest available drivers.

# Demos
The following applications are available for use:

- codec/gentenc <input> <output>
	Generic encoder for processing images. Due to the limitations that we have set in the
	compression settings, all images must have dimensions (w x h) that satisfy the following
	constraints:
	1. w mod 128 == 0
	2. h mod 128 == 0
	3. (w * h) mod (16 * 256 * 32) == 0
	
	The most applicable dimensions that satisfy these constraints are 512x512.
	
- demo/viewer
	OpenGL program that loads and displays the images produced by the encoder
	
- demo/photos_sf [-p] [-s] <directory>
- demo/photos [-p] [-s] <directory>
	An OpenGL program that batch loads all images in a folder. The "-s" flag
	denotes serial execution, and the "-p" flag is for profiling (timing information
	is reported to stdout). Testing has only been done using the same file types per folder.
	GST files have only been tested when all files use the same dimensions. The difference
	between photos and photos_sf is that photos_sf attempts to interleave the
	compressed streams for GST files in order to better batch load the images (and hence
	increase the GPU parallelism of our method).
	
- demo/demo
	A motion JPEG video player with hardcoded paths. Please refer to the source code for
	more information on how to use this demo. In short, the demo expects a series of
	frames produced by the encode in a folder "../test/dump_gtc" relative to the
	working directory. It then loads and plays each file named "frameXXXX.gtc" in sequence
	where XXXX represents the frame number.