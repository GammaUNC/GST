#include "config.h"
#include "gpu.h"

#include <cassert>
#include <iostream>
#include <fstream>

static cl_context gContext;
static cl_command_queue gCommandQueue;
static cl_kernel gKernel;

static const size_t kBlockWidth = 4;
static const size_t kBlockHeight = 4;
static const size_t kNumComponents = 4;
static const size_t kNumBlockPixels = kBlockWidth * kBlockHeight;
static const size_t kBlockSize = kNumBlockPixels * kNumComponents;

static size_t kLocalWorkSizeX = 16;
static size_t kLocalWorkSizeY = 16;
static inline size_t GetTotalWorkItems() { return kLocalWorkSizeX * kLocalWorkSizeY; }

// We have 16 pixels per work item and 4 bytes per pixel, so each work
// group will need this much local memory.
static inline size_t GetPixelBufferBytes() { return GetTotalWorkItems() * kBlockSize; }

// Thirty-two bytes for the kernel arguments...
static inline size_t GetTotalLocalMemory() { return GetPixelBufferBytes() + 32; }

static const char *clErrMsg(cl_int err) {
  const char *errMsg = "Unknown error";
  switch(err) {
  case CL_SUCCESS:                         errMsg = "Success!"; break;
  case CL_DEVICE_NOT_FOUND:                errMsg = "Device not found."; break;
  case CL_DEVICE_NOT_AVAILABLE:            errMsg = "Device not available"; break;
  case CL_COMPILER_NOT_AVAILABLE:          errMsg = "Compiler not available"; break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:   errMsg = "Memory object allocation failure"; break;
  case CL_OUT_OF_RESOURCES:                errMsg = "Out of resources"; break;
  case CL_OUT_OF_HOST_MEMORY:              errMsg = "Out of host memory"; break;
  case CL_PROFILING_INFO_NOT_AVAILABLE:    errMsg = "Profiling information not available"; break;
  case CL_MEM_COPY_OVERLAP:                errMsg = "Memory copy overlap"; break;
  case CL_IMAGE_FORMAT_MISMATCH:           errMsg = "Image format mismatch"; break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:      errMsg = "Image format not supported"; break;
  case CL_BUILD_PROGRAM_FAILURE:           errMsg = "Program build failure"; break;
  case CL_MAP_FAILURE:                     errMsg = "Map failure"; break;
  case CL_INVALID_VALUE:                   errMsg = "Invalid value"; break;
  case CL_INVALID_DEVICE_TYPE:             errMsg = "Invalid device type"; break;
  case CL_INVALID_PLATFORM:                errMsg = "Invalid platform"; break;
  case CL_INVALID_DEVICE:                  errMsg = "Invalid device"; break;
  case CL_INVALID_CONTEXT:                 errMsg = "Invalid context"; break;
  case CL_INVALID_QUEUE_PROPERTIES:        errMsg = "Invalid queue properties"; break;
  case CL_INVALID_COMMAND_QUEUE:           errMsg = "Invalid command queue"; break;
  case CL_INVALID_HOST_PTR:                errMsg = "Invalid host pointer"; break;
  case CL_INVALID_MEM_OBJECT:              errMsg = "Invalid memory object"; break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: errMsg = "Invalid image format descriptor"; break;
  case CL_INVALID_IMAGE_SIZE:              errMsg = "Invalid image size"; break;
  case CL_INVALID_SAMPLER:                 errMsg = "Invalid sampler"; break;
  case CL_INVALID_BINARY:                  errMsg = "Invalid binary"; break;
  case CL_INVALID_BUILD_OPTIONS:           errMsg = "Invalid build options"; break;
  case CL_INVALID_PROGRAM:                 errMsg = "Invalid program"; break;
  case CL_INVALID_PROGRAM_EXECUTABLE:      errMsg = "Invalid program executable"; break;
  case CL_INVALID_KERNEL_NAME:             errMsg = "Invalid kernel name"; break;
  case CL_INVALID_KERNEL_DEFINITION:       errMsg = "Invalid kernel definition"; break;
  case CL_INVALID_KERNEL:                  errMsg = "Invalid kernel"; break;
  case CL_INVALID_ARG_INDEX:               errMsg = "Invalid argument index"; break;
  case CL_INVALID_ARG_VALUE:               errMsg = "Invalid argument value"; break;
  case CL_INVALID_ARG_SIZE:                errMsg = "Invalid argument size"; break;
  case CL_INVALID_KERNEL_ARGS:             errMsg = "Invalid kernel arguments"; break;
  case CL_INVALID_WORK_DIMENSION:          errMsg = "Invalid work dimension"; break;
  case CL_INVALID_WORK_GROUP_SIZE:         errMsg = "Invalid work group size"; break;
  case CL_INVALID_WORK_ITEM_SIZE:          errMsg = "Invalid work item size"; break;
  case CL_INVALID_GLOBAL_OFFSET:           errMsg = "Invalid global offset"; break;
  case CL_INVALID_EVENT_WAIT_LIST:         errMsg = "Invalid event wait list"; break;
  case CL_INVALID_EVENT:                   errMsg = "Invalid event"; break;
  case CL_INVALID_OPERATION:               errMsg = "Invalid operation"; break;
  case CL_INVALID_GL_OBJECT:               errMsg = "Invalid OpenGL object"; break;
  case CL_INVALID_BUFFER_SIZE:             errMsg = "Invalid buffer size"; break;
  case CL_INVALID_MIP_LEVEL:               errMsg = "Invalid mip-map level"; break;
  }

  return errMsg;
}

void ContextErrorCallback(const char *errinfo, const void *, size_t, void *) {
  fprintf(stderr, "Context error: %s\n", errinfo);
}

void PrintDeviceInfo(cl_device_id device_id) {

  size_t strLen;
  const size_t kStrBufSz = 1024;
  char strBuf[kStrBufSz];

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_NAME, kStrBufSz, strBuf, &strLen);
  std::cout << "Device name: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_PROFILE, kStrBufSz, strBuf, &strLen);
  std::cout << "Device profile: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_VENDOR, kStrBufSz, strBuf, &strLen);
  std::cout << "Device vendor: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_VERSION, kStrBufSz, strBuf, &strLen);
  std::cout << "Device version: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DRIVER_VERSION, kStrBufSz, strBuf, &strLen);
  std::cout << "Device driver version: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_ADDRESS_BITS, kStrBufSz, strBuf, &strLen);
  std::cout << "Device driver address bits: " << *((cl_uint *)strBuf) << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, kStrBufSz, strBuf, &strLen);
  std::cout << "Max work group size: " << *((size_t *)strBuf) << std::endl;
  assert(*((size_t *)strBuf) >= GetTotalWorkItems());

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, kStrBufSz, strBuf, &strLen);
  std::cout << "Max work item dimensions: " << *((cl_uint *)strBuf) << std::endl;
  assert(*((cl_uint *)strBuf) >= 2);

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, kStrBufSz, strBuf, &strLen);
  size_t nSizeElements = strLen / sizeof(size_t);
  std::cout << "Max work item sizes: (";
  size_t *dimSizes = (size_t *)(strBuf);
  for(size_t j = 0; j < nSizeElements; j++) {
    std::cout << dimSizes[j] << ((j == nSizeElements - 1)? "" : ", ");
    assert(j != 0 || dimSizes[j] > kLocalWorkSizeX);
    assert(j != 1 || dimSizes[j] > kLocalWorkSizeY);
  }
  std::cout << ")" << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_EXTENSIONS, kStrBufSz, strBuf, &strLen);
  std::cout << "Device extensions: " << std::endl;
  for (size_t k = 0; k < strLen; ++k) {
    if (strBuf[k] == ',' || strBuf[k] == ' ') {
      strBuf[k] = '\0';
    }
  }

  std::cout << "  " << strBuf << std::endl;
  for (size_t k = 1; k < strLen; ++k) {
    if (strBuf[k] == '\0' && k < strLen - 1) {
      std::cout << "  " << (strBuf + k + 1) << std::endl;
    }
  }

  cl_device_type deviceType;
  size_t deviceTypeSz;
  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, &deviceTypeSz);
  if(deviceType & CL_DEVICE_TYPE_CPU) {
    std::cout << "Device driver type: CPU" << std::endl;
  }
  if(deviceType & CL_DEVICE_TYPE_GPU) {
    std::cout << "Device driver type: GPU" << std::endl;
  }
  if(deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
    std::cout << "Device driver type: ACCELERATOR" << std::endl;
  }
  if(deviceType & CL_DEVICE_TYPE_DEFAULT) {
    std::cout << "Device driver type: DEFAULT" << std::endl;
  }
}

void InitializeOpenCLKernel() {
  size_t strLen;
  const cl_uint kMaxPlatforms = 8;
  cl_platform_id platforms[kMaxPlatforms];
  cl_uint nPlatforms;
  CHECK_CL(clGetPlatformIDs, kMaxPlatforms, platforms, &nPlatforms);

#ifndef NDEBUG
  fprintf(stdout, "\n");
  fprintf(stdout, "Found %d OpenCL platform%s.\n", nPlatforms, nPlatforms == 1? "" : "s");

  for(cl_uint i = 0; i < nPlatforms; i++) {
    char strBuf[256];

    fprintf(stdout, "\n");
    fprintf(stdout, "Platform %d info:\n", i);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_PROFILE, 256, strBuf, &strLen);
    fprintf(stdout, "Platform profile: %s\n", strBuf);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_VERSION, 256, strBuf, &strLen);
    fprintf(stdout, "Platform version: %s\n", strBuf);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_NAME, 256, strBuf, &strLen);
    fprintf(stdout, "Platform name: %s\n", strBuf);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_VENDOR, 256, strBuf, &strLen);
    fprintf(stdout, "Platform vendor: %s\n", strBuf);
  }
#endif

  cl_platform_id platform = platforms[0];
  if(nPlatforms > 1) {
    assert(!"FIXME - Choose a platform");
  }

  const cl_uint kMaxDevices = 8;
  cl_device_id devices[kMaxDevices];
  cl_uint nDevices;
  CHECK_CL(clGetDeviceIDs, platform, CL_DEVICE_TYPE_GPU, kMaxDevices, devices, &nDevices);

#ifndef NDEBUG
  fprintf(stdout, "\n");
  fprintf(stdout, "Found %d device%s on platform 0.\n", nDevices, nDevices == 1? "" : "s");

  for(cl_uint i = 0; i < nDevices; i++) {
    PrintDeviceInfo(devices[i]);
  }

  std::cout << std::endl;
#endif

  // Get current CGL Context and CGL Share group
  CGLContextObj kCGLContext = CGLGetCurrentContext();
  CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

  cl_context_properties properties[] = {
    CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
    (cl_context_properties)kCGLShareGroup, 0
  };

  cl_int errCreateContext;
  //  gContext = clCreateContext(properties, nDevices, devices,
  //                             ContextErrorCallback, NULL, &errCreateContext);   
  gContext = clCreateContext(properties, 0, 0,
                             ContextErrorCallback, NULL, &errCreateContext);   
  CHECK_CL((cl_int), errCreateContext);

  cl_device_id existing_ids[16];
  size_t nDeviceIds;
  CHECK_CL(clGetContextInfo, gContext, CL_CONTEXT_DEVICES,
           sizeof(existing_ids), existing_ids, &nDeviceIds);
  nDeviceIds /= sizeof(existing_ids[0]);

  // If we're sharing an openGL context, there should really only
  // be one device ID...
  assert (nDeviceIds == 1);

  std::cout << "Num devices after shared context: " << nDeviceIds << std::endl;
  for (size_t i = 0; i < nDeviceIds; ++i) {
    std::cout << "Device " << i << ": " << std::endl;
    PrintDeviceInfo(existing_ids[i]);
  }

  cl_device_id shared_device = existing_ids[0];
  
  // If the total local memory required is greater than the minimum specified.. check!
  cl_ulong totalLocalMemory;
  CHECK_CL(clGetDeviceInfo,
           shared_device,
           CL_DEVICE_LOCAL_MEM_SIZE,
           sizeof(cl_ulong),
           &totalLocalMemory,
           &strLen);
  assert(strLen == sizeof(cl_ulong));
  assert(totalLocalMemory >= 16384);
  while(GetTotalLocalMemory() > totalLocalMemory) {
    kLocalWorkSizeX >>= 1;
    kLocalWorkSizeY >>= 1;
  }

  std::ifstream progfs(OPENCL_KERNEL_PATH);
  std::string progStr((std::istreambuf_iterator<char>(progfs)),
                      std::istreambuf_iterator<char>());
  const char *progCStr = progStr.c_str();

  cl_int errCreateProgram;
  cl_program program;
  program = clCreateProgramWithSource(gContext, 1, &progCStr, NULL, &errCreateProgram);
  CHECK_CL((cl_int), errCreateProgram);

  if(clBuildProgram(program, 1, &shared_device, "", NULL, NULL) != CL_SUCCESS) {
    size_t bufferSz;
    clGetProgramBuildInfo(program, shared_device,
                          CL_PROGRAM_BUILD_LOG, sizeof(size_t), NULL, &bufferSz);
    char *buffer = new char[bufferSz + 1];
    buffer[bufferSz] = '\0';
    clGetProgramBuildInfo(program, shared_device,
                          CL_PROGRAM_BUILD_LOG, bufferSz, buffer, NULL);
    std::cerr << "CL Compilation failed:" << std::endl;
    std::cerr << buffer + 1 << std::endl;
    abort();
  }
  CHECK_CL(gUnloadCompilerFunc, platform);

  cl_int errCreateCommandQueue;
  gCommandQueue = clCreateCommandQueue(gContext, shared_device, 0, &errCreateCommandQueue);
  CHECK_CL((cl_int), errCreateCommandQueue);

  cl_int errCreateKernel;
  gKernel = clCreateKernel(program, "compressDXT", &errCreateKernel);
  CHECK_CL((cl_int), errCreateKernel);

  CHECK_CL(clReleaseProgram, program);
}

void DestroyOpenCLKernel() {
  CHECK_CL(clReleaseKernel, gKernel);

  CHECK_CL(clReleaseCommandQueue, gCommandQueue);
  CHECK_CL(clReleaseContext, gContext);
}

void RunKernel(unsigned char *data, GLuint pbo, int x, int y, int channels) {
  unsigned char *src_data = data;
  if (3 == channels) {
    unsigned char *new_data = (unsigned char *)malloc(x * y * 4);
    int nPixels = x * y;
    for (int i = 0; i < nPixels; ++i) {
      new_data[4*i] = src_data[3*i];
      new_data[4*i + 1] = src_data[3*i + 1];
      new_data[4*i + 2] = src_data[3*i + 2];
      new_data[4*i + 3] = 0xFF;
    }

    src_data = new_data;
  }

  cl_image_format fmt;
  fmt.image_channel_data_type = CL_UNSIGNED_INT8;
  fmt.image_channel_order = CL_RGBA;

  const size_t nBlocksX = ((x + 3) / 4);
  const size_t nBlocksY = ((y + 3) / 4);

  // Upload the data to the GPU...
  cl_int errCreateImage;
  cl_mem input = gCreateImage2DFunc(gContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, &fmt,
                                    x, y, 4*x, src_data, &errCreateImage);
  CHECK_CL((cl_int), errCreateImage);

  // Finished all OpenGL calls, now we can start making OpenCL calls...
  cl_int errCreateFromGL;
  cl_mem dxt_mem = clCreateFromGLBuffer(gContext, CL_MEM_WRITE_ONLY, pbo, &errCreateFromGL);
  CHECK_CL((cl_int), errCreateFromGL);

  // Acquire lock on GL objects...
  CHECK_CL(clEnqueueAcquireGLObjects, gCommandQueue, 1, &dxt_mem, 0, NULL, NULL);

  // Set the arguments
  CHECK_CL(clSetKernelArg, gKernel, 0, sizeof(input), &input);
  CHECK_CL(clSetKernelArg, gKernel, 1, sizeof(dxt_mem), &dxt_mem);

  size_t global_work_size[2] = { nBlocksX, nBlocksY };
  CHECK_CL(clEnqueueNDRangeKernel, gCommandQueue, gKernel, 2, NULL,
           global_work_size, 0, 0, NULL, NULL);

  // Release the GL objects
  CHECK_CL(clEnqueueReleaseGLObjects, gCommandQueue, 1, &dxt_mem, 0, NULL, NULL);
  CHECK_CL(clFinish, gCommandQueue);

  // Release the buffers
  CHECK_CL(clReleaseMemObject, input);

  if (data != src_data) {
    free(src_data);
  }
}
