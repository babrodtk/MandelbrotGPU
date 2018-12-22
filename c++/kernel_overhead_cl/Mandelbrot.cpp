#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <boost/format.hpp>

#include <CL/cl2.hpp>

#ifdef _WIN32
#include <sys/timeb.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "TGA.h"

#define CL_CHECK(_expr)                                                          \
    do {                                                                         \
        cl_int _err = _expr;                                                     \
        if (_err == CL_SUCCESS)                                                  \
            break;                                                               \
        fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
        abort();                                                                 \
    } while (0)

inline double getCurrentTime() {
#ifdef WIN32
    LARGE_INTEGER f;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return t.QuadPart / (double)f.QuadPart;
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec + tv.tv_usec*1e-6;
#endif
};

static std::vector<cl::Device> devices;
static cl::Context *context;
static cl::CommandQueue *queue;

static int plIndex = 0; // use first platform
static int devIndex = 0; // use first GPU device

static const char mandelbrotKernelSource[] =
    "__kernel void mandelbrotKernel(__global float* output, unsigned int pitch, \n"
    "            unsigned int nx, unsigned int ny, \n"
    "            unsigned int iterations, \n"
    "            float x0, float y0, \n"
    "            float dx, float dy) {\n"
    "\n"
    "    //Get thread id of this thread\n"
    "    int i = get_global_id(0);\n"
    "    int j = get_global_id(1);\n"
    "\n"
    "    //Check for out of bounds\n"
    "    if (i < nx && j < ny) {\n"
    "        float x = i*dx + x0;\n"
    "        float y = j*dy + y0;\n"
    "\n"
    "        float2 z0 = (float2)(x, y);\n"
    "        float2 z = z0;\n"
    "        int k = 0;\n"
    "\n"
    "        //Loop until iterations or until it diverges\n"
    "        while (z.x*z.x + z.y*z.y < 25.0 && k < iterations) {\n"
    "            float tmp = z.x*z.x - z.y*z.y + z0.x;\n"
    "            z.y = 2 * z.x*z.y + z0.y;\n"
    "            z.x = tmp;\n"
    "           ++k;\n"
    "       }\n"
    "\n"
    "        //Write out result to GPU memory\n"
    "        if (k < iterations) {\n"
    "            __global float* row = (__global float*)((__global char*) output + j*pitch);\n"
    "            row[i] = fmod((k - log(log(sqrt(z.x*z.x + z.y*z.y)) / log(5.0)) / log(2.0)) / 100, 1.0);\n"
    "        }\n"
    "        else {\n"
    "            __global float* row = (__global float*)((__global char*) output + j*pitch);\n"
    "            row[i] = 0.0f;\n"
    "        }\n"
    "    }\n"
    "}\n\0";

std::vector<std::vector<float> > mandelbrot(
    unsigned int nx, unsigned int ny, unsigned int iterations,
    std::vector<float> x0, std::vector<float> y0,
    std::vector<float> dx, std::vector<float> dy,
    size_t block_width = 8, size_t block_height = 8) {
    cl_int error = CL_SUCCESS;
    int num_zooms = x0.size();
    assert(num_zooms == x0.size());
    assert(num_zooms == y0.size());
    assert(num_zooms == dx.size());
    assert(num_zooms == dy.size());

    //Allocate GPU data 
    std::vector<cl::Buffer> output_gpu(num_zooms);
    for (int i = 0; i < num_zooms; ++i) {
        output_gpu[i] = cl::Buffer(*context, CL_MEM_READ_WRITE,
            nx*ny*sizeof(float), NULL, &error);
        CL_CHECK(error);
    }

    // Create stream
    queue = new cl::CommandQueue(*context, devices[devIndex], CL_QUEUE_PROFILING_ENABLE, &error);
    CL_CHECK(error);

    //Create timing events
    std::vector<cl::Event> events(num_zooms);

    // create program
    cl::Program::Sources sources;
    sources.push_back(mandelbrotKernelSource);
    cl::Program program(*context, sources, &error);
    CL_CHECK(error);

    // compile program
    error = program.build(devices, 0, 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        throw std::runtime_error("Program build failed");
    }
    CL_CHECK(error);

    // create kernels
    cl::Kernel kernel(program, "mandelbrotKernel", &error);
    CL_CHECK(error);

    //Run kernel and generate images
    double enqueue_compute_start = getCurrentTime();

    for (int i = 0; i < num_zooms; ++i) {
        kernel.setArg<cl::Buffer>(0, output_gpu[i]);
        kernel.setArg<unsigned int>(1, (unsigned int) (nx*sizeof(float)));
        kernel.setArg<unsigned int>(2, nx);
        kernel.setArg<unsigned int>(3, ny);
        kernel.setArg<unsigned int>(4, iterations);
        kernel.setArg<float>(5, x0[i]);
        kernel.setArg<float>(6, y0[i]);
        kernel.setArg<float>(7, dx[i]);
        kernel.setArg<float>(8, dy[i]);

        // execute kernel
        CL_CHECK(queue->enqueueNDRangeKernel(
                 kernel, cl::NullRange, 
                 cl::NDRange(((nx + block_width - 1)/block_width)*block_width, ((ny + block_height - 1)/block_height)*block_height), 
                 cl::NDRange(block_width, block_height), 0, &events[i]));
    }
    double enqueue_compute_end = getCurrentTime();

    //Synchronize
    double sync_compute_start = getCurrentTime();
    double gpu_time_compute = 0.0;
    for (int i = 0; i < num_zooms; ++i) {
        CL_CHECK(events[i].wait());
        float milliseconds = 0;
        milliseconds = (events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>() - events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>()) * .000001;
        std::cout << "Iteration " << i << " took " << milliseconds << " ms" << std::endl;
        gpu_time_compute += milliseconds;
    }
    double sync_compute_end = getCurrentTime();
    std::cout << "Compute" << std::endl;
    std::cout << "Enqueue:  " << (enqueue_compute_end - enqueue_compute_start) << " s" << std::endl;
    std::cout << "Sync:     " << (sync_compute_end - sync_compute_start) << " s" << std::endl;
    std::cout << "CPU time: " << (enqueue_compute_end + sync_compute_end - enqueue_compute_start - sync_compute_start) << " s" << std::endl;
    std::cout << "GPU time: " << gpu_time_compute * 1.0e-3 << " s" << std::endl;

    //Allocate CPU data 
    std::vector<std::vector<float> > retval(num_zooms);
    for (int i = 0; i < num_zooms; ++i) {
        retval[i].resize(nx * ny);
    }

    //Download from GPU to CPU
    double enqueue_dl_start = getCurrentTime();
    for (int i = 0; i < num_zooms; ++i) {
        CL_CHECK(queue->enqueueReadBuffer(output_gpu[i], CL_TRUE, 0, sizeof(float) * nx * ny, retval[i].data(), 0, &events[i]));
    }
    double enqueue_dl_end = getCurrentTime();

    //Synchronize
    double sync_dl_start = getCurrentTime();
    double gpu_time_dl = 0.0;
    for (int i = 0; i < num_zooms; ++i) {
        CL_CHECK(events[i].wait());
        float milliseconds = 0;
        milliseconds = milliseconds = (events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>() - events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>()) * .000001;
        std::cout << "Iteration " << i << " took " << milliseconds << " ms" << std::endl;
        gpu_time_dl += milliseconds;
    }
    double sync_dl_end = getCurrentTime();
    std::cout << "Download" << std::endl;
    std::cout << "Enqueue:  " << (enqueue_dl_end - enqueue_dl_start) << " s" << std::endl;
    std::cout << "Sync:     " << (sync_dl_end - sync_dl_start) << " s" << std::endl;
    std::cout << "CPU time: " << (enqueue_dl_end + sync_dl_end - enqueue_dl_start - sync_dl_start) << " s" << std::endl;
    std::cout << "GPU time: " << gpu_time_dl * 1.0e-3 << " s" << std::endl;

    std::cout << "========" << std::endl;
    std::cout << "Averages" << std::endl;
    std::cout << "Enqueue compute:  " << (1.0e3*(enqueue_compute_end - enqueue_compute_start) / num_zooms) << " ms" << std::endl;
    std::cout << "Enqueue download: " << (1.0e3*(enqueue_dl_end - enqueue_dl_start) / num_zooms) << " ms" << std::endl;
    std::cout << "Kernel:           " << (gpu_time_compute / num_zooms) << " ms" << std::endl;
    std::cout << "Download:         " << (gpu_time_dl / num_zooms) << " ms" << std::endl;
    std::cout << "========" << std::endl;

    return retval;
}


int main(int argc, char* argv[]) {
    const int n = 1024;
    const int nx = 3*n;
    const int ny = 2*n;
    const int iterations = 1000;

    //Set zoom parameters
    const double x_center = -0.75 + 0.0025;
    const double y_center = 0.1;
    const double factor = 0.95;
    const int num_zooms = 50;

    //Generate zoom locations
    std::vector<float> x0({ (float) (x_center - 1.5) });
    std::vector<float> y0({ (float) (y_center - 1.0) });
    std::vector<float> dx({ (float) (3.0 / double(nx)) });
    std::vector<float> dy({ (float) (2.0 / double(ny)) });

    for (int i = 1; i < num_zooms; ++i) {
        const double new_dx = dx.back() * factor;
        const double new_dy = dy.back() * factor;
        dx.push_back(new_dx);
        dy.push_back(new_dy);

        x0.push_back(x_center - new_dx*nx/2.0);
        y0.push_back(y_center - new_dy*ny/2.0);

        std::cout << new_dx*nx << "x" << new_dy*ny << std::endl;
    }

    // init OpenCL
    cl_int error = CL_SUCCESS;
    std::vector<cl::Platform> platforms;

    CL_CHECK(cl::Platform::get(&platforms));
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platform found");

    platforms[plIndex].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        throw std::runtime_error("No OpenCL GPU device found");

    // create context
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[plIndex])(),
        0
    };
    context = new cl::Context(devices, contextProperties, NULL, 0, &error);
    CL_CHECK(error);

    std::vector<std::vector<float> > result = mandelbrot(nx, ny, iterations, x0, y0, dx, dy);

    for (int i = 0; i < result.size(); ++i) {
        std::stringstream filename;
        filename << "mandelbrot_" << i << ".tga";
        std::cout << "Writing to " << filename.str() << std::endl;
        toTGA(result[i].data(), nx, ny, filename.str());
    }

    return 0;
}

