#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cassert>
#include <boost/format.hpp>

#ifdef _WIN32
#include <sys/timeb.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "TGA.h"
#include "oclutils.h"

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
        output_gpu[i] = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE,
            nx*ny*sizeof(float), NULL, &error);
        CL_CHECK(error);
    }

    //Create timing events
    std::vector<cl::Event> events(num_zooms);

    //Run kernel and generate images
    cl::Kernel *kernel = OpenCLUtils::getKernel("mandelbrotKernel");
    double enqueue_compute_start = getCurrentTime();

    for (int i = 0; i < num_zooms; ++i) {
        kernel->setArg<cl::Buffer>(0, output_gpu[i]);
        kernel->setArg<unsigned int>(1, (unsigned int) (nx*sizeof(float)));
        kernel->setArg<unsigned int>(2, nx);
        kernel->setArg<unsigned int>(3, ny);
        kernel->setArg<unsigned int>(4, iterations);
        kernel->setArg<float>(5, x0[i]);
        kernel->setArg<float>(6, y0[i]);
        kernel->setArg<float>(7, dx[i]);
        kernel->setArg<float>(8, dy[i]);

        // execute kernel
        CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, 
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
        CL_CHECK(OpenCLUtils::elapsedMilliseconds(events[i]));
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
        CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(output_gpu[i], CL_TRUE, 0, sizeof(float) * nx * ny, retval[i].data(), 0, &events[i]));
    }
    double enqueue_dl_end = getCurrentTime();

    //Synchronize
    double sync_dl_start = getCurrentTime();
    double gpu_time_dl = 0.0;
    for (int i = 0; i < num_zooms; ++i) {
        CL_CHECK(events[i].wait());
        float milliseconds = 0;
        CL_CHECK(OpenCLUtils::elapsedMilliseconds(events[i]));
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
    const int iterations = 1;

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
    std::vector<std::pair<std::string, std::string> > sources;
    sources.push_back(std::make_pair("mandelbrotKernel", "Mandelbrot.cl"));
    OpenCLUtils::init(
                sources, CL_DEVICE_TYPE_GPU,
                (boost::format("-I %s") % ".").str());

    std::vector<std::vector<float> > result = mandelbrot(nx, ny, iterations, x0, y0, dx, dy);

    for (int i = 0; i < result.size(); ++i) {
        std::stringstream filename;
        filename << "mandelbrot_" << i << ".tga";
        std::cout << "Writing to " << filename.str() << std::endl;
        toTGA(result[i].data(), nx, ny, filename.str());
    }

    return 0;
}

