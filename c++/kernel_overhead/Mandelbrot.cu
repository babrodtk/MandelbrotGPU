#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cassert>

#ifdef _WIN32
#include <sys/timeb.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "TGA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define CUDA_SAFE_CALL(ans) { cudaSafeCall((ans), __FILE__, __LINE__); }
inline void cudaSafeCall(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "Safe call on" << file << ":" << line << " failed " << cudaGetErrorString(code);
        exit(code);
    }
}

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

__global__ void mandelbrotKernel(float* output, unsigned int pitch, 
            unsigned int nx, unsigned int ny, 
            unsigned int iterations, 
            float x0, float y0, 
            float dx, float dy) {

    //Get thread id of this thread
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    //Check for out of bounds
    if (i < nx && j < ny) {
        float x = i*dx + x0;
        float y = j*dy + y0;

        float2 z0 = make_float2(x, y);
        float2 z = z0;
        int k = 0;

        //Loop until iterations or until it diverges
        while (z.x*z.x + z.y*z.y < 25.0 && k < iterations) {
            float tmp = z.x*z.x - z.y*z.y + z0.x;
            z.y = 2 * z.x*z.y + z0.y;
            z.x = tmp;
            ++k;
        }

        //Write out result to GPU memory
        if (k < iterations) {
            float* row = (float*)((char*)output + j*pitch);
            row[i] = fmod((k - log(log(sqrt(z.x*z.x + z.y*z.y)) / log(5.0)) / log(2.0)) / 100, 1.0);
        }
        else {
            float* row = (float*)((char*)output + j*pitch);
            row[i] = 0.0f;
        }
    }
}

std::vector<float*> mandelbrot(
    unsigned int nx, unsigned int ny, unsigned int iterations,
    std::vector<float> x0, std::vector<float> y0,
    std::vector<float> dx, std::vector<float> dy,
    size_t block_width = 8, size_t block_height = 8) {
    int num_zooms = x0.size();
    assert(num_zooms == x0.size());
    assert(num_zooms == y0.size());
    assert(num_zooms == dx.size());
    assert(num_zooms == dy.size());

    //Create block dimensions and grid dimensions
    dim3 block = dim3(block_width, block_height, 1);
    dim3 grid = dim3((nx + block_width - 1) / block_width, (ny + block_height - 1) / block_height);

    //Allocate GPU data 
    std::vector<float*> output_gpu(num_zooms);
    std::vector<size_t> pitch(num_zooms);
    for (int i = 0; i < num_zooms; ++i) {
        CUDA_SAFE_CALL(cudaMallocPitch<float>(&output_gpu[i], &pitch[i], nx*sizeof(float), ny));
    }

    //Create stream
    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));

    //Create timing events
    std::vector<cudaEvent_t> start_events(num_zooms);
    std::vector<cudaEvent_t> stop_events(num_zooms);
    for (int i = 0; i < num_zooms; ++i) {
        CUDA_SAFE_CALL(cudaEventCreate(&start_events[i]));
        CUDA_SAFE_CALL(cudaEventCreate(&stop_events[i]));
    }

    //Run kernel and generate images
    double enqueue_compute_start = getCurrentTime();
    for (int i = 0; i < num_zooms; ++i) {
        //Launch kernel
        CUDA_SAFE_CALL(cudaEventRecord(start_events[i]));
        mandelbrotKernel <<<grid, block, 0, stream>>>(output_gpu[i], pitch[i],
            nx, ny, iterations,
            x0[i], y0[i],
            dx[i], dy[i]);
        CUDA_SAFE_CALL(cudaEventRecord(stop_events[i]));
    }
    double enqueue_compute_end = getCurrentTime();

    //Synchronize
    double sync_compute_start = getCurrentTime();
    double gpu_time_compute = 0.0;
    for (int i = 0; i < num_zooms; ++i) {
        CUDA_SAFE_CALL(cudaEventSynchronize(stop_events[i]));
        float milliseconds = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]));
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
    std::vector<float*> retval(num_zooms);
    for (int i = 0; i < num_zooms; ++i) {
        CUDA_SAFE_CALL(cudaMallocHost(&retval[i], nx*ny*sizeof(float)));
    }

    //Download from GPU to CPU
    double enqueue_dl_start = getCurrentTime();
    for (int i = 0; i < num_zooms; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(start_events[i]));
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&retval[i][0], nx * sizeof(float),
            output_gpu[i], pitch[i],
            nx * sizeof(float), ny,
            cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaEventRecord(stop_events[i]));
    }
    double enqueue_dl_end = getCurrentTime();

    //Synchronize
    double sync_dl_start = getCurrentTime();
    double gpu_time_dl = 0.0;
    for (int i = 0; i < num_zooms; ++i) {
        CUDA_SAFE_CALL(cudaEventSynchronize(stop_events[i]));
        float milliseconds = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]));
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
    const int iterations = 5000;
    const int num_zooms = 10;

    const int x_center = -0.75 + 0.0025;
    const int y_center = 0.1;
    const int factor = 0.95;

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

    // Choose which GPU to run on, change this on a multi-GPU system.
    CUDA_SAFE_CALL(cudaSetDevice(0));

    std::vector<float*> result = mandelbrot(nx, ny, iterations, x0, y0, dx, dy);

    for (int i = 0; i < result.size(); ++i) {
        std::stringstream filename;
        filename << "mandelbrot_" << i << ".tga";
        std::cout << "Writing to " << filename.str() << std::endl;
        toTGA(result[i], nx, ny, filename.str());
        CUDA_SAFE_CALL(cudaFreeHost(result[i]));
    }


    CUDA_SAFE_CALL(cudaDeviceReset());

    return 0;
}

