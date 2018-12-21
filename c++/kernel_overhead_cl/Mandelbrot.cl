__kernel void mandelbrotKernel(__global float* output, unsigned int pitch, 
            unsigned int nx, unsigned int ny, 
            unsigned int iterations, 
            float x0, float y0, 
            float dx, float dy) {

    //Get thread id of this thread
    int i = get_global_id(0);
    int j = get_global_id(1);

    //Check for out of bounds
    if (i < nx && j < ny) {
        float x = i*dx + x0;
        float y = j*dy + y0;

        float2 z0 = (float2)(x, y);
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
            __global float* row = (__global float*)((__global char*) output + j*pitch);
            row[i] = fmod((k - log(log(sqrt(z.x*z.x + z.y*z.y)) / log(5.0)) / log(2.0)) / 100, 1.0);
        }
        else {
            __global float* row = (__global float*)((__global char*) output + j*pitch);
            row[i] = 0.0f;
        }
    }
}
