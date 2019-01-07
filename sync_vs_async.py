import pyopencl as cl
import time
import numpy as np
import os

os.environ["PYOPENCL_CTX"] = "0"
ctx = cl.create_some_context()
print("Using ", ctx.devices[0].name)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

module = cl.Program(ctx, """
__kernel void fill(__global float *dest) {
  const int i = get_global_id(0);
  dest[i] = i+1;
}
""").build()
#fill = module.get_function("fill")

N = 32
local_size = (N, 1)
global_size = (N, 1)
cpu_data = np.empty(N, dtype=np.float32)
cpu_data.fill(-1)
gpu_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cpu_data)




iters = 10000
tic = time.time()
for i in range(iters):
    module.fill(queue, cpu_data.shape, None, gpu_data)
toc = time.time()
print("Elapsed: " + str(1e6*(toc-tic)/iters) + " microseconds per launch")

cpu_data_res = np.empty_like(cpu_data)
cl.enqueue_copy(queue, cpu_data_res, gpu_data)

print(cpu_data_res)

iters = 10000
tic = time.time()
for i in range(iters):
    module.fill(queue, cpu_data.shape, None, gpu_data)
toc = time.time()
print("Elapsed: " + str(1e6*(toc-tic)/iters) + " microseconds per launch")

cpu_data_res2 = np.empty_like(cpu_data)
cl.enqueue_copy(queue, cpu_data_res2, gpu_data)

print(cpu_data_res2)
