/* vecadd.cu — trivial vector addition kernel for the ufo-cuda example.
 *
 * extern "C" keeps the symbol unmangled so that
 * (cuda-module-get-function mod "vecadd") finds it.
 */
extern "C" __global__ void vecadd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}
