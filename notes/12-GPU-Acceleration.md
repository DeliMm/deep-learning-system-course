# GPU Acceleration

## GPU programming 

GPU have massive parallel computing units. It's programming mode is **SIMT**:

- Single instruction multiple threads 
- All threads executes the same code, but can take different path 
- Threads are grouped into blocks, Thread within the same block have shared memory 
- Blocks are grouped into a launch grid 
- A kernel executes a grid 

Example: Vector add 

```c++
void VecaddCPU(float *A, float *B, float *C, int n){
    for (int i = 0; i < n; i ++){
        C[i] = A[i] + B[i];
    }
}

__global__ void VecaddKernel(float *A, float *B, float *C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}
```
Vctor and host side

```c++
__global__ void VecaddKernel(float *A, float *B, float *C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

void VecAddCUDA(float *Acpu, float *Bcpu, float *Ccpu, int n){
    float *dA, *dB, *dC;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));
    cudaMemcpy(dA, Acpu, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bcpu, n * sizeof(float), cudaMemcpyHostToDevice);
    int thredas_per_block = 512;
    int nblocks = (n + threads_pre_block - 1) / threads_per_block;
    VecAddKernel<<<nblocks, thread_per_block>>>(dA, dB, dC, n);
    CudaMemcpy(Ccpu, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
```
real applications usually **keep data in gpu memory as long as possible**

### Example: window sum 
```c++
#define radius 2 
__global__ void WindowSumSimpleKernel(float *A, float *B, int n){
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (out_idx < n){
        float sum = 0;
        for (int dx = -RADIUS; dx <= RADIUS; ++dx){
            sum += A[dx + out_idx + RADIUS];
        }
        B[out_idx] = sum;
    }
}
```

use thread block of size 4 to cooperatively fetch the data to shared memory. 

```c++
__global__ void WindowSumShareKernel(float *A, float *B, int n){
    __shared__ float temp[ThREADS_PER_BLOCK + 2 * RADIUS];
    int base = blockDim.x * blockIdx.x;
    int out_idx = base  + threadIdx.x;
    if (base + threadIdx.x < n){
        temp[threadIdx.x] = A[base + threadIdx.x];
    }
    if (threadIdx.x < 2 * RADIUS && base + THREADS_PER_BLOCK + threadIdx.x < n){
        temp[theradIdx.x + THREADS_PER_BLOCK] = A[base + THREADS_PER_BLOCK + threadIdx.x];
    }
    __syncthreds();
    if (out_idx < n){
        float sum = 0;
        for (int dx = -RADIUS; dx <= RADIUS;  ++dx){
            sum += temp[threadIdx.x + dx + RADIUS];
        }
        B[out_idx] = sum;
    }
}
```

High level takeways

- Launch thread grid and blocks
- Cooperatively fetch common to shared memory to increase reuse

## Case study: matrix multiplication on GPU

### Thread-level: register tiling

```c++
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]){
    int ybase = blockIdx.y * blockDim.y + threadIdx.y;
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;

    float c[V][V] = {0};
    float a[V], b[V];
    for (int k = 0; k < N; ++k){
        a[:] = A[k, ybase*V : ybase*V + V];
        b[:] = B[k, xbase*V : xbase*V + v];
        for (int y = 0; y < V; ++y)
            for (int x = 0; x < V; ++x){
                c[y][x] += a[y] * b[x];
        }
        C[ybase * V: ybase*V + V, xbase*V, xbase*V + V] = c[:];
    }
}
```

### Block-level: shared memory tiling

```c++
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]){
    __shared__ float sA[S][L], sB[S][L];
    float a[V], b[V];
    int yblock = blockIdx.x;
    int xblock = blockIdx.y;

    for (int k0 = 0; k0< N;k0 += s){
        __syncthreads();
        // needs to be implemented by thread cooperative fetching
        sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
        sB[:, :] = B[k: k + S, xblock * L: xblock * L + L];
        __syncthreads();
        for (int ki = 0; ki < S; ++ki){
            a[:] = sA[ki, threadIdx.y * V: threadIdx.y * V + V];
            b[:] = sB[ki, threadIdx.x * V: threadIdx.y * V + V];
            for (int y = 0; y < V; ++y)
                for (int x = 0; x < V; ++x)
                    c[y][x] += a[y] * b[x];
        }
    }
    int xbase = blockIdx.y * blockDim.y + threadIdx.y;
    int ybase = blockIdx.x * blockDim.x + threadIdx.x;
    C[ybase*V: ybase*V + V, xbase*V: xbase*V + V] = c[:];
}
```

More GPU optimization techniques:

- Global memroy continuous read
- shared memroy bank conflict 
- software pipelining
- Warp level optimizations
- Tensor Core
