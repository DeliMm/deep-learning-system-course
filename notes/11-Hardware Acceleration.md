# Hardware Acceleration 

## General acceleration techniques 

### Vectorization 

Adding two arrays of length 256 

```c++
void vecadd(float *A, float *B, float *C){
    for (int i = 0; i < 64; ++ i){
        float4 a = load_float4(A + i * 4);
        float4 b = load_float4(B + i * 4);
        float4 c = add_float4(a, b);
        store_float(C + i * 4, c);
    }
}
```
Additional requirements: memory(A, B, C) needs to be aligned to 128 bits.


how to store a matrix in memory:

- Row major: A[i, j] => Adata[i * A.shape[1] + j]
- Column major: A[i, j] => Adata[i * A.shape[0] + i]
- Strides format: A[i, j] => Adata[i * strides[0] + j * strides[1]]

Advantages of Strides format: can perform transformation/slicing in zero copy way 

- Slice: change the begin offset and shape 
- Transpose: swap the strides 
- Boradcast: insert a stride equals 0 

Disadvantages of Stride format: memory access becomes not continuous 

- Makes vectorization harder 
- Many linear algebra opreations may require compact the array first


### Parallelization 

Executes the computation on multiple threads

```c++
void vecadd(float *A, float *B, float *C){
    for (int i = 0; i < 64; ++ i){
        float4 a = load_float4(A + i * 4);
        float4 b = load_float4(B + i * 4);
        float4 c = add_float4(a, b);
        store_float(C + i * 4, c);
    }
}
```
## Case study: matrix multiplication

```c++
dram float A[n][n], B[n][n], C[n][n];
for (int i = 0; i < n; ++ i){
    for (int j = 0; j < n;  ++ j){
        register float c = 0;
        for (int k = 0; k < n; ++ k){
            register float a = A[i][k];
            register float b = B[j][k];
            c += a * b;
        }
        C[i][j] = c;
    }
}
```
Load cost: 2 * dramspeed * n^3, Register cost: 3

### Register tiled matrix multiplication

```c++
dram float A[n/v1][n/v3][v1][v3];
dram float B[n/v2][n/v3][v2][v3];
dram float C[n/v1][n/v2][v1][v2];

for (int i = 0; i < n/v1; ++ i){
    for (int j = 0; j < n/v2;  ++ j){
        register float c[v1][v2] = 0;
        for (int k = 0; k < n/v3; ++ k){
            register float a[v1][v3] = A[i][k];
            register float b[v2][v3] = B[j][k];
            c += dot(a, b.T);
        }
        C[i][j] = c;
    }
}
```
load cost: dramspeed * (n^3 / v2 + n^3 / v1), Register cost: v1 * v3 + v2 * v3 + v1 * v2

### Cache line aware tiling 

```c++
dram float A[n/b1][b1][n];
dram float B[n/b2][b2][n];
dram float C[n/b1][n/b2][b1][b2];

for (int i = 0; i < n/b1; ++ i){
    l1cache float a[b1][n] = A[i];
    for (int j = 0; j < n/b2;  ++ j){
        l1cache b[b2][n] = B[j];
        
        C[i][j] = dot(a, b.T);
    }
}
```
Common reuse patterns: `c[i][j] = sum(A[i][k] * B[j][k], axis=k)`

Acces of A is independent of j, tile the j dimension by v enables reuse of A for v times.