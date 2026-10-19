import numpy as np
from numba import cuda


# Task 9: Strided memory access
@cuda.jit
def _copy_stride(x, out, stride):
    i = cuda.grid(1)
    if i < out.size:
        j = i * stride
        if j < x.size:
            out[i] = x[j]


def benchmark_strided_access(in_arr, out_arr, stride):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if out_arr.size:
        _copy_stride[(out_arr.size + 255) // 256, 256](
            in_arr, out_arr, stride
        )


# Task 10: Moving average with clamped boundaries
@cuda.jit
def _stencil(x, out, radius):
    i = cuda.grid(1)
    if i < x.size:
        total = 0.0
        for k in range(-radius, radius + 1):
            j = i + k
            if j < 0:
                j = 0
            if j >= x.size:
                j = x.size - 1
            total += x[j]
        out[i] = total / (2 * radius + 1)


def stencil_1d_shared(in_arr, out_arr, radius):
    if in_arr.size:
        _stencil[(in_arr.size + 255) // 256, 256](
            in_arr, out_arr, radius
        )


# Task 11: 1D convolution, same output length
@cuda.jit
def _conv(x, kernel, out):
    i = cuda.grid(1)
    if i < out.size:
        total = 0.0
        radius = kernel.size // 2
        for k in range(kernel.size):
            j = i + radius - k
            if 0 <= j < x.size:
                total += x[j] * kernel[k]
        out[i] = total


def conv_1d_shared_tiled(x, kernel, y):
    if y.size:
        _conv[(y.size + 255) // 256, 256](x, kernel, y)


# Task 12: Tiled matrix multiplication
@cuda.jit
def _matmul(A, B, C):
    tile_a = cuda.shared.array((16, 16), dtype=np.float32)
    tile_b = cuda.shared.array((16, 16), dtype=np.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * 16 + ty
    col = cuda.blockIdx.x * 16 + tx
    acc = 0.0

    for phase in range((A.shape[1] + 15) // 16):
        ak = phase * 16 + tx
        bk = phase * 16 + ty

        tile_a[ty, tx] = (
            A[row, ak]
            if row < A.shape[0] and ak < A.shape[1]
            else 0.0
        )
        tile_b[ty, tx] = (
            B[bk, col]
            if bk < B.shape[0] and col < B.shape[1]
            else 0.0
        )
        cuda.syncthreads()

        for k in range(16):
            acc += tile_a[ty, k] * tile_b[k, tx]
        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc


def matmul_tiled_shared(A, B, C):
    _matmul[
        ((C.shape[1] + 15) // 16, (C.shape[0] + 15) // 16),
        (16, 16)
    ](A, B, C)


# Task 13: Shared-memory transpose with padding
@cuda.jit
def _transpose(A, B):
    tile = cuda.shared.array((32, 33), dtype=np.float32)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if y < A.shape[0] and x < A.shape[1]:
        tile[ty, tx] = A[y, x]
    else:
        tile[ty, tx] = 0.0
    cuda.syncthreads()

    out_x = cuda.blockIdx.y * 32 + tx
    out_y = cuda.blockIdx.x * 32 + ty
    if out_y < B.shape[0] and out_x < B.shape[1]:
        B[out_y, out_x] = tile[tx, ty]


def transpose_bank_conflict_free(A, B):
    _transpose[
        ((A.shape[1] + 31) // 32, (A.shape[0] + 31) // 32),
        (32, 32)
    ](A, B)


# Task 14: Block-level sum reduction
@cuda.jit
def _reduce_sum(x, block_sums):
    shared = cuda.shared.array(256, dtype=np.float32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + tid

    shared[tid] = x[i] if i < x.size else 0.0
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        block_sums[cuda.blockIdx.x] = shared[0]


def reduction_sum_shared(in_arr, block_sums):
    blocks = (in_arr.size + 255) // 256
    if blocks:
        _reduce_sum[blocks, 256](in_arr, block_sums)


# Task 15: Block-level argmax
@cuda.jit
def _argmax(x, max_values, max_indices):
    vals = cuda.shared.array(256, dtype=np.float32)
    inds = cuda.shared.array(256, dtype=np.int64)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + tid

    if i < x.size:
        vals[tid] = x[i]
        inds[tid] = i
    else:
        vals[tid] = -np.inf
        inds[tid] = -1
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            if vals[tid + stride] > vals[tid] or (
                vals[tid + stride] == vals[tid]
                and inds[tid + stride] >= 0
                and (inds[tid] < 0 or inds[tid + stride] < inds[tid])
            ):
                vals[tid] = vals[tid + stride]
                inds[tid] = inds[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        max_values[cuda.blockIdx.x] = vals[0]
        max_indices[cuda.blockIdx.x] = inds[0]



def reduction_argmax_shared(in_arr, block_max_vals, block_max_indices):
    blocks = (in_arr.size + 255) // 256
    if blocks == 0:
        return

    _argmax[blocks, 256](
        in_arr, block_max_vals, block_max_indices
    )
