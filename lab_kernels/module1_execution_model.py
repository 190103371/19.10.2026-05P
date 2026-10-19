import numpy as np
from numba import cuda


@cuda.jit
def _vector_add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


def vector_add_1d(a, b, out):
    n = out.size
    if n:
        _vector_add_kernel[(n + 255) // 256, 256](a, b, out)


@cuda.jit
def _vector_scale_kernel(x, alpha, out):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.size:
        out[i] = x[i] * alpha
        i += stride


def vector_scale_grid_stride(x, alpha, out):
    if x.size:
        _vector_scale_kernel[64, 256](x, alpha, out)


@cuda.jit
def _matrix_add_kernel(A, B, C):
    col, row = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


def matrix_add_2d(A, B, C):
    rows, cols = C.shape
    _matrix_add_kernel[
        ((cols + 15) // 16, (rows + 15) // 16), (16, 16)
    ](A, B, C)


@cuda.jit
def _matrix_transpose_kernel(A, B):
    col, row = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        B[col, row] = A[row, col]


def matrix_transpose_naive(A, B):
    rows, cols = A.shape
    _matrix_transpose_kernel[
        ((cols + 15) // 16, (rows + 15) // 16), (16, 16)
    ](A, B)


@cuda.jit
def _volume_init_kernel(vol):
    x, y, z = cuda.grid(3)
    if z < vol.shape[0] and y < vol.shape[1] and x < vol.shape[2]:
        vol[z, y, x] = z * 10000 + y * 100 + x


def volume_init_3d(vol, D, H, W):
    _volume_init_kernel[
        ((W + 7) // 8, (H + 7) // 8, (D + 7) // 8), (8, 8, 8)
    ](vol)


@cuda.jit
def _rgb_to_grayscale_kernel(img_rgb, img_gray):
    x, y = cuda.grid(2)
    if y < img_gray.shape[0] and x < img_gray.shape[1]:
        r = img_rgb[y, x, 0]
        g = img_rgb[y, x, 1]
        b = img_rgb[y, x, 2]
        img_gray[y, x] = 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_grayscale(img_rgb, img_gray):
    h, w = img_gray.shape
    _rgb_to_grayscale_kernel[
        ((w + 15) // 16, (h + 15) // 16), (16, 16)
    ](img_rgb, img_gray)


@cuda.jit
def _vector_lerp_kernel(a, b, t, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (1.0 - t[i]) * a[i] + t[i] * b[i]


def vector_lerp(a, b, t, out):
    if out.size:
        _vector_lerp_kernel[(out.size + 255) // 256, 256](a, b, t, out)


@cuda.jit
def _vector_threshold_kernel(x, thresh, mask):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.size:
        mask[i] = 1 if x[i] >= thresh else 0
        i += stride


def vector_threshold(x, thresh, mask):
    if x.size:
        _vector_threshold_kernel[64, 256](x, thresh, mask)
