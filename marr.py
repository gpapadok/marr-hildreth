#!/usr/bin/python3
import cv2
import numpy as np


laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    The convolution operation.
    First it unravels each block of the image and stacks it in a (N*M, Kn*Km)
    matrix (where NxM and KnxKm are the dimensions of the image and kernel
    respectively). Then multiplies it with the kernel unraveled into a vector.
    This method is much more efficient that naively iterating through each block
    of the image.
    Examples:
    >>> a = np.array([[0,0,255],[0,0,255],[0,0,255]])
    >>> convolve(a, a)
    array([[     0., 130050.,      0.],
           [     0., 195075.,      0.],
           [     0., 130050.,      0.]])
    """
    krows, kcols = kernel.shape
    rows, cols = image.shape
   
    padded = np.pad(image, (krows//2, kcols//2), mode='constant')
    stacked = np.array([
        np.ravel(padded[i:i+krows, j:j+kcols])
        for i in range(rows) for j in range(cols)
    ], dtype=np.float32)

    conv = (stacked @ np.ravel(kernel)).reshape(rows, cols)
    return conv


def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Constructs a Gaussian kernel for a given variance sigma.
    Automatically adjusts kernel size to 3 sigmas.
    Examples:
    >>> gaussian_kernel(1.)
    array([[1.96412803e-05, 2.39279779e-04, 1.07237757e-03, 1.76805171e-03,
            1.07237757e-03, 2.39279779e-04, 1.96412803e-05],
           [2.39279779e-04, 2.91502447e-03, 1.30642333e-02, 2.15392793e-02,
            1.30642333e-02, 2.91502447e-03, 2.39279779e-04],
           [1.07237757e-03, 1.30642333e-02, 5.85498315e-02, 9.65323526e-02,
            5.85498315e-02, 1.30642333e-02, 1.07237757e-03],
           [1.76805171e-03, 2.15392793e-02, 9.65323526e-02, 1.59154943e-01,
            9.65323526e-02, 2.15392793e-02, 1.76805171e-03],
           [1.07237757e-03, 1.30642333e-02, 5.85498315e-02, 9.65323526e-02,
            5.85498315e-02, 1.30642333e-02, 1.07237757e-03],
           [2.39279779e-04, 2.91502447e-03, 1.30642333e-02, 2.15392793e-02,
            1.30642333e-02, 2.91502447e-03, 2.39279779e-04],
           [1.96412803e-05, 2.39279779e-04, 1.07237757e-03, 1.76805171e-03,
            1.07237757e-03, 2.39279779e-04, 1.96412803e-05]])
    >>> gaussian_kernel(.3)
    array([[2.64291611e-05, 6.83644778e-03, 2.64291611e-05],
           [6.83644778e-03, 1.76838826e+00, 6.83644778e-03],
           [2.64291611e-05, 6.83644778e-03, 2.64291611e-05]])
    >>> gaussian_kernel(-1)
    array([], shape=(0, 0), dtype=float64)
    """
    size = np.ceil(3 * sigma).astype(int)

    x = np.arange(-size, size + 1)

    X, Y = np.meshgrid(x, x)

    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * np.pi * sigma ** 2)

    return kernel


def zero_crossing(image: np.ndarray, threshold: float = 0) -> np.ndarray:
    """
    Applies the zero crossing algorithm for edge detection on an image filtered
    with a Laplacian of Gaussian kernel.
    For each pixel if two adjusted pixels on a given direction change signs and
    their difference is above a threshold it is marked as an edge pixel (255).
    Otherwise it is marked as a background pixel (0).
    Examples:
    >>> a = np.array([[-1,0,1],[0,0,0],[0,0,1]])
    >>> zero_crossing(a)
    array([[  0, 255,   0],
           [  0, 255,   0],
           [  0,   0,   0]], dtype=uint8)
    """
    N, M = image.shape

    edges = np.zeros_like(image, dtype=np.uint8)
    for i in range(N):
        for j in range(M):
            if i > 0 and i < N - 1:
                left = image[i - 1, j]
                right = image[i + 1, j]
                if left * right < 0 and np.abs(left - right) > threshold:
                    edges[i, j] = 255
            if j > 0 and j < M - 1:
                up = image[i, j + 1]
                down = image[i, j - 1]
                if up * down < 0 and np.abs(up - down) > threshold:
                    edges[i, j] = 255
            if (i > 0 and i < N - 1) and (j > 0 and j < M - 1):
                up_left = image[i - 1, j - 1]
                down_right = image[i + 1, j + 1]
                down_left = image[i - 1, j + 1]
                up_right = image[i + 1, j - 1]
                if (
                    up_left * down_right < 0
                    and np.abs(up_left - down_right) > threshold
                ):
                    edges[i, j] = 255
                elif (
                    down_left * up_right < 0
                    and np.abs(down_left - up_right) > threshold
                ):
                    edges[i, j] = 255
    return edges


def marrhildreth(
    image: np.ndarray, sigma: float = 1, threshold: float = 40
) -> np.ndarray:
    """
    Applies the Marr-Hildreth algorithm for edge detection on a given image.
    Parameters are the variance of the Gaussian kernel sigma and
    the zero crossing threshold. Normally for this algorithm the image is
    filtered with a Laplacian of Gaussian (LoG) kernel before passed to the zero
    crossing algorithm. Here the process is split in two, first filtering with
    a Gaussian kernel then with a Laplacian. Results are the same.
    Source: https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm
    Examples:
    >>> a = np.array([[0,0,255],[0,0,255],[0,0,255]])
    >>> marr_hildreth(a)
    array([[  0, 255,   0],
           [  0, 255,   0],
           [  0, 255,   0]], dtype=uint8)
    """
    g_kernel = gaussian_kernel(sigma)
    blurred = convolve(image, g_kernel)

    laplacian = convolve(blurred, laplacian_kernel)
   
    edges = zero_crossing(laplacian, threshold=threshold)
    return edges


if __name__ == "__main__":
    lena = cv2.imread("./lena.jpg", 0)
   
    edges = marrhildreth(lena, sigma=3.5, threshold=0.7)
   
    cv2.imwrite("edges.jpg", edges)
