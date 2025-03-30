import math

import numpy as np
from scipy.signal import convolve2d


def delta_fn(k: int) -> bool:
    """
    Kronecker delta function.

    Args:
        k : int

    Returns:
        1 if k is 0, 0 otherwise.
    """
    if k == 0:
        return True
    else:
        return False


def build_laplacian_kernel(kernel_size: int = 5) -> np.ndarray:
    """
    Returns a Laplacian kernel used in wavefield equation.
    Currently implemented for 2D media only.
    Uses the finite difference coefficient estimation summarized by Taylor, Cameron R.
    https://web.media.mit.edu/~crtaylor/calculator.html

    Args:
        kernel_size : int
            Size of the kernel matrix. The resulting kernel will have a shape of kernel_size x kernel_size.
            Higher values will increase the modelling accuracy with more computational cost. Defaults to 5.
    Returns:
        Numpy ndarray with a shape of kernel_size x kernel_size representing Laplacian kernel.
    """

    derivative_order: int = 2
    coef_num = kernel_size // 2
    stencil = np.tile(np.linspace(-coef_num, coef_num, kernel_size), (kernel_size, 1))
    powers = np.array([[j for _ in range(kernel_size)] for j in range(kernel_size)], dtype=np.float32)
    stencil = np.power(stencil, powers)
    factor_vector = np.array(
        [math.factorial(derivative_order) * delta_fn(i - derivative_order) for i in range(kernel_size)]
    )
    derivation_coef = np.matmul(np.linalg.inv(stencil), factor_vector)
    laplacian_matrix = np.zeros((kernel_size, kernel_size))
    laplacian_matrix[coef_num, :] += derivation_coef
    laplacian_matrix[:, coef_num] += derivation_coef
    return laplacian_matrix


def solve_one_step(
    wavefield: np.ndarray,
    tau: np.ndarray,
    kappa: np.ndarray,
    laplacian_kernel: np.ndarray,
) -> np.ndarray:
    """
    Takes a wavefield array containing previous, current and future steps,
    tau and kappa fields and pre-built Laplacian kernel to perform step forward propagation.
    Currently only supports Mur boundary condition.

    Args:
        wavefield : np.ndarray
            Wavefield of shape (3, size_x, size_y) containing previous, current and n+1 steps in size_x * size_y media.
        tau : np.ndarray
            Tau field of shape (size_x, size_y) derived from velocity model.
        kappa : np.ndarray
            Kappa field of shape (size_x, size_y) derived from velocity model.
        laplacian_kernel : np.ndarray
            Laplacian kernel to be used in propagation.
    Returns:
        Numpy ndarray of shape (3, size_x, size_y) where ndarray[0] is the propagated wavefield.
    """
    dimx, dimy = wavefield.shape[1:]
    wavefield[2] = wavefield[1]
    wavefield[1] = wavefield[0]
    boundary_size = laplacian_kernel.shape[0] // 2
    conv_result = (
        tau[boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size]
        * convolve2d(wavefield[1], laplacian_kernel, mode="valid", boundary="symm")
        + 2 * wavefield[1, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size]
        - wavefield[2, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size]
    )

    wavefield[0, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size] = conv_result
    wavefield = update_boundaries(wavefield, kappa, boundary_size)
    return wavefield


def update_boundaries(wavefield: np.ndarray, kappa: np.ndarray, boundary_size: int) -> np.ndarray:
    """
    Takes a wavefield array containing previous, current and future steps,
    kappa field and a boundary size to update wavefield values at the edges according
    to Mur boundary condition formulation.

    Args:
        wavefield : np.ndarray
            Wavefield of shape (3, size_x, size_y) containing previous, current and n+1 steps in size_x * size_y media.
        kappa : np.ndarray
            Kappa field of shape (size_x, size_y) derived from velocity model.
        boundary_size : int
            Boundary size to introduce Mur condition.
    Returns:
        Numpy ndarray of shape (3, size_x, size_y) where ndarray[0] is the wavefield with updated boundary condition.
    """
    dimx, dimy = wavefield.shape[1:]
    c = dimx - 1
    wavefield[0, dimx - boundary_size - 1 : c, 1 : dimy - 1] = wavefield[
        1, dimx - boundary_size - 2 : c - 1, 1 : dimy - 1
    ] + (kappa[dimx - boundary_size - 1 : c, 1 : dimy - 1] - 1) / (
        kappa[dimx - boundary_size - 1 : c, 1 : dimy - 1] + 1
    ) * (
        wavefield[0, dimx - boundary_size - 2 : c - 1, 1 : dimy - 1]
        - wavefield[1, dimx - boundary_size - 1 : c, 1 : dimy - 1]
    )

    c = 0
    wavefield[0, c:boundary_size, 1 : dimy - 1] = wavefield[1, c + 1 : boundary_size + 1, 1 : dimy - 1] + (
        kappa[c:boundary_size, 1 : dimy - 1] - 1
    ) / (kappa[c:boundary_size, 1 : dimy - 1] + 1) * (
        wavefield[0, c + 1 : boundary_size + 1, 1 : dimy - 1] - wavefield[1, c:boundary_size, 1 : dimy - 1]
    )

    r = dimy - 1
    wavefield[0, 1 : dimx - 1, dimy - 1 - boundary_size : r] = wavefield[
        1, 1 : dimx - 1, dimy - 2 - boundary_size : r - 1
    ] + (kappa[1 : dimx - 1, dimy - 1 - boundary_size : r] - 1) / (
        kappa[1 : dimx - 1, dimy - 1 - boundary_size : r] + 1
    ) * (
        wavefield[0, 1 : dimx - 1, dimy - 2 - boundary_size : r - 1]
        - wavefield[1, 1 : dimx - 1, dimy - 1 - boundary_size : r]
    )

    r = 0
    wavefield[0, 1 : dimx - 1, r:boundary_size] = wavefield[1, 1 : dimx - 1, r + 1 : boundary_size + 1] + (
        kappa[1 : dimx - 1, r:boundary_size] - 1
    ) / (kappa[1 : dimx - 1, r:boundary_size] + 1) * (
        wavefield[0, 1 : dimx - 1, r + 1 : boundary_size + 1] - wavefield[1, 1 : dimx - 1, r:boundary_size]
    )
    return wavefield
