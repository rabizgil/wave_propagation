import math
from typing import Literal

import numpy as np
import torch
from torch.nn.functional import conv2d

RETURN_TYPE = Literal["pt", "np"]


def delta_fn(k: int) -> int:
    """
    Kronecker delta function.

    Args:
        k : int

    Returns:
        1 if k is 0, 0 otherwise.
    """
    if k == 0:
        return 1
    else:
        return 0


def build_laplacian_kernel(
    kernel_size: int = 5, return_type: RETURN_TYPE = "pt", device="cpu"
) -> np.ndarray | torch.Tensor:
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
        torch.Tensor or numpy ndarray with a shape of kernel_size x kernel_size representing Laplacian kernel.
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
    if return_type == "pt":
        return torch.from_numpy(laplacian_matrix).to(device)
    else:
        return laplacian_matrix


def solve_one_step(
    wavefield: torch.Tensor,
    tau: torch.Tensor,
    kappa: torch.Tensor,
    laplacian_kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Takes a wavefield array containing previous, current and future steps,
    tau and kappa fields and pre-built Laplacian kernel to perform step forward propagation.
    Currently only supports Mur boundary condition.

    Args:
        wavefield : torch.Tensor
            Wavefield of shape (3, size_x, size_y) containing previous, current and n+1 steps in size_x * size_y media.
        tau : torch.Tensor
            Tau field of shape (size_x, size_y) derived from velocity model.
        kappa : torch.Tensor
            Kappa field of shape (size_x, size_y) derived from velocity model.
        laplacian_kernel : torch.Tensor
            Laplacian kernel to be used in propagation.
    Returns:
        Numpy ndarray of shape (3, size_x, size_y) where ndarray[0] is the propagated wavefield.
    """
    dimx, dimy = wavefield.shape[1:]
    wavefield[2] = wavefield[1]
    wavefield[1] = wavefield[0]
    boundary_size = laplacian_kernel.shape[0] // 2

    laplacian = tau[None, None, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size] * conv2d(
        wavefield[1][None, None, :, :], laplacian_kernel[None, None, :, :]
    )
    laplacian = torch.squeeze(laplacian)

    conv_result = (
        laplacian
        + 2 * wavefield[1, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size]
        - wavefield[2, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size]
    )

    wavefield[0, boundary_size : dimx - boundary_size, boundary_size : dimy - boundary_size] = conv_result
    wavefield = update_boundaries(wavefield, kappa, boundary_size)
    return wavefield


def update_boundaries(wavefield: torch.Tensor, kappa: torch.Tensor, boundary_size: int) -> torch.Tensor:
    """
    Takes a wavefield array containing previous, current and future steps,
    kappa field and a boundary size to update wavefield values at the edges according
    to Mur boundary condition formulation.

    Args:
        wavefield : torch.Tensor
            Wavefield of shape (3, size_x, size_y) containing previous, current and n+1 steps in size_x * size_y media.
        kappa : torch.Tensor
            Kappa field of shape (size_x, size_y) derived from velocity model.
        boundary_size : int
            Boundary size to introduce Mur condition.
    Returns:
        torch.Tensor of shape (3, size_x, size_y) where tensor[0, :, :] is the wavefield with updated boundary condition.
    """
    dimx, dimy = wavefield.shape[1:]
    c = dimx - 1
    kappa_coef = (kappa - 1) / (kappa + 1)
    wavefield[0, dimx - boundary_size - 1 : c, 1 : dimy - 1] = wavefield[
        1, dimx - boundary_size - 2 : c - 1, 1 : dimy - 1
    ] + kappa_coef[dimx - boundary_size - 1 : c, 1 : dimy - 1] * (
        wavefield[0, dimx - boundary_size - 2 : c - 1, 1 : dimy - 1]
        - wavefield[1, dimx - boundary_size - 1 : c, 1 : dimy - 1]
    )

    c = 0
    wavefield[0, c:boundary_size, 1 : dimy - 1] = wavefield[1, c + 1 : boundary_size + 1, 1 : dimy - 1] + kappa_coef[
        c:boundary_size, 1 : dimy - 1
    ] * (wavefield[0, c + 1 : boundary_size + 1, 1 : dimy - 1] - wavefield[1, c:boundary_size, 1 : dimy - 1])

    r = dimy - 1
    wavefield[0, 1 : dimx - 1, dimy - 1 - boundary_size : r] = wavefield[
        1, 1 : dimx - 1, dimy - 2 - boundary_size : r - 1
    ] + kappa_coef[1 : dimx - 1, dimy - 1 - boundary_size : r] * (
        wavefield[0, 1 : dimx - 1, dimy - 2 - boundary_size : r - 1]
        - wavefield[1, 1 : dimx - 1, dimy - 1 - boundary_size : r]
    )

    r = 0
    wavefield[0, 1 : dimx - 1, r:boundary_size] = wavefield[1, 1 : dimx - 1, r + 1 : boundary_size + 1] + kappa_coef[
        1 : dimx - 1, r:boundary_size
    ] * (wavefield[0, 1 : dimx - 1, r + 1 : boundary_size + 1] - wavefield[1, 1 : dimx - 1, r:boundary_size])
    return wavefield
