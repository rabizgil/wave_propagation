import math

import numpy as np


# https://en.wikipedia.org/wiki/Gaussian_function
def generate_gaussian_wavelet(npoints: int = 10, sigma: float = 2.4) -> np.ndarray:
    """
    Generates a Gaussian wavelet.

    Args:
        npoints : int
            Number of points in x and y directions to construct the wavelet on.
        sigma : float
            Sigma factor for the Gaussian wavelet. Affects spatial frequency.
    Returns:
        Gaussian wavelet as a numpy array of shape (npoints, npoints).
    """
    xx, yy = np.meshgrid(
        np.linspace(-npoints, npoints, 2 * npoints + 1),
        np.linspace(-npoints, npoints, 2 * npoints + 1),
    )
    wavelet = 300 / (sigma * 2 * math.pi) * (math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((xx**2 + yy**2) / (sigma**2)))
    return wavelet


def generate_ricker_wavelet(npoints: int = 10, sigma: float = 2.4) -> np.ndarray:
    """
    Generates a 2D Ricker (Mexican Hat) wavelet.

    Args:
        npoints : int
            Number of points in x and y directions to construct the wavelet on.
        sigma : float
            Sigma factor for the Ricker wavelet. Affects spatial frequency.
    Returns:
        Ricker wavelet as a numpy array of shape (npoints, npoints).
    """
    xx, yy = np.meshgrid(
        np.linspace(-npoints, npoints, 2 * npoints + 1),
        np.linspace(-npoints, npoints, 2 * npoints + 1),
    )
    wavelet = ((xx) ** 2 + (yy) ** 2) / (2 * sigma**2)
    wavelet = 300 * (1 - wavelet) * np.exp(-wavelet)
    return wavelet


def inject_wavelet(wavefield: np.ndarray, wavelet: np.ndarray, x: int, y: int, scale: int | float = 1) -> np.ndarray:
    """
    Takes a wavefield array containing previous, current and future steps,
    wavelet array and injects the given wavelet into the wavefiled at position x, y.

    Args:
        wavefield : np.ndarray
            Wavefield of shape (3, size_x, size_y) containing previous, current and n+1 steps in size_x * size_y media.
        wavelet : np.ndarray
            Wavelet array to be injected.
        x : int
            Horizontal coordinate of the injection point.
        y : int
            Vertical coordinate of the injection point.
        scale : int | float
            Scale factor to be applied to the wavelet.
    """
    w, h = wavelet.shape
    w = int(w / 2)
    h = int(h / 2)
    wavefield[:2, x - w : x + w + 1, y - h : y + h + 1] += scale * wavelet
    return wavefield
