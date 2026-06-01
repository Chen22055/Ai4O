"""自由空间传播算法。

本文件用于实现波动光学中的传播算符，例如角谱法、Fresnel FFT
传播和 Rayleigh-Sommerfeld 传播。后续应支持在不同传播后端之间
切换，并提供必要的采样/混叠检查。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .field import Field

PropagatorName = Literal["angular_spectrum", "fresnel", "rayleigh_sommerfeld"]


def angular_spectrum(field: Field, distance: float, *, keep_evanescent: bool = False) -> Field:
    """使用角谱法传播复光场。

    这是当前推荐的默认传播器，适合近场和中等传播距离。默认丢弃
    倏逝波分量，以保持数值稳定。
    """

    grid = field.grid
    wavelength = grid.wavelength
    argument = 1.0 - (wavelength * grid.FX) ** 2 - (wavelength * grid.FY) ** 2

    if keep_evanescent:
        kz_factor = np.sqrt(argument.astype(np.complex128))
    else:
        kz_factor = np.sqrt(np.clip(argument, 0.0, None))

    transfer = np.exp(1j * grid.k * distance * kz_factor)
    propagated = np.fft.ifft2(np.fft.fft2(field.data) * transfer)
    return field.with_data(propagated)


def fresnel(field: Field, distance: float) -> Field:
    """使用频域 Fresnel 近似传播复光场。"""

    if distance == 0:
        return field
    grid = field.grid
    transfer = np.exp(-1j * np.pi * grid.wavelength * distance * (grid.FX**2 + grid.FY**2))
    propagated = np.exp(1j * grid.k * distance) * np.fft.ifft2(np.fft.fft2(field.data) * transfer)
    return field.with_data(propagated)


def _fft_convolve_same(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return the center crop of the linear convolution array * kernel.

    This mirrors scipy.signal.fftconvolve(..., mode="same") for two 2D arrays
    with odd or even shapes, while keeping this module dependent only on NumPy.
    """

    if array.ndim != 2 or kernel.ndim != 2:
        raise ValueError("array and kernel must both be 2D")

    full_shape = (
        array.shape[0] + kernel.shape[0] - 1,
        array.shape[1] + kernel.shape[1] - 1,
    )
    full = np.fft.ifft2(np.fft.fft2(array, full_shape) * np.fft.fft2(kernel, full_shape))
    start_y = (kernel.shape[0] - 1) // 2
    start_x = (kernel.shape[1] - 1) // 2
    end_y = start_y + array.shape[0]
    end_x = start_x + array.shape[1]
    return full[start_y:end_y, start_x:end_x]


def rayleigh_sommerfeld(field: Field, distance: float) -> Field:
    """使用 Rayleigh-Sommerfeld 卷积积分传播复光场。

    该实现参考 `RefCode_of_Free_Space_Propagation` 中的 Matlab 思路：

    U(x, y, z) = U0(x, y) (*) h(x, y, z)

    其中采用 Rayleigh-Sommerfeld 一阶解的完整冲激响应：

    h = z / r**2 * (1 / (2*pi*r) + 1 / (j*lambda)) * exp(j*k*r)

    相比 Matlab 参考代码中的无量纲简化核，这里保留了 `1/(2*pi*r)`
    近场修正项，并使用真实波长。所有长度使用物理单位 meter。
    离散卷积结果乘以 `dx * dy`，用于近似连续面积积分。
    """

    if distance == 0:
        return field

    grid = field.grid
    r2 = grid.X**2 + grid.Y**2 + distance**2
    r = np.sqrt(r2)
    kernel = (distance / r2) * (
        1.0 / (2.0 * np.pi * r) + 1.0 / (1j * grid.wavelength)
    ) * np.exp(1j * grid.k * r)
    propagated = _fft_convolve_same(field.data, kernel) * grid.dx * grid.dy
    return field.with_data(propagated)


def propagate(field: Field, distance: float, method: PropagatorName = "angular_spectrum") -> Field:
    """统一传播入口。"""

    if method == "angular_spectrum":
        return angular_spectrum(field, distance)
    if method == "fresnel":
        return fresnel(field, distance)
    if method == "rayleigh_sommerfeld":
        return rayleigh_sommerfeld(field, distance)
    raise ValueError(f"unknown propagation method: {method}")
