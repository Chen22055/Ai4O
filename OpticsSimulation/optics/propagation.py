"""自由空间传播算法。

本文件用于实现波动光学中的传播算符，例如角谱法、Fresnel FFT
传播和 Rayleigh-Sommerfeld 传播。后续应支持在不同传播后端之间
切换，并提供必要的采样/混叠检查。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .field import Field

PropagatorName = Literal["angular_spectrum", "fresnel"]


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


def propagate(field: Field, distance: float, method: PropagatorName = "angular_spectrum") -> Field:
    """统一传播入口。"""

    if method == "angular_spectrum":
        return angular_spectrum(field, distance)
    if method == "fresnel":
        return fresnel(field, distance)
    raise ValueError(f"unknown propagation method: {method}")
