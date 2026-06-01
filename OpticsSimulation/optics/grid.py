"""空间与频率采样网格定义。

本文件负责生成波动光学仿真所需的坐标网格，例如 x/y 空间坐标、
X/Y 二维网格、fx/fy 频率坐标，以及采样间隔、物理尺寸、波长等
基础参数管理。后续传播算法和器件模型都应依赖这里统一定义的网格。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """二维标量波动光学采样网格。

    参数默认使用 SI 单位：长度为 meter，波长为 meter。
    shape 采用 NumPy 习惯，即 (ny, nx)。
    """

    shape: tuple[int, int]
    pixel_pitch: float | tuple[float, float]
    wavelength: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("shape must be a 2D tuple: (ny, nx)")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")

        if isinstance(self.pixel_pitch, tuple):
            if len(self.pixel_pitch) != 2:
                raise ValueError("pixel_pitch tuple must be (dy, dx)")
            dy, dx = self.pixel_pitch
        else:
            dy = dx = self.pixel_pitch
        if dy <= 0 or dx <= 0:
            raise ValueError("pixel_pitch values must be positive")

        object.__setattr__(self, "dy", float(dy))
        object.__setattr__(self, "dx", float(dx))

    @property
    def ny(self) -> int:
        return self.shape[0]

    @property
    def nx(self) -> int:
        return self.shape[1]

    @property
    def extent(self) -> tuple[float, float]:
        """返回物理尺寸 (height, width)。"""

        return self.ny * self.dy, self.nx * self.dx

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.nx) - self.nx / 2) * self.dx + self.center[0]

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.ny) - self.ny / 2) * self.dy + self.center[1]

    @property
    def X(self) -> np.ndarray:
        return np.meshgrid(self.x, self.y, indexing="xy")[0]

    @property
    def Y(self) -> np.ndarray:
        return np.meshgrid(self.x, self.y, indexing="xy")[1]

    @property
    def fx(self) -> np.ndarray:
        return np.fft.fftfreq(self.nx, d=self.dx)

    @property
    def fy(self) -> np.ndarray:
        return np.fft.fftfreq(self.ny, d=self.dy)

    @property
    def FX(self) -> np.ndarray:
        return np.meshgrid(self.fx, self.fy, indexing="xy")[0]

    @property
    def FY(self) -> np.ndarray:
        return np.meshgrid(self.fx, self.fy, indexing="xy")[1]

    @property
    def k(self) -> float:
        return 2 * np.pi / self.wavelength
