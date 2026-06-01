"""复光场对象与基础光场生成。

本文件用于定义标量相干光场 U(x, y)，并提供平面波、高斯光、
孔径光场、图像振幅输入、图像相位输入等构造方式。也负责强度、
相位、归一化和基础可视化等通用操作。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid


@dataclass(frozen=True)
class Field:
    """定义在 Grid 上的二维复光场。"""

    grid: Grid
    data: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.data, dtype=np.complex128)
        if array.shape != self.grid.shape:
            raise ValueError(f"field shape {array.shape} does not match grid shape {self.grid.shape}")
        object.__setattr__(self, "data", array)

    @classmethod
    def plane_wave(
        cls,
        grid: Grid,
        amplitude: complex = 1.0,
        tilt: tuple[float, float] = (0.0, 0.0),
    ) -> "Field":
        """生成带可选倾角的平面波。

        tilt 为 (kx, ky)，单位 rad/m。默认 tilt=(0,0)。
        """

        phase = tilt[0] * grid.X + tilt[1] * grid.Y
        return cls(grid, amplitude * np.exp(1j * phase))

    @classmethod
    def gaussian(cls, grid: Grid, waist: float, amplitude: complex = 1.0) -> "Field":
        """生成中心位于网格中心的高斯光场。"""

        if waist <= 0:
            raise ValueError("waist must be positive")
        radius2 = grid.X**2 + grid.Y**2
        return cls(grid, amplitude * np.exp(-radius2 / waist**2))

    @classmethod
    def from_amplitude_phase(
        cls,
        grid: Grid,
        amplitude: float | np.ndarray = 1.0,
        phase: float | np.ndarray = 0.0,
    ) -> "Field":
        """由振幅和相位构造复光场。"""

        amp = np.asarray(amplitude, dtype=float)
        ph = np.asarray(phase, dtype=float)
        return cls(grid, amp * np.exp(1j * ph))

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.data) ** 2

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.data)

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def power(self) -> float:
        """离散积分意义下的总光强。"""

        return float(np.sum(self.intensity) * self.grid.dx * self.grid.dy)

    def with_data(self, data: np.ndarray) -> "Field":
        return Field(self.grid, data)

    def normalize_power(self, target_power: float = 1.0) -> "Field":
        """按总光强归一化光场。"""

        if target_power <= 0:
            raise ValueError("target_power must be positive")
        current_power = self.power
        if current_power == 0:
            raise ValueError("cannot normalize a zero-power field")
        scale = np.sqrt(target_power / current_power)
        return self.with_data(self.data * scale)
