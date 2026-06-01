"""通用光学器件模型。

本文件用于定义薄透镜、光阑、光栅、相位掩膜、振幅掩膜等常见
光学元件。建议所有器件使用统一接口，例如 apply(field)，以便
被 OpticalSystem 顺序组合成完整光路。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .field import Field
from .propagation import PropagatorName, propagate


class OpticalElement(Protocol):
    """所有光学器件遵循的最小接口。"""

    def apply(self, field: Field) -> Field:
        ...


@dataclass(frozen=True)
class FreeSpace:
    """自由空间传播段。"""

    distance: float
    method: PropagatorName = "angular_spectrum"

    def apply(self, field: Field) -> Field:
        return propagate(field, self.distance, self.method)


@dataclass(frozen=True)
class ThinLens:
    """理想薄透镜，相位因子为 exp[-j k (x^2+y^2)/(2f)]。"""

    focal_length: float

    def apply(self, field: Field) -> Field:
        if self.focal_length == 0:
            raise ValueError("focal_length must be nonzero")
        grid = field.grid
        phase = -grid.k * (grid.X**2 + grid.Y**2) / (2 * self.focal_length)
        return field.with_data(field.data * np.exp(1j * phase))


@dataclass(frozen=True)
class CircularAperture:
    """圆形光阑。"""

    radius: float

    def apply(self, field: Field) -> Field:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        grid = field.grid
        mask = (grid.X**2 + grid.Y**2) <= self.radius**2
        return field.with_data(field.data * mask)


@dataclass(frozen=True)
class PhaseMask:
    """任意相位掩膜，phase 单位为 rad。"""

    phase: np.ndarray

    def apply(self, field: Field) -> Field:
        phase = np.asarray(self.phase, dtype=float)
        if phase.shape != field.grid.shape:
            raise ValueError("phase mask shape must match field grid shape")
        return field.with_data(field.data * np.exp(1j * phase))


@dataclass(frozen=True)
class AmplitudeMask:
    """任意振幅掩膜。"""

    amplitude: np.ndarray

    def apply(self, field: Field) -> Field:
        amplitude = np.asarray(self.amplitude, dtype=float)
        if amplitude.shape != field.grid.shape:
            raise ValueError("amplitude mask shape must match field grid shape")
        return field.with_data(field.data * amplitude)


@dataclass(frozen=True)
class BinaryAmplitudeGrating:
    """一维二值振幅光栅。

    period 为光栅周期，duty_cycle 为透光占空比。
    orientation 可选 "x" 或 "y"，表示沿哪个坐标方向周期变化。
    """

    period: float
    duty_cycle: float = 0.5
    orientation: str = "x"

    def apply(self, field: Field) -> Field:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if not 0 < self.duty_cycle <= 1:
            raise ValueError("duty_cycle must be in (0, 1]")

        coord = field.grid.X if self.orientation == "x" else field.grid.Y
        if self.orientation not in {"x", "y"}:
            raise ValueError("orientation must be 'x' or 'y'")
        normalized = np.mod(coord, self.period) / self.period
        mask = normalized < self.duty_cycle
        return field.with_data(field.data * mask)
