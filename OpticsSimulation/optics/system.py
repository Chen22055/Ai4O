"""可组合光路系统。

本文件用于定义 OpticalSystem，把自由传播段、透镜、光栅、SLM、
扩散片和相机等器件按顺序组合起来，并提供统一的 run(input_field)
仿真入口。也可在后续支持保存中间光场以便调试。
"""

from __future__ import annotations

from dataclasses import dataclass

from .elements import OpticalElement
from .field import Field


@dataclass(frozen=True)
class OpticalSystem:
    """按顺序执行的光路系统。"""

    elements: list[OpticalElement]

    def run(self, input_field: Field, *, keep_intermediate: bool = False) -> Field | list[Field]:
        """运行光路。

        keep_intermediate=False 时只返回最终光场；为 True 时返回包含
        输入场和每个器件后光场的列表。
        """

        field = input_field
        history = [field]
        for element in self.elements:
            field = element.apply(field)
            if keep_intermediate:
                history.append(field)
        return history if keep_intermediate else field
