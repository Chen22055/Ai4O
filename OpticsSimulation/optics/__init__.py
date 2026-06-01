"""光学仿真核心包。

本包集中放置波动光学仿真的基础对象、传播算法、光学器件、
SLM/扩散片/相机模型，以及光路系统组合逻辑。
"""

from .field import Field
from .grid import Grid
from .system import OpticalSystem

__all__ = ["Field", "Grid", "OpticalSystem"]
