# OpticsSimulation

本文件夹用于建立可验证、可扩展的 Python 波动光学仿真系统。当前目标是先完成基础物理仿真和验证，例如光栅衍射、透镜成像、Airy disk 等；后续再扩展到 SLM、任意相位光、QPI、D2NN 和硬件在环优化。

## 核心概念

### `configs/*.yaml` 与 `optics/field.py` 的关系

`configs/custom_phase_field.yaml` 不是光场本身，而是一次仿真实验的“参数说明书”。它描述用户想输入什么样的光场，例如波长、采样网格、振幅来源、相位来源、传播距离、输出路径等。

`optics/field.py` 才是代码层面的光场定义。它提供 `Field` 类，用复数数组表示标量相干光场：

```text
U(x, y) = A(x, y) * exp(j * phi(x, y))
```

因此后续完整流程应是：

```text
读取 config -> 生成 Grid -> 生成 amplitude 和 phase -> Field.from_amplitude_phase() -> OpticalSystem.run()
```

换句话说：

- `custom_phase_field.yaml`：告诉程序“我要什么输入光场”。
- `field.py`：负责把振幅和相位变成真正可传播的复光场对象。
- `experiments/run_custom_phase_field.py`：未来负责把配置文件和 `Field` 类连接起来。

### 如果以后输入 MNIST 灰度图

仿照 2022 QPI/D2NN 论文时，MNIST 灰度图不会直接作为普通强度图片输入，而是可以映射为纯相位物体。例如灰度图 `img` 归一化到 `[0, 1]` 后：

```text
phase(x, y) = alpha * pi * img(x, y)
amplitude(x, y) = 1
U_in(x, y) = exp(j * phase(x, y))
```

代码层面建议放在 `datasets/phase_objects.py` 中实现，例如：

```python
phase = image_to_phase(mnist_image, phase_range=(0, alpha * np.pi))
field = Field.from_amplitude_phase(grid, amplitude=1.0, phase=phase)
```

如果之后做训练，推荐新增 `training/` 或在 `experiments/run_qpi_phase_recovery.py` 中先做最小 demo：

```text
MNIST image -> phase object -> Field -> trainable phase masks / SLM layers -> Camera intensity -> loss
```

训练时目标可以是：

- 分类任务：相机不同区域代表不同数字类别。
- QPI 相位恢复任务：输出归一化强度图尽量等于输入相位图。
- 自定义任务：输出某种指定光斑、模式或边缘图。

## 文件夹架构

```text
OpticsSimulation/
  README.md
  TODO.md
  configs/
  datasets/
  examples/
  experiments/
  optics/
  outputs/
  validation/
  RefCode_of_Free_Space_Propagation/
  RefPaper/
```

## 目录与文件作用

### `configs/`

配置文件目录。用于保存不同实验的输入参数，避免把波长、距离、网格大小等参数硬编码在 Python 脚本里。

- `custom_phase_field.yaml`：自定义任意相位光场输入的通用配置模板。
- `validate_grating.yaml`：光栅衍射验证实验配置。
- `validate_lens.yaml`：透镜成像/聚焦验证实验配置。
- `qpi_phase_recovery_2022.yaml`：2022 QPI 相位恢复论文启发的可选参考配置，不是复现要求。
- `qpi_diffuser_2023.yaml`：2023 随机扩散片 QPI 论文启发的可选参考配置，不是复现要求。

### `optics/`

核心光学仿真代码。

- `__init__.py`：导出常用核心类，例如 `Grid`、`Field`、`OpticalSystem`。
- `grid.py`：定义二维空间/频率采样网格，包括坐标、频率坐标、像素间距和波长。
- `field.py`：定义复光场 `Field`，并提供平面波、高斯光、振幅相位构造、强度和相位计算。
- `propagation.py`：自由空间传播算法，目前包含角谱法和 Fresnel 传播。
- `elements.py`：通用光学元件，目前包含自由传播段、薄透镜、圆孔、相位掩膜、振幅掩膜和二值振幅光栅。
- `system.py`：可组合光路系统，把多个光学元件按顺序串起来执行。
- `slm.py`：SLM 模型预留文件，后续实现理想相位 SLM、振幅 SLM、量化 SLM 和真实 TN-SLM。
- `diffuser.py`：随机相位扩散片模型预留文件。
- `camera.py`：相机强度探测、裁剪、binning、归一化和噪声模型预留文件。
- `metrics.py`：MSE、PSNR、PCC、相位误差、衍射效率等评价指标预留文件。

### `datasets/`

输入物体和数据生成工具。

- `__init__.py`：数据集工具包入口。
- `phase_objects.py`：相位物体生成预留文件。后续可实现光栅、圆孔、几何图案、MNIST 灰度图到相位图的转换。

### `experiments/`

实验脚本目录。每个脚本负责读取配置、构造光场和光路、运行仿真、保存结果。

- `run_custom_phase_field.py`：自定义任意相位光场实验入口。
- `run_validation_grating.py`：光栅衍射验证实验入口。
- `run_validation_lens.py`：透镜成像/聚焦验证实验入口。
- `run_qpi_phase_recovery.py`：QPI 相位到强度映射的可选参考实验入口。
- `run_qpi_diffuser.py`：带随机扩散片的 QPI 可选参考实验入口。

### `validation/`

自动化物理验证测试目录。

- `test_grating_diffraction.py`：验证光栅衍射峰位置。
- `test_lens_imaging.py`：验证透镜聚焦/成像行为。
- `test_propagation_energy.py`：验证自由传播能量近似守恒。

### `examples/`

交互式示例 notebook。

- `grating_diffraction_demo.ipynb`：光栅衍射演示。
- `lens_imaging_demo.ipynb`：透镜成像/聚焦演示。

### `outputs/`

仿真输出目录。

- `outputs/figures/`：保存强度图、相位图、径向分布图等。
- `outputs/data/`：保存数值数组、指标、配置快照等。

当前已有示例输出：

- `outputs/figures/smoke_test_lens_intensity.png`
- `outputs/figures/airy_validation_intensity.png`
- `outputs/figures/airy_validation_radial_profile.png`

### `RefCode_of_Free_Space_Propagation/`

学长提供的 Matlab 自由空间传播参考代码。后续可用于对照 Rayleigh-Sommerfeld 传播实现和基础光学结果。

### `RefPaper/`

参考文献目录。当前两篇 QPI/D2NN 论文主要作为结构设计和方法启发，不作为必须复现目标。

## 当前运行环境

建议使用 conda 环境 `py_env`：

```powershell
conda run -n py_env python ...
```

如果只是运行基础模块，当前主要依赖：

- `numpy`
- `matplotlib`

后续读取 YAML、处理图像、训练 D2NN 时可能需要：

- `pyyaml`
- `pillow` 或 `opencv-python`
- `torch`

## 推荐开发顺序

1. 完善 `configs/validate_grating.yaml` 和 `experiments/run_validation_grating.py`。
2. 完成光栅衍射验证，确认传播算法和采样设置合理。
3. 完善 `configs/validate_lens.yaml` 和 `experiments/run_validation_lens.py`。
4. 完成透镜成像、Airy disk 或焦斑验证。
5. 实现 `custom_phase_field.yaml` 的读取逻辑，支持用户输入任意相位光。
6. 在 `datasets/phase_objects.py` 中实现图像到相位物体的转换。
7. 再进入 SLM、相机、D2NN 训练和真实器件建模。
