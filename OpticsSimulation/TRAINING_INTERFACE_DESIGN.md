# Training Interface Design

本文件记录当前 NumPy 波动光学仿真系统如何与未来 PyTorch 可微训练流程衔接。目标是让当前代码既能稳定做物理验证，又能自然扩展到类似 2022 QPI/D2NN 论文中的端到端训练：

```text
input phase object -> optical system -> camera/output processor -> loss -> backpropagation -> update trainable optical elements
```

## 1. 总体原则

当前 `optics/` 中的 NumPy 代码主要负责：

- 物理概念验证；
- 光栅衍射、透镜成像、Airy pattern 等标准实验；
- 与 Matlab 参考代码对齐；
- 生成可信的前向传播基准。

未来 PyTorch 代码主要负责：

- 可微传播；
- 可训练 SLM / diffractive layers；
- 输入输出数据集训练；
- loss 反向传播；
- 自动优化光学器件参数。

因此建议保持两套实现的职责清晰：

```text
NumPy version   -> validation and physical baseline
PyTorch version -> differentiable training and optimization
```

不要为了训练强行把当前 NumPy 代码改成复杂的通用 backend。短期更稳妥的做法是后续新增 `training/` 或 `optics_torch/`。

## 2. 当前需要保留的稳定接口

### 2.1 Field 构造接口

当前 NumPy 版：

```python
Field.from_amplitude_phase(grid, amplitude, phase)
```

这个概念需要保留。未来 PyTorch 版可以对应：

```python
TorchField.from_amplitude_phase(grid, amplitude, phase)
```

核心约定：

```text
U(x, y) = A(x, y) * exp(j * phi(x, y))
```

这样 MNIST、几何图案、相位光栅、随机相位等输入都可以通过同一套概念转换为复光场。

### 2.2 Optical element 接口

当前 NumPy 版器件接口：

```python
field = element.apply(field)
```

未来 PyTorch 版可以使用：

```python
field = element.forward(field)
```

或让器件继承 `torch.nn.Module` 后直接：

```python
field = element(field)
```

无论具体写法如何，核心抽象要保持：

```text
optical element: input complex field -> output complex field
```

光路系统不应该关心器件内部是固定参数还是可训练参数。

### 2.3 OpticalSystem 顺序组合接口

当前 NumPy 版：

```python
system = OpticalSystem([element1, element2, element3])
output_field = system.run(input_field)
```

未来 PyTorch 版可以对应：

```python
model = TrainableOpticalSystem([layer1, layer2, layer3])
output_field = model(input_field)
```

训练循环应类似：

```python
output_field = model(input_field)
output_image = camera(output_field)
loss = loss_fn(output_image, target)
loss.backward()
optimizer.step()
```

## 3. 不要让 NumPy 数组阻断未来训练

当前固定器件可以使用：

```python
phase: np.ndarray
```

但可训练器件不能只存普通 NumPy 数组。未来训练版应使用：

```python
torch.nn.Parameter
```

例如：

```python
class TrainablePhaseMask(torch.nn.Module):
    def __init__(self, shape):
        self.phase = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, field):
        return field * torch.exp(1j * self.phase)
```

因此建议后续新增：

```text
training/trainable_elements.py
```

或：

```text
optics_torch/elements.py
```

不要把可训练参数直接塞进当前 NumPy `PhaseMask` 中。

## 4. 传播算法需要 PyTorch 版本

当前 `optics/propagation.py` 使用：

```python
np.fft.fft2
np.fft.ifft2
np.exp
np.abs
```

这些不会被 PyTorch 自动求导追踪。训练时需要对应的 PyTorch 实现：

```python
torch.fft.fft2
torch.fft.ifft2
torch.exp
torch.abs
```

未来建议新增：

```text
training/torch_propagation.py
```

至少实现：

- differentiable angular spectrum propagation；
- differentiable Fresnel propagation；
- optional differentiable Rayleigh-Sommerfeld propagation。

训练初期建议优先用角谱法或 Fresnel，因为速度和数值稳定性通常更适合大批量训练。

## 5. Camera / OutputProcessor 与 Loss 要分层

2022 QPI/D2NN 类任务的流程不是简单输出复光场，而是：

```text
complex field -> intensity -> crop/bin/reference normalization -> compare with target
```

因此后续要分清三层：

```text
OpticalSystem:
  只负责复光场传播和器件调制。

Camera / OutputProcessor:
  负责 field -> intensity，以及 crop、binning、ROI、reference-region normalization。

Loss:
  负责 output image 与 target image 的标量误差。
```

不要把 loss 写进传播函数，也不要把 QPI 归一化写死在某个实验脚本里。

未来可新增：

```text
training/torch_camera.py
training/losses.py
```

例如 QPI 任务中：

```python
intensity = torch.abs(output_field.data) ** 2
normalized = signal_region / reference_region.mean()
loss = torch.mean((normalized - target_phase_image) ** 2)
```

## 6. MNIST 灰度图如何成为相位输入

如果后续仿照 2022 QPI/D2NN 论文，MNIST 灰度图可以被解释为纯相位物体。

设归一化灰度图为：

```text
image(x, y) in [0, 1]
```

则相位可以定义为：

```text
phase(x, y) = alpha * pi * image(x, y)
```

振幅取常数：

```text
amplitude(x, y) = 1
```

输入复光场：

```text
U_in(x, y) = exp(j * phase(x, y))
```

这类转换逻辑应该放在：

```text
datasets/phase_objects.py
```

未来 PyTorch 训练时，也可以新增：

```text
training/datasets.py
```

用于把 batch 形式的图像转换为 batch 形式的复光场。

## 7. Config 中要区分固定参数和可训练参数

未来配置文件应能描述哪些器件固定，哪些器件可训练。例如：

```yaml
optical_system:
  elements:
    - type: free_space
      distance: 0.04
      trainable: false

    - type: trainable_phase_mask
      shape: [200, 200]
      init: zeros
      phase_range: [0.0, 6.283185307179586]
      trainable: true

    - type: free_space
      distance: 0.04
      trainable: false
```

这样 config 既能描述普通仿真实验，也能描述训练实验。

## 8. 推荐新增结构

未来进入训练阶段时，建议新增：

```text
OpticsSimulation/
  training/
    __init__.py
    torch_grid.py
    torch_field.py
    torch_propagation.py
    trainable_elements.py
    trainable_system.py
    torch_camera.py
    losses.py
    datasets.py
    train_qpi.py
```

各文件职责：

- `torch_grid.py`：PyTorch 版本网格，保存坐标和频率坐标 tensor。
- `torch_field.py`：PyTorch 版本复光场。
- `torch_propagation.py`：可微自由空间传播。
- `trainable_elements.py`：可训练相位层、SLM、D2NN layer。
- `trainable_system.py`：继承 `torch.nn.Module` 的顺序光路系统。
- `torch_camera.py`：可微强度探测、crop、binning、reference normalization。
- `losses.py`：MSE、QPI loss、分类 loss、diffraction efficiency penalty。
- `datasets.py`：MNIST/图案数据到复光场的 batch 转换。
- `train_qpi.py`：训练入口脚本。

## 9. 当前代码中应避免的事情

- 不要把实验参数硬编码进核心模块。
- 不要让传播函数负责保存图片或计算 loss。
- 不要让器件类同时承担数据集读取职责。
- 不要把 NumPy 数组和 Torch tensor 混在同一个计算图里。
- 不要在当前 `optics/` 中过早引入复杂训练逻辑。

## 10. 当前代码中值得保留的事情

- `Field` 表示复光场。
- `Grid` 统一管理空间和频率采样。
- `FreeSpace`、`ThinLens`、`Grating` 等器件使用统一 `apply(field)` 接口。
- `OpticalSystem` 使用顺序列表组织光路。
- 实验参数放在 `configs/` 中。
- 输出图和验证结果由 `experiments/` 生成。
- 物理正确性由 `validation/` 和标准实验保证。

这些概念未来都可以一一映射到 PyTorch 训练系统。
