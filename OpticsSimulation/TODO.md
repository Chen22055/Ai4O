<!-- 本文件用于记录 OpticsSimulation 项目的阶段性开发任务和优先级，指导后续逐步实现、验证和扩展。 -->

# Optics Simulation TODO

## Phase 0: Project Hygiene

- [ ] Decide the package name and keep it consistent: current folder is `OpticsSimulation`.
- [ ] Add a short `README.md` explaining the simulation goal, folder layout, and how to run experiments.
- [ ] Choose dependency stack: `numpy`, `scipy`, `matplotlib`, `pyyaml`; add `torch` later for trainable D2NN.
- [ ] Add a lightweight environment file, e.g. `requirements.txt` or `pyproject.toml`.
- [ ] Define output conventions for figures, arrays, logs, and config snapshots.

## Phase 1: Core Wave Optics Modules

- [x] Implement `optics/grid.py`.
  - Create spatial coordinates `x`, `y`, `X`, `Y`.
  - Create frequency coordinates `fx`, `fy`, `FX`, `FY`.
  - Store wavelength, pixel pitch, array shape, physical size, and units.

- [x] Implement `optics/field.py`.
  - Represent a scalar coherent optical field as complex array `U`.
  - Provide constructors for plane wave, Gaussian beam, aperture field, image-derived amplitude, and image-derived phase.
  - Provide utilities for intensity, phase, normalization, and plotting.

- [x] Implement `optics/propagation.py`.
  - Start with angular spectrum propagation.
  - Add Fresnel FFT propagation.
  - [ ] Add Rayleigh-Sommerfeld propagation or a compatibility implementation based on the Matlab reference code.
  - [ ] Include sampling and aliasing checks.

- [x] Implement `optics/elements.py`.
  - Add thin lens.
  - Add aperture.
  - Add grating.
  - Add phase mask and amplitude mask.
  - Make every element expose a shared interface such as `apply(field)`.

- [x] Implement `optics/system.py`.
  - Compose multiple optical elements and propagation steps.
  - Support a simple sequential `run(input_field)` workflow.
  - Save intermediate fields optionally for debugging.

## Phase 2: Physical Validation

- [ ] Implement `configs/validate_grating.yaml`.
  - Wavelength, grid size, pixel pitch, grating period, propagation distance, expected diffraction orders.

- [ ] Implement `experiments/run_validation_grating.py`.
  - Generate a binary or sinusoidal grating.
  - Propagate to far field or lens focal plane.
  - Compare measured diffraction peak positions with `sin(theta_m) = m * lambda / d`.

- [x] Implement `configs/validate_lens.yaml`.
  - Wavelength, grid size, object distance, lens focal length, image distance.

- [x] Implement `experiments/run_validation_lens.py`.
  - Simulate thin-lens focusing and simple imaging.
  - Verify focal spot and thin-lens equation `1/f = 1/u + 1/v`.

- [ ] Implement `validation/test_propagation_energy.py`.
  - Check approximate energy conservation for free-space propagation under suitable sampling.

- [ ] Implement `validation/test_grating_diffraction.py`.
  - Programmatically assert diffraction peak positions.

- [ ] Implement `validation/test_lens_imaging.py`.
  - Programmatically assert focal location or image-plane sharpness.

## Phase 2.5: Custom Phase Field Input

- [ ] Implement `configs/custom_phase_field.yaml`.
  - Support user-defined arbitrary phase input.
  - Allow phase sources from analytic formulas, image files, NumPy arrays, or built-in generators.
  - Configure amplitude source independently from phase source.
  - Configure propagation distance and optional optical elements after the phase field.

- [ ] Implement `experiments/run_custom_phase_field.py`.
  - Read the custom phase field config.
  - Build the input field `U = A(x, y) * exp(j * phi(x, y))`.
  - Run the configured optical path.
  - Save intensity, phase, and intermediate field snapshots.

## Phase 3: Camera and Output Processing

- [ ] Implement `optics/camera.py`.
  - Convert complex field to intensity.
  - Add crop, ROI, binning, exposure scaling, clipping, and optional noise.
  - Add QPI-style reference-region normalization.

- [ ] Implement `optics/metrics.py`.
  - Add MSE.
  - Add PSNR.
  - Add Pearson correlation coefficient.
  - Add absolute phase error and percent phase error.
  - Add diffraction efficiency.

## Phase 4: SLM and Diffractive Layers

- [ ] Implement `optics/slm.py`.
  - Add ideal phase-only SLM.
  - Add amplitude-only SLM.
  - Add quantized phase levels for 6-bit, 7-bit, 8-bit testing.
  - Add a placeholder interface for calibrated TN-SLM behavior.

- [ ] Implement `optics/diffuser.py`.
  - Add random phase diffuser.
  - Support Gaussian-correlated random height maps.
  - Expose correlation length as a config parameter.

- [ ] Validate SLM-like masks with simple phase grating and lens phase patterns.

## Phase 5: Optional QPI / D2NN Paper-Inspired Demos

- [ ] Implement `datasets/phase_objects.py`.
  - Generate binary gratings.
  - Generate phase bars and simple geometric phase objects.
  - Later: convert MNIST-like images into phase objects.

- [ ] Implement `configs/qpi_phase_recovery_2022.yaml`.
  - Keep paper-inspired parameters only as optional reference values, not as a reproduction target.
  - Include input FOV, sampling pitch, layer size, number of layers, propagation distances, and output reference region if useful.

- [ ] Implement `experiments/run_qpi_phase_recovery.py`.
  - Build phase-only input object.
  - Build several phase-only diffractive layers.
  - First run without training using fixed/random masks.
  - Later add optimization.

- [ ] Implement `configs/qpi_diffuser_2023.yaml`.
  - Keep random diffuser settings only as optional reference values, not as a reproduction target.
  - Include random diffuser correlation length, object-to-diffuser distance, inter-layer distance, and output binning if useful.

- [ ] Implement `experiments/run_qpi_diffuser.py`.
  - Add random diffuser before diffractive layers.
  - Compare no-network propagation, lens imaging baseline, and diffractive-network output.

## Phase 6: Trainable D2NN

- [ ] Decide whether trainable modules live in `optics/` or a new `training/` folder.
- [ ] Port phase-only diffractive layers to PyTorch.
- [ ] Implement differentiable angular spectrum propagation.
- [ ] Train a tiny 1-layer or 2-layer phase-only network on binary phase gratings.
- [ ] Add QPI loss: normalized output intensity vs target phase image.
- [ ] Add power-efficiency penalty.
- [ ] Save trained phase masks and output comparisons.

## Phase 7: Real Device Modeling

- [ ] Collect actual SLM parameters: pixel pitch, resolution, wavelength range, polarization requirements, fill factor, transmittance.
- [ ] Add calibration data format for intensity-voltage and phase-voltage curves.
- [ ] Add `CalibratedSLM` model using measured lookup tables.
- [ ] Add TN-SLM placeholder model based on Jones matrices.
- [ ] Add hardware mismatch simulation: pixel quantization, dead pixels, phase noise, axial/lateral misalignment.

## Phase 8: Documentation and Reproducibility

- [ ] For each validation experiment, save figure plus JSON/YAML record of parameters.
- [ ] Add a short note comparing Python results with Matlab reference code.
- [ ] Add rendered examples to `outputs/figures/`.
- [ ] Keep paper-inspired configs separate from physically realistic lab configs.
