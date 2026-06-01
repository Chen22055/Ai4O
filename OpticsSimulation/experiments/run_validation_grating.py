"""光栅衍射验证实验入口。

本脚本读取 validate_grating.yaml，构造如下光路：
平面波 -> 光栅 -> 薄透镜 -> 焦距处自由传播。
透镜焦平面近似为光栅远场/Fourier 平面，因此第 m 级衍射峰的位置满足
x_m ≈ f * m * lambda / d，其中 d 为光栅周期。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OpticsSimulation.optics import Field, Grid, OpticalSystem
from OpticsSimulation.optics.elements import BinaryAmplitudeGrating, FreeSpace, SinusoidalPhaseGrating, ThinLens


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_grating(config: dict):
    grating_config = config["grating"]
    grating_type = grating_config["type"]
    if grating_type == "binary_amplitude":
        return BinaryAmplitudeGrating(
            period=float(grating_config["period"]),
            duty_cycle=float(grating_config["duty_cycle"]),
            orientation=grating_config["orientation"],
        )
    if grating_type == "sinusoidal_phase":
        return SinusoidalPhaseGrating(
            period=float(grating_config["period"]),
            phase_depth=float(grating_config["phase_depth"]),
            orientation=grating_config["orientation"],
        )
    raise ValueError(f"unknown grating type: {grating_type}")


def central_profile(intensity: np.ndarray, orientation: str, half_width: int) -> np.ndarray:
    ny, nx = intensity.shape
    if orientation == "x":
        center = ny // 2
        rows = slice(max(center - half_width, 0), min(center + half_width + 1, ny))
        return np.mean(intensity[rows, :], axis=0)
    if orientation == "y":
        center = nx // 2
        cols = slice(max(center - half_width, 0), min(center + half_width + 1, nx))
        return np.mean(intensity[:, cols], axis=1)
    raise ValueError("orientation must be 'x' or 'y'")


def find_peak_position(axis: np.ndarray, profile: np.ndarray, expected_position: float, half_width: float) -> float:
    mask = (axis >= expected_position - half_width) & (axis <= expected_position + half_width)
    if not np.any(mask):
        raise ValueError(f"no samples near expected peak position {expected_position}")
    local_axis = axis[mask]
    local_profile = profile[mask]
    return float(local_axis[np.argmax(local_profile)])


def save_figures(
    grid: Grid,
    intensity: np.ndarray,
    axis: np.ndarray,
    profile: np.ndarray,
    theory_positions: dict[int, float],
    measured_positions: dict[int, float],
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    extent_mm = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[-1] * 1e3, grid.y[0] * 1e3]

    plt.figure(figsize=(7, 5))
    plt.imshow(np.log10(intensity + 1e-8), cmap="inferno", extent=extent_mm)
    plt.colorbar(label="log10(normalized intensity)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Grating diffraction pattern at lens focal plane")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_intensity_log.png", dpi=dpi)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(axis * 1e3, profile, label="simulation central profile")
    for order, position in theory_positions.items():
        plt.axvline(position * 1e3, color="r", linestyle="--", alpha=0.45)
        plt.text(position * 1e3, 0.8, f"m={order}", rotation=90, va="center", ha="right", fontsize=8)
    for position in measured_positions.values():
        plt.axvline(position * 1e3, color="g", linestyle=":", alpha=0.75)
    plt.xlabel("position (mm)")
    plt.ylabel("normalized intensity")
    plt.yscale("log")
    plt.ylim(1e-6, 1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_central_profile.png", dpi=dpi)
    plt.close()


def run(config_path: Path) -> None:
    config = load_config(config_path)

    wavelength = float(config["simulation"]["wavelength"])
    shape = tuple(int(v) for v in config["grid"]["shape"])
    pixel_pitch = float(config["grid"]["pixel_pitch"])
    focal_length = float(config["lens"]["focal_length"])
    propagation_distance = float(config["propagation"]["distance"])
    propagation_method = config["propagation"]["method"]
    grating_period = float(config["grating"]["period"])
    orientation = config["grating"]["orientation"]

    output_dir = Path(config["output"]["directory"])
    prefix = config["output"]["prefix"]
    dpi = int(config["output"]["dpi"])
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid(shape=shape, pixel_pitch=pixel_pitch, wavelength=wavelength)
    input_field = Field.plane_wave(grid)
    system = OpticalSystem(
        [
            build_grating(config),
            ThinLens(focal_length),
            FreeSpace(propagation_distance, method=propagation_method),
        ]
    )

    output_field = system.run(input_field)
    intensity = output_field.intensity
    intensity = intensity / np.max(intensity)

    profile = central_profile(
        intensity=intensity,
        orientation=orientation,
        half_width=int(config["analysis"]["profile_average_half_width"]),
    )
    axis = grid.x if orientation == "x" else grid.y

    orders = [int(order) for order in config["analysis"]["orders"]]
    peak_search_half_width = float(config["analysis"]["peak_search_half_width"])
    theory_positions: dict[int, float] = {}
    measured_positions: dict[int, float] = {}
    errors: dict[int, float] = {}

    for order in orders:
        sine_theta = order * wavelength / grating_period
        if abs(sine_theta) >= 1:
            continue
        theta = np.arcsin(sine_theta)
        theory_position = focal_length * np.tan(theta)
        measured_position = find_peak_position(axis, profile, theory_position, peak_search_half_width)
        theory_positions[order] = float(theory_position)
        measured_positions[order] = measured_position
        errors[order] = abs(measured_position - theory_position)

    save_figures(
        grid=grid,
        intensity=intensity,
        axis=axis,
        profile=profile,
        theory_positions=theory_positions,
        measured_positions=measured_positions,
        output_dir=output_dir,
        prefix=prefix,
        dpi=dpi,
    )

    print("Grating diffraction validation completed.")
    print(f"Config: {config_path}")
    print("Optical path: plane wave -> grating -> thin lens -> free space to focal plane")
    print(f"Grating type: {config['grating']['type']}")
    print(f"Wavelength: {wavelength:.3e} m")
    print(f"Grating period: {grating_period:.3e} m")
    print(f"Focal length: {focal_length:.3e} m")
    print(f"Output directory: {output_dir}")
    print("Diffraction order peak positions:")
    for order in sorted(theory_positions):
        print(
            f"  m={order:+d}: theory={theory_positions[order] * 1e3:+.4f} mm, "
            f"measured={measured_positions[order] * 1e3:+.4f} mm, "
            f"abs_error={errors[order] * 1e6:.2f} um"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grating diffraction validation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("OpticsSimulation/configs/validate_grating.yaml"),
        help="Path to the grating validation YAML config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
