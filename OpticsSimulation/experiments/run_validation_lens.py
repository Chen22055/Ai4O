"""透镜聚焦/圆孔衍射 Airy pattern 验证实验入口。

本脚本读取 validate_lens.yaml，构造如下光路：
平面波 -> 圆孔光阑 -> 薄透镜 -> 焦距处自由传播。
输出焦平面强度图、局部裁剪图、对数强度图和径向强度曲线，并比较
仿真第一暗环位置与 Airy pattern 理论值 r1 = 1.22 * lambda * f / D。
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
from OpticsSimulation.optics.elements import CircularAperture, FreeSpace, ThinLens


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def radial_profile(intensity: np.ndarray, radius_grid: np.ndarray, max_radius: float, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, max_radius + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.empty_like(centers)

    for index, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (radius_grid >= left) & (radius_grid < right)
        profile[index] = float(np.mean(intensity[mask])) if np.any(mask) else np.nan

    return centers, profile


def estimate_first_minimum(centers: np.ndarray, profile: np.ndarray, theory_radius: float) -> float:
    search_mask = (centers > 0.65 * theory_radius) & (centers < 1.35 * theory_radius)
    if not np.any(search_mask):
        raise ValueError("no radial-profile samples are available near the theoretical first zero")
    search_profile = profile[search_mask]
    search_centers = centers[search_mask]
    return float(search_centers[np.nanargmin(search_profile)])


def save_intensity_images(
    intensity: np.ndarray,
    grid: Grid,
    output_dir: Path,
    prefix: str,
    dpi: int,
    crop_half_width: float,
) -> None:
    extent_mm = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[-1] * 1e3, grid.y[0] * 1e3]

    plt.figure(figsize=(6, 5))
    plt.imshow(intensity, cmap="inferno", extent=extent_mm)
    plt.colorbar(label="Normalized intensity")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Focal-plane intensity, linear scale")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_intensity_full_linear.png", dpi=dpi)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(intensity + 1e-8), cmap="inferno", extent=extent_mm)
    plt.colorbar(label="log10(normalized intensity)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Focal-plane intensity, log scale")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_intensity_full_log.png", dpi=dpi)
    plt.close()

    crop_mask_x = np.abs(grid.x) <= crop_half_width
    crop_mask_y = np.abs(grid.y) <= crop_half_width
    crop = intensity[np.ix_(crop_mask_y, crop_mask_x)]
    crop_extent_mm = [
        grid.x[crop_mask_x][0] * 1e3,
        grid.x[crop_mask_x][-1] * 1e3,
        grid.y[crop_mask_y][-1] * 1e3,
        grid.y[crop_mask_y][0] * 1e3,
    ]

    plt.figure(figsize=(6, 5))
    plt.imshow(crop, cmap="inferno", extent=crop_extent_mm, vmin=0.0, vmax=0.1)
    plt.colorbar(label="Normalized intensity")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Focal-plane intensity, cropped and clipped")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_intensity_crop_clipped.png", dpi=dpi)
    plt.close()


def save_radial_profile(
    centers: np.ndarray,
    profile: np.ndarray,
    theory_radius: float,
    simulated_radius: float,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(centers * 1e6, profile, label="simulation radial mean")
    plt.axvline(theory_radius * 1e6, color="r", linestyle="--", label="theory first zero")
    plt.axvline(simulated_radius * 1e6, color="g", linestyle=":", label="sim first minimum")
    plt.xlabel("radius (um)")
    plt.ylabel("normalized intensity")
    plt.yscale("log")
    plt.ylim(1e-5, 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_radial_profile.png", dpi=dpi)
    plt.close()


def run(config_path: Path) -> None:
    config = load_config(config_path)

    wavelength = float(config["simulation"]["wavelength"])
    shape = tuple(int(v) for v in config["grid"]["shape"])
    pixel_pitch = float(config["grid"]["pixel_pitch"])
    aperture_radius = float(config["aperture"]["radius"])
    focal_length = float(config["lens"]["focal_length"])
    propagation_distance = float(config["propagation"]["distance"])
    propagation_method = config["propagation"]["method"]

    output_dir = Path(config["output"]["directory"])
    prefix = config["output"]["prefix"]
    dpi = int(config["output"]["dpi"])
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid(shape=shape, pixel_pitch=pixel_pitch, wavelength=wavelength)
    input_field = Field.plane_wave(grid)
    system = OpticalSystem(
        [
            CircularAperture(aperture_radius),
            ThinLens(focal_length),
            FreeSpace(propagation_distance, method=propagation_method),
        ]
    )

    output_field = system.run(input_field)
    intensity = output_field.intensity
    intensity = intensity / np.max(intensity)

    diameter = 2.0 * aperture_radius
    theory_first_zero = 1.22 * wavelength * focal_length / diameter
    radius_grid = np.sqrt(grid.X**2 + grid.Y**2)
    centers, profile = radial_profile(
        intensity=intensity,
        radius_grid=radius_grid,
        max_radius=float(config["analysis"]["radial_max_radius"]),
        bin_width=max(grid.dx, grid.dy),
    )
    simulated_first_minimum = estimate_first_minimum(centers, profile, theory_first_zero)
    relative_error = abs(simulated_first_minimum - theory_first_zero) / theory_first_zero

    save_intensity_images(
        intensity=intensity,
        grid=grid,
        output_dir=output_dir,
        prefix=prefix,
        dpi=dpi,
        crop_half_width=float(config["analysis"]["crop_half_width"]),
    )
    save_radial_profile(
        centers=centers,
        profile=profile,
        theory_radius=theory_first_zero,
        simulated_radius=simulated_first_minimum,
        output_dir=output_dir,
        prefix=prefix,
        dpi=dpi,
    )

    print("Lens/Airy validation completed.")
    print(f"Config: {config_path}")
    print("Optical path: plane wave -> circular aperture -> thin lens -> free space to focal plane")
    print(f"Wavelength: {wavelength:.3e} m")
    print(f"Grid shape: {shape}, pixel pitch: {pixel_pitch:.3e} m")
    print(f"Aperture diameter: {diameter:.3e} m")
    print(f"Focal length: {focal_length:.3e} m")
    print(f"Theory first zero: {theory_first_zero * 1e6:.3f} um")
    print(f"Simulated first minimum: {simulated_first_minimum * 1e6:.3f} um")
    print(f"Relative error: {relative_error:.3%}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lens focusing / Airy pattern validation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("OpticsSimulation/configs/validate_lens.yaml"),
        help="Path to the lens validation YAML config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
