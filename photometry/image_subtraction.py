"""Image subtraction helpers built around HOTPANTS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from numpy.typing import NDArray

from .io import ImageStat

DEFAULT_HOTPANTS_BIN = Path("/Users/zexwu/cmd/bin/hotpants")
DEFAULT_GAUSSIANS: tuple[tuple[int, float], ...] = ((6, 0.7), (4, 1.5), (2, 3.0))


@dataclass(frozen=True)
class HotpantsSubtractionResult:
    """HOTPANTS subtraction outputs."""

    difference: NDArray
    noise: NDArray | None
    mask: NDArray[np.bool_]
    header: fits.Header
    convolved: NDArray | None
    convolved_header: fits.Header | None
    convolved_source: str | None
    command: tuple[str, ...]
    stdout: str
    stderr: str


def _fill_masked(data: NDArray, mask: NDArray[np.bool_] | None) -> NDArray:
    """Return finite float32 image data with masked pixels filled."""
    out = np.asarray(data, dtype=np.float32).copy()
    bad = ~np.isfinite(out)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)
    good = ~bad
    fill = float(np.nanmedian(out[good])) if np.any(good) else 0.0
    out[bad] = fill
    return out


def _write_image_and_mask(
    image_path: Path,
    mask_path: Path,
    data: NDArray,
    mask: NDArray[np.bool_] | None,
    header: fits.Header | None,
) -> None:
    """Write HOTPANTS-compatible image and mask files."""
    fits.PrimaryHDU(_fill_masked(data, mask), header=header).writeto(
        image_path,
        overwrite=True,
    )
    if mask is None:
        mask_data = np.zeros(np.asarray(data).shape, dtype=np.uint8)
    else:
        mask_data = np.asarray(mask, dtype=np.uint8)
    fits.PrimaryHDU(mask_data).writeto(mask_path, overwrite=True)


def _read_primary(path: Path) -> tuple[NDArray, fits.Header]:
    """Read a primary FITS image as float data plus header."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32), hdul[0].header.copy()


def hotpants_subtract(
    science_data: NDArray,
    reference_data: NDArray,
    *,
    science_header: fits.Header | None = None,
    reference_header: fits.Header | None = None,
    science_mask: NDArray[np.bool_] | None = None,
    reference_mask: NDArray[np.bool_] | None = None,
    science_stat: ImageStat | None = None,
    reference_stat: ImageStat | None = None,
    gaussian_list: Sequence[tuple[int, float]] = DEFAULT_GAUSSIANS,
    hotpants_bin: str | Path = DEFAULT_HOTPANTS_BIN,
    tmp_dir: str | Path | None = None,
    convolve: str | None = None,
    normalize: str = "t",
    kernel_half_width: int = 10,
    kernel_order: int = 2,
    background_order: int = 1,
    stamp_grid: tuple[int, int] = (10, 10),
    region_grid: tuple[int, int] = (1, 1),
    extra_args: Sequence[str] = (),
) -> HotpantsSubtractionResult:
    """Subtract ``reference`` from ``science`` with HOTPANTS.

    ``normalize="t"`` keeps the difference image on the reference/template flux
    scale, matching HOTPANTS' ``-n t`` option.
    """
    science_data = np.asarray(science_data, dtype=float)
    reference_data = np.asarray(reference_data, dtype=float)
    if science_data.shape != reference_data.shape:
        raise ValueError(
            "science and reference shapes must match, got "
            f"{science_data.shape} and {reference_data.shape}"
        )
    if convolve not in {None, "i", "t"}:
        raise ValueError(
            "convolve must be None, 'i' for science/image, or 't' for template"
        )
    if normalize not in {"i", "t", "u"}:
        raise ValueError("normalize must be 'i', 't', or 'u'")
    if len(gaussian_list) == 0:
        raise ValueError("gaussian_list must contain at least one (degree, sigma) pair")

    hotpants_bin = Path(hotpants_bin).expanduser()
    if not hotpants_bin.exists():
        raise FileNotFoundError(f"HOTPANTS binary not found: {hotpants_bin}")

    context = (
        tempfile.TemporaryDirectory(dir=tmp_dir)
        if tmp_dir is not None
        else tempfile.TemporaryDirectory()
    )
    with context as work_name:
        work = Path(work_name)
        sci_path = work / "science.fits"
        ref_path = work / "reference.fits"
        sci_mask_path = work / "science_mask.fits"
        ref_mask_path = work / "reference_mask.fits"
        diff_path = work / "difference.fits"
        noise_path = work / "difference_noise.fits"
        out_mask_path = work / "difference_mask.fits"
        conv_path = work / "convolved.fits"

        _write_image_and_mask(
            sci_path,
            sci_mask_path,
            science_data,
            science_mask,
            science_header,
        )
        _write_image_and_mask(
            ref_path,
            ref_mask_path,
            reference_data,
            reference_mask,
            reference_header,
        )

        ng_args: list[str] = [str(len(gaussian_list))]
        for degree, sigma in gaussian_list:
            ng_args.extend([str(int(degree)), f"{float(sigma):.6g}"])

        cmd = [
            str(hotpants_bin),
            "-inim",
            sci_path.name,
            "-tmplim",
            ref_path.name,
            "-outim",
            diff_path.name,
            "-imi",
            sci_mask_path.name,
            "-tmi",
            ref_mask_path.name,
            "-omi",
            out_mask_path.name,
            "-oni",
            noise_path.name,
            "-oci",
            conv_path.name,
            "-n",
            normalize,
            "-r",
            str(int(kernel_half_width)),
            "-ko",
            str(int(kernel_order)),
            "-bgo",
            str(int(background_order)),
            "-nsx",
            str(int(stamp_grid[0])),
            "-nsy",
            str(int(stamp_grid[1])),
            "-nrx",
            str(int(region_grid[0])),
            "-nry",
            str(int(region_grid[1])),
            "-ng",
            *ng_args,
            "-v",
            "0",
        ]
        if science_stat is not None:
            cmd.extend(
                [
                    "-ig",
                    str(float(science_stat.gain)),
                    "-ir",
                    str(float(science_stat.rdnoise)),
                ]
            )
        if reference_stat is not None:
            cmd.extend(
                [
                    "-tg",
                    str(float(reference_stat.gain)),
                    "-tr",
                    str(float(reference_stat.rdnoise)),
                ]
            )
        if convolve is not None:
            cmd.extend(["-c", convolve])
        cmd.extend(str(arg) for arg in extra_args)

        proc = subprocess.run(
            cmd,
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "HOTPANTS failed with exit code "
                f"{proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

        difference, header = _read_primary(diff_path)
        noise = None
        if noise_path.exists():
            noise, _ = _read_primary(noise_path)

        if out_mask_path.exists():
            out_mask, _ = _read_primary(out_mask_path)
            mask = np.asarray(out_mask != 0, dtype=bool)
        else:
            mask = ~np.isfinite(difference)

        convolved = None
        convolved_header = None
        if conv_path.exists():
            convolved, convolved_header = _read_primary(conv_path)

        return HotpantsSubtractionResult(
            difference=difference,
            noise=noise,
            mask=mask | ~np.isfinite(difference),
            header=header,
            convolved=convolved,
            convolved_header=convolved_header,
            convolved_source=convolve,
            command=tuple(cmd),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


def plot_subtraction_diagnostics(
    science_data: NDArray,
    reference_data: NDArray,
    result: HotpantsSubtractionResult,
    *,
    science_mask: NDArray[np.bool_] | None = None,
    reference_mask: NDArray[np.bool_] | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
    sigma_limit: float = 5.0,
) -> tuple[plt.Figure, NDArray]:
    """Inspect subtraction residuals in flux and noise-scaled sigma units."""
    science_data = np.asarray(science_data, dtype=float)
    reference_data = np.asarray(reference_data, dtype=float)
    difference = np.asarray(result.difference, dtype=float)
    mask = np.asarray(result.mask, dtype=bool) | ~np.isfinite(difference)
    if science_mask is not None:
        mask |= np.asarray(science_mask, dtype=bool)
    if reference_mask is not None:
        mask |= np.asarray(reference_mask, dtype=bool)

    if result.noise is not None:
        noise = np.asarray(result.noise, dtype=float)
        sigma_image = difference / np.where(noise > 0, noise, np.nan)
    else:
        valid = difference[~mask]
        scale = float(np.nanstd(valid)) if valid.size else 1.0
        sigma_image = difference / max(scale, 1.0e-6)

    valid_flux = difference[~mask & np.isfinite(difference)]
    valid_sigma = sigma_image[~mask & np.isfinite(sigma_image)]
    flux_abs = float(np.nanpercentile(np.abs(valid_flux), percentile[1])) if valid_flux.size else 1.0
    flux_abs = max(flux_abs, 1.0e-6)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax = axes.ravel()
    ax[0].imshow(
        science_data,
        origin="lower",
        cmap="gray",
        norm=simple_norm(science_data, "linear", percent=99.0),
    )
    ax[0].set_title("Science")
    ax[1].imshow(
        reference_data,
        origin="lower",
        cmap="gray",
        norm=simple_norm(reference_data, "linear", percent=99.0),
    )
    ax[1].set_title("Reference")
    ax[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-flux_abs,
        vmax=flux_abs,
    )
    ax[2].set_title("Difference flux")
    ax[3].imshow(
        sigma_image,
        origin="lower",
        cmap="coolwarm",
        vmin=-sigma_limit,
        vmax=sigma_limit,
    )
    ax[3].set_title("Difference sigma")
    ax[4].hist(valid_flux[np.isfinite(valid_flux)], bins=80, histtype="step")
    ax[4].set_title("Flux residuals")
    ax[5].hist(
        valid_sigma[np.isfinite(valid_sigma)],
        bins=80,
        range=(-sigma_limit, sigma_limit),
        histtype="step",
    )
    ax[5].set_title("Sigma residuals")

    for item in ax[:4]:
        item.set_xlabel("x")
        item.set_ylabel("y")
    fig.tight_layout()
    return fig, axes
