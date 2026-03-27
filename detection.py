"""Source-detection and FWHM-estimation helpers."""

from __future__ import annotations

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

from .catalog import Catalog


def flux_to_mag(flux: NDArray, zeropoint: float) -> NDArray:
    """Convert flux to magnitude with a given zeropoint."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux) + zeropoint
        mag = np.where(flux > 0, mag, np.nan)
    return mag


def flux_err_to_mag_err(flux: NDArray, flux_err: NDArray) -> NDArray:
    """Propagate flux error to magnitude error."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 2.5 / np.log(10.0)
        merr = k * np.abs(flux_err / flux)
        merr = np.where(flux > 0, merr, np.nan)
    return merr


def detect_star_catalog(
    data: NDArray,
    *,
    finder_fwhm: float,
    threshold_sigma: float,
    saturation_level: float,
    use_background: bool,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[Catalog, float, NDArray | None, float]:
    """Detect stars and return catalog plus background statistics.

    Returns ``(catalog, background_scalar, background2d_or_none, detect_std)``.
    """
    data = np.asarray(data, dtype=float)
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    background2d = None
    if use_background:
        min_dim = min(data.shape)
        box = max(16, min(64, min_dim // 8))
        if box <= 1:
            fill = float(np.nanmedian(data[~mask])) if np.any(~mask) else 0.0
            bkg_image = np.full_like(data, fill, dtype=float)
            background = fill
        else:
            bkg2d = Background2D(
                data,
                box_size=(box, box),
                filter_size=(3, 3),
                sigma_clip=SigmaClip(sigma=3.0),
                bkg_estimator=MedianBackground(),
                mask=mask,
            )
            bkg_image = np.asarray(bkg2d.background, dtype=float)
            background2d = bkg_image
            background = float(np.nanmedian(bkg_image))

        det_data = data - bkg_image
        _, median, std = sigma_clipped_stats(det_data, sigma=3.0, mask=mask)
    else:
        _, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
        background = float(median)
        det_data = data - median

    det_data = np.array(det_data, copy=True)
    det_data[mask] = median

    finder = DAOStarFinder(
        fwhm=finder_fwhm,
        threshold=threshold_sigma * std,
        sharplo=0.2,
        sharphi=1.2,
        roundlo=-1.0,
        roundhi=1.0,
    )
    sources = finder(det_data)

    if sources is None or len(sources) == 0:
        return Catalog(), background, background2d, float(std)

    xi = np.rint(np.asarray(sources["xcentroid"], dtype=float)).astype(int)
    yi = np.rint(np.asarray(sources["ycentroid"], dtype=float)).astype(int)
    inside = (
        (0 <= xi)
        & (xi < data.shape[1])
        & (0 <= yi)
        & (yi < data.shape[0])
    )
    sources = sources[inside]
    if len(sources) == 0:
        return Catalog(), background, background2d, float(std)

    xi = np.rint(np.asarray(sources["xcentroid"], dtype=float)).astype(int)
    yi = np.rint(np.asarray(sources["ycentroid"], dtype=float)).astype(int)
    sources = sources[~mask[yi, xi]]
    if len(sources) == 0:
        return Catalog(), background, background2d, float(std)

    sources = sources[sources["peak"] < saturation_level]
    flux = np.asarray(sources["flux"], dtype=float)
    fluxerr = np.sqrt(np.clip(flux, 0.0, None))

    catalog = Catalog.from_arrays(
        x=np.asarray(sources["xcentroid"], dtype=float),
        y=np.asarray(sources["ycentroid"], dtype=float),
        mag=flux_to_mag(flux, zeropoint=0.0),
        mag_err=flux_err_to_mag_err(flux, fluxerr),
    )
    return catalog, background, background2d, float(std)
