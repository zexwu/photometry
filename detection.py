from __future__ import annotations

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

from .catalog import StarCatalog
from .fwhm import estimate_fwhm


def flux_to_mag(flux: NDArray, zeropoint: float) -> NDArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux) + zeropoint
        mag = np.where(flux > 0, mag, np.nan)
    return mag


def flux_err_to_mag_err(flux: NDArray, flux_err: NDArray) -> NDArray:
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
) -> tuple[StarCatalog, float, NDArray | None, float]:
    background2d = None
    if use_background:
        min_dim = min(data.shape)
        box = max(16, min(64, min_dim // 8))
        if box <= 1:
            bkg_image = np.full_like(data, np.nanmedian(data), dtype=float)
            background = float(np.nanmedian(bkg_image))
        else:
            bkg2d = Background2D(
                data,
                box_size=(box, box),
                filter_size=(3, 3),
                sigma_clip=SigmaClip(sigma=3.0),
                bkg_estimator=MedianBackground(),
            )
            bkg_image = np.asarray(bkg2d.background, dtype=float)
            background2d = bkg_image
            background = float(np.nanmedian(bkg_image))

        det_data = data - bkg_image
        _, median, std = sigma_clipped_stats(det_data, sigma=3.0)
    else:
        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        background = float(median)
        det_data = data - median

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
        return StarCatalog(), background, background2d, float(std)

    sources = sources[sources["peak"] < saturation_level]
    flux = np.asarray(sources["flux"], dtype=float)
    fluxerr = np.sqrt(np.clip(flux, 0.0, None))

    catalog = StarCatalog.from_arrays(
        x=np.asarray(sources["xcentroid"], dtype=float),
        y=np.asarray(sources["ycentroid"], dtype=float),
        mag=flux_to_mag(flux, zeropoint=0.0),
        mag_err=flux_err_to_mag_err(flux, fluxerr),
    )
    return catalog, background, background2d, float(std)


def estimate_catalog_fwhm(
    data: NDArray,
    catalog: StarCatalog,
    *,
    max_stars: int,
    half_size: int,
) -> tuple[float, int]:
    ind = np.argsort(catalog.mag)[:max_stars]
    fwhm = estimate_fwhm(data, catalog.x[ind], catalog.y[ind], half_size=half_size)
    return float(fwhm), int(len(ind))
