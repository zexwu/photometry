"""Synthetic end-to-end tests for analytical Moffat PSF photometry."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from photometry.catalog import Catalog
from photometry.image import Image
from photometry.io import ImageStat
from photometry.photometry import (
    AnalyticalMoffatPSF,
    build_analytical_moffat_psf,
    run_analytical_psf_photometry,
)


@dataclass(frozen=True)
class SyntheticTruth:
    """Ground truth for one synthetic crowded stellar field."""

    psf: AnalyticalMoffatPSF
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    background2d: np.ndarray
    stat: ImageStat


def _synthetic_catalog(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Catalog:
    """Return a slightly perturbed starting catalog."""
    return Catalog.from_arrays(
        x=x + rng.normal(0.0, 0.08, len(x)),
        y=y + rng.normal(0.0, 0.08, len(y)),
        mag=np.zeros(len(x), dtype=float),
        mag_err=np.full(len(x), 0.01, dtype=float),
    )


def make_synthetic_dataset(
    seed: int = 7,
    *,
    background_gradient: bool = True,
    poisson_noise: bool = True,
) -> tuple[np.ndarray, Catalog, SyntheticTruth]:
    """Build a noisy synthetic image with controllable sky and noise model."""
    rng = np.random.default_rng(seed)
    ny, nx = 128, 128
    yy, xx = np.mgrid[:ny, :nx]

    psf = AnalyticalMoffatPSF(
        alpha_major=1.9,
        alpha_minor=1.4,
        beta=3.2,
        theta=np.deg2rad(23.0),
    )
    x = np.array([18.3, 31.1, 44.8, 58.4, 61.9, 76.2, 26.6, 52.4, 71.7, 94.0])
    y = np.array([21.5, 37.3, 19.6, 43.2, 46.0, 27.1, 69.0, 73.0, 64.8, 88.0])
    flux = np.array([7000, 6200, 8200, 8800, 4500, 6100, 7300, 6700, 7200, 5600])

    if background_gradient:
        background2d = 120.0 + 0.08 * xx + 0.05 * yy
    else:
        background2d = np.full((ny, nx), 120.0, dtype=float)
    image = np.array(background2d, copy=True)
    for xc, yc, fc in zip(x, y, flux):
        image += fc * psf.unit_flux_image(xx, yy, xc, yc)

    if poisson_noise:
        noisy = rng.poisson(np.clip(image, 0.0, None)).astype(float)
    else:
        noisy = np.array(image, copy=True)
    noisy += rng.normal(0.0, 2.5, size=image.shape)

    stat = ImageStat(
        fwhm=psf.fwhm,
        gain=1.0,
        rdnoise=2.5,
        background=float(np.median(background2d)),
        background2d=background2d,
    )
    return noisy, _synthetic_catalog(x, y, rng), SyntheticTruth(
        psf=psf,
        x=x,
        y=y,
        flux=flux,
        background2d=background2d,
        stat=stat,
    )


def _match_truth(catalog: Catalog, truth: SyntheticTruth) -> tuple[np.ndarray, np.ndarray]:
    """Match fitted catalog rows back to the synthetic truth positions."""
    tree = cKDTree(np.c_[catalog.x, catalog.y])
    dist, idx = tree.query(np.c_[truth.x, truth.y], k=1)
    return np.asarray(dist, dtype=float), np.asarray(idx, dtype=int)


def test_low_level_analytical_psf_photometry_recovers_synthetic_field() -> None:
    """Low-level builder and fitter should recover PSF shape and stellar fluxes."""
    data, catalog, truth = make_synthetic_dataset(
        background_gradient=False,
        poisson_noise=False,
    )
    psf, _, stars_used = build_analytical_moffat_psf(
        data,
        catalog,
        stat=truth.stat,
        cutout_size=19,
        max_stars=8,
    )
    fitted_catalog, result = run_analytical_psf_photometry(
        data,
        catalog,
        psf=psf,
        stat=truth.stat,
        zeropoint=25.0,
        maxiters=4,
    )

    assert stars_used >= 5
    assert abs(psf.fwhm - truth.psf.fwhm) < 0.2
    assert abs(psf.ellipticity - truth.psf.ellipticity) < 0.03
    assert abs(np.degrees(psf.theta - truth.psf.theta)) < 3.0

    dist, idx = _match_truth(fitted_catalog, truth)
    assert len(fitted_catalog) == len(truth.x)
    assert float(np.max(dist)) < 0.35

    fitted_flux = 10.0 ** (-0.4 * (fitted_catalog.mag[idx] - 25.0))
    flux_frac_err = np.abs(fitted_flux - truth.flux) / truth.flux
    assert float(np.median(flux_frac_err)) < 0.05
    assert float(np.nanstd(result.residual_image)) < 5.0


def test_image_workflow_runs_analytical_psf_photometry_on_synthetic_data() -> None:
    """High-level Image workflow should run end-to-end on synthetic data."""
    data, catalog, truth = make_synthetic_dataset(seed=11)
    header = fits.Header()
    header["GAIN"] = truth.stat.gain
    header["RDNOISE"] = truth.stat.rdnoise
    header["SEEING"] = truth.stat.fwhm

    img = Image(path="synthetic.fits", data=data, header=header)
    img.catalog = catalog
    img.stat = ImageStat(
        fwhm=truth.stat.fwhm,
        gain=truth.stat.gain,
        rdnoise=truth.stat.rdnoise,
        background=truth.stat.background,
        background2d=truth.background2d,
    )

    img.build_analytical_psf(max_stars=8).run_analytical_psf_photometry(
        zeropoint=25.0,
        maxiters=4,
        inspect=True
    )
    plt.show()

    dist, idx = _match_truth(img.catalog, truth)
    fitted_flux = 10.0 ** (-0.4 * (img.catalog.mag[idx] - 25.0))
    flux_frac_err = np.abs(fitted_flux - truth.flux) / truth.flux

    assert len(img.catalog) == len(truth.x)
    assert float(np.max(dist)) < 0.35
    assert float(np.median(flux_frac_err)) < 0.06
    assert float(np.nanmedian(img.catalog.mag_err)) < 0.03
    assert "build_analytical_psf" in img.note
    assert "run_analytical_psf_photometry" in img.note
