"""PSF/ePSF and aperture-photometry helper functions."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.visualization import simple_norm
from numpy.typing import NDArray
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.background import LocalBackground
from photutils.psf import EPSFBuilder, PSFPhotometry, extract_stars
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .catalog import Catalog
from .detection import flux_err_to_mag_err, flux_to_mag
from .io import ImageStat

GAUSSIAN_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class AnalyticalMoffatPSF:
    """Elliptical, rotated Moffat PSF normalized to unit total flux."""

    alpha_major: float
    alpha_minor: float
    beta: float
    theta: float

    @property
    def axis_ratio(self) -> float:
        """Minor-to-major axis ratio."""
        return float(self.alpha_minor / self.alpha_major)

    @property
    def ellipticity(self) -> float:
        """Simple ellipticity definition ``1 - b / a``."""
        return float(1.0 - self.axis_ratio)

    @property
    def peak(self) -> float:
        """Central surface-brightness for a unit-flux PSF."""
        return float((self.beta - 1.0) / (np.pi * self.alpha_major * self.alpha_minor))

    @property
    def fwhm_major(self) -> float:
        """FWHM along the major axis in pixels."""
        return float(
            2.0 * self.alpha_major * np.sqrt(2.0 ** (1.0 / self.beta) - 1.0)
        )

    @property
    def fwhm_minor(self) -> float:
        """FWHM along the minor axis in pixels."""
        return float(
            2.0 * self.alpha_minor * np.sqrt(2.0 ** (1.0 / self.beta) - 1.0)
        )

    @property
    def fwhm(self) -> float:
        """Equivalent circularized FWHM in pixels."""
        return float(np.sqrt(self.fwhm_major * self.fwhm_minor))

    def unit_flux_image(
        self,
        x: NDArray,
        y: NDArray,
        x0: float,
        y0: float,
    ) -> NDArray:
        """Evaluate the unit-flux PSF image on a pixel-center grid."""
        dx = np.asarray(x, dtype=float) - x0
        dy = np.asarray(y, dtype=float) - y0
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        xp = dx * cos_t + dy * sin_t
        yp = -dx * sin_t + dy * cos_t
        rr = (xp / self.alpha_major) ** 2 + (yp / self.alpha_minor) ** 2
        return self.peak * (1.0 + rr) ** (-self.beta)

    def evaluate(
        self,
        x: NDArray,
        y: NDArray,
        *,
        flux: float,
        x0: float,
        y0: float,
        background: float = 0.0,
        grad_x: float = 0.0,
        grad_y: float = 0.0,
        ref_x: float | None = None,
        ref_y: float | None = None,
    ) -> NDArray:
        """Evaluate the PSF plus an optional local background plane."""
        model = flux * self.unit_flux_image(x, y, x0, y0)
        if background != 0.0 or grad_x != 0.0 or grad_y != 0.0:
            if ref_x is None:
                ref_x = x0
            if ref_y is None:
                ref_y = y0
            model = model + background
            model = model + grad_x * (np.asarray(x, dtype=float) - ref_x)
            model = model + grad_y * (np.asarray(y, dtype=float) - ref_y)
        return model

    def stamp(self, size: int | None = None) -> NDArray:
        """Return a centered PSF stamp for visualization."""
        if size is None:
            size = int(np.ceil(8.0 * self.fwhm_major))
        size = max(15, int(size))
        if size % 2 == 0:
            size += 1
        half = size // 2
        yy, xx = np.mgrid[:size, :size]
        return self.unit_flux_image(xx, yy, half, half)


@dataclass
class AnalyticalPSFPhotometryResult:
    """Outputs from analytical PSF photometry."""

    table: Table
    model_image: NDArray
    residual_image: NDArray
    n_iterations: int


@dataclass
class _StarCutout:
    """Internal isolated-star cutout bundle."""

    index: int
    data: NDArray
    mask: NDArray[np.bool_]
    x0: float
    y0: float
    x_abs: float
    y_abs: float
    background: float
    noise: float
    flux_init: float
    shape_guess: tuple[float, float, float] | None


def _wrap_theta(theta: float) -> float:
    """Wrap an ellipse orientation to ``[-pi/2, pi/2)``."""
    return float(((theta + 0.5 * np.pi) % np.pi) - 0.5 * np.pi)


def _canonicalize_moffat_shape(
    alpha_major: float,
    alpha_minor: float,
    theta: float,
) -> tuple[float, float, float]:
    """Ensure the first axis is the major axis."""
    if alpha_minor > alpha_major:
        alpha_major, alpha_minor = alpha_minor, alpha_major
        theta += 0.5 * np.pi
    return float(alpha_major), float(alpha_minor), _wrap_theta(theta)


def _border_pixels(cut: NDArray, mask: NDArray[np.bool_] | None = None) -> NDArray:
    """Return perimeter pixels from a cutout, excluding masked entries."""
    top = cut[0, :]
    bottom = cut[-1, :]
    left = cut[1:-1, 0]
    right = cut[1:-1, -1]
    border = np.concatenate((top, bottom, left, right))
    if mask is None:
        return border[np.isfinite(border)]

    border_mask = np.concatenate(
        (mask[0, :], mask[-1, :], mask[1:-1, 0], mask[1:-1, -1])
    )
    good = np.isfinite(border) & (~border_mask)
    return border[good]


def _cutout_border_stats(
    cut: NDArray,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[float, float]:
    """Robust background and scatter estimate from cutout edges."""
    border = _border_pixels(cut, mask)
    if border.size == 0:
        finite = cut[np.isfinite(cut)]
        if finite.size == 0:
            return 0.0, 1.0
        median = float(np.nanmedian(finite))
        std = float(np.nanstd(finite))
        return median, max(std, 1.0e-6)

    median = float(np.nanmedian(border))
    std = float(np.nanstd(border))
    return median, max(std, 1.0e-6)


def _cutout_slices(
    x: float,
    y: float,
    *,
    half_size: int,
    shape: tuple[int, int],
) -> tuple[slice, slice, float, float] | None:
    """Return image slices and local subpixel center for a square cutout."""
    xi = int(np.rint(x))
    yi = int(np.rint(y))
    ny, nx = shape
    if not (
        half_size <= xi < nx - half_size and half_size <= yi < ny - half_size
    ):
        return None

    xs = slice(xi - half_size, xi + half_size + 1)
    ys = slice(yi - half_size, yi + half_size + 1)
    return ys, xs, x - xs.start, y - ys.start


def _shape_guess_from_moments(
    cut: NDArray,
    *,
    mask: NDArray[np.bool_] | None,
    x0: float,
    y0: float,
) -> tuple[float, float, float] | None:
    """Estimate major/minor Gaussian widths and rotation from moments."""
    yy, xx = np.mgrid[: cut.shape[0], : cut.shape[1]]
    background, _ = _cutout_border_stats(cut, mask)
    signal = np.clip(cut - background, 0.0, None)
    if mask is not None:
        signal = np.where(mask, 0.0, signal)

    total = float(np.sum(signal))
    if not np.isfinite(total) or total <= 0.0:
        return None

    dx = xx - x0
    dy = yy - y0
    mxx = float(np.sum(signal * dx * dx) / total)
    myy = float(np.sum(signal * dy * dy) / total)
    mxy = float(np.sum(signal * dx * dy) / total)
    cov = np.array([[mxx, mxy], [mxy, myy]], dtype=float)

    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    evals = np.clip(evals, 1.0e-4, None)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    sigma_major = float(np.sqrt(evals[0]))
    sigma_minor = float(np.sqrt(evals[1]))
    theta = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return sigma_major, sigma_minor, _wrap_theta(theta)


def _estimate_cutout_flux(
    cut: NDArray,
    *,
    mask: NDArray[np.bool_] | None,
    x0: float,
    y0: float,
    radius: float,
) -> float:
    """Crude background-subtracted flux estimate for fit initialization."""
    yy, xx = np.mgrid[: cut.shape[0], : cut.shape[1]]
    background, _ = _cutout_border_stats(cut, mask)
    signal = cut - background
    good = np.isfinite(signal)
    if mask is not None:
        good &= ~mask
    core = ((xx - x0) ** 2 + (yy - y0) ** 2) <= radius**2
    flux = float(np.sum(np.clip(signal[good & core], 0.0, None)))
    if flux <= 0.0:
        flux = float(np.sum(np.clip(signal[good], 0.0, None)))
    return max(flux, 1.0e-6)


def _least_squares_errors(fit: object) -> NDArray:
    """Approximate one-sigma parameter uncertainties from a least-squares fit."""
    jac = getattr(fit, "jac", None)
    x = getattr(fit, "x", None)
    fun = getattr(fit, "fun", None)
    if jac is None or x is None or fun is None:
        return np.full(0, np.nan, dtype=float)

    jac = np.asarray(jac, dtype=float)
    if jac.ndim != 2 or jac.shape[0] <= jac.shape[1]:
        return np.full(len(x), np.nan, dtype=float)

    dof = max(1, jac.shape[0] - jac.shape[1])
    scale = float(np.sum(np.asarray(fun, dtype=float) ** 2) / dof)
    try:
        cov = np.linalg.inv(jac.T @ jac) * scale
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jac.T @ jac) * scale

    diag = np.diag(cov)
    diag = np.where(diag >= 0.0, diag, np.nan)
    return np.sqrt(diag)


def _select_isolated_star_indices(
    data: NDArray,
    catalog: Catalog,
    *,
    cutout_size: int,
    max_stars: int,
    mask: NDArray[np.bool_] | None = None,
    isolation_radius: float | None = None,
    contamination_fraction: float = 0.25,
) -> NDArray[np.int_]:
    """Rank stars by isolation and return indices suitable for PSF building."""
    ny, nx = data.shape
    half = cutout_size // 2
    margin = half + 1
    isolation_radius = (
        1.0 * cutout_size if isolation_radius is None else float(isolation_radius)
    )

    x = np.asarray(catalog.x, dtype=float)
    y = np.asarray(catalog.y, dtype=float)
    base_good = (x >= margin) & (x < nx - margin) & (y >= margin) & (y < ny - margin)
    nn_dist = np.full(len(catalog), np.inf, dtype=float)

    if len(catalog) > 1:
        coords = np.c_[x, y]
        dists, _ = cKDTree(coords).query(coords, k=2)
        nn_dist = dists[:, 1]

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    contamination_ratio = np.full(len(catalog), np.inf, dtype=float)
    snr = np.zeros(len(catalog), dtype=float)
    for i, (xc, yc) in enumerate(zip(xi, yi)):
        if not base_good[i]:
            continue

        cut = data[
            yc - half : yc + half + 1,
            xc - half : xc + half + 1,
        ]
        if cut.shape != (cutout_size, cutout_size) or not np.isfinite(cut).all():
            base_good[i] = False
            continue

        if mask is not None:
            cut_mask = mask[
                yc - half : yc + half + 1,
                xc - half : xc + half + 1,
            ]
            if np.any(cut_mask):
                base_good[i] = False
                continue

        background, noise = _cutout_border_stats(cut, cut_mask if mask is not None else None)
        cut_sub = cut - background
        central_peak = np.max(cut_sub[half - 1 : half + 2, half - 1 : half + 2])
        if not np.isfinite(central_peak) or central_peak <= 0:
            base_good[i] = False
            continue
        snr[i] = central_peak / max(noise, 1.0e-6)

        outer = cut_sub.copy()
        outer[half - 1 : half + 2, half - 1 : half + 2] = -np.inf
        contaminant_peak = np.max(outer)
        contamination_ratio[i] = max(0.0, contaminant_peak) / central_peak

    preferred = base_good.copy()
    preferred &= nn_dist >= isolation_radius
    preferred &= contamination_ratio <= contamination_fraction

    rank = np.lexsort((catalog.mag, contamination_ratio, -nn_dist, -snr))
    preferred_idx = rank[preferred[rank]]
    base_idx = rank[base_good[rank]]

    if len(preferred_idx) >= 3:
        return np.asarray(preferred_idx[:max_stars], dtype=int)
    return np.asarray(base_idx[:max_stars], dtype=int)


def build_epsf_model(
    data: NDArray,
    catalog: Catalog,
    *,
    cutout_size: int,
    oversample: int,
    max_stars: int,
    mask: NDArray[np.bool_] | None = None,
    isolation_radius: float | None = None,
    contamination_fraction: float = 0.25,
    maxiters: int = 10,
) -> tuple[object, object, int]:
    """Build an ePSF model and return ``(epsf, stars, stars_used)``."""
    if cutout_size < 3 or cutout_size % 2 == 0:
        raise ValueError("cutout_size must be an odd integer >= 3")
    idx = _select_isolated_star_indices(
        data,
        catalog,
        cutout_size=cutout_size,
        max_stars=max_stars,
        mask=mask,
        isolation_radius=isolation_radius,
        contamination_fraction=contamination_fraction,
    )

    if len(idx) < 3:
        raise ValueError(f"not enough usable stars for ePSF: selected={len(idx)}")

    stars_tbl = Table()
    stars_tbl["x"] = catalog.x[idx]
    stars_tbl["y"] = catalog.y[idx]

    stars = extract_stars(
        NDData(data=data, mask=mask),
        stars_tbl,
        size=(cutout_size, cutout_size),
    )
    if stars is None or len(stars) < 3:
        raise ValueError(f"failed to extract enough ePSF stars: extracted={len(stars) if stars is not None else 0}")

    epsf, _ = EPSFBuilder(
        oversampling=oversample,
        maxiters=maxiters,
        progress_bar=True,
        smoothing_kernel="quartic",
    )(stars)
    if epsf is None or getattr(epsf, "data", None) is None:
        raise ValueError("ePSF builder failed to produce a valid model")

    return epsf, stars, int(len(idx))


def _extract_isolated_star_cutouts(
    data: NDArray,
    catalog: Catalog,
    *,
    cutout_size: int,
    max_stars: int,
    mask: NDArray[np.bool_] | None = None,
    isolation_radius: float | None = None,
    contamination_fraction: float = 0.25,
) -> list[_StarCutout]:
    """Extract high-quality isolated stellar cutouts for analytical PSF fitting."""
    idx = _select_isolated_star_indices(
        data,
        catalog,
        cutout_size=cutout_size,
        max_stars=max_stars,
        mask=mask,
        isolation_radius=isolation_radius,
        contamination_fraction=contamination_fraction,
    )
    if len(idx) < 3:
        raise ValueError(
            f"not enough isolated stars for analytical PSF: selected={len(idx)}"
        )

    half_size = cutout_size // 2
    radius = max(2.0, min(0.45 * cutout_size, half_size - 0.5))
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    cutouts: list[_StarCutout] = []
    for star_index in idx:
        res = _cutout_slices(
            float(catalog.x[star_index]),
            float(catalog.y[star_index]),
            half_size=half_size,
            shape=data.shape,
        )
        if res is None:
            continue
        ys, xs, x0, y0 = res
        cut = np.asarray(data[ys, xs], dtype=float)
        cut_mask = np.asarray(mask[ys, xs], dtype=bool)
        if cut.shape != (cutout_size, cutout_size) or np.any(cut_mask):
            continue
        if not np.isfinite(cut).all():
            continue

        background, noise = _cutout_border_stats(cut, cut_mask)
        shape_guess = _shape_guess_from_moments(cut, mask=cut_mask, x0=x0, y0=y0)
        flux_init = _estimate_cutout_flux(
            cut,
            mask=cut_mask,
            x0=x0,
            y0=y0,
            radius=radius,
        )
        cutouts.append(
            _StarCutout(
                index=int(star_index),
                data=cut,
                mask=cut_mask,
                x0=float(x0),
                y0=float(y0),
                x_abs=float(catalog.x[star_index]),
                y_abs=float(catalog.y[star_index]),
                background=background,
                noise=noise,
                flux_init=flux_init,
                shape_guess=shape_guess,
            )
        )

    if len(cutouts) < 3:
        raise ValueError(
            f"failed to extract enough isolated stellar cutouts: extracted={len(cutouts)}"
        )
    return cutouts


def _initial_moffat_shape(
    cutouts: list[_StarCutout],
    *,
    fwhm_guess: float | None = None,
    beta_guess: float = 3.5,
) -> tuple[float, float, float, float]:
    """Build a stable starting point for the global Moffat shape fit."""
    alpha_scale = 1.0 / (2.0 * np.sqrt(2.0 ** (1.0 / beta_guess) - 1.0))
    major_vals: list[float] = []
    minor_vals: list[float] = []
    theta_vals: list[float] = []
    eq_fwhm_vals: list[float] = []

    for cutout in cutouts:
        if cutout.shape_guess is None:
            continue
        sigma_major, sigma_minor, theta = cutout.shape_guess
        fwhm_major = GAUSSIAN_FWHM * sigma_major
        fwhm_minor = GAUSSIAN_FWHM * sigma_minor
        major_vals.append(max(fwhm_major * alpha_scale, 0.35))
        minor_vals.append(max(fwhm_minor * alpha_scale, 0.35))
        theta_vals.append(theta)
        eq_fwhm_vals.append(np.sqrt(fwhm_major * fwhm_minor))

    if fwhm_guess is None or not np.isfinite(fwhm_guess) or fwhm_guess <= 0.0:
        if eq_fwhm_vals:
            fwhm_guess = float(np.median(eq_fwhm_vals))
        else:
            fwhm_guess = 3.0

    alpha_guess = max(float(fwhm_guess) * alpha_scale, 0.35)
    alpha_major = float(np.median(major_vals)) if major_vals else alpha_guess
    alpha_minor = float(np.median(minor_vals)) if minor_vals else alpha_guess
    alpha_major, alpha_minor, _ = _canonicalize_moffat_shape(
        alpha_major,
        alpha_minor,
        0.0,
    )

    if theta_vals:
        theta_vals_arr = np.asarray(theta_vals, dtype=float)
        theta = 0.5 * np.arctan2(
            np.sum(np.sin(2.0 * theta_vals_arr)),
            np.sum(np.cos(2.0 * theta_vals_arr)),
        )
    else:
        theta = 0.0

    return alpha_major, alpha_minor, float(beta_guess), _wrap_theta(theta)


def _fit_global_moffat_shape(
    cutouts: list[_StarCutout],
    *,
    stat: ImageStat | None,
    max_position_shift: float,
    max_nfev: int,
) -> tuple[AnalyticalMoffatPSF, Table]:
    """Fit a shared elliptical Moffat profile across isolated stars."""
    cutout_size = cutouts[0].data.shape[0]
    yy, xx = np.mgrid[:cutout_size, :cutout_size]
    gain = 1.0 if stat is None else max(float(stat.gain), 1.0e-6)
    read_noise_var = 0.0 if stat is None else (float(stat.rdnoise) / gain) ** 2
    fwhm_guess = None if stat is None else float(stat.fwhm)
    if fwhm_guess is not None and np.isfinite(fwhm_guess) and fwhm_guess > 0.0:
        fwhm_sigma = max(0.15 * float(fwhm_guess), 0.2)
    else:
        fwhm_sigma = None

    alpha_major0, alpha_minor0, beta0, theta0 = _initial_moffat_shape(
        cutouts,
        fwhm_guess=fwhm_guess,
    )
    axis_ratio0 = np.clip(alpha_minor0 / alpha_major0, 0.25, 1.0)
    alpha_upper = max(0.75 * cutout_size, 4.0 * alpha_major0)

    def unpack_shape(params: NDArray) -> AnalyticalMoffatPSF:
        alpha_major = float(params[0])
        alpha_minor = float(params[0] * params[1])
        beta = float(params[2])
        theta = float(params[3])
        alpha_major, alpha_minor, theta = _canonicalize_moffat_shape(
            alpha_major,
            alpha_minor,
            theta,
        )
        return AnalyticalMoffatPSF(
            alpha_major=alpha_major,
            alpha_minor=alpha_minor,
            beta=beta,
            theta=theta,
        )

    def initial_vector(use_cutouts: list[_StarCutout]) -> tuple[NDArray, NDArray, NDArray]:
        p0 = [alpha_major0, axis_ratio0, beta0, theta0]
        lower = [0.3, 0.2, 1.2, -0.5 * np.pi]
        upper = [alpha_upper, 1.0, 12.0, 0.5 * np.pi]
        for cutout in use_cutouts:
            p0.extend([cutout.flux_init, cutout.x0, cutout.y0])
            lower.extend(
                [
                    0.0,
                    max(0.0, cutout.x0 - max_position_shift),
                    max(0.0, cutout.y0 - max_position_shift),
                ]
            )
            upper.extend(
                [
                    np.inf,
                    min(cutout_size - 1.0, cutout.x0 + max_position_shift),
                    min(cutout_size - 1.0, cutout.y0 + max_position_shift),
                ]
            )
        return (
            np.asarray(p0, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    def residuals(params: NDArray, use_cutouts: list[_StarCutout]) -> NDArray:
        psf = unpack_shape(params[:4])
        res: list[NDArray] = []
        offset = 4
        for cutout in use_cutouts:
            flux, x0, y0 = params[offset : offset + 3]
            offset += 3
            source = flux * psf.unit_flux_image(xx, yy, x0, y0)
            model = source + cutout.background
            sigma = np.sqrt(
                np.maximum(cutout.noise**2 + read_noise_var + source / gain, 1.0e-6)
            )
            good = np.isfinite(cutout.data) & (~cutout.mask)
            res.append(((cutout.data - model) / sigma)[good].ravel())
        if fwhm_sigma is not None and fwhm_guess is not None:
            res.append(np.asarray([(psf.fwhm - fwhm_guess) / fwhm_sigma], dtype=float))
            res.append(np.asarray([(psf.beta - beta0) / 2.0], dtype=float))
        return np.concatenate(res)

    def solve(use_cutouts: list[_StarCutout]) -> object:
        p0, lower, upper = initial_vector(use_cutouts)
        return least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            args=(use_cutouts,),
            loss="soft_l1",
            f_scale=1.5,
            max_nfev=max_nfev,
        )

    fit = solve(cutouts)

    def summarize(use_cutouts: list[_StarCutout], fit_result: object) -> tuple[AnalyticalMoffatPSF, Table]:
        psf = unpack_shape(np.asarray(fit_result.x[:4], dtype=float))
        err = _least_squares_errors(fit_result)
        rows = []
        offset = 4
        for i, cutout in enumerate(use_cutouts):
            flux, x0, y0 = fit_result.x[offset : offset + 3]
            flux_err = err[offset] if len(err) > offset else np.nan
            offset += 3
            source = flux * psf.unit_flux_image(xx, yy, x0, y0)
            model = source + cutout.background
            sigma = np.sqrt(
                np.maximum(cutout.noise**2 + read_noise_var + source / gain, 1.0e-6)
            )
            good = np.isfinite(cutout.data) & (~cutout.mask)
            resid = ((cutout.data - model) / sigma)[good]
            rows.append(
                (
                    cutout.index,
                    cutout.x_abs,
                    cutout.y_abs,
                    float(flux),
                    float(flux_err),
                    float(cutout.background),
                    float(np.hypot(x0 - cutout.x0, y0 - cutout.y0)),
                    float(np.sqrt(np.mean(resid**2))) if resid.size else np.nan,
                    i,
                )
            )

        table = Table(
            rows=rows,
            names=(
                "index",
                "x",
                "y",
                "flux_fit",
                "flux_err",
                "background",
                "center_shift",
                "residual_rms",
                "rank",
            ),
        )
        return psf, table

    psf, fit_table = summarize(cutouts, fit)

    rms = np.asarray(fit_table["residual_rms"], dtype=float)
    shift = np.asarray(fit_table["center_shift"], dtype=float)
    valid = np.isfinite(rms) & np.isfinite(shift) & (fit_table["flux_fit"] > 0.0)
    if np.any(valid):
        med_rms = float(np.nanmedian(rms[valid]))
        mad_rms = float(np.nanmedian(np.abs(rms[valid] - med_rms)))
        rms_limit = med_rms + 3.0 * max(mad_rms, 0.15)
        keep = valid & (rms <= rms_limit) & (shift <= max_position_shift)
    else:
        keep = valid

    if np.sum(keep) >= 3 and np.any(~keep):
        fit = solve([cutouts[int(i)] for i in np.asarray(fit_table["rank"][keep], dtype=int)])
        psf, fit_table = summarize(
            [cutouts[int(i)] for i in np.asarray(fit_table["rank"][keep], dtype=int)],
            fit,
        )

    fit_table.sort("residual_rms")
    return psf, fit_table


def build_analytical_moffat_psf(
    data: NDArray,
    catalog: Catalog,
    *,
    stat: ImageStat | None = None,
    cutout_size: int,
    max_stars: int,
    mask: NDArray[np.bool_] | None = None,
    isolation_radius: float | None = None,
    contamination_fraction: float = 0.2,
    max_position_shift: float = 1.5,
    max_nfev: int = 600,
) -> tuple[AnalyticalMoffatPSF, Table, int]:
    """Fit a robust analytical Moffat PSF from isolated field stars."""
    if cutout_size < 7 or cutout_size % 2 == 0:
        raise ValueError("cutout_size must be an odd integer >= 7")

    cutouts = _extract_isolated_star_cutouts(
        data,
        catalog,
        cutout_size=cutout_size,
        max_stars=max_stars,
        mask=mask,
        isolation_radius=isolation_radius,
        contamination_fraction=contamination_fraction,
    )
    psf, fit_table = _fit_global_moffat_shape(
        cutouts,
        stat=stat,
        max_position_shift=max_position_shift,
        max_nfev=max_nfev,
    )
    return psf, fit_table, int(len(fit_table))


def plot_epsf_cutouts(stars: object) -> None:
    """Display extracted stellar cutouts used for ePSF construction."""
    if len(stars) == 0:
        raise ValueError("no ePSF cutouts to display")

    ncols = min(5, len(stars))
    nrows = int(np.ceil(len(stars) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < len(stars):
            img = stars[i].data
            norm = simple_norm(img, "sqrt", percent=99.0)
            ax.imshow(img, origin="lower", norm=norm, cmap="viridis")
            if hasattr(stars[i], "cutout_center"):
                cx, cy = stars[i].cutout_center
                ax.plot(cx, cy, "r+", ms=8, mew=1.5)
            ax.set_title(f"{i} {stars[i].origin}", fontsize=9)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_epsf_photometry_diagnostics(
    data: NDArray,
    catalog: Catalog,
    *,
    epsf: object,
    phot: PSFPhotometry,
    stat: ImageStat,
) -> None:
    """Display the fitted sources, ePSF model, model image, and residual image."""
    model_image = phot.make_model_image(data.shape)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    norm = plt.matplotlib.colors.Normalize(*np.nanpercentile(data, [1, 99]))
    if stat.background2d is not None:
        background = np.asarray(stat.background2d, dtype=float)
    elif np.isfinite(stat.background):
        background = np.full_like(data, float(stat.background), dtype=float)
    else:
        background = np.full_like(data, float(np.nanmedian(data)), dtype=float)

    data_sub = data - background
    residual_image = data_sub - model_image

    gain = max(float(stat.gain), 1.0e-6)
    read_noise_var = (float(stat.rdnoise) / gain) ** 2
    poisson_var = np.clip(model_image + background, 0.0, None) / gain
    sigma_image = np.sqrt(poisson_var + read_noise_var)
    residual_sigma = residual_image / np.maximum(sigma_image, 1.0e-6)

    axes[0].imshow(data, origin="lower", norm=norm, cmap="viridis")
    axes[0].scatter(
        catalog.x,
        catalog.y,
        ec="red",
        fc="none",
        lw=0.7,
        s=25,
    )
    axes[0].set_title("Original image")

    axes[1].imshow(epsf.data, origin="lower", cmap="viridis")
    axes[1].set_title("ePSF image")

    axes[2].imshow(model_image, origin="lower", norm=norm, cmap="viridis")
    axes[2].set_title("Model image")

    axes[3].imshow(
        residual_sigma,
        origin="lower",
        vmin=-5.0,
        vmax=5.0,
        cmap="coolwarm",
    )
    axes[3].set_title("Residual image [sigma]")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()


def run_epsf_photometry(
    data: NDArray,
    catalog: Catalog,
    *,
    epsf: object,
    cutout_size: int,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[Catalog, PSFPhotometry, Table | None]:
    """Run ePSF fitting and return fitted catalog and photometry outputs."""
    init_params = Table()
    init_params["x_0"] = catalog.x
    init_params["y_0"] = catalog.y

    phot = PSFPhotometry(
        psf_model=epsf,
        fit_shape=cutout_size,
        localbkg_estimator=LocalBackground(inner_radius=10, outer_radius=15),
        aperture_radius=3.0,
    )
    result = phot(data, init_params=init_params, mask=mask)

    flux_fit, flux_err = result["flux_fit"], result["flux_err"]
    fitted_catalog = Catalog.from_arrays(
        x=np.asarray(result["x_fit"], dtype=float),
        y=np.asarray(result["y_fit"], dtype=float),
        mag=flux_to_mag(flux_fit, zeropoint=0.0),
        mag_err=flux_err_to_mag_err(flux_fit, flux_err),
    )
    return fitted_catalog, phot, result


def _source_patch_bounds(
    shape: tuple[int, int],
    *,
    x: float,
    y: float,
    half_size: int,
) -> tuple[int, int, int, int]:
    """Clip a square source-fitting patch to image bounds."""
    ny, nx = shape
    xi = int(np.rint(x))
    yi = int(np.rint(y))
    x0 = max(0, xi - half_size)
    x1 = min(nx, xi + half_size + 1)
    y0 = max(0, yi - half_size)
    y1 = min(ny, yi + half_size + 1)
    return x0, x1, y0, y1


def _render_analytical_psf_model(
    shape: tuple[int, int],
    *,
    psf: AnalyticalMoffatPSF,
    x: NDArray,
    y: NDArray,
    flux: NDArray,
    render_radius: float | None = None,
) -> NDArray:
    """Render a sparse full-frame model image from analytical PSF parameters."""
    model = np.zeros(shape, dtype=float)
    if render_radius is None:
        render_radius = max(8.0, 4.5 * psf.fwhm_major)
    half_size = int(np.ceil(render_radius))

    for xc, yc, fc in zip(x, y, flux):
        if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(fc) and fc > 0.0):
            continue
        x0, x1, y0, y1 = _source_patch_bounds(
            shape,
            x=float(xc),
            y=float(yc),
            half_size=half_size,
        )
        yy, xx = np.mgrid[y0:y1, x0:x1]
        model[y0:y1, x0:x1] += fc * psf.unit_flux_image(xx, yy, float(xc), float(yc))

    return model


def _fit_single_analytical_source(
    patch: NDArray,
    patch_mask: NDArray[np.bool_],
    xx: NDArray,
    yy: NDArray,
    *,
    psf: AnalyticalMoffatPSF,
    stat: ImageStat,
    x_init: float,
    y_init: float,
    flux_init: float,
    position_bound: float,
    fit_sky_plane: bool,
    max_nfev: int,
) -> tuple[NDArray, NDArray, float, bool, int]:
    """Fit one source on a local patch after subtracting neighbor models."""
    good = np.isfinite(patch) & (~patch_mask)
    n_params = 6 if fit_sky_plane else 4
    if int(np.sum(good)) <= n_params:
        return np.full(n_params, np.nan), np.full(n_params, np.nan), np.nan, False, 0

    background0, noise0 = _cutout_border_stats(patch, patch_mask)
    gain = max(float(stat.gain), 1.0e-6)
    read_noise_var = (float(stat.rdnoise) / gain) ** 2

    x_min = float(np.min(xx))
    x_max = float(np.max(xx))
    y_min = float(np.min(yy))
    y_max = float(np.max(yy))
    p0 = [max(float(flux_init), 1.0e-6), x_init, y_init, background0]
    lower = [
        0.0,
        max(x_min, x_init - position_bound),
        max(y_min, y_init - position_bound),
        -np.inf,
    ]
    upper = [
        np.inf,
        min(x_max, x_init + position_bound),
        min(y_max, y_init + position_bound),
        np.inf,
    ]
    if fit_sky_plane:
        p0.extend([0.0, 0.0])
        lower.extend([-np.inf, -np.inf])
        upper.extend([np.inf, np.inf])

    ref_x = float(x_init)
    ref_y = float(y_init)

    def residuals(params: NDArray) -> NDArray:
        flux = params[0]
        x0 = params[1]
        y0 = params[2]
        background = params[3]
        grad_x = params[4] if fit_sky_plane else 0.0
        grad_y = params[5] if fit_sky_plane else 0.0
        source = flux * psf.unit_flux_image(xx, yy, x0, y0)
        model = psf.evaluate(
            xx,
            yy,
            flux=flux,
            x0=x0,
            y0=y0,
            background=background,
            grad_x=grad_x,
            grad_y=grad_y,
            ref_x=ref_x,
            ref_y=ref_y,
        )
        sigma = np.sqrt(np.maximum(noise0**2 + read_noise_var + source / gain, 1.0e-6))
        return ((patch - model) / sigma)[good]

    fit = least_squares(
        residuals,
        np.asarray(p0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=max_nfev,
    )
    params = np.asarray(fit.x, dtype=float)
    err = _least_squares_errors(fit)
    if len(err) != len(params):
        err = np.full(len(params), np.nan, dtype=float)
    rms = float(np.sqrt(np.mean(np.asarray(fit.fun, dtype=float) ** 2)))
    success = bool(fit.success and np.isfinite(params[0]) and params[0] > 0.0)
    return params, err, rms, success, int(getattr(fit, "nfev", 0))


def run_analytical_psf_photometry(
    data: NDArray,
    catalog: Catalog,
    *,
    psf: AnalyticalMoffatPSF,
    stat: ImageStat,
    zeropoint: float,
    mask: NDArray[np.bool_] | None = None,
    fit_radius: float | None = None,
    render_radius: float | None = None,
    maxiters: int = 3,
    flux_tolerance: float = 1.0e-3,
    position_bound: float = 1.5,
    fit_sky_plane: bool = True,
    max_nfev: int = 200,
) -> tuple[Catalog, AnalyticalPSFPhotometryResult]:
    """Iteratively fit an analytical Moffat PSF to a source catalog."""
    data = np.asarray(data, dtype=float)
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    n_sources = len(catalog)
    if n_sources == 0:
        empty = Table()
        empty["x_fit"] = np.empty(0, dtype=float)
        empty["y_fit"] = np.empty(0, dtype=float)
        empty["flux_fit"] = np.empty(0, dtype=float)
        empty["flux_err"] = np.empty(0, dtype=float)
        empty["success"] = np.empty(0, dtype=bool)
        return (
            Catalog(),
            AnalyticalPSFPhotometryResult(
                table=empty,
                model_image=np.zeros_like(data, dtype=float),
                residual_image=np.array(data, copy=True),
                n_iterations=0,
            ),
        )

    fit_radius = max(4.0, 2.5 * psf.fwhm_major) if fit_radius is None else float(fit_radius)
    render_radius = (
        max(8.0, 4.5 * psf.fwhm_major) if render_radius is None else float(render_radius)
    )
    neighbor_radius = fit_radius + render_radius + position_bound
    half_size = int(np.ceil(fit_radius))
    init_half_size = max(half_size, int(np.ceil(1.5 * psf.fwhm_major)))

    x_fit = np.asarray(catalog.x, dtype=float).copy()
    y_fit = np.asarray(catalog.y, dtype=float).copy()
    flux_fit = np.full(n_sources, np.nan, dtype=float)
    flux_err = np.full(n_sources, np.nan, dtype=float)
    background = np.full(n_sources, np.nan, dtype=float)
    grad_x = np.full(n_sources, np.nan, dtype=float)
    grad_y = np.full(n_sources, np.nan, dtype=float)
    residual_rms = np.full(n_sources, np.nan, dtype=float)
    success = np.zeros(n_sources, dtype=bool)
    nfev = np.zeros(n_sources, dtype=int)

    init_flux = np.full(n_sources, np.nan, dtype=float)
    for i in range(n_sources):
        x0, x1, y0, y1 = _source_patch_bounds(
            data.shape,
            x=float(x_fit[i]),
            y=float(y_fit[i]),
            half_size=init_half_size,
        )
        patch = data[y0:y1, x0:x1]
        patch_mask = mask[y0:y1, x0:x1]
        init_flux[i] = _estimate_cutout_flux(
            patch,
            mask=patch_mask,
            x0=float(x_fit[i] - x0),
            y0=float(y_fit[i] - y0),
            radius=max(2.0, psf.fwhm),
        )
    flux_fit[:] = init_flux

    if np.any(np.isfinite(catalog.mag)):
        order = np.argsort(np.where(np.isfinite(catalog.mag), catalog.mag, np.inf))
    else:
        order = np.argsort(-np.nan_to_num(init_flux, nan=0.0))

    n_iterations = 0
    for iteration in range(max(1, int(maxiters))):
        n_iterations = iteration + 1
        coords = np.c_[x_fit, y_fit]
        tree = cKDTree(coords) if len(coords) > 1 else None
        max_relative_change = 0.0

        for idx in order:
            if not (np.isfinite(x_fit[idx]) and np.isfinite(y_fit[idx])):
                continue

            x0, x1, y0, y1 = _source_patch_bounds(
                data.shape,
                x=float(x_fit[idx]),
                y=float(y_fit[idx]),
                half_size=half_size,
            )
            patch = data[y0:y1, x0:x1]
            patch_mask = mask[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]

            neighbor_model = np.zeros_like(patch, dtype=float)
            if tree is not None:
                neighbors = tree.query_ball_point(
                    [float(x_fit[idx]), float(y_fit[idx])],
                    r=neighbor_radius,
                )
            else:
                neighbors = []
            for j in neighbors:
                if j == idx:
                    continue
                if not (
                    np.isfinite(x_fit[j])
                    and np.isfinite(y_fit[j])
                    and np.isfinite(flux_fit[j])
                    and flux_fit[j] > 0.0
                ):
                    continue
                neighbor_model += flux_fit[j] * psf.unit_flux_image(
                    xx,
                    yy,
                    float(x_fit[j]),
                    float(y_fit[j]),
                )

            target = patch - neighbor_model
            params, err, rms, ok, fit_nfev = _fit_single_analytical_source(
                target,
                patch_mask,
                xx,
                yy,
                psf=psf,
                stat=stat,
                x_init=float(x_fit[idx]),
                y_init=float(y_fit[idx]),
                flux_init=float(flux_fit[idx]),
                position_bound=position_bound,
                fit_sky_plane=fit_sky_plane,
                max_nfev=max_nfev,
            )
            nfev[idx] = fit_nfev
            residual_rms[idx] = rms
            success[idx] = ok
            if not ok:
                continue

            previous_flux = flux_fit[idx]
            flux_fit[idx] = params[0]
            x_fit[idx] = params[1]
            y_fit[idx] = params[2]
            background[idx] = params[3]
            flux_err[idx] = err[0] if len(err) > 0 else np.nan
            if fit_sky_plane:
                grad_x[idx] = params[4]
                grad_y[idx] = params[5]
            else:
                grad_x[idx] = 0.0
                grad_y[idx] = 0.0

            if np.isfinite(previous_flux) and previous_flux > 0.0:
                denom = max(abs(previous_flux), abs(flux_fit[idx]), 1.0)
                max_relative_change = max(
                    max_relative_change,
                    float(abs(flux_fit[idx] - previous_flux) / denom),
                )

        if max_relative_change < flux_tolerance:
            break

    model_image = _render_analytical_psf_model(
        data.shape,
        psf=psf,
        x=x_fit,
        y=y_fit,
        flux=flux_fit,
        render_radius=render_radius,
    )

    if stat.background2d is not None:
        background_image = np.asarray(stat.background2d, dtype=float)
    elif np.isfinite(stat.background):
        background_image = np.full_like(data, float(stat.background), dtype=float)
    else:
        finite = data[~mask]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        background_image = np.full_like(data, fill, dtype=float)
    residual_image = data - background_image - model_image

    result_table = Table()
    result_table["x_init"] = np.asarray(catalog.x, dtype=float)
    result_table["y_init"] = np.asarray(catalog.y, dtype=float)
    result_table["x_fit"] = x_fit
    result_table["y_fit"] = y_fit
    result_table["flux_fit"] = flux_fit
    result_table["flux_err"] = flux_err
    result_table["background"] = background
    result_table["grad_x"] = grad_x
    result_table["grad_y"] = grad_y
    result_table["residual_rms"] = residual_rms
    result_table["success"] = success
    result_table["nfev"] = nfev

    valid = success & np.isfinite(flux_fit) & (flux_fit > 0.0)
    fitted_catalog = Catalog.from_arrays(
        x=x_fit[valid],
        y=y_fit[valid],
        mag=flux_to_mag(flux_fit[valid], zeropoint=zeropoint),
        mag_err=flux_err_to_mag_err(flux_fit[valid], flux_err[valid]),
    )
    return fitted_catalog, AnalyticalPSFPhotometryResult(
        table=result_table,
        model_image=model_image,
        residual_image=residual_image,
        n_iterations=n_iterations,
    )


def plot_analytical_psf_photometry_diagnostics(
    data: NDArray,
    catalog: Catalog,
    *,
    psf: AnalyticalMoffatPSF,
    result: AnalyticalPSFPhotometryResult,
    stat: ImageStat,
) -> None:
    """Display analytical PSF model and residual diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    norm = plt.matplotlib.colors.Normalize(*np.nanpercentile(data, [1, 99]))
    if stat.background2d is not None:
        background = np.asarray(stat.background2d, dtype=float)
    else:
        background = np.full_like(data, float(stat.background), dtype=float)

    gain = max(float(stat.gain), 1.0e-6)
    read_noise_var = (float(stat.rdnoise) / gain) ** 2
    poisson_var = np.clip(result.model_image + background, 0.0, None) / gain
    sigma_image = np.sqrt(poisson_var + read_noise_var)
    residual_sigma = result.residual_image / np.maximum(sigma_image, 1.0e-6)

    axes[0].imshow(data, origin="lower", norm=norm, cmap="viridis")
    axes[0].scatter(catalog.x, catalog.y, ec="red", fc="none", lw=0.7, s=25)
    axes[0].set_title("Original image")

    axes[1].imshow(psf.stamp(), origin="lower", cmap="viridis")
    axes[1].set_title(
        (
            f"Moffat PSF\n"
            f"FWHM={psf.fwhm:.2f}px e={psf.ellipticity:.2f} "
            f"theta={np.degrees(psf.theta):.1f} deg"
        )
    )

    axes[2].imshow(result.model_image, origin="lower", norm=norm, cmap="viridis")
    axes[2].set_title("Model image")

    axes[3].imshow(
        residual_sigma,
        origin="lower",
        vmin=-5.0,
        vmax=5.0,
        cmap="coolwarm",
    )
    axes[3].set_title("Residual image [sigma]")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()


def _dophot_par_text(
    *,
    version: Literal["C", "fortran"],
    thresh_min: float = 2.0e8,
    thresh_max: float = 2.0e16,
    default_par: Path,
    image_name: str,
    obj_name: str,
    log_name: str,
    stat: ImageStat,
) -> str:
    """Build DoPHOT parameter-file contents for one image."""
    if version == "C":
        return f"""\
PARAMS_DEFAULT  = '{default_par}'
PARAMS_OUT      = '/dev/null'
IMAGE_IN        = '{image_name}'
LOGFILE         = '{log_name}'
LOGVERBOSITY    = 1
OBJECTS_OUT     = '{obj_name}'
ERRORS_OUT      = ' '
SHADOWFILE_OUT  = ' '
OBJECTS_IN      = ' '
IMAGE_OUT       = '{image_name.replace('.fits', '_out.fits')}'
PSFTYPE         = 'PGAUSS'
SKYTYPE         = 'PLANE'
OBJTYPE_IN      = 'COMPLETE'
OBJTYPE_OUT     = 'COMPLETE'
FWHM            = {stat.fwhm:.2f}
SKY             = {stat.background:.2f}
EPERDN          = {stat.gain}
RDNOISE         = {stat.rdnoise}
CENTINTMAX      = {thresh_max}
CTPERSAT        = {thresh_max}
ITOP            = {thresh_max}
THRESHMAX       = {thresh_max}
THRESHMIN       = {thresh_min}
APBOX_X         = {int(2.5 * stat.fwhm)}
APBOX_Y         = {int(2.5 * stat.fwhm)}
NFITBOX_X       = {int(2. * stat.fwhm)}
NFITBOX_Y       = {int(2. * stat.fwhm)}
END"""

    return f"""\
AUTOTHRESH      = 'NO'
FINISHFILE      = ' '
IMAGE_IN        = {image_name}
IMAGE_OUT       = '{image_name.replace('.fits', '_out.fits')}'
OBJECTS_OUT     = {obj_name}
PARAMS_OUT      = ' '
PARAMS_DEFAULT  = {default_par}
PSFTYPE         = 'PGAUSS'
OBJTYPE_IN      = 'COMPLETE'
OBJTYPE_OUT     = 'COMPLETE'
THRESHMIN       = {thresh_min}
THRESHMAX       = {thresh_max}
EPERDN          = {stat.gain}
RDNOISE         = {stat.rdnoise}
FWHM            = {stat.fwhm:.2f}
SKY             = {stat.background:.2f}
TOP             = {thresh_max}
END"""


def run_dophot_catalog(
    *,
    path: Path,
    data: NDArray,
    header: fits.Header,
    stat: ImageStat,
    mask: NDArray[np.bool_] | None,
    dophot_bin: Path,
    default_par: Path,
    tmp_dir: Path,
    version: Literal["C", "fortran"],
    mag_zero_point: float = 25.0,
) -> tuple[Catalog | None, float, float]:
    """Run DoPHOT and parse its object catalog.

    Returns ``(catalog_or_none, background, fwhm)``.
    """
    stem = path.stem.split(".", 1)[0]
    par_path = tmp_dir / f"{stem}.par"
    image_name = f"{stem}.fits"
    obj_name = f"{stem}.obj"
    log_name = f"{stem}.log"

    image_path = tmp_dir / image_name
    obj_path = tmp_dir / obj_name

    par_path.write_text(
        _dophot_par_text(
            stat=stat,
            version=version,
            default_par=default_par,
            image_name=image_name,
            obj_name=obj_name,
            log_name=log_name,
        ),
        encoding="utf-8",
    )

    image_data = np.asarray(data, dtype=np.float32)
    if mask is not None and np.any(mask):
        image_data = image_data.copy()
        fill_value = (
            float(stat.background)
            if np.isfinite(stat.background)
            else float(np.nanmedian(image_data[~mask]))
        )
        image_data[np.asarray(mask, dtype=bool)] = fill_value

    fits.PrimaryHDU(image_data, header=header).writeto(image_path, overwrite=True)

    subprocess.run(
        [str(dophot_bin), par_path.name],
        cwd=tmp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not obj_path.exists() or obj_path.stat().st_size <= 0:
        return None, stat.background, stat.fwhm

    data = np.loadtxt(obj_path)
    if data.ndim != 2:
        return None, stat.background, stat.fwhm

    data = data[(np.abs(data[:, 4]) < 99) & (np.abs(data[:, 5]) < 1)]
    data = data[data[:, 1] == 1]

    if len(data) == 0:
        return None, stat.background, stat.fwhm

    data = data[np.argsort(data[:, 4])]
    catalog = Catalog.from_arrays(
        x=data[:, 2] - 0.5,
        y=data[:, 3] - 0.5,
        mag=data[:, 4] + mag_zero_point,
        mag_err=data[:, 5],
    )
    out_background = float(np.median(data[:, 6]))
    out_fwhm = float(np.median((data[:, 7] * data[:, 8]) ** 0.5))
    return catalog, out_background, out_fwhm


def run_aperture_photometry(
    data: NDArray,
    catalog: Catalog,
    *,
    stat: ImageStat,
    r_ap: float,
    r_in: float,
    r_out: float,
    zeropoint: float,
    auto_scale: bool,
    mask: NDArray[np.bool_] | None = None,
) -> Catalog:
    """Run aperture photometry and return a filtered magnitude catalog."""
    if auto_scale:
        _r_ap = r_ap * stat.fwhm
        _r_in = r_in * stat.fwhm
        _r_out = r_out * stat.fwhm
    else:
        _r_ap = r_ap
        _r_in = r_in
        _r_out = r_out

    positions = np.c_[catalog.x, catalog.y]
    apertures = CircularAperture(positions, r=_r_ap)
    annuli = CircularAnnulus(positions, r_in=_r_in, r_out=_r_out)

    annulus_stats = ApertureStats(
        data,
        annuli,
        mask=mask,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
    )

    bkg_per_pixel = annulus_stats.median
    bkg_std_per_pixel = annulus_stats.std
    total_bkg_in_aperture = bkg_per_pixel * apertures.area

    phot_table = aperture_photometry(data, apertures, mask=mask)
    phot_table["flux_bkg_subtracted"] = (
        phot_table["aperture_sum"] - total_bkg_in_aperture
    )
    fluxes = phot_table["aperture_sum"].data - total_bkg_in_aperture

    mags = np.full(len(fluxes), np.nan)
    mag_errs = np.full(len(fluxes), np.nan)

    valid_flux = fluxes > 0
    source_variance_adu = np.maximum(phot_table["flux_bkg_subtracted"], 0) / stat.gain
    per_pix_var_adu = (bkg_std_per_pixel**2) + (stat.rdnoise / stat.gain) ** 2
    bkg_variance_adu = apertures.area * per_pix_var_adu
    flux_err = np.sqrt(source_variance_adu + bkg_variance_adu)

    mags[valid_flux] = flux_to_mag(fluxes[valid_flux], zeropoint=zeropoint)
    mag_errs[valid_flux] = flux_err_to_mag_err(fluxes[valid_flux], flux_err[valid_flux])

    return Catalog.from_arrays(
        x=np.asarray(phot_table["xcenter"][valid_flux], dtype=float),
        y=np.asarray(phot_table["ycenter"][valid_flux], dtype=float),
        mag=np.asarray(mags[valid_flux], dtype=float),
        mag_err=np.asarray(mag_errs[valid_flux], dtype=float),
    )
