"""PSF/ePSF and aperture-photometry helper functions."""

from __future__ import annotations

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
from scipy.spatial import cKDTree

from .catalog import Catalog
from .detection import flux_err_to_mag_err, flux_to_mag
from .io import ImageStat


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
) -> tuple[object, object, int]:
    """Build an ePSF model and return ``(epsf, stars, stars_used)``."""
    if cutout_size < 3 or cutout_size % 2 == 0:
        raise ValueError("cutout_size must be an odd integer >= 3")

    ny, nx = data.shape
    half = cutout_size // 2
    margin = half + 1
    isolation_radius = (
        1 * cutout_size if isolation_radius is None else float(isolation_radius)
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

        border = np.concatenate(
            [cut[0, :], cut[-1, :], cut[1:-1, 0], cut[1:-1, -1]]
        )
        background = np.median(border)
        noise = np.std(border)
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
        idx = preferred_idx[:max_stars]
    else:
        idx = base_idx[:max_stars]

    if len(idx) < 3:
        raise ValueError(
            f"not enough usable stars for ePSF: selected={len(idx)}"
        )

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
        maxiters=50,
        progress_bar=True,
        smoothing_kernel="quartic",
    )(stars)
    if epsf is None or getattr(epsf, "data", None) is None:
        raise ValueError("ePSF builder failed to produce a valid model")

    return epsf, stars, int(len(idx))


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
    else:
        background = np.full_like(data, float(stat.background), dtype=float)

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
