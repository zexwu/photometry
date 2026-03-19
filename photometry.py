from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.visualization import simple_norm
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.background import LocalBackground
from photutils.psf import EPSFBuilder, PSFPhotometry, extract_stars
from scipy.spatial import cKDTree

from .catalog import StarCatalog
from .detection import flux_err_to_mag_err, flux_to_mag
from .image_stat import ImageStat


def build_epsf_model(
    data: np.ndarray,
    catalog: StarCatalog,
    *,
    oversample: int,
    max_stars: int,
) -> tuple[object, object, int]:
    ny, nx = data.shape
    cutout_size = 9
    half = cutout_size // 2
    margin = half + 1

    x = np.asarray(catalog.x, dtype=float)
    y = np.asarray(catalog.y, dtype=float)
    good = (x >= margin) & (x < nx - margin) & (y >= margin) & (y < ny - margin)

    coords = np.c_[x, y]
    dists, _ = cKDTree(coords).query(coords, k=2)
    nn_dist = dists[:, 1]
    good &= nn_dist >= cutout_size

    idx = np.argsort(catalog.mag)[good][:max_stars]
    stars_tbl = Table()
    stars_tbl["x"] = catalog.x[idx]
    stars_tbl["y"] = catalog.y[idx]

    stars = extract_stars(NDData(data=data), stars_tbl, size=(cutout_size, cutout_size))
    epsf, _ = EPSFBuilder(
        oversampling=oversample,
        maxiters=50,
        progress_bar=True,
        smoothing_kernel="quartic",
    )(stars)
    return epsf, stars, int(len(idx))


def plot_epsf_cutouts(stars: object) -> None:
    ncols = min(5, len(stars))
    nrows = int(len(stars) / ncols) + 1

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


def run_epsf_photometry(
    data: np.ndarray,
    catalog: StarCatalog,
    *,
    epsf: object,
    cutout_size: int,
) -> tuple[StarCatalog, PSFPhotometry, Table]:
    init_params = Table()
    init_params["x_0"] = catalog.x
    init_params["y_0"] = catalog.y

    phot = PSFPhotometry(
        psf_model=epsf,
        fit_shape=cutout_size,
        localbkg_estimator=LocalBackground(inner_radius=10, outer_radius=15),
        aperture_radius=3.0,
    )
    result = phot(data, init_params=init_params)

    flux_fit, flux_err = result["flux_fit"], result["flux_err"]
    fitted_catalog = StarCatalog.from_arrays(
        x=np.asarray(result["x_fit"], dtype=float),
        y=np.asarray(result["y_fit"], dtype=float),
        mag=flux_to_mag(flux_fit, zeropoint=0.0),
        mag_err=flux_err_to_mag_err(flux_fit, flux_err),
    )
    return fitted_catalog, phot, result


def run_aperture_photometry(
    data: np.ndarray,
    catalog: StarCatalog,
    *,
    stat: ImageStat,
    r_ap: float,
    r_in: float,
    r_out: float,
    zeropoint: float,
    auto_scale: bool,
) -> StarCatalog:
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
        sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
    )

    bkg_per_pixel = annulus_stats.median
    bkg_std_per_pixel = annulus_stats.std
    total_bkg_in_aperture = bkg_per_pixel * apertures.area

    phot_table = aperture_photometry(data, apertures)
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

    return StarCatalog.from_arrays(
        x=np.asarray(phot_table["xcenter"][valid_flux], dtype=float),
        y=np.asarray(phot_table["ycenter"][valid_flux], dtype=float),
        mag=np.asarray(mags[valid_flux], dtype=float),
        mag_err=np.asarray(mag_errs[valid_flux], dtype=float),
    )
