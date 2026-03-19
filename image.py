from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Tuple, Literal

import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from matplotlib.axes import Axes
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground, LocalBackground
from photutils.detection import DAOStarFinder
from astropy.table import Table
from astropy.nddata import NDData, CCDData
from photutils.psf import extract_stars, EPSFBuilder
from photutils.psf import PSFPhotometry
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.visualization import simple_norm
from scipy.spatial import cKDTree
from ccdproc import cosmicray_lacosmic
from time import sleep

from .pymatch import match_stars
from .fwhm import estimate_fwhm

import warnings
from astropy.utils.exceptions import AstropyUserWarning
warnings.filterwarnings("ignore", category=AstropyUserWarning)

DETECT_ERROR = 1
APPHOT_ERROR = 2
DOPHOT_ERROR = 2
MATCH_ERROR = 3

logger = logging.getLogger(__name__)


def _empty_array() -> NDArray:
    return np.empty(0, dtype=float)


@dataclass(kw_only=True)
class Image:
    path: Path | str
    data: NDArray | None = None
    header: fits.Header | None = None
    gain: float = 1.0
    rdnoise: float = 1.0

    # catalog fields
    x: NDArray = field(default_factory=_empty_array)
    y: NDArray = field(default_factory=_empty_array)
    inst_mag: NDArray = field(default_factory=_empty_array)
    inst_mag_err: NDArray = field(default_factory=_empty_array)

    # diagnostic/metadata fields
    fwhm: float = np.nan
    background: float = np.nan
    background2d: NDArray | None = None

    meta: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    flag: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        if self.data is None or self.header is None:
            self.from_fits()

    def __repr__(self) -> str:
        shape = None if self.data is None else tuple(self.data.shape)
        return (
            f"Image(path={self.path.name!r}, shape={shape}, "
            f"nstars={self.nstars}, fwhm={self.fwhm:.2f}, background={self.background:.2f})"
        )

    def from_fits(self) -> Image:
        """Load FITS image and header from `self.path`.

        - Units: raw ADU counts assumed.
        - Populates: `data`, `header`, basic keywords (`gain`, `rdnoise`, `fwhm`).
        """
        with fits.open(self.path, memmap=False) as hdul:
            primary = hdul[0]
            data = np.asarray(primary.data, dtype=np.float32)
            # data = np.where(np.isfinite(data), data, np.nan)
            mask = ~np.isfinite(data)
            data[mask] = -1

            if data.ndim != 2:
                raise ValueError(f"Expected a 2D FITS image, got shape={data.shape}")
            self.data = data
            self.header = primary.header.copy()
            self.gain = self.header.get("GAIN", 1.0)
            self.rdnoise = self.header.get("RDNOISE", 1.0)
            self.fwhm = self.header.get("SEEING", np.nan)

        return self

    def clear(self) -> Image:

        """Clear catalog- and derived fields while keeping image data/header."""
        self.x = _empty_array()
        self.y = _empty_array()
        self.inst_mag = _empty_array()
        self.inst_mag_err = _empty_array()
        self.fwhm = np.nan
        self.background = np.nan

        self.meta.pop("detect_median", None)
        self.meta.pop("detect_std", None)
        self.meta.pop("fwhm_n_used", None)

        return self

    @property
    def nstars(self) -> int:
        """Number of sources in the in-memory catalog."""
        return int(len(self.x))

    def remove_cosmic_rays(self) -> Image:
        """Remove cosmic rays in-place using L.A.Cosmic via `ccdproc`.

        Returns self for chaining.
        """
        ccd = CCDData(self.data, unit="adu")
        cleaned = cosmicray_lacosmic(
            ccd,
            gain=self.gain,
            readnoise=self.rdnoise,
            sigclip=4.5,
            sigfrac=0.3,
            objlim=5.0,
        )

        self.data = cleaned.data
        return self

    def detect_star(
        self,
        *,
        finder_fwhm: float = 3.0,
        threshold_sigma: float = 4.0,
        saturation_level: float = 40000.0,
        background: bool = True,
    ) -> Image:
        """Detect stars with DAOStarFinder and build a simple catalog.

        - finder_fwhm: PSF FWHM in px for the DAO filter.
        - threshold_sigma: detection threshold relative to background std.
        - satval: peak count above which detections are discarded.
        - background: if True, subtract a 2D median background before detection.
        """
        data = self.data
        self.clear()

        if background:
            min_dim = min(data.shape)
            box = max(16, min(64, min_dim // 8))
            if box <= 1:
                bkg_image = np.full_like(data, np.nanmedian(data), dtype=float)
                self.background = float(np.nanmedian(bkg_image))
            else:
                bkg2d = Background2D(
                    data,
                    box_size=(box, box),
                    filter_size=(3, 3),
                    sigma_clip=SigmaClip(sigma=3.0),
                    bkg_estimator=MedianBackground(),
                )
                bkg_image = np.asarray(bkg2d.background, dtype=float)
                self.background2d = bkg_image
                self.background = float(np.nanmedian(bkg_image))

            det_data = data - bkg_image
            _, median, std = sigma_clipped_stats(det_data, sigma=3.0)
        else:
            _, median, std = sigma_clipped_stats(data, sigma=3.0)
            self.background = float(median)
            det_data = data - median

        finder = DAOStarFinder(fwhm=finder_fwhm, threshold=threshold_sigma * std,
                               sharplo=0.2,
                               sharphi=1.2,
                               roundlo=-1.0,
                               roundhi=1.0,)
        sources = finder(det_data)

        if sources is None or len(sources) == 0:
            self.meta["detect_median"] = float(median)
            self.meta["detect_std"] = float(std)
            return self

        sources = sources[sources["peak"] < saturation_level]
        x = np.asarray(sources["xcentroid"], dtype=float)
        y = np.asarray(sources["ycentroid"], dtype=float)
        flux = np.asarray(sources["flux"], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            fluxerr = np.sqrt(np.clip(flux, 0.0, None))
            inst_mag = flux_to_mag(flux, zeropoint=0.0)
            inst_mag_err = flux_err_to_mag_err(flux, fluxerr)

        self.x = x
        self.y = y
        self.inst_mag = inst_mag
        self.inst_mag_err = inst_mag_err

        self.meta["detect_median"] = float(median)
        self.meta["detect_std"] = float(std)
        return self

    def estimate_fwhm(
        self,
        *,
        max_stars: int = 512,
        half_size: int = 6,
    ) -> Image:
        """Estimate median stellar FWHM [px] from the brightest detections.

        Uses Gaussian fits on per-star cutouts (see `fwhm.estimate_fwhm`).
        """

        ind = np.argsort(self.inst_mag)[:max_stars]
        fwhm = estimate_fwhm(self.data, self.x[ind], self.y[ind], half_size=half_size)
        self.fwhm = fwhm
        return self


    def sort_by(self: Image, kwd: str | List[str] = "inst_mag") -> Image:
        """Sort the catalog in-place by one or multiple keys. """
        if isinstance(kwd, str):
            idx = np.argsort(getattr(self, kwd))
        else:
            keys = [getattr(self, k) for k in kwd]
            idx = np.lexsort(keys[::-1])  # reverse so first kwd is primary

        for k in ["x", "y", "inst_mag", "inst_mag_err"]:
            setattr(self, k, getattr(self, k)[idx])
        return self


    def show(
        self,
        ax: Axes | None = None,
        percentile: Tuple[float, float] = (1, 99),
        cmap: str = "gray",
    ) -> Tuple[plt.Figure | plt.SubFigure, plt.Axes]:
        """Quick-look image with detections overlay.

        Returns (fig, ax). Units: image in ADU, coordinates in pixels.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        vmin, vmax = np.nanpercentile(self.data, percentile)
        ax.imshow(self.data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.scatter(self.x, self.y, s=24, facecolors="none", edgecolors="limegreen", lw=0.7)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title(self.path.name)
        return fig, ax


    def run_dophot(
        self,
        dophot_bin: Path,
        default_par: Path,
        tmp_dir: Path,
        version: Literal["C", "fortran"] = "C"
    ) -> Image:
        """Run DoPHOT on this image and ingest output catalog.

        - dophot_bin: path to DoPHOT executable.
        - default_par: parameter file.
        - tmp_dir: working directory where symlink and outputs are placed.
        - version: minimal switch for parameter format ("C" or other).
        """
        stem = self.path.stem.split(".", 1)[0]

        par_path = tmp_dir / f"{stem}.par"
        image_name = f"{stem}.fits"
        obj_name = f"{stem}.obj"
        log_name = f"{stem}.log"

        image_path = tmp_dir / image_name
        obj_path = tmp_dir / obj_name

        if version == "C":
            # NOTE: Dophot_C parameter
            pm_text = f"""\
PARAMS_DEFAULT  = '{default_par}'
PARAMS_OUT      = '/dev/null'
IMAGE_IN        = '{image_name}'
LOGFILE         = '{log_name}'
LOGVERBOSITY    = 1
OBJECTS_OUT     = '{obj_name}'
ERRORS_OUT      = ' '
SHADOWFILE_OUT  = ' '
OBJECTS_IN      = ' '
IMAGE_OUT       = ' '
PSFTYPE         = 'PGAUSS'
SKYTYPE         = 'PLANE'
OBJTYPE_IN      = 'COMPLETE'
OBJTYPE_OUT     = 'COMPLETE'
FWHM            = {self.fwhm:.2f}
SKY             = {self.background:.2f}
EPERDN          = {self.gain}
RDNOISE         = {self.rdnoise}
ITOP            = 40000
ICRIT           = 10
CENTINTMAX      = 30000
CTPERSAT        = 40000
THRESHMAX       = 40000
THRESHMIN       = 200
APBOX_X         = 16
APBOX_Y         = 16
NFITBOX_X       = 12
NFITBOX_Y       = 12
END"""

        else:
            pm_text = f"""\
AUTOTHRESH      = 'NO'
FINISHFILE      = ' '
IMAGE_IN        = {image_name}
IMAGE_OUT       = {image_name.replace(".fits", "_out.fits")}
OBJECTS_OUT     = {obj_name}
PARAMS_OUT      = ' '
PARAMS_DEFAULT  = {default_par}
PSFTYPE         = 'PGAUSS'
OBJTYPE_IN      = 'COMPLETE'
OBJTYPE_OUT     = 'COMPLETE'
THRESHMIN       = 100.0
THRESHMAX       = 40000.0
EPERDN          = {self.gain}
RDNOISE         = {self.rdnoise}
FWHM            = {self.fwhm:.2f}
SKY             = {self.background:.2f}
TOP             = 40000.0
END"""

        par_path.write_text(pm_text, encoding="utf-8")

        image_path.unlink(missing_ok=True)
        image_path.symlink_to(self.path.resolve())

        subprocess.run(
            [str(dophot_bin), par_path.name],
            cwd=tmp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sleep(0.5)

        if obj_path.stat().st_size > 0:
            data = np.loadtxt(obj_path)
            if data.ndim == 2:
                data = data[(np.abs(data[:, 4]) < 99) & (np.abs(data[:, 5]) < 1)]
                data = data[data[:, 1] == 1]
                if len(data):
                    data = data[np.argsort(data[:, 4])]

                    self.x, self.y = data[:, 2] - 0.5, data[:, 3] - 0.5
                    self.inst_mag, self.inst_mag_err = data[:, 4] + 25, data[:, 5]
                    self.background = np.median(data[:, 6])
                    self.fwhm = np.median((data[:, 7] * data[:, 8]) ** 0.5)
        return self

    def build_epsf(self, oversample: int = 2, max_stars: int = 100, inspect: bool = True) -> Image:
        """Build an ePSF model from isolated bright stars.

        - oversample: ePSF oversampling factor.
        - max_stars: maximum number of stars to use.
        - inspect: if True, show cutouts used to build the ePSF.
        """
        ny, nx = self.data.shape
        cutout_size = 9
        half = cutout_size // 2

        margin = half + 1

        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)

        good = (
            (x >= margin) &
            (x <  nx - margin) &
            (y >= margin) &
            (y <  ny - margin)
        )

        coords = np.c_[x, y]
        tree = cKDTree(coords)
        dists, nn_idx = tree.query(coords, k=2)  # k=1 is self, k=2 nearest neighbor
        nn_dist = dists[:, 1]

        good &= nn_dist >= cutout_size
        idx = np.argsort(self.inst_mag)[good][:max_stars]

        stars_tbl = Table()
        stars_tbl["x"] = self.x[idx]
        stars_tbl["y"] = self.y[idx]

        nd = NDData(data=self.data)
        stars = extract_stars(nd, stars_tbl, size=(cutout_size, cutout_size))

        epsf_builder = EPSFBuilder(
            oversampling=oversample,
            maxiters=50,
            progress_bar=True,
            smoothing_kernel='quartic',
        )
        epsf, fitted_stars = epsf_builder(stars)
        self.epsf = epsf

        if inspect:
            ncols = 5
            nstars = len(stars)
            ncols = min(ncols, nstars)
            nrows = int(nstars / ncols) + 1

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
            axes = np.atleast_1d(axes).ravel()

            for i, ax in enumerate(axes):
                if i < nstars:
                    img = stars[i].data
                    norm = simple_norm(img, 'sqrt', percent=99.0)
                    ax.imshow(img, origin='lower', norm=norm, cmap='viridis')

                    # mark the cutout-center if available
                    if hasattr(stars[i], 'cutout_center'):
                        cx, cy = stars[i].cutout_center
                        ax.plot(cx, cy, 'r+', ms=8, mew=1.5)
                    ax.set_title(f'{i} {stars[i].origin}', fontsize=9)
                else:
                    ax.axis('off')

                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.show()

        return self

    def epsfphot(self, cutout_size: int = 9, show: bool = False) -> Image:
        """Perform PSF photometry using the ePSF model.

        Updates x, y to fitted centroids and computes instrumental magnitudes.
        """
        init_params = Table()
        init_params["x_0"] = self.x
        init_params["y_0"] = self.y

        localbkg = LocalBackground(inner_radius=10, outer_radius=15)
        phot = PSFPhotometry(
            psf_model=self.epsf,
            fit_shape=cutout_size,
            localbkg_estimator=localbkg,
            aperture_radius=3.0
        )

        result = phot(self.data, init_params=init_params)
        self.x = result["x_fit"]
        self.y = result["y_fit"]

        flux_fit, flux_err = result["flux_fit"], result["flux_err"]
        self.inst_mag = flux_to_mag(flux_fit, zeropoint=0.0)
        self.inst_mag_err = flux_err_to_mag_err(flux_fit, flux_err)

        if show:
            model_image = phot.make_model_image(self.data.shape)
            residual_image = phot.make_residual_image(self.data)
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))

            # original image
            norm0 = simple_norm(self.data, "sqrt", percent=99.0)
            axes[0].imshow(self.data, origin="lower", norm=norm0, cmap="viridis")
            axes[0].scatter(self.x, self.y, ec="red", fc="none", lw=0.7, s=25)
            axes[0].set_title("Original image")

            # ePSF image
            epsf_img = self.epsf.data
            norm1 = simple_norm(epsf_img, "sqrt", percent=99.0)
            axes[1].imshow(epsf_img, origin="lower", norm=norm1, cmap="viridis")
            axes[1].set_title("ePSF image")

            # fitted model image
            norm2 = simple_norm(model_image, "sqrt", percent=99.0)
            axes[2].imshow(model_image, origin="lower", norm=norm2, cmap="viridis")
            axes[2].set_title("Model image")

            # residual image
            norm3 = simple_norm(residual_image, "sqrt", percent=99.0)
            axes[3].imshow(residual_image, origin="lower", norm=norm3, cmap="viridis")
            axes[3].set_title("Residual image")

            for ax in axes:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            plt.tight_layout()
            plt.show()

        return self


    def apphot(
        self,
        r_ap: float = 1.5,
        r_in: float = 3,
        r_out: float = 6,
        zeropoint: float = 25.0,
        auto_scale: bool = True,
    ) -> Image:
        """Aperture photometry.

        Radii are in units of FWHM if `auto_scale`, else pixels.
        Magnitudes use: m = -2.5 log10(flux) + zeropoint.
        Variance model: var = flux/gain + area*(sigma_sky^2 + (rdnoise/gain)^2).
        """
        if auto_scale:
            _r_ap = r_ap * self.fwhm
            _r_in = r_in * self.fwhm
            _r_out = r_out * self.fwhm
        else:
            _r_ap = r_ap
            _r_in = r_in
            _r_out = r_out

        positions = np.c_[self.x, self.y]
        apertures = CircularAperture(positions, r=_r_ap)
        annuli = CircularAnnulus(positions, r_in=_r_in, r_out=_r_out)

        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        annulus_stats = ApertureStats(self.data, annuli, sigma_clip=sigclip)

        bkg_per_pixel = annulus_stats.median
        bkg_std_per_pixel = annulus_stats.std  # Empirical sky sigma per pixel
        total_bkg_in_aperture = bkg_per_pixel * apertures.area

        phot_table = aperture_photometry(self.data, apertures)
        phot_table['local_bkg_per_pix'] = bkg_per_pixel
        phot_table['aperture_bkg_total'] = total_bkg_in_aperture
        phot_table['flux_bkg_subtracted'] = phot_table['aperture_sum'] - total_bkg_in_aperture

        fluxes = phot_table['aperture_sum'].data - total_bkg_in_aperture

        mags = np.full(len(fluxes), np.nan)
        mag_errs = np.full(len(fluxes), np.nan)

        valid_flux = fluxes > 0
        source_variance_adu = np.maximum(phot_table['flux_bkg_subtracted'], 0) / self.gain
        per_pix_var_adu = (bkg_std_per_pixel ** 2) + (self.rdnoise / self.gain) ** 2
        bkg_variance_adu = apertures.area * per_pix_var_adu
        flux_err = np.sqrt(source_variance_adu + bkg_variance_adu)

        mags[valid_flux] = flux_to_mag(fluxes[valid_flux], zeropoint=zeropoint)
        mag_errs[valid_flux] = flux_err_to_mag_err(fluxes[valid_flux], flux_err[valid_flux])

        self.x = phot_table["xcenter"][valid_flux]
        self.y = phot_table["ycenter"][valid_flux]

        self.inst_mag = mags[valid_flux]
        self.inst_mag_err = mag_errs[valid_flux]

        return self

    def transform_to(self, img: Image, flip: bool = False, inspect: bool = False,
                     superflat_order: Tuple[int, int] = (0, 0),
                     select: Callable = lambda _:_>-np.inf) -> Image:

        if min(self.nstars, img.nstars) < 3:
            logger.error(f"Not enough stars to match (self={self.nstars}, ref={img.nstars}), drop this image.")
            self.flag = MATCH_ERROR
            return self

        sgn = -1 if flip else 1
        self_xy = np.c_[self.x * sgn, self.y]
        img_xy = np.c_[img.x, img.y]
        res = match_stars(self_xy, img_xy, 2)

        if res.inlier_count < 50 and len(img_xy) > 2 * len(self_xy):
            img_xy = img_xy[: int(1.5 * len(self_xy))]  # limit ref stars
            res = match_stars(self_xy, img_xy, 2)

        res.transform.A[:, 0] *= sgn  # adjust for flip if needed

        id1, id2 = res.pairs[:, 0], res.pairs[:, 1]
        T = res.transform

        if len(id2) < 10:
            logger.error(f"{self.path.name}: Only {len(id2):02d} matched stars found, drop this image.")
            self.flag = MATCH_ERROR
            return self

        diff = self.inst_mag[id1] - img.inst_mag[id2]
        err = (self.inst_mag_err[id1] ** 2 + img.inst_mag_err[id2] ** 2) ** 0.5
        use = (0 < err) & (err < 0.1)
        use &= select(img.inst_mag[id2])

        # Check if we need to apply a 2D polynomial superflat (order > 0 in any axis)
        superflat = sum(superflat_order) > 0
        if superflat:
            order_x, order_y = superflat_order

            # 1. Extract coordinates and magnitude differences for matched stars
            x_fit, y_fit = self.x[id1][use], self.y[id1][use]
            mag_fit = diff[use]

            # 2. Normalize coordinates to prevent matrix instability (CRITICAL for order >= 2)
            x_mean, x_std = np.mean(self.x), np.std(self.x)
            y_mean, y_std = np.mean(self.y), np.std(self.y)

            xn = (x_fit - x_mean) / x_std
            yn = (y_fit - y_mean) / y_std

            # Helper function to dynamically build the 2D polynomial basis matrix
            def get_basis(x_arr, y_arr):
                terms = []
                for i in range(order_x + 1):
                    for j in range(order_y + 1):
                        terms.append((x_arr ** i) * (y_arr ** j))
                return np.column_stack(terms)

            valid = np.ones(len(mag_fit), dtype=bool)

            # 3. Iterative Least Squares with 3-sigma clipping
            for _ in range(15):
                # Build basis only for currently valid stars
                basis = get_basis(xn[valid], yn[valid])
                coeff = np.linalg.lstsq(basis, mag_fit[valid], rcond=None)[0]

                # Evaluate on all matched stars to calculate residuals
                full_basis = get_basis(xn, yn)
                residuals = mag_fit - (full_basis @ coeff)

                # Flag new outliers beyond 3 standard deviations

                median = np.median(residuals[valid])
                mad_raw = np.median(np.abs(residuals[valid]- median))
                std = mad_raw * 1.4826  # Scale to Gaussian standard deviation

                new_valid = np.abs(residuals) < (3.0 * std)

                # Break loop if converged (no new outliers)
                if np.array_equal(valid, new_valid):
                    break
                valid = new_valid

            # 4. Evaluate the final polynomial surface on ALL stars in the catalog
            self_xn = (self.x - x_mean) / x_std
            self_yn = (self.y - y_mean) / y_std
            all_basis = get_basis(self_xn, self_yn)

            med = all_basis @ coeff
            n_used = np.sum(valid)

        else:
            _, med, std = sigma_clipped_stats(data=diff[use], sigma=3, maxiters=25, stdfunc="mad_std")

            n_used = len(diff[use])

        self.inst_mag -= med
        self.inst_mag_err = np.sqrt(self.inst_mag_err ** 2 + std ** 2 / n_used)
        self.x, self.y = T.apply(np.c_[self.x, self.y]).T

        if inspect:
            # 1. Gather data for the matched and filtered stars
            plot_x, plot_y = self.x[id1][use], self.y[id1][use]
            ref_mag = img.inst_mag[id2][use]
            plot_err = err[use]

            # 2. Extract model values correctly based on the mode
            model_vals = med[id1][use] if superflat else med

            # 3. Calculate residuals and identify outliers using the 'std' from the fit
            raw_diff = diff[use]
            residuals = raw_diff - model_vals
            inliers = np.abs(residuals) < (3.0 * std)
            outliers = ~inliers

            # 4. Set up the figure
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs = axs.ravel()
            ebar_kwargs = dict(fmt="o", fillstyle="none", markersize=4, lw=0.7, alpha=0.6)

            # --- Ax 0: Coordinate Map ---
            axs[0].scatter(img.x[id2], img.y[id2], s=15, fc="none", ec="r", lw=1, label="Ref")
            axs[0].scatter(self.x[id1], self.y[id1], s=15, marker="x", lw=1, c="b", label="Target")
            axs[0].set_title("Matched Star Coordinates")
            axs[0].set_xlabel("X [px]")
            axs[0].set_ylabel("Y [px]")
            axs[0].legend()

            # --- Ax 1: Residuals vs Ref Mag ---
            if np.any(outliers):
                axs[1].errorbar(ref_mag[outliers], residuals[outliers], yerr=plot_err[outliers], c="C1", label="Rejected", **ebar_kwargs)
            if np.any(inliers):
                axs[1].errorbar(ref_mag[inliers], residuals[inliers], yerr=plot_err[inliers], c="C0", label="Used", **ebar_kwargs)

            axs[1].axhline(0, c="r", lw=1.5, ls="--")
            axs[1].set_ylim(0.3, -0.3)
            axs[1].set_xlabel("Ref Mag")
            axs[1].set_ylabel("Residual (img - ref - model)")
            axs[1].set_title("Magnitude Residuals")

            mode_str = "Superflat 2D Plane" if superflat else "Standard Scalar"
            stat_text = f"Mode: {mode_str}\nScatter (std): {std:.3f}\nStars used: {np.sum(inliers)}"
            axs[1].text(0.05, 0.95, stat_text, transform=axs[1].transAxes, va="top", ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # --- Ax 2: Spatial Residuals (X) ---
            if np.any(outliers):
                axs[2].errorbar(plot_x[outliers], residuals[outliers], yerr=plot_err[outliers], c="C1", label="Rejected", **ebar_kwargs)
            if np.any(inliers):
                axs[2].errorbar(plot_x[inliers], residuals[inliers], yerr=plot_err[inliers], c="C0", label="Used", **ebar_kwargs)

            axs[2].axhline(0, c="r", lw=1.5, ls="--")
            axs[2].set_ylim(0.3, -0.3)
            axs[2].set_xlabel("X [px]")
            axs[2].set_ylabel("Residual (img - ref - model)")
            axs[2].set_title("Spatial Residuals (X)")

            # --- Ax 3: Spatial Residuals (Y) ---
            if np.any(outliers):
                axs[3].errorbar(plot_y[outliers], residuals[outliers], yerr=plot_err[outliers], c="C1", label="Rejected", **ebar_kwargs)
            if np.any(inliers):
                axs[3].errorbar(plot_y[inliers], residuals[inliers], yerr=plot_err[inliers], c="C0", label="Used", **ebar_kwargs)

            axs[3].axhline(0, c="r", lw=1.5, ls="--")
            axs[3].set_ylim(0.3, -0.3)
            axs[3].set_xlabel("Y [px]")
            axs[3].set_ylabel("Residual (img - ref - model)")
            axs[3].set_title("Spatial Residuals (Y)")

            fig.tight_layout()
        return self


    def dump(self, filename: str = None) -> None:
        if not filename:
            filename = self.path.stem + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> Image:
        with open(filename, "rb") as f:
            return pickle.load(f)

    def copy(self) -> Image:
        return deepcopy(self)


# ---- Small magnitude helpers -------------------------------------------------

def flux_to_mag(flux: NDArray, zeropoint: float) -> NDArray:
    """Convert flux to magnitude with given zeropoint. Returns NaN for flux <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux) + zeropoint
        mag = np.where(flux > 0, mag, np.nan)
    return mag


def flux_err_to_mag_err(flux: NDArray, flux_err: NDArray) -> NDArray:
    """Propagate flux error to magnitude error (1.0857 * dF/F)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 2.5 / np.log(10.0)
        merr = k * np.abs(flux_err / flux)
        merr = np.where(flux > 0, merr, np.nan)
    return merr
