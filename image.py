from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import os
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.modeling.fitting import TRFLSQFitter
from astropy.stats import SigmaClip, mad_std, sigma_clipped_stats
from matplotlib.axes import Axes
from numba import njit
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.psf import (CircularGaussianPRF, IterativePSFPhotometry,
                           SourceGrouper)

from .pymatch import MatchResult, match_stars


@njit(cache=True)
def _moment_vars_jit(
    data: NDArray,
    x: NDArray,
    y: NDArray,
    median: float,
    half_size: int,
) -> tuple[NDArray, NDArray]:
    n = x.size
    varx = np.empty(n, dtype=np.float64)
    vary = np.empty(n, dtype=np.float64)
    varx[:] = np.nan
    vary[:] = np.nan
    ny, nx = data.shape

    for i in range(n):
        ix = int(np.rint(x[i]))
        iy = int(np.rint(y[i]))

        x0 = 0 if ix - half_size < 0 else ix - half_size
        x1 = nx if ix + half_size + 1 > nx else ix + half_size + 1
        y0 = 0 if iy - half_size < 0 else iy - half_size
        y1 = ny if iy + half_size + 1 > ny else iy + half_size + 1
        if x1 <= x0 or y1 <= y0:
            continue

        wsum = 0.0
        mx_num = 0.0
        my_num = 0.0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                v = data[yy, xx] - median
                if np.isfinite(v) and v > 0.0:
                    wsum += v
                    mx_num += v * xx
                    my_num += v * yy

        if wsum <= 0.0:
            continue

        mx = mx_num / wsum
        my = my_num / wsum

        vx_num = 0.0
        vy_num = 0.0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                v = data[yy, xx] - median
                if np.isfinite(v) and v > 0.0:
                    dx = xx - mx
                    dy = yy - my
                    vx_num += v * dx * dx
                    vy_num += v * dy * dy

        vx = vx_num / wsum
        vy = vy_num / wsum
        if vx > 0.0 and vy > 0.0:
            varx[i] = vx
            vary[i] = vy

    return varx, vary


def _empty_array() -> NDArray:
    return np.empty(0, dtype=float)


@dataclass(kw_only=True)
class Image:
    path: Path | str
    data: NDArray | None = None
    header: Header | None = None
    filter: str = "unknown"

    # catalog fields
    x: NDArray = field(default_factory=_empty_array)
    y: NDArray = field(default_factory=_empty_array)
    inst_mag: NDArray = field(default_factory=_empty_array)
    inst_magerr: NDArray = field(default_factory=_empty_array)

    # diagnostic/metadata fields
    fwhm: float = np.nan
    fwhmx: float = np.nan
    fwhmy: float = np.nan
    background: float = np.nan
    nstars: int = -1

    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        if self.data is None or self.header is None:
            self.load()

    def __repr__(self) -> str:
        shape = None if self.data is None else tuple(self.data.shape)
        return (
            f"Image(path={self.path.name!r}, \n"
            f"filter={self.filter}, shape={shape}, "
            f"nstars={self.nstars}, fwhm={self.fwhm:.2f}, background={self.background:.2f})"
        )

    def load(self) -> Image:
        with fits.open(self.path, memmap=False) as hdul:
            primary = hdul[0]
            data = np.asarray(primary.data)
            # data = np.where(np.isfinite(data), data, np.nan)
            mask = ~np.isfinite(data)
            data[mask] = -1

            if data.ndim != 2:
                raise ValueError(f"Expected a 2D FITS image, got shape={data.shape}")
            self.data = data
            self.header = primary.header.copy()
            self.filter = self.header.get("FILTER", "unknown").strip()

        return self

    def clear(self) -> None:
        self.x = _empty_array()
        self.y = _empty_array()
        self.inst_mag = _empty_array()
        self.inst_magerr = _empty_array()
        self.nstars = -1

        self.fwhm = np.nan
        self.fwhmx = np.nan
        self.fwhmy = np.nan
        self.background = np.nan

        self.meta.pop("detect_median", None)
        self.meta.pop("detect_std", None)
        self.meta.pop("fwhm_n_used", None)

    def detect(
        self,
        *,
        threshold_sigma: float = 5.0,
        finder_fwhm: float = 3.0,
        background: bool = True,
    ) -> Image:
        data = self.data
        if data is None:
            raise ValueError("Image data is not loaded.")

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

        x = np.asarray(sources["xcentroid"], dtype=float)
        y = np.asarray(sources["ycentroid"], dtype=float)
        flux = np.asarray(sources["flux"], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            fluxerr = np.sqrt(np.clip(flux, 0.0, None))
            inst_mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
            inst_magerr = np.where(
                flux > 0,
                2.5 / np.log(10.0) * np.abs(fluxerr / flux),
                np.nan,
            )

        self.x = x
        self.y = y
        self.inst_mag = inst_mag
        self.inst_magerr = inst_magerr
        self.nstars = len(x)

        self.meta["detect_median"] = float(median)
        self.meta["detect_std"] = float(std)
        return self

    def estimate_fwhm(
        self,
        *,
        max_stars: int = 128,
        half_size: int = 6,
    ) -> Image:
        data = self.data
        x = self.x
        y = self.y

        if data is None:
            raise ValueError("Image data is not loaded.")
        if x is None or y is None or self.nstars == 0:
            raise ValueError("Run img.detect() before img.estimate_fwhm().")

        median = self.meta.get("detect_median")
        if median is None:
            _, median, _ = sigma_clipped_stats(data, sigma=3.0)

        if self.inst_mag is not None:
            order = np.argsort(self.inst_mag)
            keep = order[: min(max_stars, len(order))]
        else:
            keep = np.arange(min(max_stars, len(x)))

        varx_samples, vary_samples = _moment_vars_jit(
            np.asarray(data, dtype=np.float64),
            x[keep].astype(np.float64, copy=False),
            y[keep].astype(np.float64, copy=False),
            float(median),
            half_size,
        )

        good = np.isfinite(varx_samples) & np.isfinite(vary_samples)
        if not np.any(good):
            self.fwhmx = np.nan
            self.fwhmy = np.nan
            self.fwhm = np.nan
            self.meta["fwhm_n_used"] = 0
            return self

        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        self.fwhmx = sigma_to_fwhm * float(np.sqrt(np.nanmedian(varx_samples[good])))
        self.fwhmy = sigma_to_fwhm * float(np.sqrt(np.nanmedian(vary_samples[good])))
        self.fwhm = float(np.nanmean((self.fwhmx, self.fwhmy)))
        self.meta["fwhm_n_used"] = int(np.count_nonzero(good))
        return self

    def sort_by(self: Image, kwd: str | List[str] = "inst_mag") -> Image:
        if isinstance(kwd, str):
            idx = np.argsort(getattr(self, kwd))
        else:
            keys = [getattr(self, k) for k in kwd]
            idx = np.lexsort(keys[::-1])  # reverse so first kwd is primary

        for k in ["x", "y", "inst_mag", "inst_magerr"]:
            setattr(self, k, getattr(self, k)[idx])
        return self

    def match(
        self, ref: Image, max_distance: float = 2, flip: bool = False
    ) -> MatchResult:
        if min(self.nstars, ref.nstars) < 3:
            raise ValueError(
                "Need at least 3 stars in each image for triangle matching."
            )

        sgn = -1 if flip else 1
        img = np.c_[self.x * sgn, self.y]
        ref = np.c_[ref.x, ref.y]
        res = match_stars(img, ref, max_distance)
        return res

    def show(
        self,
        ax: Axes | None = None,
        percentile: Tuple[float, float] = (1, 99),
        cmap: str = "gray",
    ) -> Any:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        vmin, vmax = np.nanpercentile(self.data, percentile)
        ax.imshow(self.data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.scatter(self.x, self.y, s=24, facecolors="none", edgecolors="limegreen", lw=0.7)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title(self.path.name)
        return fig, ax

    def apphot(self) -> Image:
        return self

    def dophot(
        self,
        bin: Path,
        default_par: Path,
        eperdn: float,
        rdnoise: float,
        tmp_dir: Path,
    ) -> Image:
        stem = self.path.stem.split(".", 1)[0]

        par_path = tmp_dir / f"{stem}.par"
        image_name = f"{stem}.fits"
        obj_name = f"{stem}.obj"
        log_name = f"{stem}.log"

        image_path = tmp_dir / image_name
        obj_path = tmp_dir / obj_name

        # pm = {
        #     "SKY": self.background,
        #     "FWHM": self.fwhm,
        #     "EPERDN": eperdn,
        #     "RDNOISE": rdnoise,
        #     "AUTOTHRESH": "NO",
        #     "THRESH_MIN": 100,
        #     "THRESH_MAX": 65000,
        #     "NPSFMIN": min(100, self.nstars),
        #     "NPSFORDER": 1,
        #     "PARAMS_DEFAULT": str(default_par),
        #     "PSFTYPE": "PGAUSS",
        #     "OBJTYPE_OUT": "COMPLETE",
        #     "OBJTYPE_IN": "COMPLETE",
        #     "IMAGE_IN": image_name,
        #     "OBJECTS_OUT": obj_name,
        #     "PARAMS_OUT": "/dev/null",
        #     **overrides,
        # }
        #
        # par_path.write_text(
        #     "\n".join(
        #         f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
        #         for k, v in pm.items()
        #         if v is not None
        #     )
        #     + "\nEND\n",
        #     encoding="utf-8",
        # )

        # EPERDN = {eperdn}
        # RDNOISE = {rdnoise}
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

FWHM            = {self.fwhm:.1f}
SKY             = {self.background:.1f}
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
EPERDN          = {eperdn}
RDNOISE         = {rdnoise}
FWHM            = {self.fwhm:.1f}
SKY             = {self.background:.1f}
TOP             = 40000.0"""

        par_path.write_text(pm_text, encoding="utf-8")

        if not image_path.exists():
            image_path.symlink_to(self.path.resolve())

        subprocess.run(
            [str(bin), par_path.name],
            cwd=tmp_dir,
            # stdout=open(os.devnull, "wb"),
            # stderr=open(os.devnull, "wb"),
        )

        data = np.loadtxt(obj_path)
        if len(data):
            data = data[(np.abs(data[:, 4]) < 99) & (np.abs(data[:, 5]) < 1)]
            data = data[np.argsort(data[:, 4])]

            self.x, self.y = data[:, 2], data[:, 3]
            self.inst_mag, self.inst_magerr = data[:, 4], data[:, 5]
            self.background = np.median(data[:, 6])
            self.fwhmx = np.median(data[:, 7])
            self.fwhmy = np.median(data[:, 8])
            self.fwhm = (self.fwhmx * self.fwhmy) ** 0.5
        return self

    def psfphot(self):
        """
        Perform modern, robust PSF photometry using photutils.

        Parameters
        ----------
        data : 2D ndarray
            Input image.
        fwhm_estimate : float
            Estimated stellar FWHM in pixels.
        sky_estimate : float or 2D ndarray
            Estimated sky/background. Can be a scalar or a 2D image.

        Returns
        -------
        catalog : astropy.table.Table
            PSF-fit catalog. Includes fitted positions, fluxes, quality metrics,
            flags, and a convenience boolean column ``good``.
        residual_image : 2D ndarray
            Residual image after subtracting fitted PSF models.

        Notes
        -----
        - This routine assumes a roughly spatially-invariant PSF.
        - The PSF model here is a circular Gaussian PRF. For precision work on
        real data, replacing this with an empirical/image-based PSF is often
        preferable.
        - The routine is intentionally conservative in source filtering.
        """
        # ------------------------------------------------------------------
        # 0. Validate / sanitize inputs
        # ------------------------------------------------------------------
        data = np.asanyarray(self.data, dtype=float)
        data_sub = data - self.background
        fwhm = self.fwhm

        # mask invalid pixels up front
        mask = ~np.isfinite(data_sub)
        data_work = np.array(data_sub, copy=True)
        data_work[mask] = np.nan

        finite = np.isfinite(data_work)

        # ------------------------------------------------------------------
        # 1. Robust noise estimate
        # ------------------------------------------------------------------
        # Use MAD-based sigma estimate on the sky-subtracted frame.
        bkg_sigma = mad_std(data_work[finite], ignore_nan=True)

        if not np.isfinite(bkg_sigma) or bkg_sigma <= 0:
            # fallback for extremely clean/synthetic images
            bkg_sigma = np.nanstd(data_work[finite])
            if not np.isfinite(bkg_sigma) or bkg_sigma <= 0:
                raise RuntimeError("Failed to estimate background noise.")

        # ------------------------------------------------------------------
        # 2. Build a modern PSF/PRF model
        # ------------------------------------------------------------------
        # CircularGaussianPRF is a current photutils PSF/PRF model.
        psf_model = CircularGaussianPRF(flux=1.0, x_0=0.0, y_0=0.0, fwhm=fwhm)

        # ------------------------------------------------------------------
        # 3. Choose robust configuration parameters
        # ------------------------------------------------------------------
        # Fit window: odd integer, centered on source.
        fit_size = int(np.ceil(2.0 * fwhm))
        fit_size = max(fit_size, 5)
        if fit_size % 2 == 0:
            fit_size += 1
        fit_shape = (fit_size, fit_size)

        # Aperture radius for initial flux estimate
        aperture_radius = max(1.5 * fwhm, 2.0)

        # Group sources that are close enough to overlap
        grouper = SourceGrouper(min_separation=2.0 * fwhm)

        # Detection threshold:
        # slightly conservative for robustness against false positives
        threshold = 5.0 * bkg_sigma

        finder = DAOStarFinder(
            threshold=threshold,
            fwhm=fwhm,
            exclude_border=True,
            sharplo=0.2,
            sharphi=1.2,
            roundlo=-1.0,
            roundhi=1.0,
        )

        fitter = TRFLSQFitter()

        phot = IterativePSFPhotometry(
            psf_model=psf_model,
            fit_shape=fit_shape,
            finder=finder,
            grouper=grouper,
            fitter=fitter,
            fitter_maxiters=200,
            xy_bounds=(1.5, 1.5),
            maxiters=2,
            mode="new",
            aperture_radius=aperture_radius,
            localbkg_estimator=None,  # avoid double-counting sky since sky_estimate is input
            group_warning_threshold=25,
            progress_bar=True,
        )

        # ------------------------------------------------------------------
        # 4. Run photometry
        # ------------------------------------------------------------------
        catalog = phot(data_work, mask=mask)
        finite_fit = (
            np.isfinite(catalog["x_fit"])
            & np.isfinite(catalog["y_fit"])
            & np.isfinite(catalog["flux_fit"])
        )
        positive_flux = catalog["flux_fit"] > 0
        zero_flags = np.asarray(catalog["flags"]) == 0

        good = finite_fit & positive_flux & zero_flags
        catalog = catalog[good]

        self.x = catalog["x_fit"]
        self.y = catalog["y_fit"]
        self.inst_mag = -2.5 * np.log10(catalog["flux_fit"])
        self.inst_magerr = 2.5 / np.log(10.0) * np.abs(catalog["flux_fit"] / catalog["flux_fit"]) * 0.01
        # self.inst_magerr = 2.5 / np.log(10.0) * np.abs(catalog["flux_fit_err"] / catalog["flux_fit"])
        residual_image = phot.make_residual_image(data_work)

        return self


if __name__ == "__main__":
    iimg = (
        Image(path="~/cmd/ogle/img_old/Io.fits")
        .detect()
        .estimate_fwhm()
        .sort_by("inst_mag")
    )
    dophot_bin = Path("/Users/zexwu/cmd/bin/dophot")
    dophot_par = Path("/Users/zexwu/cmd/par/dophot.par")
    tmp_dir = Path("./tmp/").resolve()

    iimg = iimg.dophot(dophot_bin, dophot_par, 1, 1, tmp_dir)
