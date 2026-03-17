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
from astropy.stats import SigmaClip, sigma_clipped_stats
from matplotlib.axes import Axes
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground
from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from astropy.table import Table
from astropy.nddata import NDData, CCDData
from photutils.psf import extract_stars, EPSFBuilder
from photutils.psf import PSFPhotometry
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.visualization import simple_norm
from scipy.spatial import cKDTree
from ccdproc import cosmicray_lacosmic



from .pymatch import MatchResult, match_stars
from .fwhm import estimate_fwhm


def _empty_array() -> NDArray:
    return np.empty(0, dtype=float)


@dataclass(kw_only=True)
class Image:
    path: Path | str
    data: NDArray | None = None
    header: Header | None = None
    filter: str = "unknown"
    gain: float = 1.0
    rdnoise: float = 1.0

    # catalog fields
    x: NDArray = field(default_factory=_empty_array)
    y: NDArray = field(default_factory=_empty_array)
    peak: NDArray = field(default_factory=_empty_array)
    inst_mag: NDArray = field(default_factory=_empty_array)
    inst_magerr: NDArray = field(default_factory=_empty_array)

    # diagnostic/metadata fields
    airmass: float = np.nan
    fwhm: float = np.nan
    background: float = np.nan
    nstars: int = -1

    residual_image: NDArray | None = None

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
            data = np.asarray(primary.data, dtype=np.float32)
            # data = np.where(np.isfinite(data), data, np.nan)
            mask = ~np.isfinite(data)
            data[mask] = -1

            if data.ndim != 2:
                raise ValueError(f"Expected a 2D FITS image, got shape={data.shape}")
            self.data = data
            self.header = primary.header.copy()
            self.filter = self.header.get("FILTER", "unknown").strip()
            self.gain = self.header.get("GAIN", 1.0)
            self.rdnoise = self.header.get("RDNOISE", 1.0)
            self.airmass = self.header.get("AIRMASS", -1.0)
            self.fwhm = self.header.get("SEEING", np.nan)

        return self

    def clear(self) -> None:
        self.x = _empty_array()
        self.y = _empty_array()
        self.inst_mag = _empty_array()
        self.inst_magerr = _empty_array()
        self.nstars = -1

        self.fwhm = np.nan
        self.background = np.nan

        self.meta.pop("detect_median", None)
        self.meta.pop("detect_std", None)
        self.meta.pop("fwhm_n_used", None)

    def remove_cr(self) -> Image:
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

    def detect(
        self,
        *,
        finder_fwhm: float = 2.0,
        threshold_sigma: float = 5.0,
        satval: float = 40000.0,
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

        sources = sources[sources["peak"] < satval]
        x = np.asarray(sources["xcentroid"], dtype=float)
        y = np.asarray(sources["ycentroid"], dtype=float)
        peak = np.asarray(sources["peak"], dtype=float)
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
        self.peak = peak
        self.nstars = len(x)

        self.meta["detect_median"] = float(median)
        self.meta["detect_std"] = float(std)
        return self

    def estimate_fwhm(
        self,
        *,
        max_stars: int = 512,
        half_size: int = 6,
    ) -> Image:

        ind = np.argsort(self.inst_mag)[:max_stars]
        fwhm = estimate_fwhm(self.data, self.x[ind], self.y[ind], half_size=half_size)
        self.fwhm = fwhm
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


    def dophot(
        self,
        bin: Path,
        default_par: Path,
        tmp_dir: Path,
        version: str = "C"
    ) -> Image:
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

        if not image_path.exists():
            print(image_path.exists())
            image_path.symlink_to(self.path.resolve())

        subprocess.run(
            [str(bin), par_path.name],
            cwd=tmp_dir,
            stdout=open(os.devnull, "wb"),
            stderr=open(os.devnull, "wb"),
        )

        data = np.loadtxt(obj_path)
        data = data[(np.abs(data[:, 4]) < 99) & (np.abs(data[:, 5]) < 1)]
        data = data[data[:, 1] == 1]
        if len(data):
            data = data[np.argsort(data[:, 4])]

            self.x, self.y = data[:, 2] - 0.5, data[:, 3] - 0.5
            self.inst_mag, self.inst_magerr = data[:, 4], data[:, 5]
            self.background = np.median(data[:, 6])
            self.fwhm = np.median((data[:, 7] * data[:, 8]) ** 0.5)
        return self

    def build_epsf(self, oversample: int = 2, max_stars: int = 100, inspect: bool = True) -> Image:
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
        dists, idx = tree.query(coords, k=2)   # k=1 is self, k=2 is nearest neighbor
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
        self.inst_mag = -2.5 * np.log10(flux_fit)
        self.inst_magerr = 2.5 / np.log(10) * np.abs(flux_err / flux_fit)


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


    def apphot(self, r_ap = 1.5, r_in = 2.5, r_out = 3.5, zeropoint: float = 25, auto_scale: bool = True) -> Image:
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
        bkg_std_per_pixel = annulus_stats.std # Empirical noise (sky + read noise)
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
        bkg_variance_adu = apertures.area * (bkg_std_per_pixel ** 2)
        flux_err = np.sqrt(source_variance_adu + bkg_variance_adu)

        mags[valid_flux] = -2.5 * np.log10(fluxes[valid_flux]) + zeropoint
        mag_errs[valid_flux] = 1.0857 * (flux_err[valid_flux] / fluxes[valid_flux])

        self.x = self.x[valid_flux]
        self.y = self.y[valid_flux]
        self.inst_mag = mags[valid_flux]
        self.inst_magerr = mag_errs[valid_flux]
        return self
