"""High-level image pipeline orchestrator."""

from __future__ import annotations

import pickle
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning
from ccdproc import cosmicray_lacosmic
from numpy.typing import NDArray

from .catalog import Catalog
from .detection import detect_star_catalog
from .fwhm import estimate_fwhm as estimate_stellar_fwhm
from .io import ImageStat, load_fits_image
from .matching import (
    apply_solution,
    plot_transform_diagnostics,
    solve_catalog_transform,
)
from .photometry import (
    build_epsf_model,
    plot_epsf_cutouts,
    run_aperture_photometry,
    run_dophot_catalog,
    run_epsf_photometry,
)

warnings.filterwarnings("ignore", category=AstropyUserWarning)

IO_ERROR = 1
RMCR_ERROR = 2
DETECT_ERROR = 3
APPHOT_ERROR = 4
DOPHOT_ERROR = 5
MATCH_ERROR = 6


def _step(*, error_flag: int | None = None):
    """Decorator to convert unhandled exceptions into flagged pipeline errors."""

    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                if error_flag is not None:
                    self.flag = error_flag
                self._append_note("error", str(exc), step=method.__name__)
                raise

        return wrapped

    return decorator


@dataclass(kw_only=True)
class IO:
    def dump(self, filename: str | None = None) -> None:
        """Serialize the image object to a pickle file."""
        if not filename:
            filename = self.path.stem + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> IO:
        """Load a pickled image object."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def copy(self) -> IO:
        """Return a deep copy."""
        return deepcopy(self)


@dataclass(kw_only=True)
class Image(IO):
    """Image data plus extracted catalog and processing state."""

    path: Path | str
    data: NDArray | None = None
    header: fits.Header | None = None

    catalog: Catalog = field(default_factory=Catalog)
    stat: ImageStat = field(default_factory=ImageStat)
    note: str = ""
    flag: int = 0
    epsf: Any | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        if self.data is None or self.header is None:
            self.from_fits()

    @_step(error_flag=IO_ERROR)
    def from_fits(self) -> Image:
        """Load data/header/stat from the FITS file path."""
        self.data, self.header, self.stat = load_fits_image(self.path)
        self._append_note(
            "ok",
            f"path={self.path.name} shape={self.data.shape} gain={self.stat.gain:.3f} rdnoise={self.stat.rdnoise:.3f}",
            step="from_fits",
        )
        return self

    def _append_note(
        self, level: Literal["ok", "error"], detail: str, *, step: str
    ) -> None:
        """Append a timestamped status message to ``note``."""
        detail = detail.replace("\n", " ").strip()
        status = "OK" if level == "ok" else "ERROR"
        ts = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        entry = f"[{ts}] [{status}] {step} | {detail}"
        self.note = entry if not self.note else f"{self.note}\n{entry}"

    def _fail_step(self, step: str, detail: str, *, error_flag: int) -> None:
        """Record a non-raising pipeline failure and set the image flag."""
        self.flag = error_flag
        self._append_note("error", detail, step=step)

    def __repr__(self) -> str:
        shape = None if self.data is None else tuple(self.data.shape)
        return (
            f"Image(path={self.path.name!r}, shape={shape}, "
            f"nstars={self.catalog.nstars}, fwhm={self.stat.fwhm:.2f}, background={self.stat.background:.2f})"
        )


    def clear(self) -> Image:
        """Clear derived catalog/stat fields while keeping image/header loaded."""
        self.catalog.clear()
        self.stat.fwhm = np.nan
        self.stat.background = np.nan
        self.stat.background2d = None
        return self

    @_step(error_flag=RMCR_ERROR)
    def remove_cosmic_rays(self) -> Image:
        """Run L.A.Cosmic and update image data in place."""
        ccd = CCDData(self.data, unit="adu")
        cleaned = cosmicray_lacosmic(
            ccd,
            gain=self.stat.gain,
            readnoise=self.stat.rdnoise,
            sigclip=4.5,
            sigfrac=0.3,
            objlim=5.0,
        )
        self.data = cleaned.data
        self._append_note("ok", "lacosmic completed", step="remove_cosmic_rays")
        return self

    @_step(error_flag=DETECT_ERROR)
    def detect_star(
        self,
        *,
        finder_fwhm: float = 3.0,
        threshold_sigma: float = 4.0,
        saturation_level: float = 40000.0,
        background: bool = True,
    ) -> Image:
        """Detect stars and initialize the in-memory magnitude catalog."""
        self.clear()
        catalog, bkg, bkg2d, detect_std = detect_star_catalog(
            self.data,
            finder_fwhm=finder_fwhm,
            threshold_sigma=threshold_sigma,
            saturation_level=saturation_level,
            use_background=background,
        )
        self.catalog = catalog
        self.stat.background = bkg
        self.stat.background2d = bkg2d

        if self.catalog.nstars == 0:
            self._fail_step("detect_star", "no stars detected", error_flag=DETECT_ERROR)
            return self

        self._append_note(
            "ok",
            f"detected={self.catalog.nstars} std={detect_std:.4f}",
            step="detect_star",
        )
        return self

    @_step(error_flag=DETECT_ERROR)
    def estimate_fwhm(
        self,
        *,
        max_stars: int = 512,
        half_size: int = 6,
    ) -> Image:
        """Estimate frame FWHM from bright detected stars."""
        ind = np.argsort(self.catalog.mag)[:max_stars]
        used = int(len(ind))
        self.stat.fwhm = estimate_stellar_fwhm(
                self.data,
                self.catalog.x[ind],
                self.catalog.y[ind],
                half_size=half_size,
            )
        self._append_note(
            "ok", f"fwhm={self.stat.fwhm:.3f} n={used}", step="estimate_fwhm"
        )
        return self

    def sort_by(self: Image, kwd: str | List[str] = "mag") -> Image:
        """Sort catalog rows by one or more keys."""
        self.catalog.sort_inplace(kwd)
        return self

    def show(
        self,
        ax: plt.Axes | None = None,
        percentile: Tuple[float, float] = (1, 99),
        cmap: str = "gray",
    ) -> Tuple[plt.Figure | plt.SubFigure, plt.Axes]:
        """Plot image data with current catalog positions overlaid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        vmin, vmax = np.nanpercentile(self.data, percentile)
        ax.imshow(self.data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.scatter(
            self.catalog.x,
            self.catalog.y,
            s=24,
            facecolors="none",
            edgecolors="limegreen",
            lw=0.7,
        )
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title(self.path.name)
        return fig, ax

    @_step(error_flag=DOPHOT_ERROR)
    def run_dophot(
        self,
        dophot_bin: Path,
        default_par: Path,
        tmp_dir: Path,
        version: Literal["C", "fortran"] = "C",
    ) -> Image:
        """Execute DoPHOT and ingest its catalog/stat outputs."""
        catalog, out_background, out_fwhm = run_dophot_catalog(
            path=self.path,
            stat=self.stat,
            dophot_bin=dophot_bin,
            default_par=default_par,
            tmp_dir=tmp_dir,
            version=version,
        )

        if catalog is None:
            self._fail_step(
                "run_dophot", "empty or invalid DoPHOT output", error_flag=DOPHOT_ERROR
            )
            return self

        self.catalog = catalog
        self.stat.background = out_background
        self.stat.fwhm = out_fwhm
        self._append_note("ok", f"detected={self.catalog.nstars}", step="run_dophot")
        return self

    @_step()
    def build_epsf(
        self,
        oversample: int = 2,
        max_stars: int = 100,
        inspect: bool = True,
    ) -> Image:
        """Build an ePSF model from isolated stars."""
        self.epsf, stars, stars_used = build_epsf_model(
            self.data,
            self.catalog,
            oversample=oversample,
            max_stars=max_stars,
        )
        if inspect:
            plot_epsf_cutouts(stars)

        self._append_note(
            "ok", f"stars_used={stars_used} oversample={oversample}", step="build_epsf"
        )
        return self

    @_step()
    def epsfphot(self, cutout_size: int = 9, inspect: bool = False) -> Image:
        """Run ePSF photometry and replace the in-memory catalog."""
        self.catalog, phot, _ = run_epsf_photometry(
            self.data,
            self.catalog,
            epsf=self.epsf,
            cutout_size=cutout_size,
        )

        if inspect:
            model_image = phot.make_model_image(self.data.shape)
            residual_image = phot.make_residual_image(self.data)
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))

            norm0 = plt.matplotlib.colors.Normalize(
                *np.nanpercentile(self.data, [1, 99])
            )
            axes[0].imshow(self.data, origin="lower", norm=norm0, cmap="viridis")
            axes[0].scatter(
                self.catalog.x,
                self.catalog.y,
                ec="red",
                fc="none",
                lw=0.7,
                s=25,
            )
            axes[0].set_title("Original image")

            axes[1].imshow(self.epsf.data, origin="lower", cmap="viridis")
            axes[1].set_title("ePSF image")

            axes[2].imshow(model_image, origin="lower", cmap="viridis")
            axes[2].set_title("Model image")

            axes[3].imshow(residual_image, origin="lower", cmap="viridis")
            axes[3].set_title("Residual image")

            for ax in axes:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            plt.tight_layout()
            plt.show()

        self._append_note("ok", f"fitted={self.catalog.nstars}", step="epsfphot")
        return self

    @_step(error_flag=APPHOT_ERROR)
    def apphot(
        self,
        r_ap: float = 1.5,
        r_in: float = 3,
        r_out: float = 6,
        zeropoint: float = 25.0,
        auto_scale: bool = True,
    ) -> Image:
        """Run aperture photometry and replace the in-memory catalog."""
        self.catalog = run_aperture_photometry(
            self.data,
            self.catalog,
            stat=self.stat,
            r_ap=r_ap,
            r_in=r_in,
            r_out=r_out,
            zeropoint=zeropoint,
            auto_scale=auto_scale,
        )

        if self.nstars == 0:
            self._fail_step(
                "apphot",
                "no valid stars after aperture photometry",
                error_flag=APPHOT_ERROR,
            )
            return self

        self._append_note("ok", f"kept={self.catalog.nstars}", step="apphot")
        return self

    @_step(error_flag=MATCH_ERROR)
    def transform_to(
        self,
        img: Image,
        flip: bool = False,
        inspect: bool = False,
        superflat_order: Tuple[int, int] = (0, 0),
        select: Callable = lambda _: _ > -np.inf,
    ) -> Image:
        """Align this image catalog to a reference image catalog."""
        try:
            sol = solve_catalog_transform(
                self.catalog,
                img.catalog,
                flip=flip,
                superflat_order=superflat_order,
                select=select,
            )
        except ValueError as exc:
            self._fail_step("transform_to", str(exc), error_flag=MATCH_ERROR)
            return self

        if inspect:
            plot_transform_diagnostics(self.catalog, img.catalog, sol)

        self.catalog = apply_solution(self.catalog, sol)
        self._append_note(
            "ok",
            f"matched={len(sol.id2)} used={sol.n_used} std={sol.std:.4f}",
            step="transform_to",
        )
        return self
