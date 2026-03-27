"""High-level image pipeline orchestrator."""

from __future__ import annotations

import pickle
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astroscrappy import detect_cosmics
from numpy.typing import NDArray

from .catalog import (
    Catalog,
    apply_solution,
    plot_transform_diagnostics,
    solve_catalog_transform,
)
from .cr import detect_streak_mask
from .detection import detect_star_catalog
from .fwhm import estimate_stellar_fwhm
from .io import ImageStat, load_fits_image, write_fits_image
from .photometry import (
    build_epsf_model,
    plot_epsf_cutouts,
    plot_epsf_photometry_diagnostics,
    run_aperture_photometry,
    run_dophot_catalog,
    run_epsf_photometry,
)

warnings.filterwarnings("ignore", category=AstropyUserWarning)


class ImageFlag(IntEnum):
    """Pipeline status / failure flag for an ``Image``."""

    OK = 0
    IO_ERROR = 1
    RMCR_ERROR = 2
    DETECT_ERROR = 3
    APPHOT_ERROR = 4
    DOPHOT_ERROR = 5
    MATCH_ERROR = 6


def _step(*, error_flag: ImageFlag | None = None):
    """Decorator to convert unhandled exceptions into flagged pipeline errors."""

    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                if error_flag is not None:
                    self.flag = error_flag
                self.append_note(method.__name__, str(exc), error=True)
                raise

        return wrapped

    return decorator


@dataclass(kw_only=True)
class Image:
    """Image data plus extracted catalog and processing state."""

    path: Path | str
    data: NDArray | None = None
    mask: NDArray[np.bool_] | None = None
    header: fits.Header | None = None

    catalog: Catalog = field(default_factory=Catalog)
    stat: ImageStat = field(default_factory=ImageStat)
    note: str = ""
    flag: ImageFlag = ImageFlag.OK
    epsf: Any | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        if self.data is None and self.header is None:
            self.from_fits()
            return

        if self.data is None or self.header is None:
            raise ValueError(
                "data and header must both be provided for in-memory Image construction"
            )

        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError(f"expected a 2D image array, got shape={self.data.shape}")

        if self.mask is None:
            self.mask = np.zeros_like(self.data, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.data.shape:
                raise ValueError(
                    f"mask shape {self.mask.shape} does not match data shape {self.data.shape}"
                )

    @_step(error_flag=ImageFlag.IO_ERROR)
    def from_fits(self) -> Image:
        """Load data, mask, header, and statistics from ``self.path``."""
        self.data, self.mask, self.header, self.stat = load_fits_image(self.path)
        self.append_note(
            "from_fits",
            (
                f"path={self.path.name} shape={self.data.shape} "
                f"masked={int(np.sum(self.mask))} "
                f"gain={self.stat.gain:.3f} rdnoise={self.stat.rdnoise:.3f}"
            ),
        )
        return self

    def to_fits(
        self,
        filename: str | Path,
        overwrite: bool = True,
    ) -> None:
        """Write the current image data and mask to a FITS file."""
        if self.data is None or self.mask is None:
            raise ValueError("cannot write FITS without image data and mask")
        write_fits_image(
            filename,
            self.data,
            self.mask,
            self.header,
            self.stat,
            overwrite=overwrite,
        )

    def append_note(self, step: str, detail: str, *, error: bool = False) -> None:
        """Append a timestamped status message to ``note``."""
        detail = detail.replace("\n", " ").strip()
        status = "ERROR" if error else "OK"
        ts = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        entry = f"[{ts}] [{status}] {step} | {detail}"
        self.note = entry if not self.note else f"{self.note}\n{entry}"

    def __repr__(self) -> str:
        shape = None if self.data is None else tuple(self.data.shape)
        n_masked = 0 if self.mask is None else int(np.sum(self.mask))
        return (
            f"Image(path={self.path.name!r}, shape={shape}, "
            f"masked={n_masked}, nstars={len(self.catalog)}, "
            f"fwhm={self.stat.fwhm:.2f}, background={self.stat.background:.2f})"
        )

    def copy(self) -> Image:
        """Return a deep copy."""
        return deepcopy(self)

    def filled_data(self, fill_value: float | None = None) -> NDArray:
        """Return a temporary copy with masked pixels replaced."""
        if fill_value is None:
            valid = self.data[~self.mask]
            fill_value = float(np.nanmedian(valid)) if valid.size else 0.0

        filled = np.array(self.data, copy=True)
        filled[self.mask] = fill_value
        return filled

    def clear(self) -> Image:
        """Clear derived catalog/stat fields while keeping image/header loaded."""
        self.catalog.clear()
        self.stat.fwhm = np.nan
        self.stat.background = np.nan
        self.stat.background2d = None
        return self

    def sort_by(self: Image, kwd: str | list[str] = "mag") -> Image:
        """Sort catalog rows by one or more keys."""
        self.catalog.sort_inplace(kwd)
        return self

    def normalize(self, gain: float | None = None) -> Image:
        """Convert the image data from ADU to electrons using ``gain``."""
        if gain is not None:
            self.stat.gain = gain
        self.data = self.data.astype(np.float32) * self.stat.gain
        return self

    def trim(self, x0: int, y0: int, width: int, height: int) -> Image:
        """Trim the image and update catalog positions."""
        self.data = self.data[y0 : y0 + height, x0 : x0 + width]
        self.mask = self.mask[y0 : y0 + height, x0 : x0 + width]
        self.catalog.x -= x0
        self.catalog.y -= y0
        use = (
            (self.catalog.x >= 0)
            & (self.catalog.x < width)
            & (self.catalog.y >= 0)
            & (self.catalog.y < height)
        )
        self.catalog = self.catalog[use]
        self.append_note("trim", f"x0={x0} y0={y0} width={width} height={height}")
        return self

    def trim_edge(self, edge: int) -> Image:
        """Trim a fixed-width edge from the image and update catalog positions."""
        return self.trim(
            edge,
            edge,
            self.data.shape[1] - 2 * edge,
            self.data.shape[0] - 2 * edge,
        )

    def show(
        self,
        ax: plt.Axes | None = None,
        percentile: tuple[float, float] = (1, 99),
        cmap: str = "gray",
    ) -> tuple[plt.Figure | plt.SubFigure, plt.Axes]:
        """Plot image data with current catalog positions overlaid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        valid = self.data[~self.mask]
        if valid.size == 0:
            valid = self.data.ravel()
        vmin, vmax = np.nanpercentile(valid, percentile)
        display = np.ma.array(self.data, mask=self.mask)
        ax.imshow(display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
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

    @_step(error_flag=ImageFlag.RMCR_ERROR)
    def remove_cosmic_rays(self) -> Image:
        """Run detection for cosmic rays and add them to the mask."""
        crmask, _ = detect_cosmics(
            self.filled_data(),
            gain=self.stat.gain,
            readnoise=self.stat.rdnoise,
        )
        streak_mask, _ = detect_streak_mask(self.filled_data())
        self.mask |= crmask | streak_mask
        self.append_note(
            "remove_cosmic_rays",
            f"cosmics={int(np.sum(crmask))} masked={int(np.sum(self.mask))}",
        )
        return self

    @_step(error_flag=ImageFlag.DETECT_ERROR)
    def detect_sources(
        self,
        *,
        finder_fwhm: float = 3.0,
        threshold_sigma: float = 5.0,
        saturation_level: float = 2.0e16,
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
            mask=self.mask,
        )
        self.catalog = catalog
        self.stat.background = bkg
        self.stat.background2d = bkg2d

        if len(self.catalog) == 0:
            self.flag = ImageFlag.DETECT_ERROR
            self.append_note("detect_sources", "no stars detected", error=True)
            return self

        self.append_note(
            "detect_sources",
            f"detected={len(self.catalog)} std={detect_std:.4f}",
        )
        return self

    @_step(error_flag=ImageFlag.DETECT_ERROR)
    def estimate_fwhm(
        self,
        *,
        max_stars: int = 512,
        half_size: int = 6,
    ) -> Image:
        """Estimate frame FWHM from bright detected stars."""
        ind = np.argsort(self.catalog.mag)[:max_stars]
        used = len(ind)
        self.stat.fwhm = estimate_stellar_fwhm(
            self.data,
            self.catalog.x[ind],
            self.catalog.y[ind],
            mask=self.mask,
            half_size=half_size,
        )
        self.append_note("estimate_fwhm", f"fwhm={self.stat.fwhm:.3f} n={used}")
        return self

    @_step(error_flag=ImageFlag.DOPHOT_ERROR)
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
            data=self.data,
            header=self.header,
            stat=self.stat,
            mask=self.mask,
            dophot_bin=dophot_bin,
            default_par=default_par,
            tmp_dir=tmp_dir,
            version=version,
        )

        if catalog is None:
            self.flag = ImageFlag.DOPHOT_ERROR
            self.append_note("run_dophot", "empty or invalid DoPHOT output", error=True)
            return self

        self.catalog = catalog
        self.stat.background = out_background
        self.stat.fwhm = out_fwhm
        self.append_note("run_dophot", f"detected={len(self.catalog)}")
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
            mask=self.mask,
        )
        if inspect:
            plot_epsf_cutouts(stars)

        self.append_note(
            "build_epsf",
            f"stars_used={stars_used} oversample={oversample}",
        )
        return self

    @_step()
    def run_epsf_photometry(
        self,
        cutout_size: int = 9,
        inspect: bool = False,
    ) -> Image:
        """Run ePSF photometry and replace the in-memory catalog."""
        self.catalog, phot, _ = run_epsf_photometry(
            self.data,
            self.catalog,
            epsf=self.epsf,
            cutout_size=cutout_size,
            mask=self.mask,
        )

        if inspect:
            plot_epsf_photometry_diagnostics(
                self.data,
                self.catalog,
                epsf=self.epsf,
                phot=phot,
            )

        self.append_note("run_epsf_photometry", f"fitted={len(self.catalog)}")
        return self

    @_step(error_flag=ImageFlag.APPHOT_ERROR)
    def run_aperture_photometry(
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
            mask=self.mask,
        )

        if len(self.catalog) == 0:
            self.flag = ImageFlag.APPHOT_ERROR
            self.append_note(
                "run_aperture_photometry",
                "no valid stars after aperture photometry",
                error=True,
            )
            return self

        self.append_note("run_aperture_photometry", f"kept={len(self.catalog)}")
        return self

    @_step(error_flag=ImageFlag.MATCH_ERROR)
    def transform_to(
        self,
        img: Image,
        flip: bool = False,
        inspect: bool = False,
        superflat_order: tuple[int, int] = (0, 0),
        select: Callable = lambda _: _ > -np.inf,
    ) -> Image:
        """Align this image catalog to a reference image catalog."""
        source_catalog = self.catalog
        try:
            sol = solve_catalog_transform(
                source_catalog,
                img.catalog,
                flip=flip,
                superflat_order=superflat_order,
                select=select,
            )
        except ValueError as exc:
            self.flag = ImageFlag.MATCH_ERROR
            self.append_note("transform_to", str(exc), error=True)
            return self

        if inspect:
            plot_transform_diagnostics(source_catalog, img.catalog, sol)

        self.catalog = apply_solution(source_catalog, sol)
        self.data = sol.transform.warp_image(
            self.filled_data(),
            output_shape=img.data.shape,
        )
        self.mask = (
            sol.transform.warp_image(
                self.mask.astype(float),
                output_shape=img.data.shape,
                order=0,
                cval=1.0,
            )
            > 0.5
        )
        self.append_note(
            "transform_to",
            f"matched={len(sol.id2)} used={sol.n_used} std={sol.std:.4f}",
        )
        return self

    def dump(self, filename: str | None = None) -> None:
        """Serialize the image object to a pickle file."""
        if not filename:
            filename = self.path.stem + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> Image:
        """Load a pickled image object."""
        with open(filename, "rb") as f:
            return pickle.load(f)
