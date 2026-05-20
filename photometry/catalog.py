"""Star catalog container and common catalog-level operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from numpy.typing import NDArray

from .pymatch import Transformation, match_stars


def empty_array() -> NDArray:
    """Return an empty 1D float array used by dataclass defaults."""
    return np.empty(0, dtype=float)


@dataclass
class Catalog:
    """In-memory photometry catalog for a single image."""

    x: NDArray = field(default_factory=empty_array)
    y: NDArray = field(default_factory=empty_array)
    mag: NDArray = field(default_factory=empty_array)
    mag_err: NDArray = field(default_factory=empty_array)

    def __len__(self) -> int:
        """Number of stars currently stored in the catalog."""
        return int(len(self.x))

    def __getitem__(self, item) -> Catalog:
        """Return a sliced or masked view as a new ``Catalog``."""
        return Catalog.from_arrays(
            x=self.x[item],
            y=self.y[item],
            mag=self.mag[item],
            mag_err=self.mag_err[item],
        )

    def clear(self) -> None:
        """Reset all catalog columns to empty arrays."""
        self.x = empty_array()
        self.y = empty_array()
        self.mag = empty_array()
        self.mag_err = empty_array()

    def sort_inplace(self, keys: str | list[str] = "mag") -> None:
        """Sort the catalog in place by one or multiple column names."""
        if len(self) == 0:
            return
        if isinstance(keys, str):
            idx = np.argsort(getattr(self, keys))
        else:
            arrays = [getattr(self, key) for key in keys]
            idx = np.lexsort(arrays[::-1])

        self.x = self.x[idx]
        self.y = self.y[idx]
        self.mag = self.mag[idx]
        self.mag_err = self.mag_err[idx]

    @classmethod
    def from_arrays(
        cls,
        x: NDArray,
        y: NDArray,
        mag: NDArray,
        mag_err: NDArray,
    ) -> Catalog:
        """Create a catalog from array-like inputs and normalize to float arrays."""
        return cls(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            mag=np.asarray(mag, dtype=float),
            mag_err=np.asarray(mag_err, dtype=float),
        )

    def calibrate_to(
        self,
        ref: Catalog,
        *,
        flip: bool = False,
        select: Callable = lambda _: _ > -np.inf,
        superflat_order: tuple[int, int] = (0, 0),
        color: NDArray | None = None,
        ref_color: NDArray | None = None,
        max_mag_err: float = 0.1,
        report: bool = False,
    ) -> Catalog:
        """Return this catalog calibrated onto the reference catalog system.

        Supports:
        - coordinate matching
        - zeropoint calibration
        - optional spatial superflat correction
        - optional color term using ``color`` / ``ref_color``
        """
        fit = _solve_calibration_fit(
            self,
            ref,
            flip=flip,
            select=select,
            superflat_order=superflat_order,
            ref_color=ref_color,
            max_mag_err=max_mag_err,
        )
        if report:
            print(fit)
        return _apply_calibration_fit(self, fit, color=color)


@dataclass
class _CalibrationFit:
    """Internal calibration fit."""

    transform: Transformation
    id1: NDArray
    id2: NDArray
    use: NDArray
    diff: NDArray
    err: NDArray
    coeff: NDArray
    coeff_err: NDArray
    std: float
    n_used: int
    superflat_order: tuple[int, int]
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    ref_color_match: NDArray | None = None

    def evaluate(
        self,
        x: NDArray,
        y: NDArray,
        *,
        color: NDArray | None = None,
    ) -> NDArray:
        basis = _calibration_basis(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            color=color,
            superflat_order=self.superflat_order,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )
        return basis @ self.coeff

    def __repr__(self) -> str:
        """Return a compact fit summary."""
        lines = [
            f"matched={len(self.id2)} used={self.n_used} scatter={self.std:.4f}",
            f"zeropoint={self.coeff[0]:.4f} +/- {self.coeff_err[0]:.4f}",
        ]
        idx = 1
        if self.ref_color_match is not None:
            lines.append(f"color_term={self.coeff[idx]:.4f} +/- {self.coeff_err[idx]:.4f}")
            idx += 1
        order_x, order_y = self.superflat_order
        if order_x > 0 or order_y > 0:
            lines.append(f"superflat_order={self.superflat_order}")
            for i in range(idx, len(self.coeff)):
                lines.append(
                    f"superflat_coeff[{i - idx}]={self.coeff[i]:.4f} +/- {self.coeff_err[i]:.4f}"
                )
        return "\n".join(lines)


def _solve_calibration_fit(
    target: Catalog,
    ref: Catalog,
    *,
    flip: bool = False,
    select: Callable = lambda _: _ > -np.inf,
    superflat_order: tuple[int, int] = (0, 0),
    ref_color: NDArray | None = None,
    max_mag_err: float = 0.1,
) -> _CalibrationFit:
    """Solve coordinate matching and photometric calibration."""
    if min(len(target), len(ref)) < 3:
        raise ValueError(f"not enough stars self={len(target)} ref={len(ref)}")

    if ref_color is not None:
        ref_color = np.asarray(ref_color, dtype=float)
        if len(ref_color) != len(ref):
            raise ValueError(
                f"ref_color length {len(ref_color)} does not match ref length {len(ref)}"
            )

    sgn = -1 if flip else 1
    target_xy = np.c_[target.x * sgn, target.y]
    ref_xy = np.c_[ref.x, ref.y]
    res = match_stars(target_xy, ref_xy, 2)

    if res.inlier_count < 50 and len(ref_xy) > 2 * len(target_xy):
        ref_xy = ref_xy[: int(1.5 * len(target_xy))]
        res = match_stars(target_xy, ref_xy, 2)

    res.transform.A[:, 0] *= sgn
    id1, id2 = res.pairs[:, 0], res.pairs[:, 1]
    if len(id2) < 10:
        raise ValueError(f"matched stars={len(id2)}")

    diff = target.mag[id1] - ref.mag[id2]
    err = np.sqrt(target.mag_err[id1] ** 2 + ref.mag_err[id2] ** 2)
    use = (0 < err) & (err < max_mag_err)
    use &= select(ref.mag[id2])

    ref_color_match = None
    if ref_color is not None:
        ref_color_match = np.asarray(ref_color[id2], dtype=float)
        use &= np.isfinite(ref_color_match)

    (
        coeff,
        coeff_err,
        std,
        n_used,
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = _fit_photometric_model(
        x=target.x[id1],
        y=target.y[id1],
        diff=diff,
        use=use,
        ref_color=ref_color_match,
        superflat_order=superflat_order,
    )

    return _CalibrationFit(
        transform=res.transform,
        id1=id1,
        id2=id2,
        use=use,
        diff=diff,
        err=err,
        coeff=coeff,
        coeff_err=coeff_err,
        std=std,
        n_used=n_used,
        superflat_order=superflat_order,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        ref_color_match=ref_color_match,
    )


def _apply_calibration_fit(
    catalog: Catalog,
    fit: _CalibrationFit,
    *,
    color: NDArray | None = None,
) -> Catalog:
    """Apply a solved calibration fit to a catalog."""
    if fit.ref_color_match is not None:
        if color is None:
            raise ValueError("color is required for two-band calibration")
        color = np.asarray(color, dtype=float)
        if len(color) != len(catalog):
            raise ValueError(
                f"color length {len(color)} does not match catalog length {len(catalog)}"
            )

    correction = fit.evaluate(catalog.x, catalog.y, color=color)
    mag = catalog.mag - correction
    mag_err = np.sqrt(catalog.mag_err**2 + fit.std**2 / fit.n_used)
    x, y = fit.transform.apply(np.c_[catalog.x, catalog.y]).T
    return Catalog.from_arrays(x=x, y=y, mag=mag, mag_err=mag_err)


def plot_transform_diagnostics(
    target: Catalog,
    ref: Catalog,
    fit: _CalibrationFit,
) -> None:
    """Plot coordinate and residual diagnostics for a solved calibration."""
    plot_x = target.x[fit.id1][fit.use]
    plot_y = target.y[fit.id1][fit.use]
    ref_mag = ref.mag[fit.id2][fit.use]
    plot_err = fit.err[fit.use]
    plot_color = None if fit.ref_color_match is None else fit.ref_color_match[fit.use]

    model_vals = fit.evaluate(plot_x, plot_y, color=plot_color)
    residuals = fit.diff[fit.use] - model_vals
    inliers = np.abs(residuals) < (3.0 * fit.std)
    outliers = ~inliers

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs = axs.ravel()
    ebar_kwargs = dict(fmt="o", fillstyle="none", markersize=4, lw=0.7, alpha=0.6)

    axs[0].scatter(ref.x[fit.id2], ref.y[fit.id2], s=15, fc="none", ec="r", lw=1, label="Ref")
    _x, _y = fit.transform.apply(np.c_[target.x[fit.id1], target.y[fit.id1]]).T
    axs[0].scatter(_x, _y, s=15, marker="x", lw=1, c="b", label="Target")
    axs[0].set_title("Matched Star Coordinates")
    axs[0].set_xlabel("X [px]")
    axs[0].set_ylabel("Y [px]")
    axs[0].legend()

    if np.any(outliers):
        axs[1].errorbar(ref_mag[outliers], residuals[outliers], yerr=plot_err[outliers], c="C1", label="Rejected", **ebar_kwargs)
    if np.any(inliers):
        axs[1].errorbar(ref_mag[inliers], residuals[inliers], yerr=plot_err[inliers], c="C0", label="Used", **ebar_kwargs)
    axs[1].axhline(0, c="r", lw=1.5, ls="--")
    axs[1].set_ylim(0.3, -0.3)
    axs[1].set_xlabel("Ref Mag")
    axs[1].set_ylabel("Residual (img - ref - model)")
    axs[1].set_title("Magnitude Residuals")

    parts = [f"std={fit.std:.3f}", f"used={fit.n_used}"]
    if sum(fit.superflat_order) > 0:
        parts.append(f"superflat={fit.superflat_order}")
    if fit.ref_color_match is not None:
        parts.append(f"color={fit.coeff[1]:.3f}")
    axs[1].text(
        0.05,
        0.95,
        "\n".join(parts),
        transform=axs[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    if np.any(outliers):
        axs[3].errorbar(plot_x[outliers], residuals[outliers], yerr=plot_err[outliers], c="C1", label="Rejected", **ebar_kwargs)
    if np.any(inliers):
        axs[3].errorbar(plot_x[inliers], residuals[inliers], yerr=plot_err[inliers], c="C0", label="Used", **ebar_kwargs)
    axs[3].axhline(0, c="r", lw=1.5, ls="--")
    axs[3].set_ylim(0.3, -0.3)
    axs[3].set_xlabel("X [px]")
    axs[3].set_ylabel("Residual (img - ref - model)")
    axs[3].set_title("Spatial Residuals (X)")

    if np.any(outliers):
        axs[4].errorbar(plot_y[outliers], residuals[outliers], yerr=plot_err[outliers], c="C1", label="Rejected", **ebar_kwargs)
    if np.any(inliers):
        axs[4].errorbar(plot_y[inliers], residuals[inliers], yerr=plot_err[inliers], c="C0", label="Used", **ebar_kwargs)
    axs[4].axhline(0, c="r", lw=1.5, ls="--")
    axs[4].set_ylim(0.3, -0.3)
    axs[4].set_xlabel("Y [px]")
    axs[4].set_ylabel("Residual (img - ref - model)")
    axs[4].set_title("Spatial Residuals (Y)")

    norm = plt.Normalize(vmin=-3 * fit.std, vmax=3 * fit.std)
    axs[2].scatter(plot_x, plot_y, c=residuals, norm=norm, cmap="coolwarm", s=20, edgecolor="k", lw=0.5)
    axs[2].set_xlabel("X [px]")
    axs[2].set_ylabel("Y [px]")
    axs[2].set_title("Residuals Spatial Map")

    if plot_color:
        axs[5].plot(plot_color, residuals, "o", ms=5, alpha=0.7)
        axs[5].axhline(0, c="r", lw=1.5, ls="--")
        axs[5].set_xlabel("Color")
        axs[5].set_ylabel("Residual (img - ref - model)")
        axs[5].set_title("Color Residuals")
        fig.tight_layout()



def _poly_basis(
    x: NDArray,
    y: NDArray,
    order_x: int,
    order_y: int,
    *,
    include_constant: bool = True,
) -> NDArray:
    """Build a dense 2D polynomial design matrix."""
    terms = []
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            if not include_constant and i == 0 and j == 0:
                continue
            terms.append((x**i) * (y**j))
    return np.column_stack(terms)


def _calibration_basis(
    *,
    x: NDArray,
    y: NDArray,
    color: NDArray | None,
    superflat_order: tuple[int, int],
    x_mean: float,
    x_std: float,
    y_mean: float,
    y_std: float,
) -> NDArray:
    """Build the calibration basis matrix."""
    cols = [np.ones(len(x), dtype=float)]
    if color is not None:
        cols.append(np.asarray(color, dtype=float))

    order_x, order_y = superflat_order
    if order_x > 0 or order_y > 0:
        xn = (x - x_mean) / max(x_std, 1.0e-6)
        yn = (y - y_mean) / max(y_std, 1.0e-6)
        cols.append(
            _poly_basis(
                xn,
                yn,
                order_x,
                order_y,
                include_constant=False,
            )
        )

    out = []
    for col in cols:
        col = np.asarray(col, dtype=float)
        if col.ndim == 1:
            out.append(col[:, None])
        else:
            out.append(col)
    return np.column_stack(out)


def _fit_photometric_model(
    *,
    x: NDArray,
    y: NDArray,
    diff: NDArray,
    use: NDArray,
    ref_color: NDArray | None,
    superflat_order: tuple[int, int],
) -> tuple[NDArray, NDArray, float, int, float, float, float, float]:
    """Fit a sigma-clipped photometric model."""
    if np.sum(use) < 3:
        raise ValueError(f"not enough calibration stars={int(np.sum(use))}")

    x_fit = np.asarray(x[use], dtype=float)
    y_fit = np.asarray(y[use], dtype=float)
    diff_fit = np.asarray(diff[use], dtype=float)
    color_fit = None if ref_color is None else np.asarray(ref_color[use], dtype=float)

    x_mean = float(np.mean(x_fit))
    x_std = float(np.std(x_fit))
    y_mean = float(np.mean(y_fit))
    y_std = float(np.std(y_fit))
    basis = _calibration_basis(
        x=x_fit,
        y=y_fit,
        color=color_fit,
        superflat_order=superflat_order,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    valid = np.ones(len(diff_fit), dtype=bool)
    for _ in range(30):
        coeff = np.linalg.lstsq(basis[valid], diff_fit[valid], rcond=None)[0]
        residuals = diff_fit - (basis @ coeff)
        median = np.median(residuals[valid])
        mad = np.median(np.abs(residuals[valid] - median))
        std = max(1.4826 * mad, 1.0e-6)
        new_valid = np.abs(residuals) < (3.0 * std)
        if np.array_equal(valid, new_valid):
            break
        valid = new_valid

    basis_fit = basis[valid]
    dof = max(len(basis_fit) - basis_fit.shape[1], 1)
    residual_var = float(np.sum((diff_fit[valid] - basis_fit @ coeff) ** 2) / dof)
    cov = residual_var * np.linalg.pinv(basis_fit.T @ basis_fit)
    coeff_err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return coeff, coeff_err, float(std), int(np.sum(valid)), x_mean, x_std, y_mean, y_std


def ps1(refstars: Table) -> Table:
    """Append Johnson-Cousins magnitudes derived from PS1 photometry."""
    gmags = refstars["gmag"]
    gmagerrs = refstars["e_gmag"]
    rmags = refstars["rmag"]
    rmagerrs = refstars["e_rmag"]
    imags = refstars["imag"]
    imagerrs = refstars["e_imag"]

    bmags = gmags + 0.212 + 0.556 * (gmags - rmags) + 0.034 * (gmags - rmags) ** 2
    bmagerrs = np.sqrt(
        0.032**2
        + (1 + 0.556**2 + (2 * 0.034) ** 2) * gmagerrs**2
        + (0.556**2 + (2 * 0.034) ** 2) * rmagerrs**2
    )

    vmags = rmags + 0.005 + 0.462 * (gmags - rmags) + 0.013 * (gmags - rmags) ** 2
    vmagerrs = np.sqrt(
        0.012**2
        + (1 + 0.462**2 + (2 * 0.013) ** 2) * rmagerrs**2
        + (0.462**2 + (2 * 0.013) ** 2) * gmagerrs**2
    )

    rmags_c = rmags - 0.137 - 0.108 * (gmags - rmags) - 0.029 * (gmags - rmags) ** 2
    rmagerrs_c = np.sqrt(
        0.015**2
        + (1 + 0.108**2 + (2 * 0.029) ** 2) * rmagerrs**2
        + (0.108**2 + (2 * 0.029) ** 2) * gmagerrs**2
    )

    imags_c = imags - 0.366 - 0.136 * (gmags - rmags) - 0.018 * (gmags - rmags) ** 2
    imagerrs_c = np.sqrt(
        0.017**2
        + imagerrs**2
        + (0.136**2 + (2 * 0.018) ** 2) * rmagerrs**2
        + (0.136**2 + (2 * 0.018) ** 2) * gmagerrs**2
    )

    refstars["Bmag"] = bmags
    refstars["e_Bmag"] = bmagerrs
    refstars["Vmag"] = vmags
    refstars["e_Vmag"] = vmagerrs
    refstars["Rmag"] = rmags_c
    refstars["e_Rmag"] = rmagerrs_c
    refstars["Imag"] = imags_c
    refstars["e_Imag"] = imagerrs_c
    return refstars
