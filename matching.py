"""Catalog-to-catalog geometric/magnitude transformation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from numpy.typing import NDArray

from .catalog import Catalog
from .pymatch import Transformation, match_stars


@dataclass
class MatchSolution:
    """Container for an intermediate matching/fit solution."""

    transform: Transformation
    id1: NDArray
    id2: NDArray
    use: NDArray
    diff: NDArray
    err: NDArray
    med: NDArray | float
    std: float
    n_used: int
    superflat: bool


def solve_catalog_transform(
    target: Catalog,
    ref: Catalog,
    *,
    flip: bool,
    superflat_order: tuple[int, int],
    select: Callable,
) -> MatchSolution:
    """Solve coordinate/magnitude alignment from target catalog to reference."""
    if min(target.nstars, ref.nstars) < 3:
        raise ValueError(f"not enough stars self={target.nstars} ref={ref.nstars}")

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
    use = (0 < err) & (err < 0.1)
    use &= select(ref.mag[id2])

    superflat = sum(superflat_order) > 0
    if superflat:
        med, std, n_used = _solve_superflat(target, id1, use, diff, superflat_order)
    else:
        _, med, std = sigma_clipped_stats(
            data=diff[use],
            sigma=3,
            maxiters=25,
            stdfunc="mad_std",
        )
        n_used = len(diff[use])

    return MatchSolution(
        transform=res.transform,
        id1=id1,
        id2=id2,
        use=use,
        diff=diff,
        err=err,
        med=med,
        std=float(std),
        n_used=n_used,
        superflat=superflat,
    )


def apply_solution(catalog: Catalog, solution: MatchSolution) -> Catalog:
    """Apply solved transformation and error inflation to a full catalog."""
    mag = catalog.mag - solution.med
    mag_err = np.sqrt(catalog.mag_err**2 + solution.std**2 / solution.n_used)
    x, y = solution.transform.apply(np.c_[catalog.x, catalog.y]).T

    return Catalog.from_arrays(
        x=x,
        y=y,
        mag=mag,
        mag_err=mag_err,
    )


def plot_transform_diagnostics(
    target: Catalog,
    ref: Catalog,
    solution: MatchSolution,
) -> None:
    """Plot coordinate and residual diagnostics for a solved transform."""
    plot_x = target.x[solution.id1][solution.use]
    plot_y = target.y[solution.id1][solution.use]
    ref_mag = ref.mag[solution.id2][solution.use]
    plot_err = solution.err[solution.use]

    model_vals = (
        solution.med[solution.id1][solution.use] if solution.superflat else solution.med
    )
    residuals = solution.diff[solution.use] - model_vals
    inliers = np.abs(residuals) < (3.0 * solution.std)
    outliers = ~inliers

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    ebar_kwargs = dict(fmt="o", fillstyle="none", markersize=4, lw=0.7, alpha=0.6)

    axs[0].scatter(
        ref.x[solution.id2],
        ref.y[solution.id2],
        s=15,
        fc="none",
        ec="r",
        lw=1,
        label="Ref",
    )
    axs[0].scatter(
        target.x[solution.id1],
        target.y[solution.id1],
        s=15,
        marker="x",
        lw=1,
        c="b",
        label="Target",
    )
    axs[0].set_title("Matched Star Coordinates")
    axs[0].set_xlabel("X [px]")
    axs[0].set_ylabel("Y [px]")
    axs[0].legend()

    if np.any(outliers):
        axs[1].errorbar(
            ref_mag[outliers],
            residuals[outliers],
            yerr=plot_err[outliers],
            c="C1",
            label="Rejected",
            **ebar_kwargs,
        )
    if np.any(inliers):
        axs[1].errorbar(
            ref_mag[inliers],
            residuals[inliers],
            yerr=plot_err[inliers],
            c="C0",
            label="Used",
            **ebar_kwargs,
        )
    axs[1].axhline(0, c="r", lw=1.5, ls="--")
    axs[1].set_ylim(0.3, -0.3)
    axs[1].set_xlabel("Ref Mag")
    axs[1].set_ylabel("Residual (img - ref - model)")
    axs[1].set_title("Magnitude Residuals")

    mode_str = "Superflat 2D Plane" if solution.superflat else "Standard Scalar"
    stat_text = f"Mode: {mode_str}\nScatter (std): {solution.std:.3f}\nStars used: {np.sum(inliers)}"
    axs[1].text(
        0.05,
        0.95,
        stat_text,
        transform=axs[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    if np.any(outliers):
        axs[2].errorbar(
            plot_x[outliers],
            residuals[outliers],
            yerr=plot_err[outliers],
            c="C1",
            label="Rejected",
            **ebar_kwargs,
        )
    if np.any(inliers):
        axs[2].errorbar(
            plot_x[inliers],
            residuals[inliers],
            yerr=plot_err[inliers],
            c="C0",
            label="Used",
            **ebar_kwargs,
        )
    axs[2].axhline(0, c="r", lw=1.5, ls="--")
    axs[2].set_ylim(0.3, -0.3)
    axs[2].set_xlabel("X [px]")
    axs[2].set_ylabel("Residual (img - ref - model)")
    axs[2].set_title("Spatial Residuals (X)")

    if np.any(outliers):
        axs[3].errorbar(
            plot_y[outliers],
            residuals[outliers],
            yerr=plot_err[outliers],
            c="C1",
            label="Rejected",
            **ebar_kwargs,
        )
    if np.any(inliers):
        axs[3].errorbar(
            plot_y[inliers],
            residuals[inliers],
            yerr=plot_err[inliers],
            c="C0",
            label="Used",
            **ebar_kwargs,
        )
    axs[3].axhline(0, c="r", lw=1.5, ls="--")
    axs[3].set_ylim(0.3, -0.3)
    axs[3].set_xlabel("Y [px]")
    axs[3].set_ylabel("Residual (img - ref - model)")
    axs[3].set_title("Spatial Residuals (Y)")

    fig.tight_layout()


def _poly_basis(
    x: NDArray,
    y: NDArray,
    order_x: int,
    order_y: int,
) -> NDArray:
    """Build a dense 2D polynomial design matrix."""
    terms = []
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            terms.append((x**i) * (y**j))
    return np.column_stack(terms)


def _solve_superflat(
    target: Catalog,
    id1: NDArray,
    use: NDArray,
    diff: NDArray,
    superflat_order: tuple[int, int],
) -> tuple[NDArray, float, int]:
    """Fit iterative sigma-clipped 2D superflat correction."""
    order_x, order_y = superflat_order

    x_fit, y_fit = target.x[id1][use], target.y[id1][use]
    mag_fit = diff[use]

    x_mean, x_std = np.mean(target.x), np.std(target.x)
    y_mean, y_std = np.mean(target.y), np.std(target.y)

    xn = (x_fit - x_mean) / x_std
    yn = (y_fit - y_mean) / y_std

    valid = np.ones(len(mag_fit), dtype=bool)
    for _ in range(15):
        basis = _poly_basis(xn[valid], yn[valid], order_x, order_y)
        coeff = np.linalg.lstsq(basis, mag_fit[valid], rcond=None)[0]

        full_basis = _poly_basis(xn, yn, order_x, order_y)
        residuals = mag_fit - (full_basis @ coeff)

        median = np.median(residuals[valid])
        mad_raw = np.median(np.abs(residuals[valid] - median))
        std = mad_raw * 1.4826

        new_valid = np.abs(residuals) < (3.0 * std)
        if np.array_equal(valid, new_valid):
            break
        valid = new_valid

    self_xn = (target.x - x_mean) / x_std
    self_yn = (target.y - y_mean) / y_std
    all_basis = _poly_basis(self_xn, self_yn, order_x, order_y)

    med = all_basis @ coeff
    n_used = int(np.sum(valid))
    return med, float(std), n_used
