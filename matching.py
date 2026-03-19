from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from astropy.stats import sigma_clipped_stats
from numpy.typing import NDArray

from .catalog import StarCatalog
from .pymatch import Similarity, match_stars


@dataclass
class MatchSolution:
    transform: Similarity
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
    target: StarCatalog,
    ref: StarCatalog,
    *,
    flip: bool,
    superflat_order: tuple[int, int],
    select: Callable,
) -> MatchSolution:
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
        n_used = int(len(diff[use]))

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


def apply_solution(catalog: StarCatalog, solution: MatchSolution) -> StarCatalog:
    mag = catalog.mag - solution.med
    mag_err = np.sqrt(catalog.mag_err**2 + solution.std**2 / solution.n_used)
    x, y = solution.transform.apply(np.c_[catalog.x, catalog.y]).T

    return StarCatalog.from_arrays(
        x=x,
        y=y,
        mag=mag,
        mag_err=mag_err,
    )


def _poly_basis(
    x: NDArray,
    y: NDArray,
    order_x: int,
    order_y: int,
) -> NDArray:
    terms = []
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            terms.append((x**i) * (y**j))
    return np.column_stack(terms)


def _solve_superflat(
    target: StarCatalog,
    id1: NDArray,
    use: NDArray,
    diff: NDArray,
    superflat_order: tuple[int, int],
) -> tuple[NDArray, float, int]:
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
