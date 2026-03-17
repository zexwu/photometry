from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


GAUSSIAN_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _local_background(cut: NDArray[np.floating]) -> float:
    """Estimate a robust local background for a cutout.

    Uses the median of the perimeter pixels (top/bottom rows and left/right
    columns without double-counting corners). This avoids the stellar core
    while adapting to slow background variations.
    """
    top = cut[0, :]
    bottom = cut[-1, :]
    left = cut[1:-1, 0]
    right = cut[1:-1, -1]
    border = np.concatenate((top, bottom, left, right))
    return float(np.median(border))


def estimate_fwhm(
    data: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fwhm_init: float = 3.0,
    *,
    size: int = 7,
    maxiters: int = 100,
    fwhm_min: float = 0.5,
    fwhm_max: float = 20.0,
    use_local_bkg: bool = True,
) -> Optional[float]:
    """Estimate the median stellar FWHM from an image using a source catalog.

    Parameters
    - data: 2D array-like
      Image data (counts), treated as float.
    - sources: catalog
      Detected sources providing subpixel centroids. Accepts photutils
      DAOStarFinder output (astropy Table) or any iterable of mappings with
      keys 'xcentroid' and 'ycentroid'. The optional 'peak' key is used as
      the initial amplitude if present.
    - fwhm_init: float
      Initial FWHM guess (pixels) used to seed the Gaussian widths.
    - size: int
      Half-window size for per-star cutouts; window is (2*size+1)^2 pixels.
    - maxiters: int
      Max iterations for the Levenberg–Marquardt fitter.
    - fwhm_min, fwhm_max: float
      Acceptable per-star FWHM range (pixels) retained in the median.
    - use_local_bkg: bool
      If True, subtract a local perimeter-median background for each cutout;
      otherwise subtract the global median of the frame.

    Returns
    - float or None: Median FWHM in pixels if suitable stars are found.

    Notes
    - FWHM is computed from the fitted covariance as 2.355 * sqrt(sx*sy),
      the geometric mean across principal axes.
    - Centers are fixed to the subpixel centroids from the catalog.
    """
    data = np.asarray(data, dtype=float, order="C")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if size <= 0:
        raise ValueError("size must be a positive integer")

    # Global background (used if local background disabled)
    _, median, _ = sigma_clipped_stats(data, sigma=3.0)

    fitter = fitting.LevMarLSQFitter()
    fwhm_values: list[float] = []

    # Precompute local grid for speed
    n = 2 * size + 1
    yy, xx = np.mgrid[:n, :n]
    sx0 = sy0 = fwhm_init / GAUSSIAN_FWHM

    ny, nx = data.shape


    for xc, yc in zip(x, y):

        xi = int(round(xc))
        yi = int(round(yc))

        # Ensure cutout is fully within the image bounds
        if not (size <= xi < nx - size and size <= yi < ny - size):
            continue

        cut = data[yi - size : yi + size + 1, xi - size : xi + size + 1]
        if cut.shape != (n, n) or not np.isfinite(cut).all():
            continue

        # Local or global background subtraction
        bkg = _local_background(cut) if use_local_bkg else float(median)
        cut_sub = cut - bkg

        # Initial elliptical Gaussian
        g0 = models.Gaussian2D(
            amplitude=cut_sub.max(),
            x_mean=xc - (xi - size),
            y_mean=yc - (yi - size),
            x_stddev=sx0,
            y_stddev=sy0,
        )

        res = fitter(g0, xx, yy, cut_sub, maxiter=maxiters)
        if fitter.fit_info["ierr"] not in (1, 2, 3, 4):
            continue

        sx = float(res.x_stddev.value)
        sy = float(res.y_stddev.value)
        if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
            continue

        # Equivalent FWHM from covariance eigenvalues
        fwhm_eq = GAUSSIAN_FWHM * np.sqrt(sx * sy)
        if fwhm_min < fwhm_eq < fwhm_max:
            fwhm_values.append(float(fwhm_eq))

    if not fwhm_values:
        return None

    return float(np.median(fwhm_values))


if __name__ == "__main__":

    import time
    parser = argparse.ArgumentParser(description="Estimate median stellar FWHM from a FITS image.")
    parser.add_argument("path", type=str, help="Path to the FITS image file.")
    parser.add_argument("--size", type=int, default=7, help="Half-size of fit cutout (window is 2*size+1).")
    parser.add_argument("--maxiters", type=int, default=100, help="Max iterations for the LM fitter.")
    parser.add_argument("--fwhm-min", type=float, default=0.5, help="Minimum acceptable per-star FWHM.")
    parser.add_argument("--fwhm-max", type=float, default=20.0, help="Maximum acceptable per-star FWHM.")

    parser.add_argument("--sigma-threshold", type=float, default=5.0, help="DAOStarFinder threshold in sigma units.")
    parser.add_argument("--fwhm-init", type=float, default=3.0, help="Initial FWHM guess in pixels.")
    parser.add_argument(
        "--no-local-bkg",
        action="store_true",
        help="Use global median instead of local perimeter-median background.",
    )
    args = parser.parse_args()

    with fits.open(args.path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)

    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=args.fwhm_init, threshold=args.sigma_threshold * std)
    sources = finder(data - median)

    t0 = time.time()
    fwhm = estimate_fwhm(
        data,
        sources["xcentroid"],
        sources["ycentroid"],
        fwhm_init=args.fwhm_init,
        size=args.size,
        maxiters=args.maxiters,
        fwhm_min=args.fwhm_min,
        fwhm_max=args.fwhm_max,
        use_local_bkg=not args.no_local_bkg,
    )
    t1 = time.time()

    if fwhm is None:
        print("No suitable stars found.")
    else:
        print(f"Median FWHM: {fwhm:.3f} px")
    print(f"Execution time: {t1 - t0:.2f} s")
