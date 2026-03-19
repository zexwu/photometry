from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit

GAUSSIAN_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

def _gaussian_2d(xy, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
    x, y = xy

    # Translate
    dx = x - x_mean
    dy = y - y_mean

    # Rotate coordinates
    x_prime = dx * np.cos(theta) + dy * np.sin(theta)
    y_prime = -dx * np.sin(theta) + dy * np.cos(theta)

    # Compute Gaussian
    return amplitude * np.exp(
        -0.5 * ((x_prime / x_stddev) ** 2 + (y_prime / y_stddev) ** 2)
    )

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
    half_size: int = 7,
    maxiters: int = 100,
    fwhm_min: float = 0.5,
    fwhm_max: float = 20.0,
    use_local_bkg: bool = True,
) -> float:
    """Estimate the median stellar FWHM from an image using a source catalog.

    Parameters
    - data: 2D array-like
      Image data (counts), treated as float.
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
    if half_size <= 0:
        raise ValueError("size must be a positive integer")

    # Global background (used if local background disabled)
    _, median, _ = sigma_clipped_stats(data, sigma=3.0)

    fwhm_values: list[float] = []

    # Precompute local grid for speed
    n = 2 * half_size + 1
    yy, xx = np.mgrid[:n, :n]
    sx0 = sy0 = fwhm_init / GAUSSIAN_FWHM

    ny, nx = data.shape

    yyxx = (yy.ravel(), xx.ravel())


    for xc, yc in zip(x, y):

        xi = int(round(xc))
        yi = int(round(yc))

        # Ensure cutout is fully within the image bounds
        if not (half_size <= xi < nx - half_size and half_size <= yi < ny - half_size):
            continue

        cut = data[yi - half_size : yi + half_size + 1, xi - half_size : xi + half_size + 1]
        if cut.shape != (n, n) or not np.isfinite(cut).all():
            continue

        # Local or global background subtraction
        bkg = _local_background(cut) if use_local_bkg else float(median)
        cut_sub = cut - bkg

        p0 = [cut_sub.max(), xc - (xi - half_size), yc - (yi - half_size), sx0, sy0, 0.0]
        lower_bounds = [0, 0, 0, 1.0, 1.0, -np.pi/2]
        upper_bounds = [np.inf, 2*half_size, 2*half_size, 15.0, 15.0, np.pi/2]
        try:
            popt, _ = curve_fit(
                _gaussian_2d,
                yyxx,
                cut_sub.ravel(),
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=maxiters
            )
        except RuntimeError:
            continue # Fit failed to converge
        except ValueError:
            continue # Invalid data inputs

        _, _, _, sx, sy, _ = popt

        if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
            continue

        # Equivalent FWHM from covariance eigenvalues
        fwhm_eq = GAUSSIAN_FWHM * np.sqrt(sx * sy)
        if fwhm_min < fwhm_eq < fwhm_max:
            fwhm_values.append(float(fwhm_eq))

    if not fwhm_values:
        return fwhm_init

    return float(np.median(fwhm_values))


if __name__ == "__main__":

    import time
    import argparse
    parser = argparse.ArgumentParser(description="Estimate median stellar FWHM from a FITS image.")
    parser.add_argument("path", type=str, help="Path to the FITS image file.")
    parser.add_argument("--half-size", type=int, default=7, help="Half-size of fit cutout (window is 2*size+1).")
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

    from astropy.io import fits
    with fits.open(args.path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)

    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    from photutils.detection import DAOStarFinder
    finder = DAOStarFinder(fwhm=args.fwhm_init, threshold=args.sigma_threshold * std)
    sources = finder(data - median)

    t0 = time.time()
    fwhm = estimate_fwhm(
        data,
        sources["xcentroid"],
        sources["ycentroid"],
        fwhm_init=args.fwhm_init,
        half_size=args.half_size,
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
