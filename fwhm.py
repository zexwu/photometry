import numpy as np
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.io import fits
from photutils.detection import DAOStarFinder


def detect_stars(
    image: np.ndarray,
    *,
    init_fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    brightest: int | None = 500,
    peakmax: float | None = None,
    sharplo: float = 0.2,
    sharphi: float = 1.0,
    roundlo: float = -0.7,
    roundhi: float = 0.7,
    sigma: float = 3.0,
    maxiters: int = 5,
    min_separation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect isolated stars with DAOStarFinder.

    Returns
    -------
    x, y, flux
        Arrays of isolated source positions and DAO fluxes.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    _, median, std = sigma_clipped_stats(image, sigma=sigma, maxiters=maxiters)
    data = image - median

    finder = DAOStarFinder(
        threshold=threshold_sigma * std,
        fwhm=init_fwhm,
        brightest=brightest,
        peakmax=peakmax,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
        exclude_border=True,
    )
    tbl = finder(data)
    if tbl is None or len(tbl) == 0:
        return np.array([]), np.array([]), np.array([])

    x = np.asarray(tbl["xcentroid"], dtype=float)
    y = np.asarray(tbl["ycentroid"], dtype=float)
    flux = np.asarray(tbl["flux"], dtype=float) if "flux" in tbl.colnames else np.ones_like(x)

    if min_separation is None:
        min_separation = 3.0 * init_fwhm

    if len(x) > 1:
        xy = np.column_stack((x, y))
        d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        keep = np.sqrt(d2.min(axis=1)) > min_separation
        x = x[keep]
        y = y[keep]
        flux = flux[keep]

    return x, y, flux


def fit_star_fwhm(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fit_box: int = 15,
    init_fwhm: float = 3.0,
    sigma: float = 3.0,
    maxiters: int = 5,
    fwhm_min: float = 0.5,
    fwhm_max: float = 20.0,
    max_fit_error: float | None = None,
) ->  tuple[float, float, int]:
    """
    Fit Gaussian PSFs at given catalog positions and estimate global FWHM.

    Parameters
    ----------
    image : 2D ndarray
    x, y : array-like
        Catalog source positions in pixel coordinates.
    fit_box : int
        Odd cutout size for each fit.

    Returns
    -------
    fwhm : float
        Sigma-clipped median FWHM.
    scatter : float
        Std of clipped per-star FWHM values.
    n_used : int
        Number of stars used.
    star_fwhm : ndarray, optional
        Returned only if return_per_star=True.
    """

    if fit_box % 2 == 0:
        raise ValueError("fit_box must be odd")

    half = fit_box // 2
    yy, xx = np.mgrid[:fit_box, :fit_box]
    fitter = fitting.LevMarLSQFitter()
    GAUSSIAN_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sigma0 = init_fwhm / GAUSSIAN_FWHM

    fwhm_values = []

    ny, nx = image.shape

    for xc, yc in zip(x, y):
        xi = int(round(xc))
        yi = int(round(yc))

        x0 = xi - half
        x1 = xi + half + 1
        y0 = yi - half
        y1 = yi + half + 1

        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            continue

        cutout = image[y0:y1, x0:x1]
        if cutout.shape != (fit_box, fit_box) or not np.isfinite(cutout).all():
            continue

        border = np.empty(4 * fit_box - 4, dtype=cutout.dtype)
        border[:fit_box] = cutout[0, :]
        border[fit_box:2 * fit_box] = cutout[-1, :]
        border[2 * fit_box:3 * fit_box - 2] = cutout[1:-1, 0]
        border[3 * fit_box - 2:] = cutout[1:-1, -1]
        bkg = np.median(border)

        cutout_sub = cutout - bkg
        amp = cutout_sub.max()
        if not np.isfinite(amp) or amp <= 0:
            continue

        model = models.Gaussian2D(
            amplitude=amp,
            x_mean=xc - x0,
            y_mean=yc - y0,
            x_stddev=sigma0,
            y_stddev=sigma0,
            theta=0.0,
        )


        try:
            fit = fitter(model, xx, yy, cutout_sub, maxiter=100)
            sigma_eff = np.sqrt(fit.x_stddev.value * fit.y_stddev.value)
            fwhm = GAUSSIAN_FWHM * sigma_eff

            if not np.isfinite(fwhm) or not (fwhm_min < fwhm < fwhm_max):
                continue

            if max_fit_error is not None:
                resid = cutout_sub - fit(xx, yy)
                rms = np.sqrt(np.mean(resid * resid))
                if rms > max_fit_error:
                    continue

            fwhm_values.append(fwhm)

        except Exception:
            continue

    fwhm_values = np.asarray(fwhm_values, dtype=float)
    if fwhm_values.size == 0:
        return np.nan, np.nan, 0

    clipped = sigma_clip(fwhm_values, sigma=sigma, maxiters=maxiters, masked=True)
    used = clipped.compressed()

    if used.size == 0:
        return np.nan, np.nan, 0

    fwhm = float(np.median(used))
    scatter = float(np.std(used, ddof=1)) if used.size > 1 else 0.0
    n_used = int(used.size)

    return fwhm, scatter, n_used




image = fits.getdata(
    "./sci/zg/ztf_20210711385822_000643_zg_c10_o_q4_sciimg_ra324.5450_dec26.4666_asec1199.fits"
)

x, y, flux = detect_stars(
    image,
    init_fwhm=3.0,
    threshold_sigma=5.0,
    brightest=500,
)

fwhm, scatter, n_used = fit_star_fwhm(
    image,
    x,
    y,
    fit_box=15,
    init_fwhm=3.0,
)

print(fwhm, scatter, n_used)
