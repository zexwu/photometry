"""Image loading helpers, FITS I/O, and image-level scalar stats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray


@dataclass
class ImageStat:
    """Mutable per-image statistics and detector parameters."""

    fwhm: float = np.nan
    gain: float = 1.0
    rdnoise: float = 1.0
    background: float = np.nan
    background2d: NDArray | None = None


def load_fits_image(
    path: Path,
) -> tuple[NDArray, NDArray[np.bool_], fits.Header, ImageStat]:
    """Load image pixels, mask, header, and derived ``ImageStat`` values."""
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0]
        data = np.asarray(primary.data, dtype=np.float32)
        mask = ~np.isfinite(data)

        if data.ndim != 2:
            raise ValueError(f"Expected a 2D FITS image, got shape={data.shape}")

        header = primary.header.copy()
        if len(hdul) > 1:
            for hdu in hdul[1:]:
                if hdu.name.upper() == "MASK" and hdu.data is not None:
                    extra_mask = np.asarray(hdu.data, dtype=bool)
                    if extra_mask.shape == data.shape:
                        mask |= extra_mask
                    break

        stat = ImageStat(
            gain=float(header.get("GAIN", 1.0)),
            rdnoise=float(header.get("RDNOISE", 1.0)),
            fwhm=float(header.get("SEEING", np.nan)),
        )

    return data, mask.astype(bool), header, stat


def write_fits_image(
    filename: str | Path,
    data: NDArray,
    mask: NDArray[np.bool_],
    header: fits.Header | None,
    stat: ImageStat,
    *,
    overwrite: bool = True,
) -> None:
    """Write image pixels, detector metadata, and mask to a FITS file."""
    base_header = fits.Header() if header is None else header.copy()
    primary = fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=base_header)
    primary.header["GAIN"] = (stat.gain, "Electrons per ADU")
    primary.header["RDNOISE"] = (stat.rdnoise, "Read noise in electrons")
    mask_hdu = fits.ImageHDU(np.asarray(mask, dtype=np.uint8), name="MASK")
    fits.HDUList([primary, mask_hdu]).writeto(filename, overwrite=overwrite)
