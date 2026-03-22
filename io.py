"""Image loading helpers and image-level scalar stats."""

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


def load_fits_image(path: Path) -> tuple[NDArray, fits.Header, ImageStat]:
    """Load image pixels/header and derive initial ``ImageStat`` values."""
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0]
        data = np.asarray(primary.data, dtype=np.float32)
        data[~np.isfinite(data)] = -1

        if data.ndim != 2:
            raise ValueError(f"Expected a 2D FITS image, got shape={data.shape}")

        header = primary.header.copy()
        stat = ImageStat(
            gain=float(header.get("GAIN", 1.0)),
            rdnoise=float(header.get("RDNOISE", 1.0)),
            fwhm=float(header.get("SEEING", np.nan)),
        )

    return data, header, stat
