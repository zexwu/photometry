from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ImageStat:
    fwhm: float = np.nan
    gain: float = 1.0
    rdnoise: float = 1.0
    background: float = np.nan
    background2d: NDArray | None = None
