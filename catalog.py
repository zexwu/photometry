"""Star catalog container and common catalog-level operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


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

    @property
    def nstars(self) -> int:
        """Number of stars currently stored in the catalog."""
        return int(len(self.x))

    def clear(self) -> None:
        """Reset all catalog columns to empty arrays."""
        self.x = empty_array()
        self.y = empty_array()
        self.mag = empty_array()
        self.mag_err = empty_array()

    def sort_inplace(self, keys: str | list[str] = "mag") -> None:
        """Sort the catalog in place by one or multiple column names."""
        if self.nstars == 0:
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

    def masked(self, use: NDArray[np.bool_]) -> "Catalog":
        """Return a filtered catalog containing only rows where ``use`` is True."""
        return Catalog(
            x=np.asarray(self.x[use], dtype=float),
            y=np.asarray(self.y[use], dtype=float),
            mag=np.asarray(self.mag[use], dtype=float),
            mag_err=np.asarray(self.mag_err[use], dtype=float),
        )

    @classmethod
    def from_arrays(
        cls,
        x: NDArray,
        y: NDArray,
        mag: NDArray,
        mag_err: NDArray,
    ) -> "Catalog":
        """Create a catalog from array-like inputs and normalize to float arrays."""
        return cls(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            mag=np.asarray(mag, dtype=float),
            mag_err=np.asarray(mag_err, dtype=float),
        )
