from __future__ import annotations

import subprocess
from pathlib import Path
from time import sleep
from typing import Literal

import numpy as np
from astropy.io import fits

from .catalog import StarCatalog
from .image_stat import ImageStat


def load_fits_image(path: Path) -> tuple[np.ndarray, fits.Header, ImageStat]:
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


def _dophot_par_text(
    *,
    version: Literal["C", "fortran"],
    default_par: Path,
    image_name: str,
    obj_name: str,
    log_name: str,
    fwhm: float,
    background: float,
    gain: float,
    rdnoise: float,
) -> str:
    if version == "C":
        return f"""\
PARAMS_DEFAULT  = '{default_par}'
PARAMS_OUT      = '/dev/null'
IMAGE_IN        = '{image_name}'
LOGFILE         = '{log_name}'
LOGVERBOSITY    = 1
OBJECTS_OUT     = '{obj_name}'
ERRORS_OUT      = ' '
SHADOWFILE_OUT  = ' '
OBJECTS_IN      = ' '
IMAGE_OUT       = ' '
PSFTYPE         = 'PGAUSS'
SKYTYPE         = 'PLANE'
OBJTYPE_IN      = 'COMPLETE'
OBJTYPE_OUT     = 'COMPLETE'
FWHM            = {fwhm:.2f}
SKY             = {background:.2f}
EPERDN          = {gain}
RDNOISE         = {rdnoise}
ITOP            = 40000
ICRIT           = 10
CENTINTMAX      = 30000
CTPERSAT        = 40000
THRESHMAX       = 40000
THRESHMIN       = 200
APBOX_X         = 16
APBOX_Y         = 16
NFITBOX_X       = 12
NFITBOX_Y       = 12
END"""

    return f"""\
AUTOTHRESH      = 'NO'
FINISHFILE      = ' '
IMAGE_IN        = {image_name}
IMAGE_OUT       = {image_name.replace('.fits', '_out.fits')}
OBJECTS_OUT     = {obj_name}
PARAMS_OUT      = ' '
PARAMS_DEFAULT  = {default_par}
PSFTYPE         = 'PGAUSS'
OBJTYPE_IN      = 'COMPLETE'
OBJTYPE_OUT     = 'COMPLETE'
THRESHMIN       = 100.0
THRESHMAX       = 40000.0
EPERDN          = {gain}
RDNOISE         = {rdnoise}
FWHM            = {fwhm:.2f}
SKY             = {background:.2f}
TOP             = 40000.0
END"""


def run_dophot_catalog(
    *,
    path: Path,
    stat: ImageStat,
    dophot_bin: Path,
    default_par: Path,
    tmp_dir: Path,
    version: Literal["C", "fortran"],
) -> tuple[StarCatalog | None, float, float]:
    stem = path.stem.split(".", 1)[0]
    par_path = tmp_dir / f"{stem}.par"
    image_name = f"{stem}.fits"
    obj_name = f"{stem}.obj"
    log_name = f"{stem}.log"

    image_path = tmp_dir / image_name
    obj_path = tmp_dir / obj_name

    par_path.write_text(
        _dophot_par_text(
            version=version,
            default_par=default_par,
            image_name=image_name,
            obj_name=obj_name,
            log_name=log_name,
            fwhm=stat.fwhm,
            background=stat.background,
            gain=stat.gain,
            rdnoise=stat.rdnoise,
        ),
        encoding="utf-8",
    )

    image_path.unlink(missing_ok=True)
    image_path.symlink_to(path.resolve())

    subprocess.run(
        [str(dophot_bin), par_path.name],
        cwd=tmp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sleep(0.5)

    if not obj_path.exists() or obj_path.stat().st_size <= 0:
        return None, stat.background, stat.fwhm

    data = np.loadtxt(obj_path)
    if data.ndim != 2:
        return None, stat.background, stat.fwhm

    data = data[(np.abs(data[:, 4]) < 99) & (np.abs(data[:, 5]) < 1)]
    data = data[data[:, 1] == 1]
    if len(data) == 0:
        return None, stat.background, stat.fwhm

    data = data[np.argsort(data[:, 4])]
    catalog = StarCatalog.from_arrays(
        x=data[:, 2] - 0.5,
        y=data[:, 3] - 0.5,
        mag=data[:, 4] + 25,
        mag_err=data[:, 5],
    )
    out_background = float(np.median(data[:, 6]))
    out_fwhm = float(np.median((data[:, 7] * data[:, 8]) ** 0.5))
    return catalog, out_background, out_fwhm
