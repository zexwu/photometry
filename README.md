# photometry

Small photometry/matching codebase + image-processing pipeline.

## Tools

- `pymatch.py`: point-set matching and geometric transformation estimation.
- `fwhm.py`: stellar FWHM estimation from image cutouts.

These two files can be used independently of the pipeline.

## Pipeline modules

- `image.py`: high-level `Image` workflow orchestration.
- `io.py`: FITS loading/saving and `ImageStat` container.
- `detection.py`: source detection and flux-to-magnitude helpers.
- `photometry.py`: aperture/ePSF/DoPHOT-related photometry helpers.
- `catalog.py`: catalog container, alignment logic, and diagnostics plotting.

## Showcase

Minimal usage with aperture photometry and catalog alignment:

```python
from photometry.image import Image
import numpy as np

ref = (
    Image(path="./ref/ref_frame.fits")
    .detect_sources()
    .estimate_fwhm()
    .sort_by("mag")
    .run_aperture_photometry()
)

ref.show(percentile=(1, 99))
import matplotlib.pyplot as plt
plt.show()

img = (
    Image(path="./sci/sci_frame.fits")
    .detect_sources()
    .estimate_fwhm()
    .sort_by("mag")
    .run_aperture_photometry()
)

# geometric transformation estimation and image transformation
img = img.transform_to(ref, inspect=True)

# get target photometry
idx = np.hypot(img.catalog.x - 201, img.catalog.y - 200).argmin()
print(img.catalog.mag[idx], img.catalog.mag_err[idx])

```

## Scope

This repository focuses on local processing of astronomical images and catalog-based photometry/matching workflows. It is not packaged as a full framework.
