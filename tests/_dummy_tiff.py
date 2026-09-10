"""Helpers creating dummy TIFF files for the tests.

These are also used by the manual entry points of the test modules, so that
the files explored by hand in napari are the same ones the tests assert on.
"""

import numpy as np
from tifffile import TiffWriter, imwrite


TEST_METADATA = {'test_metadata': 'test metadata'}
# Make and Model tiff tags, which the reader promotes to layer metadata
TEST_MAKE = 'Test Make'
TEST_MODEL = 'Test Model'
MAKE_MODEL_EXTRATAGS = [
    (271, 's', 0, TEST_MAKE, True),
    (272, 's', 0, TEST_MODEL, True),
]


def dummy_image(size: int) -> np.ndarray:
    """Return a deterministic gradient image, recognisable when viewed."""
    x = np.linspace(0, 1, size, dtype=np.float32)
    pattern = np.outer(np.sin(8 * x) ** 2, x) + np.outer(x, np.cos(8 * x) ** 2)
    return (pattern * 127).astype(np.uint8)


def write_dummy_tiff(path: str, size: int = 16, extratags=()) -> np.ndarray:
    """Write a single level dummy tiff and return the image data."""
    data = dummy_image(size)
    imwrite(path, data, metadata=TEST_METADATA, extratags=extratags)
    return data


def write_pyramid_tiff(path: str, nlevels: int = 4, size: int = 512,
                       tile=(128, 128), extratags=()) -> list:
    """Write a pyramidal dummy tiff and return the data of each level.

    The full resolution image is written first, followed by nlevels - 1
    downsampled subifds, so the levels are ordered highest resolution first.
    """
    # decimate a single pattern, so the levels are a genuine pyramid
    # of the same image rather than unrelated arrays
    base = dummy_image(size)
    levels = [base[::2**level, ::2**level] for level in range(nlevels)]
    with TiffWriter(path) as tif:
        tif.write(levels[0], tile=tile, subifds=nlevels - 1,
                  metadata=TEST_METADATA, extratags=extratags)
        for level_data in levels[1:]:
            tif.write(level_data, tile=tile, subfiletype=1)
    return levels
