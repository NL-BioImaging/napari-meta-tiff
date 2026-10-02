"""Tests that real slides open as the expected napari layers.

The slides live on this machine only, so the tests are skipped wherever
the folder is missing, as it is in CI. For each slide, what napari made
of it and the extra metadata on the layer are written to a json file in
metadata_output/ at the root of the repository, so they can be read
after the run.

Run this module to execute the tests by hand:

    python -m tests.test_slides
"""

import datetime
import glob
import json
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from tifffile import TiffFile

from napari_meta_tiff._reader import get_best_tiff_serie, napari_get_reader

napari = pytest.importorskip('napari')


SLIDES_PATTERN = 'C:/Project/slides/tiff/*.tif*'

SLIDE_PATHS = sorted(glob.glob(SLIDES_PATTERN))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'metadata_output'


def to_json(value):
    """Return value as something json can hold.

    tifffile hands back what the file holds rather than plain python:
    numpy scalars and arrays, enum members naming a tag's value, bytes a
    vendor never said the encoding of, and datetimes. Each of those has an
    obvious json counterpart, so convert them rather than drop them. The
    conversion keeps every value the metadata holds: nothing is summarised
    or left out.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, bytes):
        # vendors do write text into a byte field, so keep it when it
        # reads as text, and fall back to hex, which holds the bytes
        # themselves rather than a mangled reading of them
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # an object of a vendor's own class, which json cannot hold, but
    # whose text is what a user would read off the layer anyway
    return str(value)


def napari_properties(layer) -> dict:
    """Return what napari made of the layer, as json."""
    return to_json({
        'name': layer.name,
        'multiscale': layer.multiscale,
        'rgb': layer.rgb,
        'ndim': layer.ndim,
        'dtype': str(layer.dtype),
        'level_shapes': layer.level_shapes,
        'axis_labels': layer.axis_labels,
        'units': [str(unit) for unit in layer.units],
        'scale': layer.scale,
        'translate': layer.translate,
        'extent_world': layer.extent.world,
        'contrast_limits': layer.contrast_limits,
    })


@pytest.mark.skipif(not SLIDE_PATHS, reason=f'no slides at {SLIDES_PATTERN}')
@pytest.mark.parametrize('path', SLIDE_PATHS, ids=lambda path: Path(path).name)
def test_slide_layer(path):
    """A slide opens as one image layer holding its data and metadata."""
    reader = napari_get_reader(path)
    assert reader is not None
    layer_data_list = reader(path)
    assert len(layer_data_list) == 1
    data, add_kwargs, layer_type = layer_data_list[0]
    assert layer_type == 'image'

    viewer = napari.components.ViewerModel()
    layer = viewer.add_image(data, **add_kwargs)

    # the layer holds the series the reader picked, at full resolution
    with TiffFile(path) as tif:
        series = tif.series[get_best_tiff_serie(tif)]
        shape, dtype = series.shape, series.dtype
    level0 = layer.data[0] if layer.multiscale else layer.data
    assert level0.shape == shape
    assert layer.dtype == dtype
    assert layer.multiscale == add_kwargs['multiscale']
    if layer.multiscale:
        # each level is no larger than the one before it
        sizes = [np.prod(level_shape) for level_shape in layer.level_shapes]
        assert sizes == sorted(sizes, reverse=True)

    # a corner of the lowest resolution reads, without loading it all
    lowest = layer.data[-1] if layer.multiscale else layer.data
    corner = np.asarray(lowest[tuple(slice(0, 16) for _ in lowest.shape)])
    assert corner.dtype == dtype and corner.size > 0

    # the extra metadata is on the layer, untouched
    assert layer.metadata == add_kwargs['metadata']
    assert len(layer.metadata) > 0
    # and the spatial metadata placed the layer where it said
    if 'scale' in add_kwargs:
        np.testing.assert_allclose(layer.scale, add_kwargs['scale'])
    if 'translate' in add_kwargs:
        np.testing.assert_allclose(layer.translate, add_kwargs['translate'])

    OUTPUT_DIR.mkdir(exist_ok=True)
    destination = OUTPUT_DIR / f'{Path(path).stem}.json'
    output = {
        'napari': napari_properties(layer),
        'metadata': to_json(layer.metadata),
    }
    with open(destination, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    with open(destination, encoding='utf-8') as file:
        assert json.load(file) == output


if __name__ == '__main__':
    for slide_path in SLIDE_PATHS:
        test_slide_layer(slide_path)
        print(f'{Path(slide_path).name} -> {OUTPUT_DIR}')
