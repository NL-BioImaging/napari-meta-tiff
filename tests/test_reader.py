"""Tests for the reader contribution and the layer data it returns.

Run this module to execute the tests by hand:

    python -m tests.test_reader
"""

import numpy as np
from tifffile import TiffFile

from napari_meta_tiff._reader import napari_get_reader

from tests._dummy_tiff import (TEST_EXIF, write_dummy_tiff, write_exif_tiff,
                               write_pyramid_tiff)


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""
    # write some fake data using your supported file format
    path = str(tmp_path / 'dummy.tif')
    data = write_dummy_tiff(path)

    reader = napari_get_reader(path)
    assert reader is not None
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    np.testing.assert_allclose(data, layer_data_tuple[0])


# tmp_path is a pytest fixture
def test_reader_pyramid(tmp_path):
    """Test reading a pyramidal TIFF file with 4 resolution levels."""
    nlevels = 4
    path = str(tmp_path / 'dummy_pyramid.tif')
    levels = write_pyramid_tiff(path, nlevels=nlevels)

    # make sure the file really has the expected number of levels
    with TiffFile(path) as tif:
        assert len(tif.series[0].levels) == nlevels

    reader = napari_get_reader(path)
    assert reader is not None
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # multiscale data is delivered as a list of arrays, one per level
    data = layer_data_tuple[0]
    assert isinstance(data, list) and len(data) == nlevels
    for level, level_data in enumerate(levels):
        assert data[level].shape == level_data.shape
        np.testing.assert_allclose(level_data, np.asarray(data[level]))


def test_reader_exif(tmp_path):
    """Exif fields are merged in, rather than nested behind ExifTag."""
    path = str(tmp_path / 'dummy_exif.tif')
    write_exif_tiff(path)

    # the file really does hold an Exif IFD behind the pointer tag
    with TiffFile(path) as tif:
        assert tif.pages[0].tags['ExifTag'].value

    _, add_kwargs, _ = napari_get_reader(path)(path)[0]
    metadata = add_kwargs['metadata']
    assert 'ExifTag' not in metadata
    for key, value in TEST_EXIF.items():
        assert metadata[key] == value


if __name__ == '__main__':
    from pathlib import Path
    import tempfile

    # the multiscale reader keeps the tiff open for lazy tile access,
    # so the temporary directory cannot always be removed on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)

        test_reader_pyramid(tmp_path)
        test_reader(tmp_path)
        test_reader_exif(tmp_path)
