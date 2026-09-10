import numpy as np
from tifffile import TiffFile, TiffWriter, imwrite

from napari_meta_tiff._reader import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    # we make the array an integer type to be compatible with the reader
    data = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
    metadata = {'test_metadata': 'test metadata'}
    path = str(tmp_path / 'dummy.tif')
    imwrite(path, data, metadata=metadata)

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
    size = 512
    # write a pyramidal tiff: full resolution image plus 3 downsampled subifds
    levels = [
        np.random.randint(0, 256, size=(size // 2**level, size // 2**level),
                          dtype=np.uint8)
        for level in range(nlevels)
    ]
    metadata = {'test_metadata': 'test metadata'}
    path = str(tmp_path / 'dummy_pyramid.tif')
    with TiffWriter(path) as tif:
        tif.write(levels[0], tile=(128, 128), subifds=nlevels - 1,
                  metadata=metadata)
        for level_data in levels[1:]:
            tif.write(level_data, tile=(128, 128), subfiletype=1)

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


if __name__ == '__main__':
    from pathlib import Path
    import tempfile

    # the multiscale reader keeps the tiff open for lazy tile access,
    # so the temporary directory cannot always be removed on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)

        test_reader_pyramid(tmp_path)
        test_reader(tmp_path)
