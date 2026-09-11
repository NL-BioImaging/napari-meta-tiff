"""Tests that the reader's layer data produces the expected napari layers.

The tests use napari's ViewerModel, so they need no Qt and no display.

Run this module to open the same dummy files in a real napari viewer
and explore the layers by hand:

    python -m tests.test_napari
"""

import numpy as np
import pytest

from napari_meta_tiff._reader import napari_get_reader

from tests._dummy_tiff import (
    MAKE_MODEL_EXTRATAGS,
    TEST_MAKE,
    TEST_MODEL,
    write_dummy_tiff,
    write_pyramid_tiff,
    write_vendor_tiff,
)

napari = pytest.importorskip('napari')


def add_layers(viewer, path: str) -> list:
    """Add the layers the reader returns for path to viewer."""
    reader = napari_get_reader(path)
    assert reader is not None
    layers = []
    for data, add_kwargs, layer_type in reader(path):
        assert layer_type == 'image'
        layers.append(viewer.add_image(data, **add_kwargs))
    return layers


# tmp_path is a pytest fixture
def test_napari_layer(tmp_path):
    """Test the layer created for a single level tiff."""
    path = str(tmp_path / 'dummy.tif')
    data = write_dummy_tiff(path, extratags=MAKE_MODEL_EXTRATAGS)

    viewer = napari.components.ViewerModel()
    layers = add_layers(viewer, path)

    assert len(layers) == 1 and len(viewer.layers) == 1
    layer = layers[0]
    assert isinstance(layer, napari.layers.Image)
    assert not layer.multiscale
    assert layer.ndim == 2
    assert layer.data.shape == data.shape
    np.testing.assert_array_equal(data, np.asarray(layer.data))

    # the reader promotes the tiff Make and Model tags to layer metadata
    assert layer.metadata['Make'] == TEST_MAKE
    assert layer.metadata['Model'] == TEST_MODEL


# tmp_path is a pytest fixture
def test_napari_layer_pyramid(tmp_path):
    """Test the multiscale layer created for a pyramidal tiff."""
    nlevels = 4
    path = str(tmp_path / 'dummy_pyramid.tif')
    levels = write_pyramid_tiff(path, nlevels=nlevels,
                                extratags=MAKE_MODEL_EXTRATAGS)

    viewer = napari.components.ViewerModel()
    layers = add_layers(viewer, path)

    assert len(layers) == 1 and len(viewer.layers) == 1
    layer = layers[0]
    assert layer.multiscale
    assert layer.ndim == 2

    # napari must see every level, ordered highest resolution first
    assert len(layer.data) == nlevels
    assert [tuple(shape) for shape in layer.level_shapes] == \
           [level_data.shape for level_data in levels]
    for level, level_data in enumerate(levels):
        np.testing.assert_array_equal(level_data,
                                      np.asarray(layer.data[level]))

    # the layer is placed in the world at full resolution
    assert layer.extent.world[1].tolist() == \
           [size - 1 for size in levels[0].shape]

    assert layer.metadata['Make'] == TEST_MAKE
    assert layer.metadata['Model'] == TEST_MODEL


def test_napari_layer_in_micrometres(tmp_path):
    """A layer is placed in micrometres where the metadata says so."""
    size = 64
    path = str(tmp_path / 'dummy_space.tif')
    write_vendor_tiff(path, '<Vendor><pixelWidth><value>0.25</value>'
                            '<unit>um</unit></pixelWidth><pixelHeight>'
                            '<value>0.25</value><unit>um</unit></pixelHeight>'
                            '<Stage><X><value>10</value><units>um</units></X>'
                            '<Y><value>-4</value><units>um</units></Y>'
                            '</Stage></Vendor>', size=size)

    viewer = napari.components.ViewerModel()
    layer = add_layers(viewer, path)[0]

    assert tuple(layer.scale) == (0.25, 0.25)
    assert tuple(layer.translate) == (-4, 10)     # ordered y, x
    assert [str(unit) for unit in layer.units] == ['micrometer'] * 2
    assert tuple(layer.axis_labels) == ('Y', 'X')
    # a quarter of a micrometre per pixel, from where the stage was:
    # the far corner is the last pixel, not one past it
    assert layer.extent.world[1].tolist() == [-4 + (size - 1) * 0.25,
                                              10 + (size - 1) * 0.25]


def test_napari_layer_without_spatial_metadata(tmp_path):
    """A file saying nothing about space is left in pixels."""
    path = str(tmp_path / 'dummy.tif')
    write_dummy_tiff(path)

    viewer = napari.components.ViewerModel()
    layer = add_layers(viewer, path)[0]

    assert tuple(layer.scale) == (1, 1)
    assert tuple(layer.translate) == (0, 0)
    assert [str(unit) for unit in layer.units] == ['pixel'] * 2


if __name__ == '__main__':
    import tempfile
    from pathlib import Path

    # the multiscale reader keeps the tiff open for lazy tile access,
    # so the temporary directory cannot always be removed on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)

        single_path = str(tmp_path / 'dummy.tif')
        write_dummy_tiff(single_path, size=256,
                         extratags=MAKE_MODEL_EXTRATAGS)
        pyramid_path = str(tmp_path / 'dummy_pyramid.tif')
        write_pyramid_tiff(pyramid_path, size=4096,
                           extratags=MAKE_MODEL_EXTRATAGS)

        viewer = napari.Viewer()
        # open through the plugin, so the napari.yaml registration
        # is exercised as well
        for path in (single_path, pyramid_path):
            viewer.open(path, plugin='napari-meta-tiff')

        for layer in viewer.layers:
            print(f'{layer.name}: multiscale={layer.multiscale} '
                  f'levels={len(layer.data) if layer.multiscale else 1} '
                  f'dtype={layer.dtype}')
            print(f'  metadata: {layer.metadata}')

        napari.run()
