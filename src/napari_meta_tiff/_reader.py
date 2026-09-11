"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import logging
from tifffile import TiffFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from napari_meta_tiff._metadata import (get_extra_metadata,
                                        get_pixel_size_um, get_position_um)


logger = logging.getLogger(__name__)

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]

# tifffile's axis code for the samples of a pixel, the colour of an rgb
# image, as opposed to the axes over which the image extends
SAMPLES_AXIS = 'S'


def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        return None     # list of paths not supported

    # the get_reader function should make as many checks as possible
    # (without loading the full file) to determine if it can read
    # the path.
    try:
        # use a context manager so the file handle is not left open
        with TiffFile(path):
            pass
    # napari_get_reader should never raise an exception, because napari
    # raises its own specific errors depending on what plugins are
    # available for the given path
    except OSError:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path: PathLike) -> List[LayerData]:
    # Reused from napari_tiff_reader
    with TiffFile(path) as tif:
        try:
            layerdata = tifffile_reader(tif)
        except Exception:
            # log the reason, so a failure in the tifffile reader is not
            # silently hidden by the fallback
            logger.warning('tifffile reader failed for %s, '
                           'falling back to imagecodecs', path, exc_info=True)
            layerdata = imagecodecs_reader(path)
    return layerdata


def get_best_tiff_serie(tif: TiffFile) -> int:
    """Return the index of the series holding the most image detail.

    A file often carries the same scene more than once: a thumbnail or
    an overview beside the image, or an RGB rendering beside the raw
    greyscale data it was made from. Rank them so the richest is read.
    """
    return max(range(len(tif.series)),
               key=lambda index: series_detail(tif.series[index]))


def series_detail(series: Any) -> Tuple[int, bool, int]:
    """Return how much detail a series holds, as a sort key.

    Pixels come first, so a thumbnail never wins over the image itself.
    Between series of the same size, a greyscale one is preferred over
    an RGB one: vendors write the RGB series as a rendering for display,
    while the greyscale series is the measurement it came from, often at
    a higher precision, which the last term then ranks.
    """
    # series.axes names each axis with a single letter code, which
    # series.sizes spells out ('sample'), so pair the codes with the shape
    sizes = dict(zip(series.axes, series.shape))
    # S is the samples axis, which holds colour rather than image extent
    pixels = 1
    for axis, size in sizes.items():
        if axis != SAMPLES_AXIS:
            pixels *= size
    samples = sizes.get(SAMPLES_AXIS, 1)
    return (pixels, samples == 1, series.dtype.itemsize)


def tifffile_reader(tif: TiffFile) -> List[LayerData]:
    # Reused from napari_tiff_reader - but always open as lazy zarr
    """Return napari LayerData from image series in TIFF file."""
    import zarr
    series_index = get_best_tiff_serie(tif)
    store = tif.aszarr(multiscales=True, series=series_index)
    group = zarr.open_group(store=store, mode='r')
    # group iteration order is arbitrary; the multiscales attrs give the
    # authoritative level order (highest resolution first)
    try:
        datasets = group.attrs['ome']['multiscales'][0]['datasets']
    except KeyError:
        datasets = group.attrs['multiscales'][0]['datasets']

    levels = [group[dataset['path']] for dataset in datasets]
    # a single level is passed as a plain array: napari would read a
    # one-element list as an extra dimension
    multiscale = len(levels) > 1
    data = levels if multiscale else levels[0]

    # napari does not use this extra `metadata` layer attribute
    # but storing the information on the layer next to the data
    # will allow users to access it and use it themselves if they wish
    metadata = get_extra_metadata(tif)
    metadata_kwargs = {
        "metadata": metadata,
        # state this explicitly rather than letting napari infer it
        "multiscale": multiscale,
    }
    metadata_kwargs.update(spatial_kwargs(tif, tif.series[series_index],
                                          metadata))

    return [(data, metadata_kwargs, "image")]


def spatial_kwargs(tif: TiffFile, series: Any, metadata: Dict) -> Dict:
    """Return what the metadata says about where the image sits.

    The pixel size and the stage position are measurements of the scene
    rather than of the array, so they are handed to napari as the scale
    and the translate of the layer, which puts the axes in micrometres
    and two images of one sample where they belong relative to each
    other. A multiscale layer is scaled by its highest level, from which
    napari works out the rest.
    """
    axes = [axis.lower() for axis in series.axes]
    shape = dict(zip(axes, series.shape))
    pixel_size = get_pixel_size_um(tif, metadata, shape)
    position = get_position_um(metadata)

    kwargs = {"axis_labels": tuple(series.axes)}
    if pixel_size:
        kwargs["scale"] = tuple(pixel_size.get(axis, 1.0) for axis in axes)
        # the samples of a pixel, and any axis nothing was said about,
        # stay in pixels rather than being called micrometres
        kwargs["units"] = tuple('um' if axis in pixel_size else 'pixel'
                                for axis in axes)
    if position:
        kwargs["translate"] = tuple(position.get(axis, 0.0) for axis in axes)
    return kwargs


def imagecodecs_reader(path: PathLike) -> List[LayerData]:
    # Reused from napari_tiff_reader
    """Return napari LayerData from first page in TIFF file."""
    from imagecodecs import imread

    return [(imread(path), {}, "image")]
