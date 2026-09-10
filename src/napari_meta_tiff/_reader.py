"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

from enum import Enum
import logging
from tifffile import TiffFile, xml2dict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]

# TIFF reserves tag codes at or above this for a vendor's private use:
# https://www.awaresystems.be/imaging/tiff/tifftags/private.html
PRIVATE_TAG_CODE = 32768

# baseline tags naming the instrument that produced the image, which
# vendors that do not use a private tag at all still fill in
IDENTITY_TAG_NAMES = ('Make', 'Model', 'Software', 'HostComputer')


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


def tifffile_reader(tif: TiffFile) -> List[LayerData]:
    # Reused from napari_tiff_reader - but always open as lazy zarr
    """Return napari LayerData from image series in TIFF file."""
    import zarr
    store = tif.aszarr(multiscales=True)
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
    metadata_kwargs = {
        "metadata": get_extra_metadata(tif),
        # state this explicitly rather than letting napari infer it
        "multiscale": multiscale,
    }

    return [(data, metadata_kwargs, "image")]


def imagecodecs_reader(path: PathLike) -> List[LayerData]:
    # Reused from napari_tiff_reader
    """Return napari LayerData from first page in TIFF file."""
    from imagecodecs import imread

    return [(imread(path), {}, "image")]



def get_extra_metadata(tif: TiffFile) -> Dict[str, Any]:
    """Return the vendor metadata in a TIFF file.

    Rather than reading the tags of particular vendors, every private tag
    is collected and normalised the same way, so that vendors which are
    not known here are picked up as well. TIFF reserves tag codes at or
    above 32768 for a vendor's own use, which is where instrument
    metadata ends up, whereas the baseline tags below that hold the
    bookkeeping needed to decode the pixels.
    """
    if tif.is_ome and tif.ome_metadata:
        return unwrap_metadata(xml2dict(tif.ome_metadata))

    extra_metadata = {}
    for page in tif.pages:
        for tag in page.tags.values():
            if tag.code >= PRIVATE_TAG_CODE:
                value = unwrap_metadata(tag.value)
                if value not in (None, '', {}):
                    # key by tag name, so several vendor tags in one file
                    # cannot overwrite each other
                    extra_metadata.setdefault(tag.name, value)
            elif tag.name in IDENTITY_TAG_NAMES:
                extra_metadata.setdefault(tag.name, tag.value)
    return extra_metadata


def unwrap_metadata(value: Any) -> Any:
    """Reduce a metadata value to the fields it actually holds.

    Vendors store their metadata as an xml document, as a nested mapping,
    or as a mapping behind a single key naming their own schema, so
    reduce all of those to the fields themselves. References to xml
    schemas are dropped, as they describe the document rather than the
    image.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        if '<?xml' not in value.lower():
            return value
        value = xml2dict(value)
    if not isinstance(value, dict):
        return value

    value = {key: item for key, item in value.items()
             if not is_schema_reference(item)}
    # a lone key naming the vendor's schema, such as OME, FeiImage or
    # Fibics, only nests the fields one level deeper
    if len(value) == 1:
        (item,) = value.values()
        if isinstance(item, dict):
            return item
    return value


def is_schema_reference(value: Any) -> bool:
    """Return whether value points at an xml schema rather than data."""
    return isinstance(value, str) and '.xsd' in value.lower()
