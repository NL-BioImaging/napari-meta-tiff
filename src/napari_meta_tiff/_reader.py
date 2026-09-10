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


def get_extra_metadata(tif):
    extra_metadata = {}
    if tif.is_ome and tif.ome_metadata:
        metadata = metadata_to_dict(tif.ome_metadata)
    else:
        metadata = {key: value for page in tif.pages for key, value in tags_to_dict(page.tags).items()
                     if key not in ('StripOffsets', 'StripByteCounts', 'TileOffsets', 'TileByteCounts', 'JPEGTables')}

    if 'FEI_TITAN' in metadata:
        extra_metadata = metadata.pop('FEI_TITAN')
        if isinstance(extra_metadata, str) and '<?xml' in extra_metadata.lower():
            extra_metadata = metadata_to_dict(extra_metadata)
        if 'FeiImage' in extra_metadata:
            extra_metadata = extra_metadata['FeiImage']
        extra_metadata = {key: value for key, value in extra_metadata.items()
                                if not (isinstance(value, str) and '.xsd' in value.lower())}
        metadata['FeiImage'] = extra_metadata
        metadata['manufacturer'] = 'FEI'
        instrument = extra_metadata.get('instrument', extra_metadata)
        metadata['model'] = instrument.get('edition', instrument.get('type', 'Titan'))
        metadata['serial'] = instrument.get('uniqueID')
    elif 'FEI_HELIOS' in metadata:
        extra_metadata = metadata['FEI_HELIOS']
        metadata['manufacturer'] = 'FEI'
        metadata['model'] = extra_metadata.get('System', {}).get('ProductName', 'Helios')
    elif 'FibicsXML' in metadata:
        extra_metadata = metadata.pop('FibicsXML')
        if isinstance(extra_metadata, str) and '<?xml' in extra_metadata.lower():
            extra_metadata = metadata_to_dict(extra_metadata)
        if 'Fibics' in extra_metadata:
            extra_metadata = extra_metadata['Fibics']
        extra_metadata = {key: value for key, value in extra_metadata.items()
                                if not (isinstance(value, str) and '.xsd' in value.lower())}
        metadata['Fibics'] = extra_metadata
        application_version = extra_metadata.get('Application', {}).get('Version', '').split()
        if len(application_version) >= 2:
            metadata['manufacturer'] = application_version[0]
            metadata['model'] = application_version[1]
    elif 'OlympusSIS' in metadata:
        extra_metadata = metadata['OlympusSIS']
    else:
        if 'Make' in metadata:
            extra_metadata['Make'] = metadata['Make']
        if 'Model' in metadata:
            extra_metadata['Model'] = metadata['Model']

    return extra_metadata


def metadata_to_dict(xml_metadata):
    metadata = xml2dict(xml_metadata)
    if 'OME' in metadata:
        metadata = metadata['OME']
    return metadata


def tags_to_dict(tags):
    """
    Converts TIFF tags to a dictionary.

    Args:
        tags: TIFF tags object.

    Returns:
        dict: Tag name-value mapping.
    """
    tag_dict = {}
    for tag in tags.values():
        value = tag.value
        if isinstance(value, Enum):
            value = value.name
        tag_dict[tag.name] = value
    return tag_dict
