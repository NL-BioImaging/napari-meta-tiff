"""Measure whether the reader really lazy loads large multiscale tiffs.

This is a diagnostic script, not a test: it prints measurements to be read
by a human rather than asserting thresholds, because the numbers depend on
the machine, the tiff layout and the tifffile version.

It counts the bytes actually pulled off disk by monkey patching every read
path on tifffile's FileHandle, so the figures are real IO rather than
timings, which are dominated by the page cache on a warm run.

Run it with:

    python -m testing.lazy_loading

Pass a path to measure a real file instead of a written one:

    python -m testing.lazy_loading C:/Project/slides/AMC_EM/B/1.tif
"""

import argparse
import os
import tempfile

import numpy as np
import tifffile
from tifffile import TiffFile, TiffWriter

from napari_meta_tiff._reader import (get_extra_metadata,
                                      napari_get_reader)


SIZE = 8192         # a level 0 of SIZE**2 bytes, 67 MB for uint8
TILE = 256
ROWSPERSTRIP = 64
NLEVELS = 5


class ReadCounter:
    """Count the bytes read from disk through tifffile's FileHandle.

    Patching FileHandle catches every read tifffile makes, including the
    ones the zarr store issues per chunk, which is what we want to know.
    """

    METHODS = ('read', 'readinto', 'read_array', 'read_segments')

    def __init__(self):
        self.bytes = 0
        self._original = {name: getattr(tifffile.FileHandle, name)
                          for name in self.METHODS}

    def __enter__(self):
        for name, func in self._original.items():
            setattr(tifffile.FileHandle, name, self._wrap(name, func))
        return self

    def __exit__(self, *exc_info):
        for name, func in self._original.items():
            setattr(tifffile.FileHandle, name, func)

    def _add(self, item):
        # read_segments yields (segment, index) pairs, or with flat=False
        # a list of those, so walk into whatever comes back
        if isinstance(item, (bytes, bytearray, memoryview)):
            self.bytes += len(item)
        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                self._add(sub_item)

    def _wrap(self, name, func):
        def wrapper(handle, *args, **kwargs):
            result = func(handle, *args, **kwargs)
            if name == 'read_segments':
                # a generator: count as it is consumed, not when created
                def counting():
                    for item in result:
                        self._add(item)
                        yield item
                return counting()
            if name == 'read':
                self._add(result)
            elif name == 'readinto':
                self.bytes += result or 0
            elif name == 'read_array':
                self.bytes += getattr(result, 'nbytes', 0)
            return result
        return wrapper

    def reset(self):
        self.bytes = 0

    @property
    def mb(self):
        return self.bytes / 1e6

    def report(self, label):
        """Print and reset the counter, returning the bytes counted."""
        print(f'  {label:<44} {self.mb:9.3f} MB')
        read = self.bytes
        self.reset()
        return read


def write_pyramid(path, tile=TILE, rowsperstrip=None):
    """Write a large pyramidal tiff, either tiled or striped."""
    layout = ({'tile': (tile, tile)} if tile
              else {'rowsperstrip': rowsperstrip})
    base = np.random.randint(0, 256, size=(SIZE, SIZE), dtype=np.uint8)
    with TiffWriter(path, bigtiff=True) as tif:
        tif.write(base, subifds=NLEVELS - 1, **layout)
        for level in range(1, NLEVELS):
            tif.write(base[::2**level, ::2**level], subfiletype=1, **layout)
    return os.path.getsize(path) / 1e6


def report_reader(path, file_mb, counter):
    """What the reader itself reads, opening and accessing by chunk."""
    print('\nreader')
    reader = napari_get_reader(path)
    counter.report('napari_get_reader (probe)')
    layer_data = reader(path)
    data, add_kwargs, _ = layer_data[0]
    counter.report(f'reader_function on a {file_mb:.0f} MB file')
    # a single level file is delivered as a plain array rather than a list
    levels = data if add_kwargs['multiscale'] else [data]
    print(f'    multiscale={add_kwargs["multiscale"]}, levels={len(levels)}, '
          f'level 0 {levels[0].shape}, chunks={levels[0].chunks}')

    level0_mb = levels[0].nbytes / 1e6
    np.asarray(levels[0][(0,) * levels[0].ndim])
    one_chunk = counter.report('one pixel of level 0')
    region_slice = tuple(slice(size // 4, size // 4 + 1024)
                         for size in levels[0].shape[-2:])
    np.asarray(levels[0][(Ellipsis,) + region_slice])
    region = counter.report('1024x1024 region of level 0')
    np.asarray(levels[-1][:])
    counter.report(f'entire lowest level {levels[-1].shape}')

    print(f'    level 0 is {level0_mb:.0f} MB, so one pixel costs '
          f'{level0_mb * 1e6 / max(one_chunk, 1):.0f}x less than the level')
    print(f'    the region read {region / (1024 * 1024):.2f}x its own pixels')

    with TiffFile(path) as tif:
        tif.asarray()
    counter.report('TiffFile.asarray() for comparison')


def report_napari_layer(path, file_mb, counter):
    """What a napari layer reads: on creation, per level, and per viewport."""
    import napari

    print('\nnapari layer')
    viewer = napari.components.ViewerModel()
    data, add_kwargs, _ = napari_get_reader(path)(path)[0]
    layer = viewer.add_image(data, **add_kwargs)
    counter.report(f'add_image on a {file_mb:.0f} MB file')
    print(f'    data_level={layer.data_level}, '
          f'corner_pixels={layer.corner_pixels.tolist()}')

    layer._update_thumbnail()
    counter.report('thumbnail generation')

    for level in reversed(range(len(layer.level_shapes))):
        layer.data_level = level
        layer.refresh()
        shape = tuple(int(size) for size in layer.level_shapes[level])
        counter.report(f'data_level={level} {shape}')

    # reads scale with the viewport, which is the point of the pyramid:
    # napari only asks for full resolution over a small region
    extent = [int(size) for size in layer.level_shapes[0]]
    for corners, label in (
        ([[0] * len(extent), [1024] * len(extent)],
         'level 0, 1024x1024 viewport'),
        ([[0] * len(extent), extent], 'level 0, whole extent (worst case)'),
    ):
        layer.data_level = 0
        layer.corner_pixels = np.array(corners)
        layer.refresh()
        np.asarray(layer._slice.image.view)   # as the renderer would
        counter.report(label)
    level0 = layer.data[0] if layer.multiscale else layer.data
    level0_mb = level0.nbytes / 1e6
    print(f'    level 0 in full is {level0_mb:.1f} MB, so the worst '
          'case genuinely needs every tile')

    # what a user of the plugin actually reaches for: the metadata as it
    # sits on the layer, which is where napari hands it back to them
    print('layer.metadata')
    print(layer.metadata)
    return layer.metadata


def report_granularity(tmpdir, counter):
    """Chunk granularity follows the tiff layout, not the reader."""
    print('\nchunk granularity')
    layouts = (
        (f'tiled {TILE}x{TILE}', {'tile': TILE}),
        (f'striped, {ROWSPERSTRIP} rows', {'tile': None,
                                           'rowsperstrip': ROWSPERSTRIP}),
    )
    for label, layout in layouts:
        path = os.path.join(tmpdir, f'{label.split()[0]}.tif')
        write_pyramid(path, **layout)
        data, _, _ = napari_get_reader(path)(path)[0]
        counter.reset()     # do not count opening the file
        np.asarray(data[0][0, 0])
        print(f'  {label:<22} chunks={str(data[0].chunks):<14} '
              f'one pixel {counter.mb:7.3f} MB')
        counter.reset()
    print('    a strip spans the full image width, so a striped image cannot '
          'be read\n    in smaller pieces than a full width band')


def report_metadata(path, counter):
    """What the reader found in the file's vendor metadata."""
    print('\nmetadata')
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)
    counter.report('reading the metadata')
    print_metadata(metadata)
    return metadata


def print_metadata(metadata):
    """Print one line per metadata field, truncating long values."""
    if not metadata:
        print('  no vendor metadata found')
        return
    for key, value in metadata.items():
        text = str(value)
        if len(text) > 120:
            text = text[:117] + '...'
        print(f'  {key:<22} {text}')


def report_space(path, counter):
    """What the metadata says about where the image sits."""
    print('\nspace')
    data, add_kwargs, _ = napari_get_reader(path)(path)[0]
    counter.report('reading the pixel size and position')
    if 'scale' not in add_kwargs:
        print('  no pixel size found, so the layer stays in pixels')
    for key in ('axis_labels', 'scale', 'units', 'translate'):
        if key in add_kwargs:
            print(f'  {key:<22} {add_kwargs[key]}')


def report_file(path, counter):
    """Run the reports that do not need a file of a known layout."""
    file_mb = os.path.getsize(path) / 1e6
    print(f'{path}\n  {file_mb:.1f} MB on disk')
    report_metadata(path, counter)
    report_space(path, counter)
    report_reader(path, file_mb, counter)
    report_napari_layer(path, file_mb, counter)


def main(paths=()):
    if paths:
        # a real file says nothing about granularity, because its layout
        # is whatever the vendor wrote, so only the written files below
        # compare a tiled against a striped layout
        with ReadCounter() as counter:
            for path in paths:
                report_file(path, counter)
        return

    # the reader keeps the tiff open for lazy tile access, so the temporary
    # directory cannot always be removed on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        path = os.path.join(tmpdir, 'pyramid.tif')
        print(f'{SIZE}x{SIZE} uint8, {NLEVELS} levels, {TILE}x{TILE} tiles, '
              'uncompressed')
        file_mb = write_pyramid(path)
        print(f'  wrote {file_mb:.1f} MB')

        with ReadCounter() as counter:
            report_reader(path, file_mb, counter)
            report_napari_layer(path, file_mb, counter)
            report_granularity(tmpdir, counter)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'paths', nargs='*', metavar='TIFF',
        help='tiff files to measure; without any, a pyramid is written '
             'and measured instead')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args().paths)
