"""Extract the metadata the reader finds in tiff files into json.

This is a diagnostic script, not a test: it runs get_extra_metadata over
a set of files and writes what it found to one json file per tiff, so the
metadata of a whole slide folder can be read, diffed and searched without
opening napari or the files themselves.

Run it with:

    python -m testing.extract_metadata "C:/Project/slides/tiff/*.tif*"

The shell on Windows does not expand a wildcard, so patterns are expanded
here, which also lets a directory be passed instead of a pattern. The json
is written to output/, one file per tiff:

    python -m testing.extract_metadata C:/Project/slides/tiff -o output
"""

import argparse
import datetime
import glob
import json
import logging
import os
from enum import Enum

import numpy as np
from tifffile import TiffFile

from napari_meta_tiff._metadata import get_extra_metadata


logger = logging.getLogger(__name__)

# the extensions a tiff is written under, so that passing a directory
# picks up the images in it rather than everything beside them
TIFF_PATTERNS = ('*.tif', '*.tiff', '*.svs', '*.ndpi', '*.scn',
                 '*.qptiff', '*.btf')

OUTPUT_DIR = 'output'


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


def extract_tiff_metadata(path):
    """Return the metadata the reader finds in a tiff, as json."""
    with TiffFile(path) as tif:
        return to_json(get_extra_metadata(tif))


def find_files(patterns):
    """Expand the patterns into the tiff files they name.

    A pattern is expanded here rather than by the shell, which on Windows
    does not do it, and a directory is taken to mean the tiffs in it.
    """
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            matches = [match
                       for tiff_pattern in TIFF_PATTERNS
                       for match in glob.glob(os.path.join(pattern,
                                                           tiff_pattern))]
        else:
            matches = glob.glob(pattern)
        if not matches:
            logger.warning('nothing matched %s', pattern)
        paths.extend(match for match in matches if os.path.isfile(match))
    # a file matched by more than one pattern is still read once, in the
    # order the patterns named it
    return list(dict.fromkeys(paths))


def output_path(path, output_dir, taken):
    """Return where the json for a tiff goes, without overwriting another.

    Files of one name can live in different folders while the output is
    flat, so a repeated name is numbered rather than silently replacing
    the file written before it.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    candidate = name
    index = 1
    while candidate in taken:
        index += 1
        candidate = f'{name}_{index}'
    taken.add(candidate)
    return os.path.join(output_dir, f'{candidate}.json')


def main(patterns, output_dir=OUTPUT_DIR):
    paths = find_files(patterns)
    if not paths:
        print('no tiff files found')
        return 1

    os.makedirs(output_dir, exist_ok=True)
    taken = set()
    failed = 0
    for path in paths:
        destination = output_path(path, output_dir, taken)
        try:
            metadata = extract_tiff_metadata(path)
        except Exception as exception:
            # one unreadable file should not stop the rest of a folder
            logger.warning('could not read %s', path, exc_info=True)
            print(f'{os.path.basename(path):<40} failed: {exception}')
            failed += 1
            continue
        with open(destination, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        print(f'{os.path.basename(path):<40} {len(metadata):>4} fields  '
              f'-> {destination}')

    print(f'\nwrote {len(paths) - failed} of {len(paths)} files to '
          f'{os.path.abspath(output_dir)}')
    return 1 if failed else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'patterns', nargs='+', metavar='TIFF',
        help=r'tiff files, directories, or patterns such as '
             r'C:\Project\slides\tiff\*.tif*')
    parser.add_argument(
        '-o', '--output-dir', default=OUTPUT_DIR,
        help=f'where the json files are written (default: {OUTPUT_DIR})')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    arguments = parse_args()
    raise SystemExit(main(arguments.patterns, arguments.output_dir))
