"""Tests for the metadata extraction script, run over real slides.

The slides live on this machine only, so the test is skipped wherever the
folder is missing, as it is in CI. The json is written to output/ at the
root of the repository, so it can be read after the run.

Run this module to execute the tests by hand:

    python -m testing.test_extract_metadata
"""

import glob
import json
from pathlib import Path

import pytest

from testing.extract_metadata import main


SLIDES_PATTERN = 'C:/Project/slides/tiff/*.tif*'

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'


@pytest.mark.skipif(not glob.glob(SLIDES_PATTERN),
                    reason=f'no slides at {SLIDES_PATTERN}')
def test_extract_metadata():
    """Every slide is read, and its metadata written to a json file."""
    assert main([SLIDES_PATTERN], str(OUTPUT_DIR)) == 0

    for path in glob.glob(SLIDES_PATTERN):
        destination = OUTPUT_DIR / f'{Path(path).stem}.json'
        assert destination.is_file()
        with open(destination, encoding='utf-8') as file:
            assert isinstance(json.load(file), dict)


if __name__ == '__main__':
    test_extract_metadata()
