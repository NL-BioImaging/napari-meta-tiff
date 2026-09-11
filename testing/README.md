# testing

Diagnostic scripts for exploring the reader by hand. These are deliberately
not tests: they print measurements for a human to read instead of asserting
thresholds, so they are not collected by pytest and do not run in CI.

Automated tests live in [`../tests`](../tests).

## lazy_loading.py

Checks that the reader really lazy loads, which matters most for large
multiscale and tiled images.

```
python -m testing.lazy_loading
```

Pass one or more paths to measure real files instead:

```
python -m testing.lazy_loading C:/Project/slides/AMC_EM/B/1.tif
```

For a real file it first prints the vendor metadata the reader found and
what that metadata says about space, the pixel size and stage position
the layer is placed by, and then the same read measurements, skipping only the granularity comparison,
which needs files written to a known layout.

Without a path it writes an 8192x8192 uint8 BigTIFF pyramid of 5 levels and counts the
bytes actually pulled off disk, by wrapping every read path on tifffile's
`FileHandle`. Byte counts are used rather than timings, which on a warm run
say more about the page cache than about the reader.

It reports:

- what opening a file costs, before any pixel is touched
- what a single pixel, a chunk and a viewport sized region cost, against
  the size of the level they come from, with `TiffFile.asarray()` as an
  eager baseline
- what a napari layer reads when it is created, when the level changes and
  as the viewport shrinks
- how chunk granularity differs between a tiled and a striped file

Reading a single pixel of a tiled image costs one tile. A striped image
cannot be read in pieces smaller than a full width band, so the same access
is as large as the strip; that follows from the file layout rather than from
this plugin.
