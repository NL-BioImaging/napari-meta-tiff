"""Tests for reading what the metadata says about space.

Run this module to execute the tests by hand:

    python -m tests.test_metadata
"""

import pytest
from tifffile import TiffFile, imwrite

from napari_meta_tiff.metadata import (get_extra_metadata,
                                       get_spatial_metadata,
                                       resolution_pixel_size)

from tests._dummy_tiff import (dummy_image, write_ome_tiff,
                               write_resolution_tiff, write_vendor_tiff)


# what one pixel of the image below measures, in micrometres
SHAPE = {'x': 1024, 'y': 512}


def spatial(tif):
    """Where the image sits, as the reader reads it."""
    return get_spatial_metadata(tif, get_extra_metadata(tif), SHAPE)


def test_garbled_micro_sign_is_repaired(tmp_path):
    """A micro sign written in GBK reads as one, not as '¦Ì'."""
    path = str(tmp_path / 'garbled.tif')
    # Helios writes the Greek mu in GBK, which tifffile reads as latin-1
    write_vendor_tiff(path, '<Vendor><Beam><HFW>21.12μm</HFW></Beam>'
                            '</Vendor>'.encode('gbk'))
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)
    assert metadata['65000']['Vendor']['Beam']['HFW'] == '21.12µm'


@pytest.mark.parametrize('xml', [
    '<Vendor><pixelsizex>1.6528e-9</pixelsizex>'
    '<pixelsizey>1.6528e-9</pixelsizey></Vendor>',
    # FEI Titan names the axes inside the pixel size
    '<Metadata><BinaryResult><PixelSize><X>1.6528e-9</X><Y>1.6528e-9</Y>'
    '</PixelSize></BinaryResult></Metadata>',
])
def test_pixel_size_from_vendor_fields(tmp_path, xml):
    """A vendor stating the pixel size is read, metres where unstated."""
    path = str(tmp_path / 'vendor.tif')
    write_vendor_tiff(path, xml)
    with TiffFile(path) as tif:
        pixel_size = spatial(tif)['pixel_size']
    assert pixel_size == pytest.approx({'x': 1.6528e-3, 'y': 1.6528e-3})


def test_pixel_size_from_field_of_view(tmp_path):
    """The width of the image divided by its pixels is a pixel size."""
    path = str(tmp_path / 'fov.tif')
    write_vendor_tiff(path, '<Vendor><Scan><FOV_X><value>129.044885430288'
                            '</value><units>um</units></FOV_X></Scan>'
                            '</Vendor>')
    with TiffFile(path) as tif:
        pixel_size = spatial(tif)['pixel_size']
    # a field of view stated for one axis describes square pixels
    expected = 129.044885430288 / SHAPE['x']
    assert pixel_size == {'x': expected, 'y': expected}


def test_pixel_size_ignores_pixel_counts(tmp_path):
    """Fields counting pixels are not fields measuring one."""
    path = str(tmp_path / 'counts.tif')
    write_vendor_tiff(path, '<Vendor><PixelXDimension>5120</PixelXDimension>'
                            '<SamplesPerPixel>1</SamplesPerPixel>'
                            '<Scan><ResolutionX>8448</ResolutionX></Scan>'
                            '</Vendor>')
    with TiffFile(path) as tif:
        assert spatial(tif)['pixel_size'] == {}


def keys_anywhere(value):
    """Yield every key in a nested metadata structure."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_anywhere(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_anywhere(item)


def test_baseline_tags_pass_through_but_the_pixel_layout(tmp_path):
    """Every baseline tag describing the image is metadata, its pixel
    layout is not: the strips, tiles and compression only decode it."""
    path = str(tmp_path / 'baseline.tif')
    imwrite(path, dummy_image(32), tile=(16, 16), compression='zlib',
            resolution=(4, 4), resolutionunit='CENTIMETER',
            datetime='2023:11:07 12:24:09', software='Acme 1.0',
            description='a plain description', metadata=None)
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)

    assert metadata['ImageWidth'] == metadata['ImageLength'] == 32
    assert metadata['XResolution'] == (4, 1)
    assert metadata['ResolutionUnit'] == 'CENTIMETER'
    assert metadata['DateTime'] == '2023:11:07 12:24:09'
    assert metadata['Software'] == 'Acme 1.0'
    assert metadata['ImageDescription'] == 'a plain description'
    layout = {'TileOffsets', 'TileByteCounts', 'TileWidth', 'TileLength',
              'Compression', 'Predictor', 'PlanarConfiguration'}
    assert not layout & set(metadata)


def test_ome_metadata_without_its_pixel_layout(tmp_path):
    """An OME document is the metadata, but where its pixels are written."""
    path = str(tmp_path / 'layout.ome.tif')
    write_ome_tiff(path, x=(0.325, 'µm'), y=(0.325, 'µm'))
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)

    pixels = metadata['OME']['Image']['Pixels']
    assert pixels['PhysicalSizeX'] == 0.325
    assert not {'TiffData', 'BinData', 'BigEndian',
                'Interleaved'} & set(keys_anywhere(metadata))


def test_pixel_size_from_ome(tmp_path):
    """OME states the physical size, in micrometres unless it says."""
    path = str(tmp_path / 'stated.ome.tif')
    write_ome_tiff(path, x=(0.325, 'µm'), y=(0.325, 'µm'))
    with TiffFile(path) as tif:
        assert spatial(tif)['pixel_size'] == {'x': 0.325, 'y': 0.325}

    path = str(tmp_path / 'bare.ome.tif')
    write_ome_tiff(path, x=(0.325, None), y=(0.325, None))
    with TiffFile(path) as tif:
        assert spatial(tif)['pixel_size'] == {'x': 0.325, 'y': 0.325}


def test_resolution_in_centimetres(tmp_path):
    """A resolution per centimetre means what it says."""
    path = str(tmp_path / 'cm.tif')
    write_resolution_tiff(path, (25000, 1))
    with TiffFile(path) as tif:
        assert resolution_pixel_size(tif) == {'x': 0.4, 'y': 0.4}


def test_degenerate_resolution_is_per_metre(tmp_path):
    """A resolution that cancels out is read as pixels per metre.

    Delmic writes 250000000/250000000 per centimetre, where only the
    numerator carries the 4 nm pixel the instrument images at.
    """
    path = str(tmp_path / 'degenerate.tif')
    write_resolution_tiff(path, (250000000, 250000000))
    with TiffFile(path) as tif:
        assert resolution_pixel_size(tif) == {'x': 0.004, 'y': 0.004}


def test_implausible_resolution_is_dropped(tmp_path):
    """A resolution no image could have is not a measurement."""
    path = str(tmp_path / 'implausible.tif')
    # one pixel per metre, which is the 1/1 an overview is written with
    write_resolution_tiff(path, (1, 1))
    with TiffFile(path) as tif:
        assert spatial(tif)['pixel_size'] == {}


def test_screen_resolution_is_dropped(tmp_path):
    """The resolution a writer fills in for display is ignored."""
    for dots_per_inch in (72, 96):
        path = str(tmp_path / f'{dots_per_inch}dpi.tif')
        write_resolution_tiff(path, (dots_per_inch, 1), unit='INCH')
        with TiffFile(path) as tif:
            assert resolution_pixel_size(tif) == {}


def test_position_from_stage(tmp_path):
    """A stage position is read, metres where no unit is given."""
    path = str(tmp_path / 'stage.tif')
    write_vendor_tiff(path, '<Vendor><Stage><X><value>10.5</value>'
                            '<units>um</units></X><Y><value>-3.5</value>'
                            '<units>um</units></Y></Stage></Vendor>')
    with TiffFile(path) as tif:
        assert spatial(tif)['position'] == {'x': 10.5,
                                                            'y': -3.5}

    path = str(tmp_path / 'bare_stage.tif')
    write_vendor_tiff(path, '<Vendor><samplePosition><x>0.002</x>'
                            '<y>-0.001</y></samplePosition></Vendor>')
    with TiffFile(path) as tif:
        assert spatial(tif)['position'] == {'x': 2000.0,
                                                            'y': -1000.0}


def test_position_in_millimetres_from_stage(tmp_path):
    """Ciqtek writes its stage position in millimetres without saying so.

    Its mapping rule states the unit: 18.9417 beside a 110x110x69 mm
    stage is millimetres, where metres would put the sample further
    away than the stage can reach.
    """
    path = str(tmp_path / 'travel.tif')
    write_vendor_tiff(path, '<Vendor><Stage><StageType>'
                            'TMCM-GYLZ-4000X-FIVE-110x110x69</StageType>'
                            '<StagePosX>18.9417</StagePosX>'
                            '<StagePosY>11.1248</StagePosY></Stage></Vendor>')
    with TiffFile(path) as tif:
        position = spatial(tif)['position']
    assert position == pytest.approx({'x': 18941.7, 'y': 11124.8})


def test_position_ignores_other_fields(tmp_path):
    """Fields that are not a position in space are left alone."""
    path = str(tmp_path / 'other.tif')
    write_vendor_tiff(path, '<Vendor><ScanInfo><TrayPosition>Rear'
                            '</TrayPosition></ScanInfo><Scan><sourceTilt>'
                            '<x>-0.012</x></sourceTilt></Scan></Vendor>')
    with TiffFile(path) as tif:
        assert spatial(tif)['position'] == {}


def test_rotations_read_apart(tmp_path):
    """The scan rotation turns the image; the stage rotation the sample."""
    path = str(tmp_path / 'rotations.tif')
    write_vendor_tiff(path, '<Vendor><Beam><ElectricRotate>-135'
                            '</ElectricRotate></Beam><Stage><StagePosR>-30'
                            '</StagePosR></Stage></Vendor>')
    with TiffFile(path) as tif:
        result = spatial(tif)
    assert (result['rotation'], result['stage_rotation']) == (-135.0, -30.0)
    # neither turns up as a position
    assert result['position'] == {}


if __name__ == '__main__':
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)

        test_garbled_micro_sign_is_repaired(tmp_path)
        test_pixel_size_from_vendor_fields(tmp_path)
        test_pixel_size_from_field_of_view(tmp_path)
        test_pixel_size_ignores_pixel_counts(tmp_path)
        test_pixel_size_from_ome(tmp_path)
        test_resolution_in_centimetres(tmp_path)
        test_degenerate_resolution_is_per_metre(tmp_path)
        test_implausible_resolution_is_dropped(tmp_path)
        test_screen_resolution_is_dropped(tmp_path)
        test_position_from_stage(tmp_path)
        test_position_in_millimetres_from_stage(tmp_path)
        test_position_ignores_other_fields(tmp_path)
        test_rotations_read_apart(tmp_path)
