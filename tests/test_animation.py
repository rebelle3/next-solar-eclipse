#!/usr/bin/env python3
"""Tests for the shadow animation example.

Runs under pytest, or standalone with ``python3 tests/test_animation.py``.

The frame data is asserted on directly rather than through the renderer: the
question is whether the shadow is in the right place at the right time, and
going through matplotlib to ask it would be testing matplotlib.  One rendered
frame is checked at the end, and only for the things a picture can settle --
that the umbra is dark and that it landed on the pixel it should have.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'examples'))

import animate_shadow as anim  # noqa: E402

from eclipsepath import geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402

KERNEL = os.environ.get('ECLIPSEPATH_KERNEL')
COUNTRIES = os.environ.get('ECLIPSEPATH_COUNTRIES')
_cache = {}


def ephemeris():
    if 'e' not in _cache:
        g.set_lunar_radius('espenak')
        _cache['e'] = Ephemeris(KERNEL)
    return _cache['e']


def built(date='2027-08-02', count=16, step=2.0):
    key = (date, count, step)
    if key not in _cache:
        eclipse = anim.find_eclipse(ephemeris(), date, samples=60)
        _cache[key] = (eclipse, anim.frames_for(eclipse, count, step))
    return _cache[key]


def test_frames_span_the_eclipse_exactly():
    eclipse, frames = built()
    assert len(frames) == 16
    assert frames[0]['tt'] == eclipse.tt_first_contact
    assert frames[-1]['tt'] == eclipse.tt_last_contact
    assert all(b['tt'] > a['tt'] for a, b in zip(frames, frames[1:]))


def test_the_shadow_only_ever_moves_forwards():
    """Consecutive centres step east along the track, never back.

    A frame computed from the wrong instant, or a longitude wrapped the wrong
    way, shows up here as a step backwards or a stride out of all proportion
    to its neighbours.

    Only the frames with the umbra down are asked this.  Before it lands there
    is no axis intersection to follow and ``shadow_point`` gives the surface
    point nearest the axis instead, which slides along the limb under a motion
    of its own and does go backwards.
    """
    eclipse, frames = built()
    centres = [f['centre'] for f in frames if f['central']]
    strides = []
    for (lat1, lon1), (lat2, lon2) in zip(centres, centres[1:]):
        lon2 = lon2 - 360.0 * round((lon2 - lon1) / 360.0)
        assert lon2 > lon1, ((lat1, lon1), (lat2, lon2))
        strides.append(float(g.geodesic_distance(lat1, lon1, lat2, lon2)))
    strides = np.array(strides)
    assert strides.max() < 6.0 * strides.min(), (strides.min(), strides.max())


def test_the_umbra_appears_and_leaves_once():
    """Totality is one unbroken run of frames, not a flicker.

    The umbra reaching the ground, going away and coming back would mean the
    footprint calculation was dropping out on some frames.
    """
    _eclipse, frames = built()
    flags = [f['central'] for f in frames]
    assert not flags[0] and not flags[-1]
    runs = [key for key, _ in __import__('itertools').groupby(flags)]
    assert runs == [False, True, False], runs
    assert sum(flags) > 4, sum(flags)


def test_coverage_peaks_in_the_middle_and_dies_at_the_ends():
    """Deepest in the middle, nothing at either end, one plateau between.

    Not ``argmax``: the deepest coverage is a flat 1 for every frame with the
    umbra on the ground, so argmax returns whichever of them came first.  What
    can be asked is that those frames are one unbroken run and that the run
    sits in the middle.
    """
    _eclipse, frames = built()
    deepest = np.array([f['deepest'] for f in frames])
    assert deepest[0] < 0.02 and deepest[-1] < 0.02, (deepest[0], deepest[-1])
    assert deepest.max() > 0.999, deepest.max()
    plateau = np.nonzero(deepest > deepest.max() - 1e-9)[0]
    assert plateau[-1] - plateau[0] == len(plateau) - 1, plateau
    middle = 0.5 * (plateau[0] + plateau[-1])
    assert 3 < middle < len(frames) - 4, middle


def test_the_view_holds_every_frame_of_the_shadow():
    _eclipse, frames = built()
    west, east, south, north = anim.visible_extent(frames)
    assert east > west and north > south
    for frame in frames:
        grid = frame['grid']
        rows, cols = np.nonzero(grid.obscuration > 0.0)
        if not rows.size:
            continue
        assert south <= grid.latitudes[rows.min()], frame['tt']
        assert north >= grid.latitudes[rows.max()], frame['tt']


def test_a_shadow_over_the_antimeridian_does_not_claim_the_whole_world():
    """The 2028 total crosses the seam, which a naive extent reads as global.

    Taking the smallest and largest longitude touched would find points at both
    ends of the array and fit a view 360 degrees wide; measuring outwards from
    the shadow's own centre does not.
    """
    _eclipse, frames = built('2028-07-22', count=12, step=2.0)
    west, east, _south, _north = anim.visible_extent(frames)
    assert east - west < 320.0, (west, east)
    # It genuinely is a wide eclipse; the point is that it is not the world.
    assert east - west > 100.0, (west, east)


def test_hours_are_not_printed_as_minutes():
    assert anim.hours_minutes(6.0 * 60.0) == '6m'
    assert anim.hours_minutes(3600.0) == '1h 00m'
    assert anim.hours_minutes(5.0 * 3600.0 + 13.0 * 60.0) == '5h 13m'


def test_a_rendered_frame_puts_the_umbra_where_the_geometry_says():
    """Render for real, then find the darkest pixel and read back its position.

    This is the one thing only the renderer can be wrong about: every transform
    between a latitude and a pixel.  The frame is inverted from image
    coordinates back to degrees and compared with the computed centre.
    """
    if not COUNTRIES or not os.path.exists(COUNTRIES):
        print('   (skipped: set ECLIPSEPATH_COUNTRIES to a Natural Earth file)')
        return
    import tempfile
    from PIL import Image

    eclipse, frames = built()
    middle = max(range(len(frames)), key=lambda i: frames[i]['deepest'])
    frames = frames[middle:middle + 1]
    extent = anim.visible_extent(built()[1])
    with tempfile.TemporaryDirectory() as folder:
        out = os.path.join(folder, 'one.gif')
        anim.render(eclipse, frames, ephemeris().timescale,
                    anim.land_rings(COUNTRIES), out, extent, fps=1, dpi=100)
        image = np.asarray(Image.open(out).convert('RGB'), dtype=float)

    # The axes occupy a known rectangle of the figure; invert it rather than
    # hunting for the frame, so a change in layout fails loudly here.
    height, width = image.shape[:2]
    margin, header, footer = 0.55, 1.15, 0.95
    map_width = 9.6 - 2.0 * margin
    west, east, south, north = extent
    map_height = min(map_width * (north - south) / (east - west), 7.5)
    figure_height = map_height + header + footer
    left = margin / 9.6 * width
    right = left + map_width / 9.6 * width
    bottom = height - footer / figure_height * height
    top = bottom - map_height / figure_height * height

    # Look for the umbra's own colour rather than for darkness.  The darkest
    # pixel in the frame is the black text in the clock, which is inside the
    # axes and would win every time.
    want = np.array([13.0, 63.0, 120.0])          # UMBRA, '#0d3f78'
    patch = image[int(top):int(bottom), int(left):int(right)]
    distance = np.linalg.norm(patch - want, axis=2)
    row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
    lon = west + (column + 0.5) / patch.shape[1] * (east - west)
    lat = north - (row + 0.5) / patch.shape[0] * (north - south)
    want_lat, want_lon = frames[0]['centre']
    offset = float(g.geodesic_distance(lat, lon, want_lat, want_lon))
    # One screen pixel is about 40 km across at this size, and the umbra is
    # 250 km wide with a flat dark middle, so the darkest pixel is somewhere
    # inside it rather than exactly at the centre.
    assert offset < 200.0, (offset, (lat, lon), (want_lat, want_lon))
    # And it is the umbra that was found, not a coincidence: dark, and bluer
    # than it is red, which nothing else on a beige and grey map is.  The exact
    # colour cannot be asked for -- a GIF has 256 of them and the one written
    # is whichever survived quantisation.
    red, green, blue = patch[row, column]
    assert blue > red + 30.0, (red, green, blue)
    assert (red + green + blue) / 3.0 < 130.0, (red, green, blue)


def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failures = []
    for name, test in tests:
        try:
            test()
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print('FAIL %s: %s' % (name, exc))
        else:
            print('ok   %s' % name)
    print('\n%d passed, %d failed' % (len(tests) - len(failures), len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
