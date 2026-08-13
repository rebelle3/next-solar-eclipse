#!/usr/bin/env python3
"""Tests for the instantaneous shadow footprint used by the visualisers.

Runs under pytest, or standalone with ``python3 tests/test_shadow.py``.

Every test here is an agreement between the new instantaneous view and
something the package already computes another way -- the shadow's centre, the
traced band's limits, a site's contact times.  A picture drawn from this module
and a row of the path table are two cuts through one geometry, and if they ever
disagree one of them is lying.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import circumstances as cc  # noqa: E402
from eclipsepath import eclipse as ec, finder, shadow  # noqa: E402
from eclipsepath import geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402
from eclipsepath.observer import circumstances_at  # noqa: E402

KERNEL = os.environ.get('ECLIPSEPATH_KERNEL')
_cache = {}


def ephemeris():
    if 'e' not in _cache:
        g.set_lunar_radius('espenak')
        _cache['e'] = Ephemeris(KERNEL)
    return _cache['e']


def eclipse_on(year, month, day, threshold=1.0, samples=60):
    """One fully analysed eclipse, computed once and shared between tests."""
    key = (year, month, day, threshold, samples)
    if key not in _cache:
        ephem = ephemeris()
        ts = ephem.timescale
        events = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                                    ts.utc(year, month, day + 1).tt)
        assert events, 'no eclipse found on %04d-%02d-%02d' % (year, month, day)
        _cache[key] = ec.analyse(ephem, events[0], threshold=threshold,
                                 samples=samples)
    return _cache[key]


def total_2027():
    return eclipse_on(2027, 8, 2)


def gridded(eclipse, tt=None, step=1.0):
    key = ('grid', id(eclipse), tt, step)
    if key not in _cache:
        _cache[key] = shadow.obscuration_grid(
            eclipse.window, eclipse.tt_greatest if tt is None else tt, step)
    return _cache[key]


def distance_to_curve(lat, lon, curve_lat, curve_lon, per_segment=40):
    """Distance from a point to a polyline, not merely to its nearest vertex.

    The umbra is drawn as a ray fan, so near sunrise, where it stretches into a
    long ellipse, consecutive vertices are kilometres apart along the curve.
    Measuring to vertices alone would charge that spacing to the geometry.
    """
    curve_lat = np.asarray(curve_lat, float)
    curve_lon = np.asarray(curve_lon, float)
    fraction = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
    dense_lat = (curve_lat[:-1] + fraction * np.diff(curve_lat)).ravel()
    dense_lon = (curve_lon[:-1] + fraction * np.diff(curve_lon)).ravel()
    return float(g.geodesic_distance(lat, lon, dense_lat, dense_lon).min())


def polygon_contains(latitudes, longitudes, lat, lon):
    """Even-odd test in the plane the contour was drawn in."""
    inside = False
    n = len(latitudes)
    for i in range(n - 1):
        x1, y1 = longitudes[i], latitudes[i]
        x2, y2 = longitudes[i + 1], latitudes[i + 1]
        if (y1 > lat) != (y2 > lat):
            crossing = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if crossing > lon:
                inside = not inside
    return inside


# --- the field --------------------------------------------------------------

def test_grid_peak_finds_the_greatest_eclipse():
    eclipse = total_2027()
    grid = gridded(eclipse)
    depth, lat, lon = grid.peak
    assert abs(depth - eclipse.obscuration) < 1e-9, depth
    # Only the depth is pinned down.  The field is a flat 1 right across the
    # umbra, so which grid point wins the argmax is arbitrary; all that can be
    # asked is that it is one of them.
    outline = shadow.umbra_outline(eclipse.window, grid.tt)
    assert polygon_contains(outline[0], outline[1], lat, lon), (lat, lon)


def test_night_side_is_masked_but_the_raw_field_is_kept():
    eclipse = total_2027()
    grid = gridded(eclipse)
    down = grid.sun_altitude <= g.HORIZON_ALTITUDE_DEG
    assert down.any() and (~down).any()
    assert np.all(grid.obscuration[down] == 0.0)
    # Somewhere the Sun is below the horizon and the Moon is still in front of
    # it; that is the difference the mask exists to make.
    assert grid.raw[down].max() > 0.0, grid.raw[down].max()


def test_field_matches_a_site_computed_the_ordinary_way():
    eclipse = total_2027()
    seen = circumstances_at(eclipse, eclipse.latitude, eclipse.longitude)
    assert seen is not None
    here = shadow.visible_obscuration(eclipse.window, seen['tt_maximum'],
                                      eclipse.latitude, eclipse.longitude)
    assert abs(float(here) - seen['obscuration']) < 1e-9, here


# --- contours ---------------------------------------------------------------

def test_every_contour_vertex_sits_on_its_level():
    eclipse = total_2027()
    grid = gridded(eclipse)
    levels = [0.0001, 0.25, 0.5, 0.75, 0.95]
    for level, lines in shadow.contours(eclipse.window, grid, levels).items():
        assert lines, 'no contour at %g' % level
        for lat, lon, _closed in lines:
            value = shadow.visible_obscuration(eclipse.window, grid.tt, lat, lon)
            assert np.abs(value - level).max() < 1e-6, (level,
                                                        np.abs(value - level).max())


def test_contours_are_closed_and_nested():
    eclipse = total_2027()
    grid = gridded(eclipse)
    levels = [0.0001, 0.25, 0.5, 0.75, 0.95]
    lines = shadow.contours(eclipse.window, grid, levels)
    for level, drawn in lines.items():
        for lat, lon, closed in drawn:
            assert closed, 'open contour at %g' % level
            assert lat[0] == lat[-1] and abs(lon[0] - lon[-1]) < 1e-9
    _, peak_lat, peak_lon = grid.peak
    for level, drawn in lines.items():
        for lat, lon, _closed in drawn:
            assert polygon_contains(lat, lon, peak_lat, peak_lon), level
    for inner, outer in zip(levels[1:], levels[:-1]):
        outer_lat, outer_lon, _ = lines[outer][0]
        for lat, lon, _closed in lines[inner]:
            for point in range(0, len(lat), 5):
                assert polygon_contains(outer_lat, outer_lon, lat[point],
                                        lon[point]), (inner, outer, point)


def test_the_outer_contour_is_where_the_partial_eclipse_begins():
    """Sites just inside the 0% outline are in eclipse; just outside, not.

    Checked against contact times computed for each site individually, which
    is the package's other, entirely separate answer to the same question.
    """
    eclipse = total_2027()
    grid = gridded(eclipse)
    lat, lon, _closed = shadow.contours(eclipse.window, grid, [1e-4])[1e-4][0]
    _, peak_lat, peak_lon = grid.peak
    tested = 0
    for index in range(0, len(lat) - 1, 23):
        bearing = g.initial_bearing(peak_lat, peak_lon, lat[index], lon[index])
        for offset, expected in ((-60.0, True), (60.0, False)):
            probe_lat, probe_lon = g.great_circle_destination(
                lat[index], lon[index], bearing, offset)
            seen = circumstances_at(eclipse, float(probe_lat), float(probe_lon))
            if seen is None:
                # Outside the penumbra altogether, or the Sun never comes up:
                # either way the site is not in eclipse now, which is only
                # informative for the point we expect to be outside.
                assert not expected, (index, offset)
                continue
            covered = (seen['tt_first_contact'] <= grid.tt
                       <= seen['tt_last_contact'])
            here = float(shadow.visible_obscuration(
                eclipse.window, grid.tt, probe_lat, probe_lon))
            assert covered == (here > 0.0), (index, offset, covered, here)
            if expected:
                assert here > 0.0, (index, offset, here)
            tested += 1
    assert tested >= 8, tested


def test_nothing_is_drawn_before_the_eclipse_starts():
    eclipse = total_2027()
    before = eclipse.tt_first_contact - 600.0 / 86400.0
    grid = shadow.obscuration_grid(eclipse.window, before, step=1.0)
    assert grid.obscuration.max() == 0.0, grid.obscuration.max()
    assert shadow.contours(eclipse.window, grid, [1e-4])[1e-4] == []


def test_a_contour_crossing_the_antimeridian_stays_one_piece():
    """The seam at 180 degrees is a wrap in the grid, not a cut.

    The 2028 total crosses the Pacific, so its outline spans the join; a grid
    that treated the last column as an edge would hand back two arcs.
    """
    eclipse = eclipse_on(2028, 7, 22)
    tt = (eclipse.tt_first_contact + 0.75
          * (eclipse.tt_last_contact - eclipse.tt_first_contact))
    grid = gridded(eclipse, tt=tt, step=1.0)
    lines = shadow.contours(eclipse.window, grid, [0.4])[0.4]
    assert len(lines) == 1, len(lines)
    lat, lon, closed = lines[0]
    assert closed
    wrapped = (lon + 180.0) % 360.0 - 180.0
    assert wrapped.max() > 170.0 and wrapped.min() < -170.0, (wrapped.min(),
                                                              wrapped.max())
    # Continuous as handed back: consecutive vertices a degree or so apart, not
    # one pair of them leaping the width of the map.
    assert np.abs(np.diff(lon)).max() < 10.0, np.abs(np.diff(lon)).max()


# --- the umbra --------------------------------------------------------------

def test_umbra_is_centred_on_the_shadow_point():
    eclipse = total_2027()
    lat, lon = shadow.umbra_outline(eclipse.window, eclipse.tt_greatest)
    centre_lat = float(np.mean(lat[:-1]))
    centre_lon = float(np.mean(lon[:-1]))
    offset = g.geodesic_distance(centre_lat, centre_lon,
                                 eclipse.latitude, eclipse.longitude)
    assert float(offset) < 1.0, offset


def test_the_umbra_edge_is_exactly_the_edge_of_totality():
    eclipse = total_2027()
    lat, lon = shadow.umbra_outline(eclipse.window, eclipse.tt_greatest)
    state = cc.state_at(eclipse.window, g.geodetic_to_itrf(lat, lon),
                        eclipse.tt_greatest)
    depth = cc.central_depth(state)
    assert np.abs(depth).max() < 1e-9, np.abs(depth).max()
    covered = shadow.visible_obscuration(eclipse.window, eclipse.tt_greatest,
                                         lat, lon)
    assert covered.min() > 0.9999, covered.min()


def test_the_umbra_is_cut_off_at_the_sunrise_line():
    """At first contact the cone overruns the limb; the outline must not.

    Every vertex should be on one of the two boundaries -- the edge of the
    cone, or the edge of the daylight -- and none should be beyond either.
    """
    eclipse = total_2027()
    point = eclipse.central_band.points[0]
    lat, lon = shadow.umbra_outline(eclipse.window, point.tt)
    state = cc.state_at(eclipse.window, g.geodetic_to_itrf(lat, lon), point.tt)
    depth = cc.central_depth(state)
    altitude = state['sun_altitude']
    assert altitude.min() > g.HORIZON_ALTITUDE_DEG - 1e-6, altitude.min()
    # Vertices held back by the sunrise line are well inside the cone, which is
    # the whole point of the clip; only the ones the cone itself stops sit on
    # its edge.
    on_cone = np.abs(depth) < 1e-9
    on_horizon = np.abs(altitude - g.HORIZON_ALTITUDE_DEG) < 1e-6
    assert (on_cone | on_horizon).all(), int((~(on_cone | on_horizon)).sum())
    # This particular instant is the interesting one: both kinds are present.
    assert on_cone.any() and on_horizon.any(), (on_cone.sum(), on_horizon.sum())


def test_the_umbra_stays_inside_the_band_it_sweeps():
    """What the umbra covers now is part of what the band covers overall.

    The band's limits are not a cut across the umbra.  ``_trace_band`` asks
    whether a place reaches totality *at any time* during the eclipse, so its
    edges are the envelope of the whole family of moving umbrae; the outline
    here is one member of that family, frozen.

    The check is made just inside the outline rather than on it.  ``reaches``
    answers by sampling 181 instants across three and a half hours, and a place
    exactly on the umbra's edge is total for an interval shrinking to nothing,
    which no sampling of the time axis can be expected to catch.  A few
    kilometres in, totality lasts long enough to be seen.
    """
    eclipse = total_2027()
    band = eclipse.central_band
    assert band is not None and band.points
    coarse = np.linspace(eclipse.tt_first_contact, eclipse.tt_last_contact,
                         ec.PREDICATE_TIME_SAMPLES)
    checked = 0
    for point in band.points[::7]:
        lat, lon = shadow.umbra_outline(eclipse.window, point.tt)
        centre_lat, centre_lon = np.mean(lat[:-1]), np.mean(lon[:-1])
        inner_lat = centre_lat + 0.9 * (lat[:-1] - centre_lat)
        inner_lon = centre_lon + 0.9 * (lon[:-1] - centre_lon)
        swept = cc.reaches(eclipse.window, coarse, inner_lat, inner_lon,
                           cc.central_depth)
        assert swept.all(), (point.tt, int((~swept).sum()))
        checked += 1
    assert checked >= 5, checked


def test_the_band_limits_graze_the_umbra_and_never_enter_it():
    """The envelope touches each instantaneous umbra without crossing it.

    Never inside is the invariant and it holds to machine precision.  How far
    outside is a question about conditioning: with the Sun overhead the two
    curves run within 100 m of each other, and as it drops towards the horizon
    the shadow edge meets the ground ever more obliquely, so the same vanishing
    difference in the geometry buys kilometres on the map and the touch point
    slides along the track.  That spread is grazing incidence, not slack in
    either calculation.
    """
    eclipse = total_2027()
    for point in eclipse.central_band.points[::7]:
        lat, lon = shadow.umbra_outline(eclipse.window, point.tt)
        limits = (np.array([point.north_latitude, point.south_latitude]),
                  np.array([point.north_longitude, point.south_longitude]))
        state = cc.state_at(eclipse.window, g.geodetic_to_itrf(*limits),
                            point.tt)
        # Outside or on the umbra, never within it.
        assert cc.central_depth(state).max() <= 1e-12, cc.central_depth(state)
        gap = max(distance_to_curve(limits[0][i], limits[1][i], lat, lon)
                  for i in (0, 1))
        assert gap < 12.0, (point.tt, gap, state['sun_altitude'])
        if min(state['sun_altitude']) > 70.0:
            assert gap < 0.2, (point.tt, gap, state['sun_altitude'])


def test_no_umbra_before_it_lands():
    eclipse = total_2027()
    assert shadow.umbra_outline(eclipse.window, eclipse.tt_first_contact) is None


def test_a_partial_eclipse_has_no_umbra_at_all():
    """2029 January is partial everywhere; nothing should be drawn as umbra."""
    eclipse = eclipse_on(2029, 1, 14, threshold=0.5)
    assert not eclipse.central
    for fraction in (0.2, 0.5, 0.8):
        tt = (eclipse.tt_first_contact + fraction
              * (eclipse.tt_last_contact - eclipse.tt_first_contact))
        assert shadow.umbra_outline(eclipse.window, tt) is None, fraction


# --- the terminator ---------------------------------------------------------

def test_terminator_sits_on_the_horizon():
    eclipse = total_2027()
    lat, lon = shadow.terminator(eclipse.window, eclipse.tt_greatest)
    state = cc.state_at(eclipse.window, g.geodetic_to_itrf(lat, lon),
                        eclipse.tt_greatest)
    error = np.abs(state['sun_altitude'] - g.HORIZON_ALTITUDE_DEG)
    assert error.max() < 1e-6, error.max()


def test_the_subsolar_point_has_the_sun_overhead():
    eclipse = total_2027()
    lat, lon = shadow.subsolar_point(eclipse.window, eclipse.tt_greatest)
    state = cc.state_at(eclipse.window,
                        g.geodetic_to_itrf(np.array([lat]), np.array([lon])),
                        eclipse.tt_greatest)
    assert abs(float(state['sun_altitude'][0]) - 90.0) < 0.02, state['sun_altitude']


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
