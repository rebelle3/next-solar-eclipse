#!/usr/bin/env python3
"""Test suite for eclipsepath.

Runs under pytest, or standalone with ``python3 tests/test_eclipsepath.py``.
The reference values come from the NASA/Espenak Five Millennium Catalog
(``tests/data/nasa_catalog.json``); path-level checks against NASA's published
path tables live in ``tests/verify_against_nasa.py``.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import circumstances as cc  # noqa: E402
from eclipsepath import catalog, eclipse as ec, finder, output  # noqa: E402
from eclipsepath import geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris, EclipseWindow  # noqa: E402
from eclipsepath.observer import circumstances_at  # noqa: E402

KERNEL = os.environ.get('ECLIPSEPATH_KERNEL')
_cache = {}


def ephemeris():
    if 'e' not in _cache:
        g.set_lunar_radius('espenak')
        _cache['e'] = Ephemeris(KERNEL)
    return _cache['e']


def reference():
    with open(os.path.join(HERE, 'data', 'nasa_catalog.json')) as handle:
        return json.load(handle)['eclipses']


def scanned():
    """The 2026-2037 scan, computed once and shared between tests."""
    if 'scan' not in _cache:
        ephem = ephemeris()
        ts = ephem.timescale
        _cache['scan'] = finder.find_events(ephem, ts.utc(2026, 1, 1).tt,
                                            ts.utc(2037, 1, 1).tt)
    return _cache['scan']


# --- geometry ---------------------------------------------------------------

def test_geodetic_roundtrip():
    lats = np.array([0.0, 51.6, -33.9, 89.9, -89.0, 45.0])
    lons = np.array([0.0, -3.9, 151.2, 12.0, -179.9, 179.9])
    heights = np.array([0.0, 0.1, -0.05, 0.0, 2.0, 0.0])
    xyz = g.geodetic_to_itrf(lats, lons, heights)
    back_lat, back_lon, back_h = g.itrf_to_geodetic(xyz)
    assert np.allclose(back_lat, lats, atol=1e-9)
    assert np.allclose(back_lon, lons, atol=1e-9)
    assert np.allclose(back_h, heights, atol=1e-9)


def test_ellipsoid_shape():
    """A point at the pole sits on the semi-minor axis, the equator on a."""
    assert abs(g.norm(g.geodetic_to_itrf(90.0, 0.0)) - g.EARTH_B_KM) < 1e-9
    assert abs(g.norm(g.geodetic_to_itrf(0.0, 0.0)) - g.EARTH_A_KM) < 1e-9


def test_geodesic_distance_known_values():
    # One degree of longitude at the equator, and a widely quoted city pair.
    assert abs(g.geodesic_distance(0, 0, 0, 1) - 111.3195) < 0.001
    assert abs(g.geodesic_distance(51.5074, -0.1278, 48.8566, 2.3522) - 343.9) < 0.5
    # A quarter of a meridian is close to a quarter of the polar circumference.
    assert abs(g.geodesic_distance(0, 0, 90, 0) - 10001.966) < 0.01


def test_obscuration_limits():
    r_sun, r_moon = 0.0045, 0.0047
    assert g.obscuration(r_sun, r_moon, 0.0) == 1.0                 # total
    assert g.obscuration(r_sun, r_moon, r_moon + r_sun) == 0.0      # no contact
    assert g.obscuration(r_sun, r_moon, r_moon - r_sun) == 1.0      # 2nd contact
    # An annular eclipse can only ever hide the square of the diameter ratio.
    ann = g.obscuration(0.0047, 0.0045, 0.0)
    assert abs(ann - (0.0045 / 0.0047) ** 2) < 1e-12
    # Equal discs exactly aligned hide everything; half-overlap is monotone.
    assert abs(g.obscuration(0.0045, 0.0045, 0.0) - 1.0) < 1e-12
    values = [g.obscuration(r_sun, r_moon, s)
              for s in np.linspace(0.0, r_sun + r_moon, 40)]
    assert all(a >= b - 1e-15 for a, b in zip(values, values[1:]))


def test_obscuration_matches_numeric_integration():
    """Check the closed-form lens area against a Monte-Carlo estimate."""
    rng = np.random.default_rng(20260812)
    r_sun, r_moon, sep = 0.00465, 0.00450, 0.00300
    x = rng.uniform(-r_sun, r_sun, 400000)
    y = rng.uniform(-r_sun, r_sun, 400000)
    inside_sun = x ** 2 + y ** 2 <= r_sun ** 2
    inside_moon = (x - sep) ** 2 + y ** 2 <= r_moon ** 2
    estimate = np.count_nonzero(inside_sun & inside_moon) / np.count_nonzero(inside_sun)
    assert abs(g.obscuration(r_sun, r_moon, sep) - estimate) < 0.002


def test_magnitude_conventions():
    r_sun, r_moon = 0.0045, 0.0047
    assert g.magnitude(r_sun, r_moon, r_sun + r_moon) == 0.0
    assert abs(g.magnitude(r_sun, r_moon, 0.0) - r_moon / r_sun) < 1e-12
    # Halfway between first and second contact, half the diameter is covered.
    half = (r_sun + r_moon + abs(r_moon - r_sun)) / 2.0
    assert abs(g.magnitude(r_sun, r_moon, half) - 0.5) < 1e-12


def test_bearing_and_destination_are_inverse():
    lat, lon, bearing, distance = 40.0, -20.0, 73.0, 500.0
    to_lat, to_lon = g.great_circle_destination(lat, lon, bearing, distance)
    assert abs(g.initial_bearing(lat, lon, to_lat, to_lon) - bearing) < 1e-6


def test_align_broadcasts_component_arrays():
    vec = np.ones((3, 5))
    assert g.align(vec, 1, 2).shape == (3, 1, 5)
    mat = np.ones((3, 3, 5))
    assert g.align(mat, 2, 2).shape == (3, 3, 1, 5)


# --- ephemeris and interpolation --------------------------------------------

def test_chebyshev_window_matches_kernel():
    ephem = ephemeris()
    ts = ephem.timescale
    tt0 = ts.utc(2026, 8, 12, 15).tt
    tt1 = ts.utc(2026, 8, 12, 21).tt
    window = EclipseWindow(ephem, tt0, tt1)
    tt = np.linspace(tt0, tt1, 97)
    exact_sun, exact_moon = ephem.sun_moon(ephem.time(tt))
    exact_rot = ephem.rotation(ephem.time(tt))
    sun, moon, rot = window.at(tt)
    assert np.abs(sun - exact_sun).max() < 0.01        # km
    assert np.abs(moon - exact_moon).max() < 0.001     # km
    # Rotation error, expressed as displacement at the Earth's surface.
    assert np.abs(rot - exact_rot).max() * g.EARTH_A_KM < 1e-4


def test_delta_t_override_shifts_longitude_only():
    ephem = ephemeris()
    ts = ephem.timescale
    events = finder.find_events(ephem, ts.utc(2026, 8, 1).tt, ts.utc(2026, 8, 31).tt)
    base = ec.analyse(ephem, events[0], samples=3)
    shifted = Ephemeris(KERNEL, delta_t=ephem.delta_t_at(events[0].tt_greatest) + 10.0)
    moved = ec.analyse(shifted, events[0], samples=3)
    assert abs(moved.latitude - base.latitude) < 0.01
    # A larger delta-T means the Earth has turned less far by a given TT, so
    # the shadow lands further east.
    expected = 10.0 * 360.98564736629 / 86400.0
    assert abs((moved.longitude - base.longitude) - expected) < 0.01, (
        moved.longitude - base.longitude)


# --- event detection against the NASA catalog -------------------------------

def test_finds_every_catalogued_eclipse():
    assert len(scanned()) == len(reference())


def test_greatest_eclipse_times_match_catalog():
    ephem = ephemeris()
    ts = ephem.timescale
    worst = 0.0
    for event, ref in zip(scanned(), reference()):
        year, month, day = (int(v) for v in ref['date'].split('-'))
        hour, minute, second = (int(v) for v in ref['td'].split(':'))
        # Compared in TT: the catalogue's UT times assume its own delta-T.
        expected = ts.tt(year, month, day, hour, minute, second).tt
        worst = max(worst, abs(event.tt_greatest - expected) * 86400.0)
    assert worst < 1.0, 'worst greatest-eclipse error %.2f s' % worst


def test_types_gamma_saros_and_magnitude_match_catalog():
    ephem = ephemeris()
    for event, ref in zip(scanned(), reference()):
        result = ec.analyse(ephem, event, samples=3, trace_limits=False)
        assert result.kind == ref['type'], ref['date']
        assert abs(result.gamma - ref['gamma']) < 0.0005, ref['date']
        assert result.saros == ref['saros'], ref['date']
        assert result.lunation == ref['luna'], ref['date']
        assert abs(result.magnitude - ref['mag']) < 0.0006, ref['date']
        assert abs(result.latitude - ref['lat']) <= 0.6, ref['date']
        delta_lon = (result.longitude - ref['lon'] + 180.0) % 360.0 - 180.0
        assert abs(delta_lon) <= 0.6, ref['date']


def test_path_width_and_duration_match_catalog():
    """Central eclipses only; the catalogue has no path for a partial."""
    ephem = ephemeris()
    for event, ref in zip(scanned(), reference()):
        if ref['width'] is None:
            continue
        result = ec.analyse(ephem, event, samples=3)
        assert abs(result.central_duration_seconds - ref['dur']) < 1.0, ref['date']
        # Tabulated widths come from a tangent-plane approximation that
        # under-reports a wide band at low Sun altitude, so grazing eclipses
        # get a looser bound; see verify_against_nasa.py.
        tolerance = 40.0 if ref['alt'] < 20 else 3.0
        assert abs(result.path_width_km - ref['width']) < tolerance, (
            '%s: %.1f vs %d' % (ref['date'], result.path_width_km, ref['width']))


def test_partial_eclipses_have_no_central_path():
    ephem = ephemeris()
    for event, ref in zip(scanned(), reference()):
        if ref['type'] != 'P':
            continue
        result = ec.analyse(ephem, event, samples=20)
        assert result.central_band is None
        assert not result.central
        assert result.peak_obscuration < 1.0


def test_annular_never_reaches_full_coverage():
    ephem = ephemeris()
    for event, ref in zip(scanned(), reference()):
        if ref['type'] != 'A':
            continue
        result = ec.analyse(ephem, event, samples=20)
        assert result.peak_obscuration < 1.0
        assert not result.meets_threshold          # default threshold is 1.0
        assert result.central_band is not None     # but it still has a path
        assert result.central_band.label == 'annularity'


# --- paths ------------------------------------------------------------------

def test_totality_band_is_consistent():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, threshold=1.0, samples=60)
    assert result.coverage_band is result.central_band   # identical for a total
    grid = np.linspace(result.tt_first_contact, result.tt_last_contact,
                       ec.PREDICATE_TIME_SAMPLES)
    for point in result.coverage_band.points[2:-2]:
        # Every centre-line sample must actually see totality...
        peak = cc.peak_eclipse(result.window,
                               g.geodetic_to_itrf(point.latitude, point.longitude),
                               grid)
        assert peak['obscuration'][0] >= 1.0 - 1e-9
        # ...and both limits should sit right on the edge of it.
        for lat, lon in ((point.north_latitude, point.north_longitude),
                         (point.south_latitude, point.south_longitude)):
            edge = cc.peak_eclipse(result.window, g.geodetic_to_itrf(lat, lon),
                                   grid)['obscuration'][0]
            assert 0.999 < edge <= 1.0, edge
        # A little beyond the limit, totality must be lost.
        heading = g.initial_bearing(point.south_latitude, point.south_longitude,
                                    point.north_latitude, point.north_longitude)
        out_lat, out_lon = g.great_circle_destination(
            point.north_latitude, point.north_longitude, heading, 6.0)
        outside = cc.peak_eclipse(result.window,
                                  g.geodetic_to_itrf(out_lat, out_lon),
                                  grid)['obscuration'][0]
        assert outside < 1.0
        assert point.width_km > 0.0
        assert point.duration_seconds > 0.0


def test_central_band_spans_exactly_the_umbral_phase():
    """Every centre-line point must be central at its own timestamp."""
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2027, 7, 15).tt,
                               ts.utc(2027, 8, 15).tt)[0]
    result = ec.analyse(ephem, event, samples=40)
    band = result.central_band
    assert abs(band.points[0].tt - result.tt_central_start) < 1e-9
    assert abs(band.points[-1].tt - result.tt_central_end) < 1e-9
    assert result.tt_first_contact < result.tt_central_start
    assert result.tt_central_end < result.tt_last_contact
    # The axis touches the globe throughout, and misses just outside.
    _, inside = ec.shadow_point(result.window,
                                np.array([p.tt for p in band.points]))
    assert inside.all()
    margin = 30.0 / 86400.0
    _, outside = ec.shadow_point(
        result.window, np.array([result.tt_central_start - margin,
                                 result.tt_central_end + margin]))
    assert not outside.any()
    # Totality at each centre-line point brackets that point's own timestamp.
    for point in band.points:
        assert point.tt_start - 1e-9 <= point.tt <= point.tt_end + 1e-9
        span = (point.tt_end - point.tt_start) * 86400.0
        assert abs(span - point.duration_seconds) < 0.05


def test_band_widens_as_threshold_falls():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    widths = []
    for threshold in (1.0, 0.95, 0.9):
        result = ec.analyse(ephem, event, threshold=threshold, samples=24)
        middle = result.coverage_band.points[len(result.coverage_band.points) // 2]
        widths.append(middle.width_km)
    assert widths[0] < widths[1] < widths[2]


def test_threshold_band_edges_meet_the_threshold():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, threshold=0.9, samples=24)
    grid = np.linspace(result.tt_first_contact, result.tt_last_contact,
                       ec.PREDICATE_TIME_SAMPLES)
    for point in result.coverage_band.points[2:-2]:
        for lat, lon in ((point.north_latitude, point.north_longitude),
                         (point.south_latitude, point.south_longitude)):
            peak = cc.peak_eclipse(result.window, g.geodetic_to_itrf(lat, lon),
                                   grid)['obscuration'][0]
            assert abs(peak - 0.9) < 0.002, peak


def test_path_sampling_does_not_move_the_path():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2027, 7, 15).tt,
                               ts.utc(2027, 8, 15).tt)[0]
    coarse = ec.analyse(ephem, event, samples=30)
    fine = ec.analyse(ephem, event, samples=90)
    assert abs(coarse.path_width_km - fine.path_width_km) < 0.01
    assert abs(coarse.central_duration_seconds
               - fine.central_duration_seconds) < 0.01


def test_representation_does_not_depend_on_sampling():
    """A borderline threshold must not flip between band and region."""
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    for threshold in (0.9, 0.85, 0.8):
        choices = set()
        for samples in (16, 24, 40, 80):
            result = ec.analyse(ephem, event, threshold=threshold,
                                samples=samples)
            choices.add(result.coverage_band is not None)
        assert len(choices) == 1, 'threshold %g flipped: %s' % (threshold, choices)


def test_region_replaces_the_band_once_it_stops_being_a_strip():
    ephem = ephemeris()
    ts = ephem.timescale
    total = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    partial = finder.find_events(ephem, ts.utc(2029, 1, 1).tt,
                                 ts.utc(2029, 1, 31).tt)[0]
    # Totality is a strip, so it stays a band.
    strip = ec.analyse(ephem, total, threshold=1.0, samples=30)
    assert strip.coverage_band is not None and strip.coverage_region is None
    # A partial eclipse's coverage area never is.
    blob = ec.analyse(ephem, partial, threshold=0.6, samples=30)
    assert blob.coverage_band is None and blob.coverage_region is not None
    assert blob.central_band is None
    # Nothing is traced for a threshold the eclipse never reaches.
    none = ec.analyse(ephem, partial, threshold=0.95, samples=30)
    assert none.coverage_band is None and none.coverage_region is None


def test_region_boundary_sits_on_the_threshold():
    """Away from the terminator every vertex must be exactly on the contour."""
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2029, 1, 1).tt,
                               ts.utc(2029, 1, 31).tt)[0]
    result = ec.analyse(ephem, event, threshold=0.6, samples=30)
    region = result.coverage_region
    # Refinement only ever adds vertices to the even sweep it starts from.
    assert len(region.points) >= ec.REGION_RAYS
    grid = np.linspace(result.tt_first_contact, result.tt_last_contact,
                       ec.PREDICATE_TIME_SAMPLES)
    def peak_at(lat, lon):
        return cc.peak_eclipse(result.window, g.geodetic_to_itrf(lat, lon),
                               grid)['obscuration'][0]

    def peak_along(point, offset):
        return peak_at(*g.great_circle_destination(
            region.centre_latitude, region.centre_longitude,
            point.azimuth, point.distance_km + offset))

    checked, grazing = 0, 0
    for point in region.points:
        inward = peak_along(point, -0.5)
        outward = peak_along(point, +0.5)
        holds = inward >= 0.6 and outward < 0.6
        if point.sun_altitude > 5.0:
            # Away from the terminator the edge is exact: the threshold is met
            # just inside it and not met just outside.
            assert holds, (point.azimuth, inward, outward)
            checked += 1
        else:
            # Near the terminator peak coverage is nearly flat, so the edge is
            # poorly conditioned and may wobble by a few kilometres.  Anything
            # that fails the invariant has to be here, not out in the open.
            grazing += 1
    assert checked > 40, 'expected much of the outline to be a clean contour'
    assert grazing, 'expected part of the outline to run along the terminator'

    # The outline must also be smooth, not merely accurate: an edge drawn from
    # vertices hundreds of kilometres apart is a row of facets even when every
    # one of them is on the contour.  This is what the refinement guarantees.
    lat = np.array([p.latitude for p in region.points])
    lon = np.array([p.longitude for p in region.points])
    gap = g.geodesic_distance(lat, lon, np.roll(lat, -1), np.roll(lon, -1))
    assert gap.max() <= ec.REGION_MAX_GAP_KM * 1.05, gap.max()
    assert result.coverage_region.centre_latitude == region.centre_latitude
    centre = cc.peak_eclipse(result.window,
                             g.geodetic_to_itrf(region.centre_latitude,
                                                region.centre_longitude),
                             grid)['obscuration'][0]
    assert centre >= 0.6


def test_region_detects_a_pole_inside_it():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2029, 6, 1).tt,
                               ts.utc(2029, 6, 30).tt)[0]
    checked = False
    for threshold, expected in ((0.20, False), (0.15, True)):
        result = ec.analyse(ephem, event, threshold=threshold, samples=20)
        assert result.coverage_region.encloses_pole is expected, threshold
        if not checked:
            checked = True
            grid = np.linspace(result.tt_first_contact, result.tt_last_contact,
                               ec.PREDICATE_TIME_SAMPLES)
            pole = cc.peak_eclipse(result.window, g.geodetic_to_itrf(90.0, 0.0),
                                   grid)['obscuration'][0]
            assert 0.15 < pole < 0.20, pole


def test_region_outputs():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2029, 1, 1).tt,
                               ts.utc(2029, 1, 31).tt)[0]
    result = ec.analyse(ephem, event, threshold=0.6, samples=30)
    collection = output.to_geojson([result], ts)
    kinds = [f['properties']['feature'] for f in collection['features']]
    assert 'coverage-region' in kinds
    for feature in collection['features']:
        if feature['properties']['feature'] != 'coverage-region':
            continue
        coords = feature['geometry']['coordinates']
        steps = [abs(coords[i][0] - coords[i - 1][0])
                 for i in range(1, len(coords))]
        assert not steps or max(steps) < 180.0
    rows = output.to_csv([result], ts).strip().split('\n')
    assert rows[0] == ','.join(output.CSV_COLUMNS)
    assert all(row.split(',')[3] == 'boundary' for row in rows[1:])
    assert len(rows) - 1 == len(result.coverage_region.points)
    detail = output.to_json([result], ts)['eclipses'][0]
    assert [x['kind'] for x in detail['geometries']] == ['region']
    assert len(detail['geometries'][0]['boundary']) == ec.REGION_RAYS
    text = output.format_text([result], ts, 0.6, path_rows=5)
    assert 'region of 60% obscuration' in text
    assert 'terminator' in text


def test_both_path_and_region_are_reported_together():
    """A central eclipse filtered at a lower threshold keeps its path too."""
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 2, 1).tt,
                               ts.utc(2026, 2, 28).tt)[0]
    result = ec.analyse(ephem, event, threshold=0.6, samples=30)
    assert result.central_band.label == 'annularity'
    assert result.coverage_region is not None
    products = output.geometries(result)
    assert len(products) == 2
    assert [output.is_region(x) for x in products] == [False, True]



# --- observers --------------------------------------------------------------

def test_cardiff_sees_a_deep_partial_in_2026():
    """South Wales saw about 93% of the Sun covered on 2026 Aug 12."""
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, samples=8)
    seen = circumstances_at(result, 51.4816, -3.1791)
    assert 0.92 < seen['obscuration'] < 0.94
    assert seen['central_seconds'] == 0.0              # partial only
    # timeanddate.com quotes maximum at 19:13 BST, i.e. 18:13 UTC.
    maximum = ts.tt_jd(seen['tt_maximum']).utc_strftime('%H:%M:%S')
    assert '18:13:00' <= maximum < '18:14:00', maximum
    assert seen['tt_first_contact'] < seen['tt_maximum'] < seen['tt_last_contact']
    assert 5400 < seen['partial_seconds'] < 7800


def test_centre_line_observer_sees_totality():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, samples=40)
    middle = result.coverage_band.points[len(result.coverage_band.points) // 2]
    seen = circumstances_at(result, middle.latitude, middle.longitude)
    assert seen['obscuration'] >= 1.0 - 1e-9
    assert abs(seen['central_seconds'] - middle.duration_seconds) < 0.2
    assert seen['tt_second_contact'] < seen['tt_third_contact']


def test_night_side_sees_nothing():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, samples=8)
    assert circumstances_at(result, -33.87, 151.21) is None      # Sydney


# --- output -----------------------------------------------------------------

def test_split_runs_cuts_at_the_antimeridian():
    assert output.split_runs([170.0, 175.0, 179.0, -179.0, -175.0]) == [(0, 3), (3, 5)]
    assert output.split_runs([0.0, 1.0, 2.0]) == [(0, 3)]
    # A wrap in any one series cuts them all.
    assert output.split_runs([1.0, 2.0, 3.0], [179.0, -179.0, -178.0]) == [(1, 3)]


def test_unwrap_keeps_cross_sections_contiguous():
    assert abs(output.unwrap_to(179.5, -179.5) - 180.5) < 1e-9
    assert abs(output.unwrap_to(-179.5, 179.5) - (-180.5)) < 1e-9
    assert abs(output.unwrap_to(10.0, 12.0) - 12.0) < 1e-9


def test_geojson_is_well_formed():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2028, 7, 1).tt,
                               ts.utc(2028, 8, 1).tt)[0]
    result = ec.analyse(ephem, event, samples=60)
    collection = output.to_geojson([result], ts)
    assert collection['type'] == 'FeatureCollection'
    kinds = {f['properties']['feature'] for f in collection['features']}
    assert {'greatest-eclipse', 'shadow-track', 'centre-line', 'northern-limit',
            'southern-limit', 'band'} <= kinds
    for feature in collection['features']:
        geometry = feature['geometry']
        coords = (geometry['coordinates'][0] if geometry['type'] == 'Polygon'
                  else [geometry['coordinates']] if geometry['type'] == 'Point'
                  else geometry['coordinates'])
        assert len(coords) >= 1
        for lon, lat in coords:
            assert -90.0 <= lat <= 90.0
            assert -185.0 <= lon <= 185.0     # limits may unwrap slightly
        steps = [abs(coords[i][0] - coords[i - 1][0])
                 for i in range(1, len(coords))]
        assert not steps or max(steps) < 180.0, 'geometry wraps the globe'
        if geometry['type'] == 'Polygon':
            assert coords[0] == coords[-1], 'polygon ring is not closed'
        if geometry['type'] == 'LineString':
            assert len(feature['properties']['times']) == len(coords)


def test_csv_rows_match_the_band():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, samples=30)
    text = output.to_csv([result], ts)
    lines = text.strip().split('\n')
    assert lines[0] == ','.join(output.CSV_COLUMNS)
    assert len(lines) - 1 == len(result.coverage_band.points)
    first = lines[1].split(',')
    assert first[0] == '2026-08-12' and first[1] == 'T'
    assert first[3] == 'cross-section'
    assert abs(float(first[5]) - result.coverage_band.points[0].latitude) < 1e-4


def test_json_round_trips():
    ephem = ephemeris()
    ts = ephem.timescale
    event = finder.find_events(ephem, ts.utc(2026, 8, 1).tt,
                               ts.utc(2026, 8, 31).tt)[0]
    result = ec.analyse(ephem, event, samples=12)
    payload = json.loads(output.dumps(output.to_json([result], ts, {'x': 1})))
    entry = payload['eclipses'][0]
    assert entry['type'] == 'T'
    assert entry['date'] == '2026-08-12'
    assert entry['band']['label'] == 'totality'
    assert len(entry['band']['points']) == len(result.coverage_band.points)


def test_duration_formatting():
    assert output.format_duration(138.1) == '2m18.1s'
    assert output.format_duration(0) == '-'
    assert output.format_duration(627.4) == '10m27.4s'


# --- search API -------------------------------------------------------------

def test_search_filters_and_orders():
    results = catalog.search('2026-01-01', years=2, threshold=1.0, samples=6,
                             ephemeris=ephemeris())
    dates = [r.tt_greatest for r in results]
    assert dates == sorted(dates)
    assert [r.kind for r in results] == ['A', 'T', 'A', 'T']
    assert [r.meets_threshold for r in results] == [False, True, False, True]


def test_search_rejects_range_outside_the_ephemeris():
    try:
        catalog.search('2400-01-01', years=1, ephemeris=ephemeris())
    except ValueError as exc:
        assert 'ephemeris' in str(exc)
    else:
        raise AssertionError('expected a ValueError')


def test_search_rejects_backwards_range():
    try:
        catalog.search('2030-01-01', end='2029-01-01', ephemeris=ephemeris())
    except ValueError as exc:
        assert 'after' in str(exc)
    else:
        raise AssertionError('expected a ValueError')


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
