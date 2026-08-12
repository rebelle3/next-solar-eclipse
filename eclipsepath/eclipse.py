"""Full analysis of a single solar eclipse: circumstances, path and limits.

The central idea is that everything is expressed as "maximum obscuration seen
over the whole event", evaluated directly from vector geometry.  The visible
region for a coverage threshold X is then

    G(X) = { p on Earth : max_t obscuration(p, t) >= X }

and its outline is traced by bisecting outwards from a spine that follows the
darkest point on the globe from first to last contact.  For X = 1 the spine is
the classic central line and G is the path of totality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import circumstances as cc
from . import geometry as g
from .ephemeris import EclipseWindow
from .finder import RawEvent, axis_metrics

WINDOW_PAD_DAYS = 6.0 / 1440.0
SEARCH_LIMIT_KM = 12000.0
NEW_MOON_EPOCH_TT = 2451550.09766   # 2000 Jan 6 new moon, TT Julian day
SYNODIC_MONTH = 29.530588861


@dataclass
class PathPoint:
    """One sample of the shadow's track: the darkest point on the globe."""
    tt: float
    latitude: float
    longitude: float
    sun_altitude: float
    obscuration: float
    magnitude: float
    central: bool


@dataclass
class BandPoint:
    """One cross-section of a traced band, perpendicular to the track."""
    tt: float
    latitude: float
    longitude: float
    sun_altitude: float
    obscuration: float
    magnitude: float
    north_latitude: float
    north_longitude: float
    south_latitude: float
    south_longitude: float
    width_km: float
    duration_seconds: float
    tt_start: float
    tt_end: float
    transverse: bool = True


@dataclass
class Band:
    """A strip of the Earth's surface where some depth criterion is met."""
    label: str
    points: List[BandPoint] = field(default_factory=list)
    truncated: bool = False

    def __bool__(self):
        return bool(self.points)


@dataclass
class Eclipse:
    tt_greatest: float
    tt_first_contact: float
    tt_last_contact: float
    kind: str
    gamma: float
    magnitude: float
    obscuration: float
    latitude: float
    longitude: float
    sun_altitude: float
    central: bool
    lunation: int
    saros: int
    threshold: float
    path: List[PathPoint] = field(default_factory=list)
    central_band: Optional[Band] = None
    coverage_band: Optional[Band] = None
    central_duration_seconds: float = 0.0
    path_width_km: float = float('nan')
    peak_obscuration: float = 0.0
    tt_central_start: float = float('nan')
    tt_central_end: float = float('nan')
    window: Optional[EclipseWindow] = None

    @property
    def meets_threshold(self):
        return self.peak_obscuration >= self.threshold - 1e-9


def lunation_number(tt):
    return int(round((tt - NEW_MOON_EPOCH_TT) / SYNODIC_MONTH))


def saros_number(lunation):
    """Saros series of a solar eclipse from its lunation number.

    Successive members of a series are 223 lunations apart and the series
    numbering advances by 38 per lunation, modulo 223.
    """
    return int((38 * lunation + 112) % 223)


def _north_vector(window, tt):
    """True celestial north (the ITRS z-axis) expressed in ICRF."""
    rot = window.rotation(np.atleast_1d(tt))
    return rot[2, :, ...]


def shadow_point(window, tt):
    """Darkest point on the globe at ``tt`` plus whether the axis reaches it.

    Returns ``(itrf_xyz, central)``.  When the axis pierces the ellipsoid that
    intersection is the answer; otherwise it is the surface point nearest the
    axis, which is where the deepest partial eclipse is seen.
    """
    tt = np.atleast_1d(np.asarray(tt, float))
    sun, moon, rot = window.at(tt)
    moon_i = g.rotate_to_itrf(rot, moon)
    sun_i = g.rotate_to_itrf(rot, sun)
    direction = moon_i - sun_i
    point, hit = g.line_ellipsoid_intersection(moon_i, direction)
    fallback = g.closest_surface_point_to_axis(moon_i, direction)
    return np.where(hit, point, fallback), hit


def umbra_sign(window, tt, itrf_point):
    """Positive where the umbra reaches the ground, negative in the antumbra."""
    tt = np.atleast_1d(np.asarray(tt, float))
    sun, moon, rot = window.at(tt)
    site = g.rotate_to_icrf(g.align(rot, 2, itrf_point.ndim - 1),
                            itrf_point)
    s_km = g.norm(g.align(moon, 1, itrf_point.ndim - 1) - site)
    return g.umbra_radius(sun, moon, s_km)


def analyse(ephem, event: RawEvent, threshold=1.0, samples=120,
            time_samples=601, trace_limits=True):
    """Turn a bracketed event into a fully described :class:`Eclipse`."""
    window = EclipseWindow(ephem,
                           event.tt_first_contact - WINDOW_PAD_DAYS,
                           event.tt_last_contact + WINDOW_PAD_DAYS)
    grid = np.linspace(event.tt_first_contact, event.tt_last_contact, time_samples)

    # --- circumstances at greatest eclipse ---------------------------------
    tg = np.array([event.tt_greatest])
    sun, moon, _ = window.at(tg)
    north = _north_vector(window, tg)
    _, gamma, _ = axis_metrics(sun, moon, north)
    point, hit = shadow_point(window, tg)
    if not bool(hit[0]):
        point = _refine_closest_point(window, tg, point)
    lat, lon, _ = g.itrf_to_geodetic(point)
    state = cc.state_at(window, point, tg)

    kind = _classify(window, grid, event, state, bool(hit[0]))
    lun = lunation_number(event.tt_greatest)

    eclipse = Eclipse(
        tt_greatest=event.tt_greatest,
        tt_first_contact=event.tt_first_contact,
        tt_last_contact=event.tt_last_contact,
        kind=kind,
        gamma=float(gamma[0]),
        magnitude=float(state['magnitude'][0]),
        obscuration=float(state['obscuration'][0]),
        latitude=float(lat[0]),
        longitude=float(lon[0]),
        sun_altitude=float(state['sun_altitude'][0]),
        central=bool(hit[0]),
        lunation=lun,
        saros=saros_number(lun),
        threshold=threshold,
        window=window,
    )

    if trace_limits:
        _build_path(eclipse, window, grid, threshold, samples)
    return eclipse


def _classify(window, grid, event, greatest_state, central):
    """T / A / H / P following the usual catalogue convention."""
    point, hit = shadow_point(window, grid)
    if not hit.any():
        mag = cc.state_at(window, point, grid)['magnitude']
        return 'T' if np.nanmax(mag) >= 1.0 else 'P'
    signs = umbra_sign(window, grid, point)[hit]
    if (signs > 0).any() and (signs < 0).any():
        return 'H'
    return 'T' if float(np.median(signs)) > 0 else 'A'


def _pattern_search(score, lat, lon, iterations=60, step=2.0, floor=1e-7):
    """Maximise ``score(lat, lon)`` by a shrinking-step search on the surface.

    Used to place the darkest point when the shadow axis misses the Earth, so
    the track stays defined right through a grazing partial eclipse.  Vectorised
    over sites: every candidate site takes its own independent walk.
    """
    lat = np.array(lat, float, copy=True)
    lon = np.array(lon, float, copy=True)
    best = score(lat, lon)
    for _ in range(iterations):
        improved = False
        for d_lat, d_lon in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            trial_lat = np.clip(lat + d_lat, -90.0, 90.0)
            trial_lon = lon + d_lon
            value = score(trial_lat, trial_lon)
            take = value > best
            lat = np.where(take, trial_lat, lat)
            lon = np.where(take, trial_lon, lon)
            best = np.where(take, value, best)
            improved = improved or bool(np.any(take))
        if not improved:
            step *= 0.5
            if step < floor:
                break
    return g.geodetic_to_itrf(lat, (lon + 540.0) % 360.0 - 180.0)


def _refine_closest_point(window, tt, start_point):
    """Surface point nearest the shadow axis (how catalogues place a partial)."""
    lat, lon, _ = g.itrf_to_geodetic(start_point)
    sun, moon, rot = window.at(tt)
    moon_itrf = g.rotate_to_itrf(rot, moon)
    axis = g.unit(moon_itrf - g.rotate_to_itrf(rot, sun))

    def score(trial_lat, trial_lon):
        offset = g.geodetic_to_itrf(trial_lat, trial_lon) - moon_itrf
        return -g.norm(offset - axis * g.dot(offset, axis))

    return _pattern_search(score, lat, lon)


def _refine_max_obscuration(window, tt, start_point):
    """Surface point seeing the deepest eclipse at this instant."""
    lat, lon, _ = g.itrf_to_geodetic(start_point)

    def score(trial_lat, trial_lon):
        state = cc.state_at(window, g.geodetic_to_itrf(trial_lat, trial_lon), tt)
        return np.where(state['sun_altitude'] >= 0.0, state['obscuration'], -1.0)

    return _pattern_search(score, lat, lon, floor=1e-6)


# --- path tracing -----------------------------------------------------------

PREDICATE_TIME_SAMPLES = 181


def _track(window, tt):
    """Darkest point on the globe at each of ``tt``, and whether it is central."""
    point, hit = shadow_point(window, tt)
    missed = ~hit
    if missed.any():
        point[:, missed] = _refine_max_obscuration(window, tt[missed],
                                                   point[:, missed])
    return point, hit


def umbral_contacts(window, tt_lo, tt_hi, steps=1200):
    """When the shadow axis first and last touches the globe.

    Sampling the central path between these instants keeps every centre-line
    point genuinely central: outside them the axis misses the Earth entirely,
    and the darkest point on the globe is only seeing a partial eclipse.
    """
    tt = np.linspace(tt_lo, tt_hi, steps)
    _, hit = shadow_point(window, tt)
    if not hit.any():
        return None
    first = int(np.argmax(hit))
    last = len(hit) - 1 - int(np.argmax(hit[::-1]))

    def touches(x):
        return shadow_point(window, np.atleast_1d(x))[1]

    def edge(outside, inside):
        for _ in range(50):
            middle = 0.5 * (outside + inside)
            if touches(middle)[0]:
                inside = middle
            else:
                outside = middle
        return inside

    start = tt[0] if first == 0 else edge(tt[first - 1], tt[first])
    end = tt[-1] if last == len(tt) - 1 else edge(tt[last + 1], tt[last])
    return float(start), float(end)


def _build_path(eclipse, window, grid, threshold, samples):
    tt = np.linspace(eclipse.tt_first_contact, eclipse.tt_last_contact, samples)
    point, hit = _track(window, tt)
    lat, lon, _ = g.itrf_to_geodetic(point)
    state = cc.state_at(window, point, tt)
    eclipse.path = [
        PathPoint(tt=float(tt[i]), latitude=float(lat[i]), longitude=float(lon[i]),
                  sun_altitude=float(state['sun_altitude'][i]),
                  obscuration=float(state['obscuration'][i]),
                  magnitude=float(state['magnitude'][i]),
                  central=bool(hit[i]))
        for i in range(len(tt))]

    coarse = np.linspace(grid[0], grid[-1], PREDICATE_TIME_SAMPLES)
    eclipse.peak_obscuration = float(
        cc.peak_eclipse(window, point, coarse)['obscuration'].max())

    umbral = umbral_contacts(window, eclipse.tt_first_contact,
                             eclipse.tt_last_contact) if hit.any() else None
    if umbral:
        eclipse.tt_central_start, eclipse.tt_central_end = umbral
        central_tt = np.linspace(umbral[0], umbral[1], samples)
        central_point, _ = shadow_point(window, central_tt)
        central_lat, central_lon, _ = g.itrf_to_geodetic(central_point)
        eclipse.central_band = _trace_band(
            window, coarse, central_tt, central_lat, central_lon,
            cc.central_depth, _central_label(eclipse))

    if threshold > 0.0:
        same = (eclipse.kind in ('T', 'H') and threshold >= 1.0
                and eclipse.central_band is not None)
        eclipse.coverage_band = (
            eclipse.central_band if same else
            _trace_band(window, coarse, tt, lat, lon,
                        cc.obscuration_depth(threshold),
                        '%g%% obscuration' % (threshold * 100)))

    _fill_greatest_metrics(eclipse, window, coarse)


def _central_label(eclipse):
    return {'T': 'totality', 'A': 'annularity',
            'H': 'totality/annularity'}.get(eclipse.kind, 'central')


def _trace_band(window, coarse, tt, lat, lon, depth, label, iterations=36):
    """Trace the edges of the region where ``depth`` is met, per track sample."""
    n = len(tt)
    heading = _spine_heading(window, tt)
    inside = _meets(window, coarse, lat, lon, depth)
    if not inside.any():
        return Band(label=label)

    edges, capped = {}, np.zeros(n, bool)
    for side, offset in (('north', -90.0), ('south', +90.0)):
        bearing = (heading + offset) % 360.0
        lo = np.zeros(n)
        hi = np.full(n, SEARCH_LIMIT_KM)
        far_lat, far_lon = g.great_circle_destination(lat, lon, bearing, hi)
        capped |= _meets(window, coarse, far_lat, far_lon, depth)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            m_lat, m_lon = g.great_circle_destination(lat, lon, bearing, mid)
            ok = _meets(window, coarse, m_lat, m_lon, depth)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        distance = 0.5 * (lo + hi)
        edges[side] = g.great_circle_destination(lat, lon, bearing, distance)

    north_lat, north_lon = edges['north']
    south_lat, south_lon = edges['south']
    transverse = _cross_section_is_transverse(lat, lon, north_lat, north_lon,
                                              south_lat, south_lon)
    width = g.geodesic_distance(north_lat, north_lon, south_lat, south_lon)
    width = np.where(capped | ~transverse, np.nan, width)
    duration, t_start, t_end = _depth_duration(window, coarse, lat, lon, depth)
    state = cc.state_at(window, g.geodetic_to_itrf(lat, lon), tt)

    band = Band(label=label, truncated=bool(capped.any()))
    band.points = [
        BandPoint(tt=float(tt[i]), latitude=float(lat[i]), longitude=float(lon[i]),
                  sun_altitude=float(state['sun_altitude'][i]),
                  obscuration=float(state['obscuration'][i]),
                  magnitude=float(state['magnitude'][i]),
                  north_latitude=float(north_lat[i]),
                  north_longitude=float(north_lon[i]),
                  south_latitude=float(south_lat[i]),
                  south_longitude=float(south_lon[i]),
                  width_km=float(width[i]), duration_seconds=float(duration[i]),
                  tt_start=float(t_start[i]), tt_end=float(t_end[i]),
                  transverse=bool(transverse[i]))
        for i in range(n) if inside[i]]
    return band


BALANCE_LIMIT = 0.25


def _cross_section_is_transverse(lat, lon, north_lat, north_lon,
                                 south_lat, south_lon):
    """Is this cut genuinely across the band rather than along it?

    Where a track meets the terminator the perpendicular to it runs nearly
    along the edge of the band instead of across it, and the two limits land
    wildly unequal distances from the centre.  The width of such a cut is not
    a path width, so it is reported as unknown rather than as a number several
    times too large.  The same happens on the sunrise/sunset edge of a wide
    low-threshold band, which is bounded by the terminator on one side.
    """
    to_north = g.geodesic_distance(lat, lon, north_lat, north_lon)
    to_south = g.geodesic_distance(lat, lon, south_lat, south_lon)
    smaller = np.minimum(to_north, to_south)
    larger = np.maximum(to_north, to_south)
    return smaller >= BALANCE_LIMIT * np.maximum(larger, 1e-9)


def _meets(window, coarse, lat, lon, depth):
    xyz = g.geodetic_to_itrf(lat, lon)
    return cc.peak_eclipse(window, xyz, coarse, depth=depth)['depth'] >= 0.0


HEADING_DELTA_DAYS = 2.0 / 86400.0


def _spine_heading(window, tt):
    """Local direction of shadow travel, from a short central difference.

    Differencing neighbouring path samples is not good enough where the track
    curves sharply (a path passing close to a pole can swing tens of degrees in
    a couple of minutes), which would tilt the perpendicular used to measure
    the path width.  A two-second baseline follows the actual shadow velocity.
    """
    before, _ = shadow_point(window, tt - HEADING_DELTA_DAYS)
    after, _ = shadow_point(window, tt + HEADING_DELTA_DAYS)
    lat_a, lon_a, _ = g.itrf_to_geodetic(before)
    lat_b, lon_b, _ = g.itrf_to_geodetic(after)
    return g.initial_bearing(lat_a, lon_a, lat_b, lon_b)


def _depth_duration(window, coarse, lat, lon, depth, iterations=36):
    """How long the criterion holds at each site, and between which instants."""
    xyz = g.geodetic_to_itrf(lat, lon)
    peak = cc.peak_eclipse(window, xyz, coarse, depth=depth)
    reaches = peak['depth'] >= 0.0
    t_peak = peak['t']

    def holds(t):
        state = cc.state_at(window, xyz, t)
        return (depth(state) >= 0.0) & (state['sun_altitude'] >= 0.0)

    bounds = []
    for outer_bound in (coarse[0], coarse[-1]):
        outer = np.full(len(np.atleast_1d(lat)), outer_bound)
        inner = t_peak.copy()
        for _ in range(iterations):
            mid = 0.5 * (outer + inner)
            ok = holds(mid)
            inner = np.where(ok, mid, inner)
            outer = np.where(ok, outer, mid)
        bounds.append(0.5 * (outer + inner))
    start, end = bounds
    duration = np.where(reaches, (end - start) * 86400.0, 0.0)
    return duration, np.where(reaches, start, np.nan), np.where(reaches, end, np.nan)


def _fill_greatest_metrics(eclipse, window, coarse):
    """Path width and central duration at greatest eclipse, as catalogues quote."""
    tt = np.array([eclipse.tt_greatest])
    point, hit = shadow_point(window, tt)
    if not bool(hit[0]):
        return
    lat, lon, _ = g.itrf_to_geodetic(point)
    band = _trace_band(window, coarse, tt, lat, lon, cc.central_depth, 'greatest')
    if band.points:
        eclipse.path_width_km = band.points[0].width_km
    contacts = cc.contact_times(window, point[:, 0], coarse)
    eclipse.central_duration_seconds = contacts['central_seconds']
