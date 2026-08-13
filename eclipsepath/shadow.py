"""The Moon's shadow on the Earth at one instant.

The rest of the package cuts the geometry along the shadow's track: where the
centre is at each moment, how wide the band is there, what one site sees over
the whole event.  A picture of an eclipse wants the perpendicular cut through
the same geometry -- the whole world at a single instant -- and nothing here
produced that.

Everything below is that cut.  The field being drawn is the fraction of the
Sun's disc covered, evaluated wherever it is asked for by the same
:func:`eclipsepath.circumstances.state_at` the rest of the package uses, so a
frame of the animation and a row of the path table cannot disagree.

Contours come from marching squares over a grid, then each vertex is placed
exactly by bisection against the real field.  The grid decides how many
wiggles the outline can have; the bisection decides where they sit.  That
split matters, because a grid coarse enough to be cheap is nowhere near fine
enough to put a contour within a kilometre on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import circumstances as cc
from . import geometry as g

# Corner bits of a marching-squares cell, anticlockwise from the lower left,
# and the edges between them.  ``EDGES[case]`` lists the segments to emit as
# pairs of edge numbers; the two ambiguous cases are resolved from the value at
# the cell's centre and appear in ``SADDLE`` instead.
EDGES = {
    0: (), 15: (),
    1: ((0, 3),), 14: ((0, 3),),
    2: ((0, 1),), 13: ((0, 1),),
    3: ((1, 3),), 12: ((1, 3),),
    4: ((1, 2),), 11: ((1, 2),),
    6: ((0, 2),), 9: ((0, 2),),
    7: ((2, 3),), 8: ((2, 3),),
}
SADDLE = {
    # (case, centre is inside) -> segments
    (5, True): ((0, 1), (2, 3)), (5, False): ((0, 3), (1, 2)),
    (10, True): ((0, 3), (1, 2)), (10, False): ((0, 1), (2, 3)),
}


@dataclass
class Grid:
    """The coverage field over the whole globe at one instant.

    ``obscuration`` is masked by visibility and ``raw`` is not; see
    :func:`visible_obscuration` for why the distinction is worth keeping.
    Longitudes run from -180 up to but not including +180: the seam is a real
    edge of the grid, closed by wrapping rather than by a duplicate column.
    """
    tt: float
    latitudes: np.ndarray
    longitudes: np.ndarray
    obscuration: np.ndarray
    raw: np.ndarray
    sun_altitude: np.ndarray

    @property
    def step(self):
        return float(self.longitudes[1] - self.longitudes[0])

    @property
    def peak(self):
        """(obscuration, latitude, longitude) at a deepest visible grid point.

        *A* deepest, not *the* deepest.  Through totality the field is a flat 1
        across the whole umbra, and through annularity flat at whatever the
        discs allow, so the location this returns is an arbitrary point inside
        that plateau -- which is the same reason the time of maximum is
        undefined on the central line.  The depth is meaningful; when the place
        is wanted, take it from :func:`eclipsepath.eclipse.shadow_point`.
        """
        flat = int(np.argmax(self.obscuration))
        i, j = np.unravel_index(flat, self.obscuration.shape)
        return (float(self.obscuration[i, j]), float(self.latitudes[i]),
                float(self.longitudes[j]))


def visible_obscuration(window, tt, latitude, longitude):
    """Fraction of the Sun covered where it is up, and zero where it is not.

    The mask is not decoration.  The geometry will happily report the Moon
    squarely over the Sun at a place where the Sun is below the horizon the
    whole time, and drawing that would claim an eclipse nobody can see;
    :func:`eclipsepath.observer.circumstances_at` returns ``None`` for exactly
    the same reason.  Zero rather than NaN, so that an outline closes along the
    terminator instead of running off the end of the world.
    """
    lat = np.asarray(latitude, float)
    lon = np.asarray(longitude, float)
    state = cc.state_at(window, g.geodetic_to_itrf(lat, lon), tt)
    up = state['sun_altitude'] > g.HORIZON_ALTITUDE_DEG
    return np.where(up, state['obscuration'], 0.0)


def obscuration_grid(window, tt, step=1.0):
    """Sample the coverage field over the globe on a regular grid."""
    latitudes = np.linspace(-90.0, 90.0, int(round(180.0 / step)) + 1)
    longitudes = np.arange(-180.0, 180.0, step)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    shape = lat_mesh.shape
    state = cc.state_at(window,
                        g.geodetic_to_itrf(lat_mesh.ravel(), lon_mesh.ravel()),
                        tt)
    raw = state['obscuration'].reshape(shape)
    altitude = state['sun_altitude'].reshape(shape)
    masked = np.where(altitude > g.HORIZON_ALTITUDE_DEG, raw, 0.0)
    return Grid(float(tt), latitudes, longitudes, masked, raw, altitude)


def _cell_segments(field, level):
    """Marching squares over a longitude-periodic field.

    Returns segments as pairs of *edge identities* rather than as coordinates.
    An edge is shared by exactly the two cells either side of it, so two cells
    that meet there produce the same identity and the segments chain together
    without any tolerance on coordinates.
    """
    inside = field >= level
    # Wrapping the last column onto the first closes the seam at the
    # antimeridian, which otherwise cuts every outline that crosses it in two.
    a = inside[:-1, :]
    b = np.roll(inside, -1, axis=1)[:-1, :]
    c = np.roll(inside, -1, axis=1)[1:, :]
    d = inside[1:, :]
    case = a + 2 * b + 4 * c + 8 * d
    values = (field[:-1, :], np.roll(field, -1, axis=1)[:-1, :],
              np.roll(field, -1, axis=1)[1:, :], field[1:, :])
    centre = sum(values) / 4.0

    columns = field.shape[1]
    segments = []
    for code in range(1, 15):
        rows, cols = np.nonzero(case == code)
        if not rows.size:
            continue
        for i, j in zip(rows.tolist(), cols.tolist()):
            if code in (5, 10):
                pairs = SADDLE[(code, bool(centre[i, j] >= level))]
            else:
                pairs = EDGES[code]
            segments.extend((_edge_identity(i, j, p, columns),
                             _edge_identity(i, j, q, columns))
                            for p, q in pairs)
    return segments


def _edge_identity(i, j, edge, columns):
    """Canonical name for a cell edge, so neighbouring cells agree on it.

    ``('h', i, j)`` runs east along latitude row ``i`` from column ``j``;
    ``('v', i, j)`` runs north up column ``j`` from row ``i``.
    """
    if edge == 0:
        return ('h', i, j)
    if edge == 1:
        return ('v', i, (j + 1) % columns)
    if edge == 2:
        return ('h', i + 1, j)
    return ('v', i, j)


def _edge_point(identity, field, latitudes, longitudes, level):
    """Where the level crosses an edge, by linear interpolation on the grid."""
    kind, i, j = identity
    columns = longitudes.size
    step = float(longitudes[1] - longitudes[0])
    if kind == 'h':
        lo, hi = field[i, j], field[i, (j + 1) % columns]
        t = _fraction(lo, hi, level)
        return float(latitudes[i]), float(longitudes[j] + t * step)
    lo, hi = field[i, j], field[i + 1, j]
    t = _fraction(lo, hi, level)
    return (float(latitudes[i] + t * (latitudes[i + 1] - latitudes[i])),
            float(longitudes[j]))


def _fraction(lo, hi, level):
    if hi == lo:
        return 0.5
    return float(np.clip((level - lo) / (hi - lo), 0.0, 1.0))


def _chain(segments):
    """Join segments end to end into polylines, closed ones marked as such.

    Each edge identity is shared by at most two cells, so the segment graph has
    maximum degree two and the walk is unambiguous.
    """
    links = {}
    for p, q in segments:
        links.setdefault(p, []).append(q)
        links.setdefault(q, []).append(p)
    seen_edges = set()
    chains = []

    def walk(start, first):
        chain = [start]
        previous, current = start, first
        while True:
            chain.append(current)
            options = [n for n in links.get(current, []) if n != previous]
            if not options:
                return chain, False
            previous, current = current, options[0]
            if current == start:
                return chain, True

    # Open chains first, so that a loop broken by a grid edge is not walked
    # from its middle and left looking like two separate pieces.
    ends = [k for k, v in links.items() if len(v) == 1]
    for start in ends:
        if start in seen_edges:
            continue
        chain, closed = walk(start, links[start][0])
        seen_edges.update(chain)
        chains.append((chain, closed))
    for start in links:
        if start in seen_edges:
            continue
        chain, closed = walk(start, links[start][0])
        seen_edges.update(chain)
        chains.append((chain, closed))
    return chains


def _unwrap(longitudes):
    """Make a run of longitudes continuous, so a seam crossing does not streak."""
    out = list(longitudes[:1])
    for lon in longitudes[1:]:
        out.append(lon - 360.0 * round((lon - out[-1]) / 360.0))
    return out


def _refine(window, tt, points, level, span_km=200.0, iterations=26):
    """Place each vertex exactly, by bisecting the real field across the edge.

    Marching squares can only put a vertex where a straight line between two
    grid samples crosses the level, which on a one-degree grid is tens of
    kilometres out.  Bisecting the true field along the same direction fixes
    that, and costs one vectorised evaluation per iteration for the whole
    outline at once.

    Bisection does not need the field to be continuous, only to change sign,
    which is what makes this work at the terminator too: there the field jumps
    from zero to whatever the eclipse is doing, and the bisection converges on
    the jump, i.e. on sunrise or sunset.
    """
    if not points:
        return points
    lat = np.array([p[0] for p in points])
    lon = np.array([p[1] for p in points])
    # Search across the outline, which is the direction the value changes
    # fastest: along the local gradient of the field.
    delta = span_km / 111.32
    east = visible_obscuration(window, tt, lat, lon + delta / np.maximum(
        np.cos(np.radians(lat)), 1e-6))
    west = visible_obscuration(window, tt, lat, lon - delta / np.maximum(
        np.cos(np.radians(lat)), 1e-6))
    north = visible_obscuration(window, tt, np.clip(lat + delta, -90.0, 90.0), lon)
    south = visible_obscuration(window, tt, np.clip(lat - delta, -90.0, 90.0), lon)
    gy, gx = north - south, east - west
    scale = np.hypot(gx, gy)
    flat = scale < 1e-12
    gx = np.where(flat, 1.0, gx / np.where(flat, 1.0, scale))
    gy = np.where(flat, 0.0, gy / np.where(flat, 1.0, scale))

    def sample(t):
        step_lat = np.clip(lat + t * gy * delta, -90.0, 90.0)
        step_lon = lon + t * gx * delta / np.maximum(
            np.cos(np.radians(lat)), 1e-6)
        return visible_obscuration(window, tt, step_lat, step_lon) - level

    lo, hi = np.full(lat.shape, -1.0), np.full(lat.shape, 1.0)
    f_lo, f_hi = sample(lo), sample(hi)
    usable = np.sign(f_lo) != np.sign(f_hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = sample(mid)
        left = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(left, mid, lo)
        f_lo = np.where(left, f_mid, f_lo)
        hi = np.where(left, hi, mid)
    t = np.where(usable, 0.5 * (lo + hi), 0.0)
    lat_out = np.clip(lat + t * gy * delta, -90.0, 90.0)
    lon_out = lon + t * gx * delta / np.maximum(np.cos(np.radians(lat)), 1e-6)
    return list(zip(lat_out.tolist(), lon_out.tolist()))


def contours(window, grid, levels, refine=True, min_points=3):
    """Outlines of the coverage field, one list of polylines per level.

    Each polyline is ``(latitudes, longitudes, closed)`` with longitudes made
    continuous, so a caller can draw it without deciding what to do at the
    antimeridian.
    """
    result = {}
    for level in levels:
        field = grid.obscuration
        segments = _cell_segments(field, level)
        lines = []
        for chain, closed in _chain(segments):
            if len(chain) < min_points:
                continue
            points = [_edge_point(e, field, grid.latitudes, grid.longitudes,
                                  level) for e in chain]
            if refine:
                points = _refine(window, grid.tt, points, level,
                                 span_km=1.5 * grid.step * 111.32)
            lat = [p[0] for p in points]
            lon = _unwrap([p[1] for p in points])
            if closed:
                lat.append(lat[0])
                lon.append(lon[0] - 360.0 * round((lon[0] - lon[-1]) / 360.0))
            lines.append((np.array(lat), np.array(lon), closed))
        result[level] = lines
    return result


def umbra_outline(window, tt, rays=180, reach_km=None, iterations=44):
    """The edge of the umbra (or antumbra) on the ground, as a closed curve.

    Not contoured.  The umbra is a hundred-odd kilometres across, so a grid
    fine enough to round it properly would be far finer than anything the rest
    of the picture needs.  It is convex and small, which is exactly the case
    where casting rays out from the centre is both cheap and smooth -- the
    opposite of the wide coverage regions, where rays are the wrong tool.

    The footprint is the shadow cone cut by the lit half of the globe, not the
    cone alone.  Near the beginning and end of an eclipse the cone crosses the
    limb, and the part beyond it lands where the Sun has not yet risen: at the
    2027 eclipse's first contact that is a third of the outline, reaching ten
    degrees into the night side.  Drawing it would put the Moon's shadow on
    ground the Sun was not lighting.

    This is the shadow *now*, which is not the band of totality.  The band's
    northern and southern limits are the envelope of every umbra the eclipse
    casts, so they lie a little outside this curve and touch it -- at the
    grazing incidence near sunrise, several kilometres outside.

    Returns ``(latitudes, longitudes)`` closed on itself, or ``None`` where the
    axis misses the Earth and there is no umbra to draw.
    """
    from . import eclipse as ec
    point, hit = ec.shadow_point(window, np.atleast_1d(float(tt)))
    if not bool(hit[0]):
        return None
    lat0, lon0, _ = g.itrf_to_geodetic(point)
    lat0, lon0 = float(lat0[0]), float(lon0[0])

    def inside(lat, lon):
        state = cc.state_at(window, g.geodetic_to_itrf(lat, lon), float(tt))
        return ((cc.central_depth(state) > 0.0)
                & (state['sun_altitude'] > g.HORIZON_ALTITUDE_DEG))

    if not inside(np.array([lat0]), np.array([lon0]))[0]:
        return None                      # a central eclipse whose umbra misses
    azimuth = np.linspace(0.0, 360.0, rays, endpoint=False)
    lo = np.zeros(rays)
    # Out to the antipode, where the Sun is certainly down given that it is up
    # at the centre.  A ray from a lit point to its antipode crosses into night
    # once and stays there, so there is exactly one sign change to find and no
    # reach to choose badly.
    hi = np.full(rays, reach_km if reach_km else np.pi * g.EARTH_A_KM)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lat, lon = g.great_circle_destination(lat0, lon0, azimuth, mid)
        within = inside(lat, lon)
        lo = np.where(within, mid, lo)
        hi = np.where(within, hi, mid)
    lat, lon = g.great_circle_destination(lat0, lon0, azimuth,
                                          0.5 * (lo + hi))
    lon = np.array(_unwrap(lon.tolist()))
    return (np.append(lat, lat[0]),
            np.append(lon, lon[0] - 360.0 * round((lon[0] - lon[-1]) / 360.0)))


def subsolar_point(window, tt):
    """Where the Sun is straight up, in degrees."""
    sun, _, rot = window.at(np.atleast_1d(float(tt)))
    xyz = g.rotate_to_itrf(rot, sun)[:, 0]
    direction = xyz / g.norm(xyz)
    return (float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(direction[1], direction[0]))))


def terminator(window, tt, points=361, iterations=40):
    """The sunrise/sunset line: everywhere the Sun sits on the apparent horizon.

    A circle of fixed angular radius about the subsolar point would be within
    about a dozen kilometres, the Earth not being round; bisecting the real
    altitude along each ray costs no more and is right instead.  The altitude
    falls away steadily from the subsolar point, so one crossing per ray.
    """
    lat0, lon0 = subsolar_point(window, tt)
    azimuth = np.linspace(0.0, 360.0, points)
    lo = np.zeros(points)
    hi = np.full(points, np.pi * g.EARTH_A_KM)      # to the antipode
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lat, lon = g.great_circle_destination(lat0, lon0, azimuth, mid)
        state = cc.state_at(window, g.geodetic_to_itrf(lat, lon), float(tt))
        up = state['sun_altitude'] > g.HORIZON_ALTITUDE_DEG
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    lat, lon = g.great_circle_destination(lat0, lon0, azimuth, 0.5 * (lo + hi))
    return lat, np.array(_unwrap(lon.tolist()))
