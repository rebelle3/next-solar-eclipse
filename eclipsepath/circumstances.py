"""Local circumstances: what an observer at a given place actually sees.

Everything is vectorised over ``P`` observing sites and ``T`` sample times so
that path-limit searches (which are bisections over hundreds of trial sites)
stay fast.  Observer positions are passed in as ITRF vectors because a site is
fixed in ITRF while the Sun and Moon move in ICRF.
"""

from __future__ import annotations

import numpy as np

from . import geometry as g

GOLDEN = 0.6180339887498949


def observe(window, tt, itrf_xyz):
    """Apparent Sun/Moon geometry seen from ``itrf_xyz`` at times ``tt``.

    ``itrf_xyz`` is ``(3, ...)`` and ``tt`` broadcasts against its trailing
    dimensions.  Returns ``(r_sun, r_moon, separation, sun_altitude_deg)``.
    """
    sun, moon, rot = window.at(tt)
    ndim = max(itrf_xyz.ndim - 1, np.ndim(tt))
    sun = g.align(sun, 1, ndim)
    moon = g.align(moon, 1, ndim)
    rot = g.align(rot, 2, ndim)
    itrf_xyz = g.align(itrf_xyz, 1, ndim)
    site = g.rotate_to_icrf(rot, itrf_xyz)
    v_sun = sun - site
    v_moon = moon - site
    r_sun = g.angular_radius(g.norm(v_sun), g.R_SUN_KM)
    r_moon = g.angular_radius(g.norm(v_moon), _lunar_radius(v_sun, v_moon))
    sep = g.separation(v_sun, v_moon)
    up = g.rotate_to_icrf(rot, g.unit(_zenith_of(itrf_xyz)))
    alt = 90.0 - np.degrees(g.separation(v_sun, up))
    return r_sun, r_moon, sep, alt


def _lunar_radius(v_sun, v_moon):
    """The Moon's radius toward the Sun, from a limb profile if one is set.

    What decides whether the Sun is covered is the height of the lunar horizon
    on the side the Sun is peeping out from, so the profile is sampled at the
    position angle of the Sun as seen from the Moon's centre.  With no profile
    set this is just the constant radius and the Moon stays a circle.
    """
    if g.LIMB_PROFILE is None:
        return g.R_MOON_KM
    line_of_sight = g.unit(v_moon)
    pole = np.zeros_like(line_of_sight)
    pole[2] = 1.0
    north = g.unit(pole - line_of_sight * g.dot(pole, line_of_sight))
    east = np.cross(line_of_sight, north, axis=0)
    offset = g.unit(v_sun) - line_of_sight
    angle = np.degrees(np.arctan2(g.dot(offset, east), g.dot(offset, north)))
    return g.LIMB_PROFILE.radius_at(angle)


def _zenith_of(itrf_xyz):
    """Geodetic zenith for an ITRF surface point (differs from the radius vector)."""
    scale = np.array([1.0, 1.0, 1.0 / (1.0 - g.EARTH_E2)]).reshape(
        (3,) + (1,) * (itrf_xyz.ndim - 1))
    return itrf_xyz * scale


def _separation_at(window, itrf_col, tt):
    """Sun-Moon separation for sites ``(3, P, 1)`` at per-site times ``(P,)``."""
    sun, moon, rot = window.at(tt)
    site = g.rotate_to_icrf(rot, itrf_col[:, :, 0])
    return g.separation(sun - site, moon - site)


def _golden_min(func, lo, hi, iterations=32):
    """Vectorised golden-section minimisation of a unimodal ``func``."""
    lo = np.array(lo, float, copy=True)
    hi = np.array(hi, float, copy=True)
    x1 = hi - GOLDEN * (hi - lo)
    x2 = lo + GOLDEN * (hi - lo)
    f1, f2 = func(x1), func(x2)
    for _ in range(iterations):
        take_left = f1 < f2
        hi = np.where(take_left, x2, hi)
        lo = np.where(take_left, lo, x1)
        x1 = hi - GOLDEN * (hi - lo)
        x2 = lo + GOLDEN * (hi - lo)
        f1, f2 = func(x1), func(x2)
    return 0.5 * (lo + hi)


def _bisect(func, lo, hi, iterations=48):
    """Vectorised bisection for a root of ``func`` bracketed by ``lo``/``hi``."""
    lo = np.array(lo, float, copy=True)
    hi = np.array(hi, float, copy=True)
    f_lo = func(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(same, mid, lo)
        f_lo = np.where(same, f_mid, f_lo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def central_depth(state):
    """Positive where the Moon's disc lies wholly inside the Sun's, or vice versa.

    This is the umbral/antumbral criterion: it defines the path of totality for
    a total eclipse and the path of annularity for an annular one, which is
    what published eclipse paths and path widths refer to.
    """
    return np.abs(state['r_moon'] - state['r_sun']) - state['separation']


def obscuration_depth(threshold):
    """Criterion for "at least ``threshold`` of the Sun's area is covered"."""
    def depth(state):
        return state['obscuration'] - threshold
    return depth


def peak_eclipse(window, itrf_xyz, tt_grid, depth=None, require_visible=True):
    """Deepest eclipse seen at each of ``P`` sites over the whole event.

    ``depth`` scores how deep the eclipse is; the returned state is taken at
    whichever instant maximises it.  The peak is normally at minimum
    separation, but a site that only catches the eclipse around sunrise or
    sunset peaks exactly as the Sun reaches the horizon, so those instants are
    candidates too and the best of them all wins.

    Which horizon crossing is taken is chosen by index, so this function can
    step slightly where that choice flips from one site to its neighbour.  A
    search for the edge of a region inherits that as a wobble of a few km,
    which is visible at the ends of a region where the edge is nearly flat.
    Evaluating every crossing removes it but costs far more than it is worth;
    see the note in eclipse._trace_region.
    """
    if depth is None:
        depth = obscuration_depth(0.0)
    itrf_xyz = np.asarray(itrf_xyz, float)
    if itrf_xyz.ndim == 1:
        itrf_xyz = itrf_xyz.reshape(3, 1)
    col = itrf_xyz[:, :, None]
    n_sites = itrf_xyz.shape[1]
    n_times = len(tt_grid)

    r_sun, r_moon, sep, alt = observe(window, tt_grid, col)
    grid_state = {'r_sun': r_sun, 'r_moon': r_moon, 'separation': sep,
                  'sun_altitude': alt,
                  'obscuration': g.obscuration(r_sun, r_moon, sep),
                  'magnitude': g.magnitude(r_sun, r_moon, sep)}
    horizon = g.HORIZON_ALTITUDE_DEG
    visible = alt >= horizon if require_visible else np.ones_like(alt, bool)
    scores = np.where(visible, depth(grid_state), -np.inf)

    best_t = tt_grid[np.argmax(scores, axis=1)]
    best = _evaluate(window, itrf_xyz, best_t, depth, require_visible)

    # Candidate A: the true minimum-separation instant.
    j = np.argmin(sep, axis=1)
    lo = tt_grid[np.maximum(j - 1, 0)]
    hi = tt_grid[np.minimum(j + 1, n_times - 1)]
    t_a = _golden_min(lambda tt: _separation_at(window, col, tt), lo, hi)
    best = _better(best, _evaluate(window, itrf_xyz, t_a, depth, require_visible))

    # Candidate B: the horizon crossing nearest that instant, if there is one.
    crossing = visible[:, :-1] != visible[:, 1:]
    if crossing.any():
        idx = np.arange(n_times - 1)[None, :]
        cost = np.where(crossing, np.abs(idx - j[:, None]), n_times * 10)
        k = np.argmin(cost, axis=1)
        has = crossing[np.arange(n_sites), k]
        t_b = _bisect(lambda tt: _altitude_at(window, col, tt),
                      tt_grid[k], tt_grid[k + 1], 30)
        t_b = np.where(has, t_b, best['t'])
        best = _better(best, _evaluate(window, itrf_xyz, t_b, depth,
                                       require_visible, tolerance=1e-9))
    return best


def _evaluate(window, itrf_xyz, tt, depth, require_visible, tolerance=0.0):
    state = state_at(window, itrf_xyz, tt)
    state['t'] = tt
    state['visible'] = state['sun_altitude'] >= g.HORIZON_ALTITUDE_DEG - tolerance
    score = depth(state)
    state['depth'] = np.where(state['visible'], score, -np.inf) if require_visible else score
    return state


def _better(a, b):
    take = b['depth'] > a['depth']
    return {key: np.where(take, b[key], a[key]) for key in a}


def _altitude_at(window, itrf_col, tt):
    """Sun altitude relative to the apparent horizon, so its root is sunrise."""
    sun, _, rot = window.at(tt)
    site = g.rotate_to_icrf(rot, itrf_col[:, :, 0])
    up = g.rotate_to_icrf(rot, g.unit(_zenith_of(itrf_col[:, :, 0])))
    altitude = 90.0 - np.degrees(g.separation(sun - site, up))
    return altitude - g.HORIZON_ALTITUDE_DEG


def state_at(window, itrf_xyz, tt):
    """Full local state for ``P`` sites, each at its own instant ``tt`` (P,)."""
    r_sun, r_moon, sep, alt = observe(window, tt, itrf_xyz)
    return {'r_sun': r_sun, 'r_moon': r_moon, 'separation': sep,
            'sun_altitude': alt,
            'obscuration': g.obscuration(r_sun, r_moon, sep),
            'magnitude': g.magnitude(r_sun, r_moon, sep)}


def contact_times(window, itrf_xyz, tt_grid):
    """First/second/third/fourth contact times for a single site.

    Returns a dict with ``c1``..``c4`` (TT Julian days, NaN when the contact
    does not occur) plus the central-phase duration in seconds.
    """
    col = np.asarray(itrf_xyz, float).reshape(3, 1)
    r_sun, r_moon, sep, alt = observe(window, tt_grid, col[:, :, None])
    r_sun, r_moon, sep, alt = (a[0] for a in (r_sun, r_moon, sep, alt))
    outer = sep - (r_sun + r_moon)
    inner = sep - np.abs(r_moon - r_sun)

    def cross(values, pivot_index, direction):
        """Bracket a sign change moving away from ``pivot_index``."""
        rng = (range(pivot_index, 0, -1) if direction < 0
               else range(pivot_index, len(values) - 1))
        for i in rng:
            a, b = (i - 1, i) if direction < 0 else (i, i + 1)
            if values[a] * values[b] <= 0.0:
                return tt_grid[a], tt_grid[b]
        return None

    pivot = int(np.argmin(sep))
    result = {}
    for name, series, direction in (('c1', outer, -1), ('c4', outer, +1),
                                    ('c2', inner, -1), ('c3', inner, +1)):
        bracket = cross(series, pivot, direction)
        if bracket is None or series[pivot] > 0.0:
            result[name] = np.nan
            continue
        func = _outer_func(window, col) if series is outer else _inner_func(window, col)
        result[name] = float(_bisect(func, np.array([bracket[0]]),
                                     np.array([bracket[1]]))[0])
    result['central_seconds'] = (
        (result['c3'] - result['c2']) * 86400.0
        if np.isfinite(result['c2']) and np.isfinite(result['c3']) else 0.0)
    result['partial_seconds'] = (
        (result['c4'] - result['c1']) * 86400.0
        if np.isfinite(result['c1']) and np.isfinite(result['c4']) else 0.0)
    return result


def _outer_func(window, col):
    def f(tt):
        r_sun, r_moon, sep, _ = observe(window, tt, col[:, :, None])
        return (sep - (r_sun + r_moon))[0]
    return f


def _inner_func(window, col):
    def f(tt):
        r_sun, r_moon, sep, _ = observe(window, tt, col[:, :, None])
        return (sep - np.abs(r_moon - r_sun))[0]
    return f


def reaches(window, tt_grid, latitude, longitude, depth):
    """Does the deepest eclipse at these places satisfy ``depth``?"""
    xyz = g.geodetic_to_itrf(latitude, longitude)
    return peak_eclipse(window, xyz, tt_grid, depth=depth)['depth'] >= 0.0
