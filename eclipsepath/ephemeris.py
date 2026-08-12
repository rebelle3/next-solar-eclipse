"""Ephemeris access and fast local interpolation.

The eclipse algorithms need Sun/Moon positions and the Earth-orientation
rotation at many thousands of instants inside a single eclipse.  Evaluating the
JPL kernel that often is slow, so :class:`EclipseWindow` samples the exact
quantities a few dozen times across the event and fits Chebyshev series to
them.  Over a few hours those functions are extremely smooth, and
``tests/test_interpolation.py`` pins the residuals at the sub-metre level.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.polynomial import chebyshev as C
from skyfield.api import load, load_file
from skyfield.framelib import itrs

DEFAULT_KERNEL = 'de440s.bsp'
KERNEL_URL = ('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/'
              + DEFAULT_KERNEL)


def _kernel_search_paths(name):
    return [name,
            os.path.join(os.path.expanduser('~'), '.eclipsepath', name),
            os.path.join(os.path.expanduser('~'), 'eph', name),
            os.path.join(os.path.dirname(__file__), name)]


class Ephemeris:
    """Wraps a JPL kernel plus the timescale, with geocentric accessors."""

    def __init__(self, kernel_path=None, download=True, delta_t=None):
        self.timescale = load.timescale()
        if delta_t is not None:
            # delta-T sets how far the Earth has turned at a given TT, so it
            # shifts a path in longitude (about 0.46 km per second at the
            # equator) without moving it in latitude or changing TT timings.
            self.timescale.delta_t_function = lambda tt: float(delta_t)
        self.delta_t_override = delta_t
        path = kernel_path or self._locate(download)
        self.path = path
        self.kernel = load_file(path)
        self.earth = self.kernel['earth']
        self.sun = self.kernel['sun']
        self.moon = self.kernel['moon']

    @staticmethod
    def _locate(download):
        for candidate in _kernel_search_paths(DEFAULT_KERNEL):
            if os.path.exists(candidate):
                return candidate
        if not download:
            raise FileNotFoundError(
                f'{DEFAULT_KERNEL} not found; pass --kernel or allow download')
        target = os.path.join(os.path.expanduser('~'), '.eclipsepath')
        os.makedirs(target, exist_ok=True)
        dest = os.path.join(target, DEFAULT_KERNEL)
        load.download(KERNEL_URL, filename=dest)
        return dest

    def coverage(self):
        """(jd_start, jd_end) of the loaded kernel, in TT-ish Julian days."""
        starts, ends = [], []
        for segment in self.kernel.segments:
            lo, hi = segment.spk_segment.start_jd, segment.spk_segment.end_jd
            starts.append(lo)
            ends.append(hi)
        return max(starts), min(ends)

    def time(self, tt_jd):
        return self.timescale.tt_jd(tt_jd)

    def delta_t_at(self, tt_jd):
        """Seconds of TT - UT1 assumed at ``tt_jd``."""
        t = self.time(tt_jd)
        return float((t.tt - t.ut1) * 86400.0)

    def sun_moon(self, t):
        """Geocentric astrometric Sun and Moon position vectors, km.

        Astrometric (light-time corrected, no aberration) positions are what
        shadow geometry wants: they place each body where it was when the light
        now arriving at Earth left it.
        """
        e = self.earth.at(t)
        return (e.observe(self.sun).position.km,
                e.observe(self.moon).position.km)

    @staticmethod
    def rotation(t):
        """ICRF -> ITRF rotation matrix, shaped (3, 3, n)."""
        return itrs.rotation_at(t)


class EclipseWindow:
    """Chebyshev model of one eclipse's geometry over ``[tt0, tt1]`` (TT days).

    All the eclipse maths calls into this rather than the kernel, which makes
    the per-point time scans and boundary bisections cheap enough to vectorise.
    """

    def __init__(self, ephem: Ephemeris, tt0, tt1, degree=16, nodes=None):
        self.ephem = ephem
        self.tt0 = float(tt0)
        self.tt1 = float(tt1)
        self.degree = degree
        n = nodes or (2 * degree + 6)
        # Chebyshev-Lobatto nodes: near-optimal conditioning for the fit.
        x = np.cos(np.linspace(np.pi, 0.0, n))
        tt = self._unscale(x)
        t = ephem.time(tt)
        sun, moon = ephem.sun_moon(t)
        rot = ephem.rotation(t)
        stacked = np.concatenate([sun, moon, rot.reshape(9, -1)], axis=0).T
        self._coef = C.chebfit(x, stacked, degree)

    def _scale(self, tt):
        return (2.0 * (np.asarray(tt, float) - self.tt0)
                / (self.tt1 - self.tt0) - 1.0)

    def _unscale(self, x):
        return self.tt0 + (np.asarray(x, float) + 1.0) * (self.tt1 - self.tt0) / 2.0

    def at(self, tt):
        """Return (sun, moon, rot) for scalar or array ``tt`` in TT Julian days."""
        values = C.chebval(self._scale(tt), self._coef)
        sun, moon = values[0:3], values[3:6]
        rot = values[6:15].reshape((3, 3) + np.shape(tt))
        return sun, moon, rot

    def sun_moon(self, tt):
        values = C.chebval(self._scale(tt), self._coef[:, :6])
        return values[0:3], values[3:6]

    def rotation(self, tt):
        values = C.chebval(self._scale(tt), self._coef[:, 6:15])
        return values.reshape((3, 3) + np.shape(tt))
