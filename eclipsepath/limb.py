"""The Moon's real limb, from laser altimetry.

Every other part of this package treats the Moon as a sphere.  It is not: the
edge that cuts off the Sun is a mountainous horizon whose radius varies by a
couple of kilometres with position angle, and which rotates slowly as libration
turns the Moon relative to us.  That is the whole reason two different mean
radii are in use — see :mod:`eclipsepath.geometry`.

A profile is the radius of that horizon as a function of position angle, built
by asking a lunar elevation model what the ground height is at each point that
happens to lie on the limb at a given moment.

Needs two downloads, both public domain:

* a LOLA gridded elevation model, ``ldem_64.img`` and its ``.lbl``, from the
  NASA Planetary Data System
* lunar orientation kernels from JPL NAIF — ``moon_080317.tf``,
  ``pck00011.tpc`` and ``moon_pa_de421_1900-2050.bpc``
"""

from __future__ import annotations

import os

import numpy as np

from . import geometry as g

LOLA_REFERENCE_KM = 1737.4       # radius the LOLA grid measures heights from
LOLA_SCALE_M = 0.5               # stored value -> metres above that radius
DEFAULT_SAMPLES = 3600           # position angles per profile

SEARCH_DIRS = [os.curdir,
               os.path.join(os.path.expanduser('~'), '.eclipsepath'),
               os.path.join(os.path.expanduser('~'), 'eph')]


def _find(name):
    for directory in SEARCH_DIRS:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        '%s not found; see eclipsepath.limb for where to download it' % name)


class LunarElevation:
    """A LOLA cylindrical elevation grid, read lazily off disk."""

    def __init__(self, image=None, label=None):
        image = image or _find('ldem_64.img')
        label = label or (os.path.splitext(image)[0] + '.lbl')
        meta = self._read_label(label)
        self.lines = meta['LINES']
        self.samples = meta['LINE_SAMPLES']
        self.resolution = meta['MAP_RESOLUTION']
        self.line_offset = meta['LINE_PROJECTION_OFFSET']
        self.sample_offset = meta['SAMPLE_PROJECTION_OFFSET']
        # Memory-mapped: a limb only ever touches a few thousand of the 265
        # million samples, so there is no reason to hold the grid in memory.
        self.grid = np.memmap(image, dtype='<i2', mode='r',
                              shape=(self.lines, self.samples))

    @staticmethod
    def _read_label(path):
        wanted = {'LINES', 'LINE_SAMPLES', 'MAP_RESOLUTION',
                  'LINE_PROJECTION_OFFSET', 'SAMPLE_PROJECTION_OFFSET'}
        found = {}
        with open(path) as handle:
            for line in handle:
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                if key in wanted:
                    found[key] = float(value.strip().split()[0].strip('<>'))
        missing = wanted - set(found)
        if missing:
            raise ValueError('label %s is missing %s' % (path, sorted(missing)))
        return {k: (int(v) if k in ('LINES', 'LINE_SAMPLES') else v)
                for k, v in found.items()}

    def radius_km(self, latitude_deg, longitude_deg):
        """Lunar radius at selenographic coordinates, in km."""
        sample = self.sample_offset + longitude_deg * self.resolution
        line = self.line_offset - latitude_deg * self.resolution
        col = np.clip(np.round(sample).astype(int), 0, self.samples - 1)
        row = np.clip(np.round(line).astype(int), 0, self.lines - 1)
        raw = np.asarray(self.grid[row, col], dtype=float)
        return LOLA_REFERENCE_KM + raw * LOLA_SCALE_M / 1000.0


class LunarOrientation:
    """Rotation from ICRF into the Moon's mean-Earth body-fixed frame."""

    def __init__(self, frame_kernel=None, constants=None, orientation=None):
        from skyfield.planetarylib import PlanetaryConstants
        pc = PlanetaryConstants()
        pc.read_text(open(frame_kernel or _find('moon_080317.tf'), 'rb'))
        pc.read_text(open(constants or _find('pck00011.tpc'), 'rb'))
        pc.read_binary(open(orientation
                            or _find('moon_pa_de421_1900-2050.bpc'), 'rb'))
        self.frame = pc.build_frame_named('MOON_ME_DE421')

    def rotation_at(self, t):
        return self.frame.rotation_at(t)


class LimbProfile:
    """Lunar radius against position angle, for one instant."""

    def __init__(self, angles_deg, radii_km):
        self.angles_deg = np.asarray(angles_deg, float)
        self.radii_km = np.asarray(radii_km, float)

    def radius_at(self, position_angle_deg):
        """Interpolate the profile, wrapping round the limb."""
        angle = np.mod(position_angle_deg, 360.0)
        return np.interp(angle, self.angles_deg, self.radii_km,
                         period=360.0)

    @property
    def mean_km(self):
        return float(self.radii_km.mean())

    def summary(self):
        return {'mean_km': self.mean_km,
                'min_km': float(self.radii_km.min()),
                'max_km': float(self.radii_km.max()),
                'range_km': float(np.ptp(self.radii_km)),
                'mean_k': self.mean_km / g.EARTH_A_KM}


def sky_basis(direction):
    """North and east unit vectors on the sky, around a line of sight.

    Position angle is measured from celestial north through east.  Both the
    profile and the lookups that use it are built on this same basis, so the
    choice of reference direction cancels out.
    """
    line_of_sight = g.unit(direction)
    pole = np.array([0.0, 0.0, 1.0])
    north = pole - line_of_sight * np.dot(pole, line_of_sight)
    north = g.unit(north)
    east = np.cross(line_of_sight, north)
    return north, east


def position_angle(direction, offset):
    """Position angle of ``offset`` seen looking along ``direction``."""
    north, east = sky_basis(direction)
    return np.degrees(np.arctan2(np.dot(offset, east), np.dot(offset, north)))


def build_profile(elevation, orientation, t, moon_from_observer,
                  samples=DEFAULT_SAMPLES):
    """The limb profile seen from ``moon_from_observer`` at time ``t``.

    The limb is where the line of sight grazes the Moon, so its points are the
    ones perpendicular to that line.  Walking round them by position angle and
    asking the elevation model for each one's height gives the horizon that
    actually cuts off the Sun.
    """
    direction = g.unit(np.asarray(moon_from_observer, float))
    north, east = sky_basis(direction)
    rotation = np.asarray(orientation.rotation_at(t))

    angles = np.linspace(0.0, 360.0, samples, endpoint=False)
    radians = np.radians(angles)
    # Each limb point, as a direction from the Moon's centre.
    sky = (np.cos(radians)[None, :] * north[:, None]
           + np.sin(radians)[None, :] * east[:, None])
    body = rotation @ sky
    latitude = np.degrees(np.arcsin(np.clip(body[2], -1.0, 1.0)))
    longitude = np.degrees(np.arctan2(body[1], body[0]))
    return LimbProfile(angles, elevation.radius_km(latitude, longitude))


def load(image=None, label=None, **kernels):
    """Convenience: the elevation grid and orientation together."""
    return LunarElevation(image, label), LunarOrientation(**kernels)


def profile_for(ephem, tt, samples=DEFAULT_SAMPLES, cache={}):
    """Build the limb profile for the moment ``tt`` (TT Julian day).

    Libration turns the Moon by well under a degree over a single eclipse, so
    one profile taken at greatest eclipse serves the whole event.
    """
    if 'data' not in cache:
        cache['data'] = load()
    elevation, orientation = cache['data']
    t = ephem.time(tt)
    moon_from_earth = (ephem.moon - ephem.earth).at(t).position.km
    return build_profile(elevation, orientation, t, moon_from_earth, samples)
