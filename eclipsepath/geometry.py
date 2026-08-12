"""Pure-numpy geometry used for solar-eclipse shadow calculations.

Everything here is frame-agnostic vector maths on arrays shaped ``(3, ...)``
(the convention Skyfield uses).  No ephemeris or time handling lives here, so
each function is trivially testable in isolation.

Coordinates are kilometres, angles radians unless a name says otherwise.
"""

from __future__ import annotations

import numpy as np

# --- Adopted constants ------------------------------------------------------
# Solar radius matching the semi-diameter of 959.63" at 1 au that is used by the
# NASA/Espenak eclipse canon.
R_SUN_KM = 696000.0

# The Moon's radius as k = R_moon / R_earth_equatorial.  Two values are in use:
# the IAU (1982) mean radius, which averages over limb peaks and valleys, and a
# smaller "mean minimum" radius adopted by Espenak for umbral and antumbral
# contacts because it better reproduces observed path limits and contact times.
# The latter is the default here, so results line up with published catalogues.
K_MOON_IAU = 0.2725076
K_MOON_ESPENAK = 0.2722810
LUNAR_RADIUS_POLICIES = {'espenak': K_MOON_ESPENAK, 'iau': K_MOON_IAU}
K_MOON = K_MOON_ESPENAK

# IAU/WGS84 reference ellipsoid.
EARTH_A_KM = 6378.137
EARTH_F = 1.0 / 298.257223563
EARTH_B_KM = EARTH_A_KM * (1.0 - EARTH_F)
EARTH_E2 = EARTH_F * (2.0 - EARTH_F)

R_MOON_KM = K_MOON * EARTH_A_KM  # 1736.6 km


def set_lunar_radius(policy_or_k):
    """Select the lunar radius by policy name ('espenak', 'iau') or by k."""
    global K_MOON, R_MOON_KM
    K_MOON = LUNAR_RADIUS_POLICIES.get(policy_or_k, policy_or_k)
    K_MOON = float(K_MOON)
    R_MOON_KM = K_MOON * EARTH_A_KM
    return K_MOON


# --- Small vector helpers ---------------------------------------------------

def norm(v):
    """Euclidean length along the leading axis of a ``(3, ...)`` array."""
    return np.sqrt(np.einsum('i...,i...->...', v, v))


def unit(v):
    return v / norm(v)


def dot(a, b):
    return np.einsum('i...,i...->...', a, b)


def separation(a, b):
    """Angle between two ``(3, ...)`` vectors, numerically safe near 0 and pi."""
    c = np.cross(a, b, axis=0)
    return np.arctan2(norm(c), dot(a, b))


def align(arr, lead_axes, target_ndim):
    """Left-pad the broadcast dimensions of a ``(3, ...)`` / ``(3, 3, ...)`` array.

    Plain numpy broadcasting cannot be used directly on these arrays because
    the leading component axes would be aligned against trailing broadcast
    axes.  ``align`` inserts length-1 axes just after the component axes so
    that arrays carrying different numbers of broadcast dimensions (sites,
    times) combine the way they read.
    """
    shape = arr.shape[lead_axes:]
    pad = max(target_ndim - len(shape), 0)
    return arr.reshape(arr.shape[:lead_axes] + (1,) * pad + shape)


def rotate_to_itrf(R, v):
    """Apply an ICRF->ITRF rotation ``R`` shaped (3, 3, ...) to ``v`` (3, ...)."""
    return np.einsum('ij...,j...->i...', R, v)


def rotate_to_icrf(R, v):
    """Inverse of :func:`rotate_to_itrf` (the rotation is orthonormal)."""
    return np.einsum('ji...,j...->i...', R, v)


# --- Geodetic <-> geocentric cartesian --------------------------------------

def geodetic_to_itrf(lat_deg, lon_deg, height_km=0.0):
    """Geodetic coordinates on the reference ellipsoid to ITRF xyz in km."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = EARTH_A_KM / np.sqrt(1.0 - EARTH_E2 * sin_lat ** 2)
    x = (n + height_km) * cos_lat * np.cos(lon)
    y = (n + height_km) * cos_lat * np.sin(lon)
    z = (n * (1.0 - EARTH_E2) + height_km) * sin_lat
    return np.array(np.broadcast_arrays(x, y, z))


def itrf_to_geodetic(xyz):
    """ITRF xyz (km) to (lat_deg, lon_deg, height_km) using Bowring's method."""
    x, y, z = xyz[0], xyz[1], xyz[2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Bowring's initial parametric latitude, then one closed-form step which is
    # accurate to well under a millimetre for near-surface points.
    theta = np.arctan2(z * EARTH_A_KM, p * EARTH_B_KM)
    ep2 = (EARTH_A_KM ** 2 - EARTH_B_KM ** 2) / EARTH_B_KM ** 2
    lat = np.arctan2(z + ep2 * EARTH_B_KM * np.sin(theta) ** 3,
                     p - EARTH_E2 * EARTH_A_KM * np.cos(theta) ** 3)
    for _ in range(2):  # refine; converges immediately at these eccentricities
        n = EARTH_A_KM / np.sqrt(1.0 - EARTH_E2 * np.sin(lat) ** 2)
        height = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - EARTH_E2 * n / (n + height)))
    n = EARTH_A_KM / np.sqrt(1.0 - EARTH_E2 * np.sin(lat) ** 2)
    height = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), height


# --- Sun/Moon disc overlap --------------------------------------------------

def angular_radius(distance_km, body_radius_km):
    return np.arcsin(np.clip(body_radius_km / distance_km, -1.0, 1.0))


def obscuration(r_sun, r_moon, sep):
    """Fraction of the solar *disc area* hidden by the Moon.

    ``r_sun``/``r_moon`` are apparent radii and ``sep`` the apparent separation
    of the two centres, all in radians.  Standard circle-circle overlap; the
    discs are small enough that treating them as planar is exact to ~1e-9.
    """
    r_sun, r_moon, sep = np.broadcast_arrays(
        np.asarray(r_sun, float), np.asarray(r_moon, float), np.asarray(sep, float))
    d = np.maximum(sep, 1e-30)
    # Guard the arccos arguments so the "no overlap" / "fully inside" branches
    # never evaluate a NaN before np.where selects them away.
    c1 = np.clip((d ** 2 + r_sun ** 2 - r_moon ** 2) / (2.0 * d * r_sun), -1.0, 1.0)
    c2 = np.clip((d ** 2 + r_moon ** 2 - r_sun ** 2) / (2.0 * d * r_moon), -1.0, 1.0)
    tri = np.clip((-d + r_sun + r_moon) * (d + r_sun - r_moon)
                  * (d - r_sun + r_moon) * (d + r_sun + r_moon), 0.0, None)
    area = (r_sun ** 2 * np.arccos(c1) + r_moon ** 2 * np.arccos(c2)
            - 0.5 * np.sqrt(tri))
    frac = area / (np.pi * r_sun ** 2)
    frac = np.where(sep >= r_sun + r_moon, 0.0, frac)          # no contact
    frac = np.where(sep <= r_moon - r_sun, 1.0, frac)          # total
    frac = np.where(sep <= r_sun - r_moon, (r_moon / r_sun) ** 2, frac)  # annular
    # The lens-area branch can overshoot 1 by ~1e-12 right at second contact.
    return np.clip(frac, 0.0, 1.0)


def magnitude(r_sun, r_moon, sep):
    """Eclipse magnitude: fraction of the solar *diameter* covered.

    Follows the usual convention that during totality/annularity the magnitude
    is reported as the ratio of apparent diameters.
    """
    r_sun, r_moon, sep = np.broadcast_arrays(
        np.asarray(r_sun, float), np.asarray(r_moon, float), np.asarray(sep, float))
    mag = (r_sun + r_moon - sep) / (2.0 * r_sun)
    mag = np.where(sep >= r_sun + r_moon, 0.0, mag)
    mag = np.where(sep <= np.abs(r_moon - r_sun), r_moon / r_sun, mag)
    return mag


# --- Shadow cone geometry ---------------------------------------------------

def shadow_axis(sun, moon):
    """Unit vector along the shadow axis, pointing from the Sun past the Moon."""
    return unit(moon - sun)


def penumbra_radius(sun, moon, s_km):
    """Radius of the penumbral cone ``s_km`` beyond the Moon along the axis."""
    dist = norm(moon - sun)
    sin_f1 = (R_SUN_KM + R_MOON_KM) / dist
    tan_f1 = sin_f1 / np.sqrt(1.0 - sin_f1 ** 2)
    return R_MOON_KM + s_km * tan_f1


def umbra_radius(sun, moon, s_km):
    """Umbral cone radius; negative past the apex, i.e. inside the antumbra."""
    dist = norm(moon - sun)
    sin_f2 = (R_SUN_KM - R_MOON_KM) / dist
    tan_f2 = sin_f2 / np.sqrt(1.0 - sin_f2 ** 2)
    return R_MOON_KM - s_km * tan_f2


def axis_earth_distance(sun, moon):
    """Signed clearance (km) between the penumbral cone and the Earth's surface.

    Negative means the penumbra is touching the globe, i.e. an eclipse is in
    progress somewhere.  Used only as a fast scanning criterion, so the sphere
    approximation to the ellipsoid is deliberate (with the polar radius, which
    keeps it conservative).
    """
    u = shadow_axis(sun, moon)
    s0 = -dot(moon, u)                      # along-axis distance to closest point
    d = norm(moon + u * s0)                 # perpendicular miss distance
    r_pen = penumbra_radius(sun, moon, np.maximum(s0, 0.0))
    behind = s0 < 0.0                       # Earth on the sunward side of the Moon
    return np.where(behind, 1e9, d - (EARTH_A_KM + r_pen))


def line_ellipsoid_intersection(origin, direction):
    """Nearest intersection of a ray with the reference ellipsoid.

    ``origin`` and ``direction`` are ITRF ``(3, ...)`` arrays.  Returns
    ``(point, hit)`` where ``point`` is the intersection (NaN-filled where the
    ray misses) and ``hit`` is a boolean array.
    """
    scale = np.array([1.0, 1.0, EARTH_A_KM / EARTH_B_KM]).reshape(
        (3,) + (1,) * (origin.ndim - 1))
    o = origin * scale
    dvec = unit(direction * scale)
    b = dot(o, dvec)
    c = dot(o, o) - EARTH_A_KM ** 2
    disc = b ** 2 - c
    hit = disc >= 0.0
    root = np.sqrt(np.where(hit, disc, 0.0))
    s = -b - root                            # near-side (sunward) intersection
    point = (o + dvec * s) / scale
    return np.where(hit, point, np.nan), hit


def closest_surface_point_to_axis(origin, direction):
    """Point on the ellipsoid closest to a ray that misses it (ITRF)."""
    scale = np.array([1.0, 1.0, EARTH_A_KM / EARTH_B_KM]).reshape(
        (3,) + (1,) * (origin.ndim - 1))
    o = origin * scale
    dvec = unit(direction * scale)
    perp = o - dvec * dot(o, dvec)
    return unit(perp) * EARTH_A_KM / scale


# --- Distances and bearings on the ellipsoid --------------------------------

def great_circle_destination(lat_deg, lon_deg, bearing_deg, distance_km,
                             radius_km=EARTH_A_KM):
    """Spherical direct problem; used only to step search rays sideways."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    brg = np.radians(bearing_deg)
    ang = distance_km / radius_km
    sin_lat2 = np.sin(lat) * np.cos(ang) + np.cos(lat) * np.sin(ang) * np.cos(brg)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
    lon2 = lon + np.arctan2(np.sin(brg) * np.sin(ang) * np.cos(lat),
                            np.cos(ang) - np.sin(lat) * sin_lat2)
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def initial_bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, in degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def densify(lat, lon, spacing_km=0.25):
    """Resample a lat/lon polyline along great circles to a target spacing.

    Two errors are being avoided.  Interpolating linearly in latitude and
    longitude would cut a large chord wherever the line is strongly bent, which
    near a pole dwarfs anything measured against the result.  And because
    :func:`distance_to_polyline` only sees the sample points, a coarse spacing
    ``s`` inflates a true distance ``d`` to roughly ``hypot(d, s/2)``, so the
    spacing has to stay well below the distances of interest.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if lat.size < 2:
        return lat, lon
    unit_xyz = unit(geodetic_to_itrf(lat, lon))
    out_lat, out_lon = [], []
    for i in range(lat.size - 1):
        a, b = unit_xyz[:, i], unit_xyz[:, i + 1]
        angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        steps = max(2, int(np.ceil(angle * EARTH_A_KM / spacing_km)) + 1)
        fraction = np.linspace(0.0, 1.0, steps)
        if angle < 1e-12:
            points = np.outer(a, np.ones(steps))
        else:
            points = (np.sin((1.0 - fraction) * angle) * a[:, None]
                      + np.sin(fraction * angle) * b[:, None]) / np.sin(angle)
        piece_lat, piece_lon, _ = itrf_to_geodetic(points * EARTH_A_KM)
        out_lat.append(piece_lat)
        out_lon.append(piece_lon)
    return np.concatenate(out_lat), np.concatenate(out_lon)


def distance_to_polyline(lat, lon, curve_lat, curve_lon):
    """Shortest distance from a point to an already densified polyline."""
    return float(geodesic_distance(lat, lon, curve_lat, curve_lon).min())


def geodesic_distance(lat1, lon1, lat2, lon2):
    """Vincenty inverse distance in km on the reference ellipsoid."""
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (lat1, lon1, lat2, lon2)))
    a, b, f = EARTH_A_KM, EARTH_B_KM, EARTH_F
    u1 = np.arctan((1 - f) * np.tan(np.radians(lat1)))
    u2 = np.arctan((1 - f) * np.tan(np.radians(lat2)))
    ll = np.radians(lon2 - lon1)
    lam = ll.copy()
    sin_u1, cos_u1 = np.sin(u1), np.cos(u1)
    sin_u2, cos_u2 = np.sin(u2), np.cos(u2)
    sin_sig = cos_sig = sigma = sin_alpha = cos2_sig_m = cos2_alpha = None
    for _ in range(100):
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)
        sin_sig = np.sqrt((cos_u2 * sin_lam) ** 2
                          + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2)
        cos_sig = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = np.arctan2(sin_sig, cos_sig)
        sin_alpha = np.where(sin_sig == 0, 0.0,
                             cos_u1 * cos_u2 * sin_lam / np.where(sin_sig == 0, 1, sin_sig))
        cos2_alpha = 1 - sin_alpha ** 2
        cos2_sig_m = np.where(cos2_alpha == 0, 0.0,
                              cos_sig - 2 * sin_u1 * sin_u2
                              / np.where(cos2_alpha == 0, 1, cos2_alpha))
        c = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = ll + (1 - c) * f * sin_alpha * (
            sigma + c * sin_sig * (cos2_sig_m + c * cos_sig
                                   * (-1 + 2 * cos2_sig_m ** 2)))
        if np.all(np.abs(lam - lam_prev) < 1e-12):
            break
    u2_ = cos2_alpha * (a ** 2 - b ** 2) / b ** 2
    aa = 1 + u2_ / 16384 * (4096 + u2_ * (-768 + u2_ * (320 - 175 * u2_)))
    bb = u2_ / 1024 * (256 + u2_ * (-128 + u2_ * (74 - 47 * u2_)))
    d_sig = bb * sin_sig * (cos2_sig_m + bb / 4 * (
        cos_sig * (-1 + 2 * cos2_sig_m ** 2)
        - bb / 6 * cos2_sig_m * (-3 + 4 * sin_sig ** 2)
        * (-3 + 4 * cos2_sig_m ** 2)))
    return b * aa * (sigma - d_sig)
