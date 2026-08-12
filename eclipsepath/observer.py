"""What a specific place on Earth sees during a given eclipse."""

from __future__ import annotations

import numpy as np

from . import circumstances as cc
from . import geometry as g
from .eclipse import PREDICATE_TIME_SAMPLES


def circumstances_at(eclipse, latitude, longitude, height_km=0.0):
    """Local circumstances for one site, or ``None`` if nothing is visible."""
    window = eclipse.window
    if window is None:
        raise ValueError('eclipse was computed without retaining its window')
    grid = np.linspace(eclipse.tt_first_contact, eclipse.tt_last_contact,
                       PREDICATE_TIME_SAMPLES)
    xyz = g.geodetic_to_itrf(latitude, longitude, height_km).reshape(3, 1)
    peak = cc.peak_eclipse(window, xyz, grid)
    if float(peak['obscuration'][0]) <= 0.0:
        return None
    contacts = cc.contact_times(window, xyz[:, 0], grid)
    return {
        'latitude': latitude,
        'longitude': longitude,
        'obscuration': float(peak['obscuration'][0]),
        'magnitude': float(peak['magnitude'][0]),
        'sun_altitude': float(peak['sun_altitude'][0]),
        'tt_maximum': float(peak['t'][0]),
        'tt_first_contact': contacts['c1'],
        'tt_last_contact': contacts['c4'],
        'tt_second_contact': contacts['c2'],
        'tt_third_contact': contacts['c3'],
        'central_seconds': contacts['central_seconds'],
        'partial_seconds': contacts['partial_seconds'],
    }
