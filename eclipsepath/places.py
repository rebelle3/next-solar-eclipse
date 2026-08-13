"""Which inhabited places are worth naming on a map of one eclipse.

Two questions, and only the first is astronomy.  What each place sees is a
matter of running the same circumstances the rest of the package computes.
Which of the seven thousand to actually name is a matter of judgement, and the
judgement here is that a map should be dense along the track and sparse away
from it: the towns the shadow crosses are the story even when they are small,
while five hundred miles off only a city anyone has heard of earns a label.

So places are sorted into bands by what they see, each band gets its own
minimum spacing on the ground, and within a band the larger place wins.  A
town of forty thousand in the path of totality beats a capital at ninety per
cent, which is the right way round: the first is where you would go.
"""

from __future__ import annotations

import json

import numpy as np

from . import circumstances as cc
from . import eclipse as ec
from . import geometry as g

# (floor on the fraction of the Sun covered, kilometres to the next label).
# Totality is handled separately and packs tighter still.
BANDS = ((0.95, 220.0), (0.85, 380.0), (0.70, 550.0), (0.40, 800.0))
CENTRAL_SPACING_KM = 110.0
DEFAULT_FLOOR = 0.40
DEFAULT_LIMIT = 90


def load(path):
    """Populated places from a Natural Earth GeoJSON, as plain dicts."""
    with open(path) as handle:
        data = json.load(handle)
    places = []
    for feature in data.get('features', [data]):
        properties = feature['properties']
        longitude, latitude = feature['geometry']['coordinates'][:2]
        places.append({
            'name': properties.get('name') or properties.get('nameascii') or '?',
            'country': properties.get('adm0name') or '',
            'latitude': float(latitude),
            'longitude': float(longitude),
            'population': int(properties.get('pop_max') or 0),
        })
    return places


def seen_by(eclipse, places, floor=DEFAULT_FLOOR):
    """Peak coverage at every place at once, and whether it is ever central.

    One vectorised pass over the whole list rather than a site at a time: the
    point of the coarse pass is to throw away the ninety-odd per cent of the
    world that sees nothing before spending anything on the rest.
    """
    latitudes = np.array([p['latitude'] for p in places])
    longitudes = np.array([p['longitude'] for p in places])
    xyz = g.geodetic_to_itrf(latitudes, longitudes)
    grid = np.linspace(eclipse.tt_first_contact, eclipse.tt_last_contact,
                       ec.PREDICATE_TIME_SAMPLES)
    peak = cc.peak_eclipse(eclipse.window, xyz, grid)
    covered = np.where(peak['visible'], peak['obscuration'], 0.0)

    # Only somewhere already at the top of the range can be central, so the
    # finer time grid -- which is what it takes to catch a place whose totality
    # lasts a few seconds -- is run over a handful of candidates.
    central = np.zeros(len(places), bool)
    close = np.nonzero(covered > 0.98)[0]
    if close.size:
        fine = np.linspace(eclipse.tt_first_contact, eclipse.tt_last_contact,
                           4 * ec.PREDICATE_TIME_SAMPLES)
        central[close] = cc.reaches(eclipse.window, fine, latitudes[close],
                                    longitudes[close], cc.central_depth)
    keep = covered >= floor
    return covered, central, keep


def select(eclipse, places, floor=DEFAULT_FLOOR, limit=DEFAULT_LIMIT):
    """Choose the places to name, dense on the track and thinning outwards."""
    covered, central, keep = seen_by(eclipse, places, floor)
    ranked = []
    for index in np.nonzero(keep)[0]:
        index = int(index)
        depth = float(covered[index])
        if central[index]:
            band, spacing = -1, CENTRAL_SPACING_KM
        else:
            band, spacing = next((i, s) for i, (f, s) in enumerate(BANDS)
                                 if depth >= f)
        ranked.append((band, -places[index]['population'], index, spacing,
                       depth))
    ranked.sort(key=lambda row: (row[0], row[1]))

    chosen, chosen_lat, chosen_lon = [], [], []
    for _band, _population, index, spacing, depth in ranked:
        if len(chosen) >= limit:
            break
        place = places[index]
        if chosen_lat:
            near = g.geodesic_distance(place['latitude'], place['longitude'],
                                       np.array(chosen_lat),
                                       np.array(chosen_lon)).min()
            if near < spacing:
                continue
        chosen.append((index, depth, bool(central[index])))
        chosen_lat.append(place['latitude'])
        chosen_lon.append(place['longitude'])
    return chosen


def describe(eclipse, places, chosen):
    """Exact circumstances for the chosen places, in the order given.

    The coarse pass above is for choosing; the timings a label quotes come from
    :func:`eclipsepath.observer.circumstances_at`, which searches for the
    deepest instant properly and bisects for the contacts.
    """
    from .observer import circumstances_at
    out = []
    for index, depth, is_central in chosen:
        place = places[index]
        seen = circumstances_at(eclipse, place['latitude'], place['longitude'])
        if seen is None:
            continue
        out.append({
            'name': place['name'],
            'country': place['country'],
            'latitude': round(place['latitude'], 4),
            'longitude': round(place['longitude'], 4),
            'population': place['population'],
            'obscuration': round(float(seen['obscuration']), 5),
            'magnitude': round(float(seen['magnitude']), 4),
            'sun_altitude': round(float(seen['sun_altitude']), 1),
            'central_seconds': round(float(seen['central_seconds']), 1),
            'central': bool(is_central and seen['central_seconds'] > 0.0),
            'tt_first_contact': seen['tt_first_contact'],
            'tt_maximum': seen['tt_maximum'],
            'tt_last_contact': seen['tt_last_contact'],
        })
    return out
