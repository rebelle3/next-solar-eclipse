"""Tracing the outline of an area that meets a coverage threshold.

Only about drawing.  Whether a place reaches a threshold is decided in
:mod:`eclipsepath.circumstances`; everything here is about turning that
predicate into a closed outline at a resolution suited to a map, which is a
presentation concern with its own knobs and no bearing on any computed
circumstance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import circumstances as cc
from . import geometry as g

REGION_RAYS = 180
REGION_LIMIT_KM = 16000.0


@dataclass
class RegionPoint:
    """One vertex of a coverage region's outline."""
    azimuth: float
    distance_km: float
    latitude: float
    longitude: float
    tt: float
    sun_altitude: float


@dataclass
class Region:
    """A closed area of the surface where a coverage threshold is met.

    Used instead of a band wherever the area is not strip-shaped: below about
    80% coverage it broadens into a blob thousands of kilometres across, and a
    partial eclipse's is bounded on one side by the terminator, so "northern
    and southern limits" would be meaningless for either.
    """
    label: str
    points: List[RegionPoint] = field(default_factory=list)
    centre_latitude: float = float('nan')
    centre_longitude: float = float('nan')
    encloses_pole: bool = False

    def __bool__(self):
        return bool(self.points)


def trace_region(window, coarse, centre_lat, centre_lon, depth, label,
                 rays=REGION_RAYS, iterations=30):
    """Outline of the area reaching ``threshold``, swept out from its centre.

    Rays are cast from the deepest point of the eclipse and each is searched
    for where coverage falls through the threshold.  Part of the outline is
    usually the terminator rather than a coverage contour, where coverage stops
    abruptly because the Sun has set rather than thinning.

    Towards the two ends of a long area the edge runs steeply against the
    bearing — 892 km per degree, measured on the 2027 eclipse — so evenly
    spaced rays land far apart there and the outline shows facets.  Two ways
    of sampling this parametrisation harder were tried and both dropped.
    Splitting the widest gaps improved the median vertex spacing from 83 km to
    26 km without changing the rendered map, since what is visible comes from
    the largest gaps, and cost five times the work.  Probing outwards for a
    later crossing was worse than useless: near the terminator peak coverage
    is flat enough that a single spurious reading drags the vertex clean
    outside the area, which is how it was caught.  Resolving those ends means
    not parametrising the edge by bearing from a centre at all — contouring a
    grid would do it.
    """
    centre_lat, centre_lon = float(centre_lat), float(centre_lon)

    def bisect(azimuth, lo, hi):
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            mid_lat, mid_lon = g.great_circle_destination(centre_lat, centre_lon,
                                                          azimuth, mid)
            ok = cc.reaches(window, coarse, mid_lat, mid_lon, depth)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return 0.5 * (lo + hi)

    def edge_at(azimuth):
        return bisect(azimuth, np.zeros(len(azimuth)),
                      np.full(len(azimuth), REGION_LIMIT_KM))

    azimuth = np.linspace(0.0, 360.0, rays, endpoint=False)
    distance = edge_at(azimuth)

    edge_lat, edge_lon = g.great_circle_destination(centre_lat, centre_lon,
                                                    azimuth, distance)
    edge = cc.peak_eclipse(window, g.geodetic_to_itrf(edge_lat, edge_lon), coarse)

    region = Region(label=label, centre_latitude=centre_lat,
                    centre_longitude=centre_lon)
    region.points = [
        RegionPoint(azimuth=float(azimuth[i]), distance_km=float(distance[i]),
                    latitude=float(edge_lat[i]), longitude=float(edge_lon[i]),
                    tt=float(edge['t'][i]),
                    sun_altitude=float(edge['sun_altitude'][i]))
        for i in range(len(azimuth))]
    poles = cc.reaches(window, coarse, np.array([90.0, -90.0]), np.array([0.0, 0.0]),
                   depth)
    region.encloses_pole = bool(poles.any())
    return region
