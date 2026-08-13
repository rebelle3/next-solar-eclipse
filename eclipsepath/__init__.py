"""Solar eclipse paths from JPL ephemerides.

Typical use::

    from eclipsepath import search
    for eclipse in search('2026-08-01', years=10, threshold=1.0):
        print(eclipse.kind, eclipse.latitude, eclipse.longitude)
"""

from .catalog import search, TYPE_NAMES
from .eclipse import (Band, BandPoint, Eclipse, PathPoint, Region,
                      RegionPoint, analyse, coverage_region)
from .ephemeris import Ephemeris, EclipseWindow
from .observer import circumstances_at

__version__ = '1.0.0'
__all__ = ['search', 'analyse', 'Eclipse', 'Band', 'BandPoint', 'PathPoint',
           'Region', 'RegionPoint', 'Ephemeris', 'EclipseWindow',
           'circumstances_at', 'coverage_region', 'TYPE_NAMES',
           '__version__']
