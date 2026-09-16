"""Turning a satellite fix into the metres the rest of the stack works in.

Measured in L2-GAP (2026-09-16), not assumed. Town10HD's OpenDRIVE header declares

    +proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m

and sampling CARLA's own `transform_to_geolocation` over a 400 m grid gives a mapping whose
worst departure from a straight line is 0.0000 m. So over the area the van drives in it is
exactly linear and inverts exactly:

    lat = lat0 + DLAT_DY * y          DLAT_DY is NEGATIVE
    lon = lon0 + DLON_DX * x

THE SIGN IS THE WHOLE POINT. CARLA's world is left-handed and its y axis runs SOUTH, so
latitude DECREASES as y increases. Getting that backwards mirrors the map about the equator
and every fix lands on the wrong side of the van -- which is exactly the kind of error that
looks like "the filter is badly tuned" for a week.

Checked against CARLA truth over 2,002 live fixes on a real drive: median error 0.0000 m.

The constants belong to a map, not to the world. `from_world()` asks CARLA for the map's own
origin; the defaults below are Town10HD's, which is what we drive on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Degrees of latitude per metre of CARLA +y. NEGATIVE: +y is south.
DLAT_DY = -8.983e-06
#: Degrees of longitude per metre of CARLA +x.
DLON_DX = 8.983e-06
#: Town10HD's georeference origin.
LAT0 = 0.0
LON0 = 0.0

#: 1 / 8.983e-06 = 111,320 m per degree, which is the equatorial figure for a spherical
#: earth of radius 6,378,137 m. Recorded so that a different map's numbers can be sanity
#: checked against it rather than merely trusted.
METRES_PER_DEGREE = 1.0 / abs(DLAT_DY)


@dataclass(frozen=True)
class GeoFrame:
    """One map's mapping between latitude/longitude and local metres."""
    lat0: float = LAT0
    lon0: float = LON0
    dlat_dy: float = DLAT_DY
    dlon_dx: float = DLON_DX

    @classmethod
    def from_world(cls, carla_map, carla_module) -> "GeoFrame":
        """Ask CARLA itself, rather than trusting the constants above.

        Two finite differences a kilometre apart, which is far enough that floating point
        noise in the conversion cannot matter and near enough to stay inside the map.
        """
        g0 = carla_map.transform_to_geolocation(carla_module.Location(x=0.0, y=0.0, z=0.0))
        gx = carla_map.transform_to_geolocation(carla_module.Location(x=1000.0, y=0.0, z=0.0))
        gy = carla_map.transform_to_geolocation(carla_module.Location(x=0.0, y=1000.0, z=0.0))
        return cls(lat0=g0.latitude, lon0=g0.longitude,
                   dlat_dy=(gy.latitude - g0.latitude) / 1000.0,
                   dlon_dx=(gx.longitude - g0.longitude) / 1000.0)

    def to_xy(self, lat: float, lon: float):
        """A satellite fix, in the local metres the planner and perception use."""
        return ((lon - self.lon0) / self.dlon_dx,
                (lat - self.lat0) / self.dlat_dy)

    def to_latlon(self, x: float, y: float):
        """...and back again, for checking the round trip."""
        return (self.lat0 + self.dlat_dy * y,
                self.lon0 + self.dlon_dx * x)

    def metres_per_degree_lat(self) -> float:
        return 1.0 / abs(self.dlat_dy)

    def metres_per_degree_lon(self) -> float:
        return 1.0 / abs(self.dlon_dx)

    def degrees_lat_for(self, metres: float) -> float:
        """How many degrees of latitude a distance is -- for setting sensor noise, which
        CARLA's blueprint wants in degrees while every requirement we have is in metres."""
        return abs(metres * self.dlat_dy)

    def degrees_lon_for(self, metres: float) -> float:
        return abs(metres * self.dlon_dx)


#: The default frame: Town10HD, the map the van drives on.
DEFAULT = GeoFrame()


def to_xy(lat: float, lon: float):
    return DEFAULT.to_xy(lat, lon)


def to_latlon(x: float, y: float):
    return DEFAULT.to_latlon(x, y)


def bearing_to_yaw(compass_rad: float) -> float:
    """CARLA's compass is north-referenced; the stack's yaw is +x-referenced.

    Measured in L2-GAP against truth on a parked van: compass_deg - yaw_deg = 90.000,
    exactly. So yaw = compass - 90 degrees, wrapped.
    """
    a = compass_rad - math.pi / 2.0
    return math.atan2(math.sin(a), math.cos(a))
