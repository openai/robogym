"""Module to provide functionality related to measurement units, since not all robots work with angles or in radians,
and other components may want to provide hybrid data representations, such as plots or reports."""
from __future__ import annotations

from enum import Enum

import numpy as np


class MeasurementUnit(Enum):
    """Units of measurement known to this check."""

    # values are important since they are pickled
    RADIANS = 1
    DEGREES = 2
    METERS = 3
    MILLIMETERS = 4
    SECONDS = 5
    MILLISECONDS = 6

    def shortname(self):
        if self == MeasurementUnit.RADIANS:
            return "rad"
        elif self == MeasurementUnit.DEGREES:
            return "deg"
        elif self == MeasurementUnit.METERS:
            return "m"
        elif self == MeasurementUnit.MILLIMETERS:
            return "mm"
        elif self == MeasurementUnit.SECONDS:
            return "s"
        elif self == MeasurementUnit.MILLISECONDS:
            return "ms"
        raise RuntimeError(f"Shortname for '{self}' is not specified")

    def convert_to(self, data: np.ndarray, to_units: MeasurementUnit) -> np.ndarray:
        """Poor man's conversion from self units to 'to_units' for supported parts. It may only support a
        subset of all possible conversions, and will raise an error if the requested one is not supported.

        :param data: Data to convert.
        :param to_units: Units to convert to.
        :return: Result of converting the data to the new units, from the current units.
        :raises RuntimeError: If the conversion pair is not supported.
        """
        if self == to_units:
            return data

        if self == MeasurementUnit.RADIANS and to_units == MeasurementUnit.DEGREES:
            return np.rad2deg(data)
        elif self == MeasurementUnit.DEGREES and to_units == MeasurementUnit.RADIANS:
            return np.deg2rad(data)
        elif self == MeasurementUnit.METERS and to_units == MeasurementUnit.MILLIMETERS:
            return np.multiply(data, 1000)
        elif self == MeasurementUnit.MILLIMETERS and to_units == MeasurementUnit.METERS:
            return np.multiply(data, 1e-3)
        elif (
            self == MeasurementUnit.SECONDS and to_units == MeasurementUnit.MILLISECONDS
        ):
            return np.multiply(data, 1000)
        raise RuntimeError(f"Can't convert between units from {self} to {to_units}")
