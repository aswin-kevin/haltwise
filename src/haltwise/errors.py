"""Exceptions raised by haltwise. All inherit from `HaltwiseError`."""


class HaltwiseError(Exception):
    """Base class for every haltwise error."""


class CalibrationError(HaltwiseError):
    """Calibration is impossible with the data given (usually: too few successful runs)."""


class NotCalibratedError(HaltwiseError):
    """A monitor was asked for decisions before it had a stopping threshold."""
