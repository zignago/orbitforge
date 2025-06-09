"""Geometry utilities for OrbitForge."""

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop


def calculate_volume_m3(shape) -> float:
    """Calculate volume of a shape in cubic meters.

    Args:
        shape: An OpenCascade shape object

    Returns:
        float: Volume in cubic meters
    """
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)  # Use static method to avoid deprecation
    return props.Mass()  # This is actually volume since we didn't set density
