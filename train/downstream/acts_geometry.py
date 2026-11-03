"""
acts_geometry.py
----------------
Builds a minimal cylindrical TPC geometry for ACTS that matches the
sPHENIX TPC as parameterised in vertexhead.py.

Geometry constants (must match vertexhead.py exactly):
    r_lim   = (31.37, 75.38)  cm  -- inner/outer TPC radius
    eta_lim = (-2.0, 2.0)         -- pseudo-rapidity acceptance
    Bz      = 1.4              T   -- solenoid field strength

Public API
----------
build_tpc_geometry(n_layers=48, Bz=1.4)
    -> (geometry, magnetic_field, surfaces)  or  None if ACTS unavailable

get_surface_for_hit(r_cm, surfaces)
    -> nearest acts.Surface  or  None
"""

from __future__ import annotations
import math
import warnings
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# TPC constants (keep in sync with vertexhead.py)
# ---------------------------------------------------------------------------
R_MIN_CM   = 31.37   # cm - innermost pad row
R_MAX_CM   = 75.38   # cm - outermost pad row
ETA_MAX    = 2.0     # |eta| acceptance
BZ_DEFAULT = 1.4     # T

# Half-length in z from |eta|_max at innermost layer
Z_HALF_CM = R_MIN_CM * math.sinh(ETA_MAX)   # ~90.6 cm

# ---------------------------------------------------------------------------
# Optional ACTS import
# ---------------------------------------------------------------------------
try:
    import acts
    import acts.examples
    _ACTS_AVAILABLE = True
except ImportError:
    _ACTS_AVAILABLE = False
    acts = None


def _check_acts() -> bool:
    if not _ACTS_AVAILABLE:
        warnings.warn(
            "ACTS Python bindings not found. Install ACTS with pybind11 support "
            "and ensure the Python package is on sys.path. "
            "All ACTS-based refinement will be skipped.",
            ImportWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tpc_geometry(
    n_layers: int = 48,
    Bz: float = BZ_DEFAULT,
) -> Optional[Tuple]:
    """
    Construct a concentric-cylinder approximation of the sPHENIX TPC
    suitable for ACTS track fitting.

    Parameters
    ----------
    n_layers : int
        Number of cylindrical layers (default 48); radii linearly spaced
        from R_MIN_CM to R_MAX_CM.
    Bz : float
        Solenoid field strength in Tesla (default 1.4 T).

    Returns
    -------
    (geometry, magnetic_field, surfaces)  or  None if ACTS unavailable.
        geometry       -- acts.TrackingGeometry
        magnetic_field -- acts.ConstantBField
        surfaces       -- list[acts.Surface] sorted by radius (cm)
    """
    if not _check_acts():
        return None

    mm = acts.UnitConstants.mm   # ACTS internal length unit is mm

    # Layer radii linearly spaced
    radii_cm = [
        R_MIN_CM + (R_MAX_CM - R_MIN_CM) * i / (n_layers - 1)
        for i in range(n_layers)
    ]

    # Build one CylinderLayer per pad row
    layer_configs = []
    for r in radii_cm:
        lconf                    = acts.CylinderLayerConfig()
        lconf.rMin               = (r - 0.5) * 10.0 * mm
        lconf.rMax               = (r + 0.5) * 10.0 * mm
        lconf.halfLengthZ        = Z_HALF_CM * 10.0 * mm
        lconf.binsPhi            = 1
        lconf.binsZ              = 1
        lconf.surfaceArrayCreator = acts.SurfaceArrayCreator()
        layer_configs.append(lconf)

    cvb_config                  = acts.CylinderVolumeBuilder.Config()
    cvb_config.layerEnvelopeR   = (1.0 * mm, 1.0 * mm)
    cvb_config.layerEnvelopeZ   = 1.0 * mm
    cvb_config.buildToRadiusZero = False
    cvb_config.volumeName       = "TPC"
    cvb_config.layers           = layer_configs

    cvb  = acts.CylinderVolumeBuilder(cvb_config)
    tgb  = acts.TrackingGeometryBuilder()
    tgb.volumeBuilders = [cvb]
    geometry = tgb.trackingGeometry()

    # Constant solenoid field (along z)
    magnetic_field = acts.ConstantBField(
        acts.Vector3(0.0, 0.0, Bz * acts.UnitConstants.T)
    )

    # Collect and sort sensitive surfaces by radius
    gctx = acts.GeometryContext()
    surfaces = sorted(
        list(geometry.sensitiveElements()),
        key=lambda s: s.center(gctx).head(2).norm(),  # transverse r
    )

    return geometry, magnetic_field, surfaces


def get_surface_for_hit(r_cm: float, surfaces) -> Optional[object]:
    """
    Return the ACTS cylindrical surface whose radius is closest to r_cm.

    Parameters
    ----------
    r_cm     : float
        Hit radius in cm.
    surfaces : list[acts.Surface]
        Sorted surface list returned by build_tpc_geometry.

    Returns
    -------
    acts.Surface or None
    """
    if surfaces is None or len(surfaces) == 0:
        return None
    gctx = acts.GeometryContext()
    r_mm = r_cm * 10.0
    return min(
        surfaces,
        key=lambda s: abs(s.center(gctx).head(2).norm() - r_mm),
    )
