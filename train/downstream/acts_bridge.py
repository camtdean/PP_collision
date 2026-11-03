"""
acts_bridge.py
--------------
Converts FM4NPP downstream tensors to ACTS data types (event-by-event,
CPU).  No gradient graph is involved; call these at evaluation time only.

Main public functions
---------------------
hits_to_measurements(x_cm, y_cm, z_cm, r_cm, surfaces)
    -> acts.MeasurementContainer

track_params_to_bound(track_params_dict, B, surfaces)
    -> list[acts.BoundTrackParameters]   (one per valid track)

assigned_to_prototracks(is_assigned)
    -> acts.ProtoTrackContainer

All functions return None gracefully when ACTS is unavailable.
"""

from __future__ import annotations

import warnings
from typing import Optional, List

import numpy as np

try:
    import acts
    import acts.examples
    _ACTS_AVAILABLE = True
except ImportError:
    _ACTS_AVAILABLE = False
    acts = None

from acts_geometry import get_surface_for_hit, BZ_DEFAULT


def _check_acts() -> bool:
    if not _ACTS_AVAILABLE:
        warnings.warn(
            "ACTS not available; bridge functions are no-ops.",
            ImportWarning, stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Helper: ACTS geometry/calibration contexts
# ---------------------------------------------------------------------------

def _default_contexts():
    """Return (gctx, mctx, cctx) with default (empty) contexts."""
    gctx = acts.GeometryContext()
    mctx = acts.MagneticFieldContext()
    cctx = acts.CalibrationContext()
    return gctx, mctx, cctx


# ---------------------------------------------------------------------------
# 1.  Hits -> MeasurementContainer
# ---------------------------------------------------------------------------

def hits_to_measurements(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    r_cm: np.ndarray,
    surfaces,
) -> Optional[object]:
    """
    Build an ACTS MeasurementContainer from per-hit Cartesian coordinates.

    Parameters
    ----------
    x_cm, y_cm, z_cm : np.ndarray shape (N,)
        Hit positions in cm.
    r_cm : np.ndarray shape (N,)
        Transverse radius in cm (used to select the nearest cylinder surface).
    surfaces : list[acts.Surface]
        Sorted list from acts_geometry.build_tpc_geometry.

    Returns
    -------
    acts.MeasurementContainer or None
    """
    if not _check_acts():
        return None

    gctx = acts.GeometryContext()
    mm   = acts.UnitConstants.mm

    container = acts.MeasurementContainer()

    for i in range(len(x_cm)):
        surf = get_surface_for_hit(float(r_cm[i]), surfaces)
        if surf is None:
            continue

        # Global hit position in ACTS mm
        global_pos = acts.Vector3(
            float(x_cm[i]) * 10.0 * mm,
            float(y_cm[i]) * 10.0 * mm,
            float(z_cm[i]) * 10.0 * mm,
        )

        # Project to local (loc0=r*phi, loc1=z) on the cylinder surface
        local2d = surf.globalToLocal(gctx, global_pos, acts.Vector3(0, 0, 0))
        if local2d is None:
            continue
        local_pos = acts.Vector2(local2d[0], local2d[1])

        # Simple diagonal covariance: 200 um in r-phi, 1 mm in z
        cov = acts.ActsSquareMatrix2()
        cov[0, 0] = (0.02 * mm) ** 2   # r-phi resolution
        cov[1, 1] = (0.1  * mm) ** 2   # z resolution

        meas = acts.Measurement2D(
            acts.SubspaceIndices2D([acts.BoundIndices.eBoundLoc0,
                                    acts.BoundIndices.eBoundLoc1]),
            local_pos,
            cov,
        )
        container.addMeasurement(meas, surf.geometryId())

    return container


# ---------------------------------------------------------------------------
# 2.  Track parameter dict -> BoundTrackParameters seeds
# ---------------------------------------------------------------------------

def track_params_to_bound(
    track_params_dict: dict,
    surfaces,
    Bz: float = BZ_DEFAULT,
) -> Optional[List]:
    """
    Convert the output of vertexhead._track_params_from_hits to a list of
    acts.BoundTrackParameters, one per valid (usable) track.

    Parameters
    ----------
    track_params_dict : dict
        Output of _track_params_from_hits for a *single* event (no batch dim):
          circle_centre : (T, 2)   – (cx, cy) in cm
          R_s           : (T,)     – signed radius in cm
          zslope        : (T,)     – dz/ds
          z0            : (T,)     – z intercept in cm
          pT            : (T,)     – transverse momentum in GeV/c
          usable        : (T,)     – bool mask
          track_position: (T, 3)   – innermost hit (x,y,z) cm
    surfaces : list[acts.Surface]
    Bz : float
        Solenoid field in Tesla (used only to derive q from sign of R_s).

    Returns
    -------
    list of (acts.BoundTrackParameters, int track_idx)  or  None
    """
    if not _check_acts():
        return None

    import math as _math
    gctx = acts.GeometryContext()
    mm   = acts.UnitConstants.mm
    GeV  = acts.UnitConstants.GeV

    usable     = np.array(track_params_dict["usable"],     dtype=bool)   # (T,)
    R_s        = np.array(track_params_dict["R_s"],        dtype=float)  # (T,)
    pT         = np.array(track_params_dict["pT"],         dtype=float)  # (T,)
    zslope     = np.array(track_params_dict["zslope"],     dtype=float)  # (T,)
    z0         = np.array(track_params_dict["z0"],         dtype=float)  # (T,)
    trk_pos    = np.array(track_params_dict["track_position"], dtype=float)  # (T,3)

    results = []
    for t in range(len(usable)):
        if not usable[t]:
            continue

        # Charge sign from signed radius
        q = 1 if R_s[t] > 0 else -1
        pT_GeV = float(pT[t])
        if pT_GeV < 1e-6:
            continue

        # phi at reference point from circle centre
        cx, cy = (
            float(track_params_dict["circle_centre"][t, 0]),
            float(track_params_dict["circle_centre"][t, 1]),
        )
        x0 = float(trk_pos[t, 0])
        y0 = float(trk_pos[t, 1])
        z0_val = float(trk_pos[t, 2])

        # Direction of momentum at reference: tangent to helix
        dx = x0 - cx
        dy = y0 - cy
        r  = _math.hypot(dx, dy)
        if r < 1e-6:
            continue
        # Tangent (perpendicular to radius vector, sign depends on charge)
        phi_mom = _math.atan2(-q * dx, q * dy)

        # theta from zslope: tan(lambda) = zslope => theta = pi/2 - lambda
        lam = _math.atan(float(zslope[t]))
        theta_val = _math.pi / 2.0 - lam
        if theta_val <= 0 or theta_val >= _math.pi:
            theta_val = _math.pi / 2.0

        # q/p
        p_GeV = pT_GeV / _math.sin(theta_val)
        qop   = q / (p_GeV * GeV)

        # Reference surface: cylinder nearest to reference point radius
        r0_cm = _math.hypot(x0, y0)
        ref_surf = get_surface_for_hit(r0_cm, surfaces)
        if ref_surf is None:
            continue

        # Local coordinates on the reference surface
        gpos = acts.Vector3(x0 * 10.0 * mm, y0 * 10.0 * mm, z0_val * 10.0 * mm)
        loc2d = ref_surf.globalToLocal(gctx, gpos, acts.Vector3(0, 0, 0))
        if loc2d is None:
            continue

        # BoundVector: (loc0, loc1, phi, theta, qop, t)
        bv = acts.BoundVector()
        bv[acts.BoundIndices.eBoundLoc0]   = loc2d[0]
        bv[acts.BoundIndices.eBoundLoc1]   = loc2d[1]
        bv[acts.BoundIndices.eBoundPhi]    = phi_mom
        bv[acts.BoundIndices.eBoundTheta]  = theta_val
        bv[acts.BoundIndices.eBoundQOverP] = qop
        bv[acts.BoundIndices.eBoundTime]   = 0.0

        # Loose diagonal covariance
        cov = acts.BoundSquareMatrix()
        cov.setZero()
        cov[acts.BoundIndices.eBoundLoc0,   acts.BoundIndices.eBoundLoc0]   = (1.0 * mm) ** 2
        cov[acts.BoundIndices.eBoundLoc1,   acts.BoundIndices.eBoundLoc1]   = (1.0 * mm) ** 2
        cov[acts.BoundIndices.eBoundPhi,    acts.BoundIndices.eBoundPhi]    = (0.05) ** 2
        cov[acts.BoundIndices.eBoundTheta,  acts.BoundIndices.eBoundTheta]  = (0.05) ** 2
        cov[acts.BoundIndices.eBoundQOverP, acts.BoundIndices.eBoundQOverP] = (0.1 * qop) ** 2
        cov[acts.BoundIndices.eBoundTime,   acts.BoundIndices.eBoundTime]   = (1.0) ** 2

        bp = acts.BoundTrackParameters(
            ref_surf.getSharedPtr(),
            bv,
            cov,
            acts.ParticleHypothesis.pion(),
        )
        results.append((bp, t))

    return results


# ---------------------------------------------------------------------------
# 3.  is_assigned bool array -> ProtoTrackContainer
# ---------------------------------------------------------------------------

def assigned_to_prototracks(is_assigned: np.ndarray) -> Optional[object]:
    """
    Convert the boolean assignment array from VertexHead to an ACTS
    ProtoTrackContainer.

    Parameters
    ----------
    is_assigned : np.ndarray shape (N_hits, N_tracks)
        Boolean array; is_assigned[h, t] == True means hit h belongs to
        track t (for a single event, batch dimension already removed).

    Returns
    -------
    acts.ProtoTrackContainer or None
    """
    if not _check_acts():
        return None

    N_hits, N_tracks = is_assigned.shape
    container = acts.ProtoTrackContainer()

    for t in range(N_tracks):
        hit_indices = np.where(is_assigned[:, t])[0].tolist()
        if len(hit_indices) < 3:
            continue
        proto = acts.ProtoTrack(hit_indices)
        container.push_back(proto)

    return container
