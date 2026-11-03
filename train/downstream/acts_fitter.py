"""
acts_fitter.py
--------------
Wraps ACTS Kalman Filter (KF) track fitting and Adaptive Multi-Vertex
Finder (AMVF) vertex finding for single-event evaluation.

Public API
----------
fit_tracks(bound_params_list, measurements, geometry, magnetic_field)
    -> list[acts.ConstTrackContainer]

find_vertex_amvf(track_container, magnetic_field)
    -> np.ndarray shape (3,) [x, y, z] in cm  or  None
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

from acts_geometry import BZ_DEFAULT


def _check_acts() -> bool:
    if not _ACTS_AVAILABLE:
        warnings.warn(
            "ACTS not available; fitter functions are no-ops.",
            ImportWarning, stacklevel=3,
        )
        return False
    return True


def _default_kf_options(gctx, mctx, cctx):
    """Build default KalmanFitterOptions."""
    extensions = acts.KalmanFitterExtensions()
    extensions.calibrator.connect(acts.makePassThroughCalibrator())
    extensions.smoother.connect(acts.gainMatrixSmoother())
    extensions.updater.connect(acts.gainMatrixUpdater())

    opts = acts.KalmanFitterOptions(gctx, mctx, cctx, extensions)
    opts.multipleScattering = True
    opts.energyLoss         = True
    return opts


# ---------------------------------------------------------------------------
# 1.  Kalman Filter track fitting
# ---------------------------------------------------------------------------

def fit_tracks(
    bound_params_list,
    measurements,
    geometry,
    magnetic_field,
) -> Optional[List]:
    """
    Run the ACTS Kalman Filter on each track seed independently.

    Parameters
    ----------
    bound_params_list : list of (acts.BoundTrackParameters, int track_idx)
        Output of acts_bridge.track_params_to_bound.
    measurements : acts.MeasurementContainer
        All hit measurements for this event.
    geometry : acts.TrackingGeometry
    magnetic_field : acts.ConstantBField

    Returns
    -------
    list of (acts.ConstTrackContainer, int track_idx)  or  None
    """
    if not _check_acts():
        return None

    if bound_params_list is None or len(bound_params_list) == 0:
        return []

    gctx, mctx, cctx = (
        acts.GeometryContext(),
        acts.MagneticFieldContext(),
        acts.CalibrationContext(),
    )

    # Propagator: straight-line + Runge-Kutta stepper in the solenoid
    stepper    = acts.EigenStepper(magnetic_field)
    navigator  = acts.Navigator(geometry)
    propagator = acts.Propagator(stepper, navigator)

    kf   = acts.KalmanFitter(propagator)
    opts = _default_kf_options(gctx, mctx, cctx)

    results = []
    for bp, t_idx in bound_params_list:
        try:
            track_container = acts.TrackContainer(
                acts.VectorTrackContainer(), acts.VectorMultiTrajectory()
            )
            result = kf.fit(measurements, bp, opts, track_container)
            if result.ok():
                const_tc = acts.ConstTrackContainer(track_container)
                results.append((const_tc, t_idx))
        except Exception as exc:
            warnings.warn(f"KF fit failed for track {t_idx}: {exc}", RuntimeWarning)

    return results


# ---------------------------------------------------------------------------
# 2.  AMVF vertex finding
# ---------------------------------------------------------------------------

def find_vertex_amvf(
    fitted_tracks,
    magnetic_field,
) -> Optional[np.ndarray]:
    """
    Run the ACTS Adaptive Multi-Vertex Finder on fitted tracks.

    Parameters
    ----------
    fitted_tracks : list of (acts.ConstTrackContainer, int)
        Output of fit_tracks.  Only the track containers are used.
    magnetic_field : acts.ConstantBField

    Returns
    -------
    np.ndarray shape (3,) [x, y, z] in cm  or  None if no vertex found.
    """
    if not _check_acts():
        return None

    if fitted_tracks is None or len(fitted_tracks) == 0:
        return None

    gctx, mctx = acts.GeometryContext(), acts.MagneticFieldContext()
    mm = acts.UnitConstants.mm

    # Collect InputTracks from fitted containers
    input_tracks = []
    for tc, _ in fitted_tracks:
        for track in tc:
            input_tracks.append(acts.InputTrack(track))

    if len(input_tracks) < 2:
        return None

    # AMVF configuration
    amvf_cfg                       = acts.AdaptiveMultiVertexFinder.Config()
    amvf_cfg.looseConstrValue      = 1e3
    amvf_cfg.maxIterations         = 200
    amvf_cfg.doSmoothing           = True
    amvf_cfg.useVertexCovForIPEstimation = False

    # Linear track parametrisation at the beam line
    lpf_cfg                  = acts.FullBilloirVertexFitter.Config()
    lpf_cfg.maxIterations    = 50
    billoirFitter            = acts.FullBilloirVertexFitter(lpf_cfg)

    linearizer_cfg           = acts.HelicalTrackLinearizer.Config()
    linearizer_cfg.bField    = magnetic_field
    linearizer                = acts.HelicalTrackLinearizer(linearizer_cfg)

    amvf_cfg.vertexFitter   = billoirFitter
    amvf_cfg.trackLinearizer = linearizer

    amvf  = acts.AdaptiveMultiVertexFinder(amvf_cfg)
    vstate = acts.AdaptiveMultiVertexFinder.State(magnetic_field, mctx)

    vertices = amvf.find(input_tracks, acts.VertexingOptions(gctx, mctx), vstate)

    if not vertices or len(vertices) == 0:
        return None

    # Pick vertex with highest sum-pT^2 of assigned tracks
    best = max(
        vertices,
        key=lambda v: sum(
            t.transverseMomentum() ** 2 for t in v.tracks()
        ),
    )

    pos = best.position()   # acts.Vector4 in mm
    # Convert mm -> cm
    return np.array([pos[0] / 10.0, pos[1] / 10.0, pos[2] / 10.0])


# ---------------------------------------------------------------------------
# 3.  Convenience: full pipeline for one event
# ---------------------------------------------------------------------------

def run_acts_for_event(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    r_cm: np.ndarray,
    track_params_dict: dict,
    is_assigned: np.ndarray,
    geometry,
    magnetic_field,
    surfaces,
    Bz: float = BZ_DEFAULT,
) -> dict:
    """
    Full ACTS pipeline for a single event.

    Parameters
    ----------
    x_cm, y_cm, z_cm, r_cm : np.ndarray shape (N_hits,)
        Un-normalised hit coordinates in cm.
    track_params_dict : dict
        Output of vertexhead._track_params_from_hits (single event, no batch).
    is_assigned : np.ndarray shape (N_hits, N_tracks) bool
    geometry, magnetic_field, surfaces
        From acts_geometry.build_tpc_geometry.
    Bz : float

    Returns
    -------
    dict with keys:
        vertex_estimate   : np.ndarray (3,) in cm  or  None
        n_fitted_tracks   : int
        fit_is_valid      : bool
    """
    from acts_bridge import (
        hits_to_measurements,
        track_params_to_bound,
    )

    measurements   = hits_to_measurements(x_cm, y_cm, z_cm, r_cm, surfaces)
    bound_params   = track_params_to_bound(track_params_dict, surfaces, Bz=Bz)

    if measurements is None or bound_params is None:
        return {"vertex_estimate": None, "n_fitted_tracks": 0, "fit_is_valid": False}

    fitted = fit_tracks(bound_params, measurements, geometry, magnetic_field)

    if not fitted:
        return {"vertex_estimate": None, "n_fitted_tracks": 0, "fit_is_valid": False}

    vertex_cm = find_vertex_amvf(fitted, magnetic_field)

    return {
        "vertex_estimate": vertex_cm,
        "n_fitted_tracks": len(fitted),
        "fit_is_valid":    vertex_cm is not None,
    }
