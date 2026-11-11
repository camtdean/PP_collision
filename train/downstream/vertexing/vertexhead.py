"""
vertexhead.py: VertexHead, vertex reconstruction from track-finder output and the training interface: TrackBuilder
(tracks.py) then the primary and secondary vertices (primaryvertex.py's PrimaryVertices, secondaryvertex.py's
SecondaryVertices, both on vertexfit.py's VertexFit, the shared fit). The PV is the first vertex the AMVF keeps,
refitted with the SVs sharing its tracks. ACTS's hard cuts are soft (sigmoids of width cut_softness_*);
hard_cuts=True restores them (the check against ACTS).

Inputs: class_probs (B, T, 2), mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N);
B events, N hits, T track slots, K = max_sv_per_event. Positions in cm.
"""
import functools
import inspect
import os

import torch
import torch.nn as nn

from .helix_tools import _expand, _impact_at_points
from .primaryvertex import PrimaryVertices
from .secondaryvertex import SecondaryVertices
from .tracks import TrackBuilder
from .vertexfit import VertexFit

_PARTS = (TrackBuilder, VertexFit, PrimaryVertices, SecondaryVertices)


@functools.cache
def _setting_names(part):
    """The keyword settings part.__init__ takes (each named: VertexHead routes settings by these names)."""
    parameters = inspect.signature(part.__init__).parameters
    if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in parameters.values()):
        raise TypeError(f"{part.__name__}.__init__ takes *args/**kwargs; VertexHead needs every setting named")
    return frozenset(parameters) - {"self", "vertex_fit"}


if sum(len(_setting_names(part)) for part in _PARTS) != len(frozenset().union(*map(_setting_names, _PARTS))):
    raise TypeError("TrackBuilder, VertexFit, PrimaryVertices and SecondaryVertices must not share a setting's name")


class VertexHead(nn.Module):
    """
    TrackBuilder (self.tracks) then the primary and secondary vertices (self.primary, self.secondary, on the fit
    self.vertex_fit). Takes their settings as keywords.

    Args:
        find_secondary, refit_pv: SVs, and the PV refitted together with them (tracks shared).
    """
    def __init__(self, find_secondary: bool = True, refit_pv: bool = True, **settings):
        super().__init__()
        unknown = sorted(set(settings) - self.setting_names())
        if unknown:
            raise TypeError(f"VertexHead has no settings {unknown}")
        part = lambda cls, *args: cls(*args, **{k: v for k, v in settings.items() if k in _setting_names(cls)})  # noqa: E731
        self.find_secondary, self.refit_pv = find_secondary, refit_pv
        self.tracks, self.vertex_fit = part(TrackBuilder), part(VertexFit)
        self.primary, self.secondary = part(PrimaryVertices, self.vertex_fit), part(SecondaryVertices, self.vertex_fit)

    @staticmethod
    def setting_names():
        """The settings VertexHead takes: its own and those of TrackBuilder, VertexFit and the two vertex searches."""
        return frozenset({"find_secondary", "refit_pv"}).union(*map(_setting_names, _PARTS))

    @classmethod
    def from_detector(cls, path, **overrides):
        """
        A VertexHead from a detector file (scripts/acts/detectors/<name>/<name>.yaml): use_acts when acts.image or
        acts.command is set, its vertex_head block as further settings; keyword arguments override it.
        """
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))
        resolve = lambda p: p if p is None else os.path.join(base, os.path.expanduser(p))  # noqa: E731 - absolute stays
        field, acts, hit_errors = d.get("field", {}), d.get("acts", {}) or {}, d.get("hit_errors_cm", {})
        silicon = d.get("silicon_hit_errors_cm") or {}
        # unset keys (None) take the parts' defaults
        kwargs = {
            "Bz": field.get("bz_tesla"),
            "beam_position": d.get("beam_position_cm"),
            "normalization": d.get("normalization"),
            "hit_sigma_rphi": hit_errors.get("rphi"),
            "hit_sigma_z": hit_errors.get("z"),
            "silicon_max_radius": silicon.get("max_radius"),
            "silicon_hit_sigma_rphi": silicon.get("rphi"),
            "silicon_hit_sigma_z": silicon.get("z"),
            "use_acts": bool(acts.get("image") or acts.get("command")),
            "acts_image": resolve(acts.get("image")),
            "acts_command": acts.get("command"),
            "field_map": resolve(field.get("map")),
            "field_map_kind": field.get("map_kind"),
            "acts_geometry": resolve(acts.get("geometry")),
            "acts_geometry_config": resolve(acts.get("geometry_config")),
            "acts_material_map": resolve(acts.get("material_map")),
            "acts_energy_loss": acts.get("energy_loss"),
            "acts_max_steps": acts.get("max_steps"),
            "acts_threads": acts.get("threads"),
            "acts_particle_pdg": acts.get("particle_pdg"),
            "acts_particle_mass": acts.get("particle_mass"),
            "acts_kalman_refit": acts.get("kalman_refit"),
            "acts_kalman_start_covariance_scale": acts.get("kalman_start_covariance_scale"),
            "acts_kalman_fallback_covariance_scale": acts.get("kalman_fallback_covariance_scale"),
            "acts_kalman_material": acts.get("kalman_material"),
            **(d.get("vertex_head") or {}),
        }
        unknown = sorted(set(d.get("vertex_head") or {}) - cls.setting_names())
        if unknown:
            raise ValueError(f"{path}: vertex_head sets {unknown}, which VertexHead does not take")
        return cls(**{**{k: v for k, v in kwargs.items() if v is not None}, **overrides})

    def forward(self, class_probs=None, mask_probs=None, points=None, padding_mask=None, track_parameters=None,
                track_covariance=None):
        """
        Args:
            class_probs: (B, T, 2) P(empty), P(real); always required (P(real) is the base track weight).
            mask_probs: (B, N, T); points: (B, N, 4) normalized hits; padding_mask: (B, N), True = real: the hits
                (unused with track_parameters).
            track_parameters, track_covariance: (B, T, 6), (B, T, 6, 6) given perigee tracks instead of hits
                (ACTS units: mm).

        Returns:
            dict (the PV: AMVF's first vertex; with find_secondary and refit_pv, refitted together with the SVs where
            the event has one, the SV outputs then from that refit and measured from the refitted PV):
                vertex_estimate (B, 3), chi_square, fit_is_valid (B,), vertex_covariance (B, 3, 3),
                vertex_from_fallback (B,) found with a looser track preselection (ACTS's kept none),
                vertex_fallback_step (B,) which (0: as ACTS, -1: none); its quality: vertex_n_tracks (B,)
                tracks with weight >= 0.5, vertex_ndf (B,) 2 sum(weights) - 3.
                primary_vertices (B, V, 3) all vertices the finder kept, in the order found (V = max_vertices;
                the PV first), primary_vertex_valid, primary_vertex_n_tracks (B, V),
                track_primary_vertex (B, T) each track's vertex (-1: none).
                track_parameters (B, T, 6) cm, track_position, track_direction (B, T, 3), track_weight,
                track_quality_mask, track_input_mask (the vertex fit's tracks: a weight, a fit and a finite
                covariance), track_sigma (B, T, 2) d0, z0 errors (cm) x track_error_scale, track_vertex_weight
                (B, T) each track's weight in the PV fit, track_chi2 (its 3D compatibility), track_dca (B, T) and
                track_closest_approach_point (B, T, 3) (the helix's 3D closest approach to the PV);
                acts_failed (B, T) with use_acts: usable by the hit fit, lost by ACTS;
                acts_refit_failed (B, T) with the Kalman refit: usable by the hit fit, propagated from its state (refit failed);
                track_covariance (B, T, 6, 6): the ACTS perigee covariance (mm), with use_acts (x
                acts_kalman_fallback_covariance_scale where the refit failed) or as given.
                With find_secondary: sv_estimate (B, K, 3), sv_covariance (B, K, 3, 3), sv_is_valid, sv_chi2,
                sv_chi2_per_dof, sv_displacement (decay length, cm), sv_decay_length_significance, sv_dira (cosine
                of the angle between the flight path and the momentum), sv_pair_dca (the seed pair's), sv_n_tracks
                (B, K), sv_momentum (B, K, 3) GeV, sv_track_mask, sv_track_weight (B, K, T), displaced_mask,
                displaced_weight (B, T) the candidates.
        """
        tracks, inputs = self.tracks(class_probs, mask_probs, points, padding_mask, track_parameters, track_covariance)
        bad = tracks["track_parameters"][..., 0].isnan()                 # tracks without a fit
        # the vertices kept: the first slots, in the order found (the PV: slot 0)
        state, kept, step = self.primary(inputs)
        valid = step >= 0
        pv_state = {k: v[:, :1] for k, v in state.items()}
        pv = pv_state["position"][:, 0].masked_fill(~valid.unsqueeze(-1), torch.nan)

        result = {}
        if self.find_secondary:
            # the other primary vertices' tracks are theirs (weight x (1 - their fit weight))
            others = (state["weight"] * (state["links"] > 0.5) * (kept & (torch.arange(
                kept.shape[1], device=kept.device) > 0)).unsqueeze(-1)).amax(dim=1)
            result, sv_state, sv_ok = self.secondary({**inputs, "weight": inputs["weight"] * (others < 0.5)}, pv, pv_state)
            if self.refit_pv and sv_ok.any():
                # the PV and its SVs fitted together (tracks shared) where the event has an SV; the pileup list too
                joint = {k: torch.cat([v, sv_state[k]], dim=1) for k, v in pv_state.items()}
                self.vertex_fit.fit(joint, torch.cat([(valid & sv_ok.any(dim=1)).unsqueeze(1), sv_ok], dim=1), inputs)
                pv_state, sv_state = {k: v[:, :1] for k, v in joint.items()}, {k: v[:, 1:] for k, v in joint.items()}
                state = {k: torch.cat([pv_state[k], v[:, 1:]], dim=1) for k, v in state.items()}
                pv = pv_state["position"][:, 0].masked_fill(~valid.unsqueeze(-1), torch.nan)
            # the SV outputs, measured from the (refitted) PV
            result = {**self.secondary.outputs(sv_state, sv_ok, pv, pv_state["covariance"][:, 0]), **result}

        # the PV's outputs; per-track ones NaN for tracks without a fit
        weight = (pv_state["weight"] * (pv_state["links"] > 0.5))[:, 0]
        loc, axes = _impact_at_points(inputs["perigee"], inputs["bz"], inputs["beam"], pv.nan_to_num().unsqueeze(1))
        per_track = lambda v: v.masked_fill(_expand(bad | ~valid.unsqueeze(-1), v), torch.nan)  # noqa: E731
        result.update({
            "vertex_estimate": pv, "fit_is_valid": valid, "vertex_from_fallback": step > 0,
            "vertex_fallback_step": step,
            "vertex_covariance": pv_state["covariance"][:, 0].masked_fill(~valid.view(-1, 1, 1), torch.nan),
            "chi_square": pv_state["chi2"][:, 0].masked_fill(~valid, torch.nan),
            "vertex_ndf": pv_state["ndf"][:, 0].masked_fill(~valid, torch.nan),
            "track_vertex_weight": weight.masked_fill(bad, 0.0), "vertex_n_tracks": (weight >= 0.5).sum(-1),
            "track_chi2": per_track(self.vertex_fit.compatibility(pv_state, pv.nan_to_num().unsqueeze(1)).squeeze(1)),
            "track_dca": per_track(loc.norm(dim=-1)),
            "track_closest_approach_point": per_track(pv.nan_to_num().unsqueeze(1) + (loc.unsqueeze(-1) * axes).sum(-2)),
        })
        # all primary vertices (pileup), in the order found; each track's (weight >= 0.5, else -1)
        fit_weight = state["weight"] * (state["links"] > 0.5) * kept.unsqueeze(-1)
        best, index = fit_weight.max(dim=1)
        result.update({
            "primary_vertices": state["position"].masked_fill(~kept.unsqueeze(-1), torch.nan),
            "primary_vertex_valid": kept,
            "primary_vertex_n_tracks": (fit_weight >= 0.5).sum(-1),
            "track_primary_vertex": torch.where(best >= 0.5, index, -1),
        })
        return {**tracks, **result}
