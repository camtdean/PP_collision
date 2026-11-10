"""
vertexhead.py: VertexHead, vertex reconstruction from track-finder output and the training interface: TrackBuilder
(tracks.py) then the primary and secondary vertices (primaryvertex.py's PrimaryVertices, secondaryvertex.py's
SecondaryVertices, both on vertexfit.py's VertexFit: settings and the shared fit).

Primary vertices: ACTS's adaptive multi-vertex finder (AMVF) with the settings of its shipped example
(AdaptiveMultiVertexFinderAlgorithm, Gaussian track-density seeder): seed, preselect tracks, fit all connected
vertices together (adaptive multi-vertex fit: Kalman vertex updates on linearised helices, annealed weights shared
between vertices), keep or delete the new vertex, repeat on the tracks left. The PV is the first vertex kept.
Secondary vertices: sPHENIX's SecondaryVertexFinder pair selection (circle intersections of displaced tracks), each
seed fitted with the same vertex fit; the PV is then refitted with the SVs sharing tracks. ACTS's hard cuts are soft
(sigmoids of width cut_softness_*); hard_cuts=True restores them (the check against ACTS).

Inputs: class_probs (B, T, 2), mask_probs (B, N, T), normalized points (B, N, 4), padding_mask (B, N);
B events, N hits, T track slots, K = max_sv_per_event. Positions in cm.
"""
import inspect
import math
import os

import torch

from .helix_tools import _expand, _impact_with_jacobian
from .primaryvertex import _NO_ORDER, PrimaryVertices
from .secondaryvertex import SecondaryVertices
from .tracks import TrackBuilder
from .vertexfit import VertexFit


def _setting_names(part):
    return set(inspect.signature(part.__init__).parameters) - {"self"}


class VertexHead(PrimaryVertices, SecondaryVertices):
    """
    TrackBuilder (self.tracks) then the primary and secondary vertices. Takes the settings of TrackBuilder and VertexFit
    as keywords (beam_position goes to both).
    """
    def __init__(self, **settings):
        unknown = sorted(set(settings) - self.setting_names())
        if unknown:
            raise TypeError(f"VertexHead has no settings {unknown}")
        super().__init__(**{k: v for k, v in settings.items() if k in _setting_names(VertexFit)})
        self.tracks = TrackBuilder(**{k: v for k, v in settings.items() if k in _setting_names(TrackBuilder)})

    @staticmethod
    def setting_names():
        """The settings VertexHead takes: those of TrackBuilder and VertexFit."""
        return _setting_names(TrackBuilder) | _setting_names(VertexFit)

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
        # unset keys (None) take TrackBuilder's / VertexFit's defaults
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
                primary_vertices (B, V, 3) all vertices the finder kept, in the order found (V = max_vertices + 1;
                the PV first, pv_index 0), primary_vertex_valid, primary_vertex_n_tracks (B, V), pv_index (B,),
                track_primary_vertex (B, T) each track's vertex (-1: none).
                track_parameters (B, T, 6) cm, track_position, track_direction (B, T, 3), track_weight,
                track_quality_mask, track_sigma (B, T, 2) d0, z0 errors (cm) x track_error_scale, track_vertex_weight
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
        tracks = self.tracks(class_probs, mask_probs, points, padding_mask, track_parameters, track_covariance)
        dtype = torch.float64
        bad = tracks["_bad"]
        track_weight = tracks["track_weight"].to(dtype)
        inputs = {"perigee": tracks["track_parameters"][..., :5].to(dtype).nan_to_num(),
                  "covariance": tracks["_covariance"].to(dtype), "bz": tracks["_bz"].to(dtype),
                  "n_hits": tracks["_n_hits"].to(dtype), "n_silicon_hits": tracks["_n_silicon_hits"].to(dtype)}
        inputs["input"] = (track_weight > 0) & ~bad & torch.isfinite(inputs["covariance"]).all(dim=(-2, -1))
        eye = torch.eye(5, dtype=dtype, device=track_weight.device)
        inputs["covariance"] = torch.where(inputs["input"][..., None, None], inputs["covariance"], eye)
        inputs["perigee"] = torch.where(inputs["input"].unsqueeze(-1), inputs["perigee"],
                                        inputs["perigee"].new_tensor([0.0, 0.0, 0.0, math.pi / 2, 1.0]))
        track_weight = track_weight * inputs["input"]
        n_events = track_weight.shape[0]
        events = torch.arange(n_events, device=track_weight.device)

        state, kept, order = self._find_primary(inputs, track_weight,
                                                (self.tracks_max_z_interval, self.tracks_max_significance))
        step = torch.where(kept.any(dim=1), 0, -1)
        for k, preselection in enumerate(self.fallback_preselection if self.pv_fallback and not self.hard_cuts else ()):
            # events without a vertex (only those): searched again with a looser preselection
            missing = (step < 0).nonzero().squeeze(1)
            if not len(missing):
                break
            again, again_kept, again_order = self._find_primary({key: v[missing] for key, v in inputs.items()},
                                                                track_weight[missing], preselection)
            use = again_kept.any(dim=1)
            found = (missing[use],)
            state = {key: value.index_put(found, again[key][use]) for key, value in state.items()}
            kept, order = kept.index_put(found, again_kept[use]), order.index_put(found, again_order[use])
            step = step.index_put(found, step.new_tensor(k + 1))
        pv_slot = order.argmin(dim=1)
        slot_state = {k: v[events, pv_slot] for k, v in state.items()}
        valid, from_fallback = step >= 0, step > 0
        pv = slot_state["position"].masked_fill(~valid.unsqueeze(-1), torch.nan)

        result = {}
        if self.find_secondary:
            pv_state = {k: v.unsqueeze(1) for k, v in slot_state.items()}
            # the other primary vertices' tracks are theirs (weight x (1 - their fit weight))
            others = (state["weight"] * (state["links"] > 0.5) * (kept & (order != order.gather(1, pv_slot[:, None]))
                                                                 ).unsqueeze(-1)).amax(dim=1)
            secondary, sv_state, sv_ok = self._find_secondary(inputs, track_weight * (others < 0.5), pv, pv_state)
            result.update(secondary)
            if self.refit_pv and sv_ok.any():
                # the PV and its SVs fitted together (tracks shared); used where the event has an SV
                joint = {k: torch.cat([v, sv_state[k]], dim=1) for k, v in pv_state.items()}
                which = torch.cat([(valid & sv_ok.any(dim=1)).unsqueeze(1), sv_ok], dim=1)
                self._fit(joint, which, track_weight, inputs)
                refit = which[:, 0]
                slot_state = {k: torch.where(_expand(refit, v), joint[k][:, 0], v) for k, v in slot_state.items()}
                state = {k: v.index_put((events, pv_slot), slot_state[k]) for k, v in state.items()}   # pileup list too
                pv = slot_state["position"].masked_fill(~valid.unsqueeze(-1), torch.nan)
                # the SV outputs from the same refit, measured from the refitted PV (the SV selection is unchanged)
                joint_sv = self._sv_outputs({k: v[:, 1:] for k, v in joint.items()}, sv_ok, pv, slot_state["covariance"])
                for key, value in joint_sv.items():
                    result[key] = torch.where(_expand(refit, value), value, result[key])

        # the PV's outputs; per-track ones NaN for tracks without a fit
        weight = slot_state["weight"] * (slot_state["links"] > 0.5)
        track_chi2 = self._compatibility({k: v.unsqueeze(1) for k, v in slot_state.items()}, pv.nan_to_num().unsqueeze(1)
                                         ).squeeze(1)
        loc, axes, _ = _impact_with_jacobian(inputs["perigee"], inputs["bz"], inputs["perigee"].new_tensor(
            self.beam_position), pv.nan_to_num().unsqueeze(1))
        closest = pv.nan_to_num().unsqueeze(1) + (loc.unsqueeze(-1) * axes).sum(-2)
        per_track = lambda v: v.masked_fill(_expand(bad | ~valid.unsqueeze(-1), v), torch.nan)  # noqa: E731
        result.update({
            "vertex_estimate": pv, "fit_is_valid": valid, "vertex_from_fallback": from_fallback,
            "vertex_fallback_step": step,
            "vertex_covariance": slot_state["covariance"].masked_fill(~valid.view(-1, 1, 1), torch.nan),
            "chi_square": slot_state["chi2"].masked_fill(~valid, torch.nan),
            "vertex_ndf": slot_state["ndf"].masked_fill(~valid, torch.nan),
            "track_vertex_weight": weight.masked_fill(bad, 0.0), "vertex_n_tracks": (weight >= 0.5).sum(-1),
            "track_chi2": per_track(track_chi2), "track_dca": per_track(loc.norm(dim=-1)),
            "track_closest_approach_point": per_track(closest),
        })
        # all primary vertices (pileup), in the order found; each track's (weight >= 0.5, else -1)
        rank = torch.where(kept, order, order.new_tensor(_NO_ORDER)).argsort(dim=1)
        vertices_kept = kept.gather(1, rank)
        fit_weight = (state["weight"] * (state["links"] > 0.5)).gather(1, rank.unsqueeze(-1).expand_as(state["weight"]))
        fit_weight = fit_weight * vertices_kept.unsqueeze(-1)
        best, index = fit_weight.max(dim=1)
        result.update({
            "primary_vertices": state["position"].gather(1, rank.unsqueeze(-1).expand(-1, -1, 3)).masked_fill(
                ~vertices_kept.unsqueeze(-1), torch.nan),
            "primary_vertex_valid": vertices_kept,
            "primary_vertex_n_tracks": (fit_weight >= 0.5).sum(-1),
            "pv_index": torch.zeros_like(pv_slot),
            "track_primary_vertex": torch.where(best >= 0.5, index, -1),
        })
        return {**{k: v for k, v in tracks.items() if not k.startswith("_")}, **result}
