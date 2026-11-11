"""
secondaryvertex.py: SecondaryVertices, VertexHead's secondary vertices: sPHENIX's SecondaryVertexFinder pair
selection (circle intersections of displaced tracks), each seed fitted with VertexFit's vertex fit.
"""
import math

import torch
import torch.nn as nn

from .helix_tools import _expand, _helix_at, _helix_centre, _perigee_at_points, _soft_above, _soft_below
from .vertexfit import _events


class SecondaryVertices(nn.Module):
    """
    VertexHead's secondary-vertex search and outputs, on the vertex fit vertex_fit (VertexFit). Defaults: sPHENIX's
    SecondaryVertexFinder; cm.

    Args:
        sv_min_track_hits: SV track candidates' TPC hits; displaced from the PV: tracks with silicon hits by
            sPHENIX's impact cuts sv_min_track_dca_xy, sv_min_track_dca_z (cm), TPC-only tracks (their resolution
            is too coarse for those) by compatibility with the PV above sv_min_track_pv_chi2; tracks of the other
            primary vertices are not candidates.
        sv_max_radius, sv_max_projected_dz, sv_max_pair_dca, sv_min_path (cm): pair seeds at circle intersections.
        sv_max_track_chi2: further tracks attached to a seed; sv_max_chi2_per_dof; sv_merge_distance (cm): one SV per
            group closer than this; max_sv_per_event.
    """
    def __init__(
        self,
        vertex_fit,
        sv_min_track_dca_xy: float = 0.02,
        sv_min_track_dca_z: float = 0.02,
        sv_min_track_hits: int = 20,
        sv_min_track_pv_chi2: float = 9.0,
        sv_max_radius: float = 40.0,
        sv_max_projected_dz: float = 1.0,
        sv_max_pair_dca: float = 0.5,
        sv_min_path: float = 0.2,
        sv_max_track_chi2: float = 9.0,
        sv_max_chi2_per_dof: float = 10.0,
        sv_merge_distance: float = 0.1,
        max_sv_per_event: int = 6,
    ):
        super().__init__()
        self.vertex_fit = vertex_fit
        self.sv_min_track_dca_xy = sv_min_track_dca_xy
        self.sv_min_track_dca_z = sv_min_track_dca_z
        self.sv_min_track_hits = sv_min_track_hits
        self.sv_min_track_pv_chi2 = sv_min_track_pv_chi2
        self.sv_max_radius = sv_max_radius
        self.sv_max_projected_dz = sv_max_projected_dz
        self.sv_max_pair_dca = sv_max_pair_dca
        self.sv_min_path = sv_min_path
        self.sv_max_track_chi2 = sv_max_track_chi2
        self.sv_max_chi2_per_dof = sv_max_chi2_per_dof
        self.sv_merge_distance = sv_merge_distance
        self.max_sv_per_event = max_sv_per_event

    def _sv_seeds(self, tracks, candidate, pv):
        """
        Pair seeds (B, P, 3) and their pair DCA (B, P) of candidate tracks (B, T): the x-y intersections of their
        circles within sv_max_radius, the helix z there within sv_max_projected_dz, the helix tangents there closer
        than sv_max_pair_dca, the midpoint further than sv_min_path from the PV (soft; score (B, P)); and the pairs'
        tracks (B, P, 2).
        """
        perigee = tracks["perigee"]
        centre, rho = _helix_centre(perigee, tracks["bz"], tracks["beam"])
        n_tracks = perigee.shape[1]
        i, j = torch.triu_indices(n_tracks, n_tracks, offset=1, device=perigee.device)
        c1, c2, r1, r2 = centre[:, i], centre[:, j], rho[:, i].abs(), rho[:, j].abs()
        d = (c2 - c1).norm(dim=-1)
        meets = (d > 0) & (d <= r1 + r2) & (d >= (r1 - r2).abs()) & candidate[:, i] & candidate[:, j]
        d = torch.where(meets, d, 1.0)
        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        h = (r1 * r1 - a * a).clamp(min=0).sqrt()
        along = (c2 - c1) / d.unsqueeze(-1)
        across = torch.stack([-along[..., 1], along[..., 0]], -1)
        base = c1 + a.unsqueeze(-1) * along
        points = torch.stack([base + h.unsqueeze(-1) * across, base - h.unsqueeze(-1) * across], 2)   # (B, pairs, 2, 2)
        # each pair's helices at its two intersections
        x1, u1 = _helix_at(perigee[:, i].unsqueeze(2).expand(-1, -1, 2, -1), rho[:, i].unsqueeze(2), centre[:, i].unsqueeze(2), points)
        x2, u2 = _helix_at(perigee[:, j].unsqueeze(2).expand(-1, -1, 2, -1), rho[:, j].unsqueeze(2), centre[:, j].unsqueeze(2), points)
        # the tangent lines' closest approach
        w0, b = x1 - x2, (u1 * u2).sum(-1)
        d1, d2 = (u1 * w0).sum(-1), (u2 * w0).sum(-1)
        denominator = (1 - b * b).clamp(min=1e-12)
        s1, s2 = (b * d2 - d1) / denominator, (d2 - b * d1) / denominator
        p1, p2 = x1 + s1.unsqueeze(-1) * u1, x2 + s2.unsqueeze(-1) * u2
        pair_dca, midpoint = (p1 - p2).norm(dim=-1), (p1 + p2) / 2
        cm = self.vertex_fit.width("cm")
        score = (meets.unsqueeze(-1).to(d.dtype)
                 * _soft_below(points.norm(dim=-1), self.sv_max_radius, cm)
                 * _soft_below((x1[..., 2] - x2[..., 2]).abs(), self.sv_max_projected_dz, cm)
                 * _soft_below(pair_dca, self.sv_max_pair_dca, cm)
                 * _soft_above((midpoint - pv.unsqueeze(1).unsqueeze(1)).norm(dim=-1), self.sv_min_path, cm))
        n_events = perigee.shape[0]
        pairs = torch.stack([i, j], -1).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 2)
        return (midpoint.reshape(n_events, -1, 3), pair_dca.reshape(n_events, -1), score.reshape(n_events, -1),
                pairs.unsqueeze(0).expand(n_events, -1, -1))

    def forward(self, tracks, pv, pv_state):
        """SV finding after the PV (tracks' weight: without the other primary vertices' tracks): sv_pair_dca,
        displaced_mask, displaced_weight; the SV states (B, K = max_sv_per_event) and which are valid."""
        vertex_fit, track_weight = self.vertex_fit, tracks["weight"]
        n_events, n_tracks = track_weight.shape
        dtype, device = track_weight.dtype, track_weight.device
        K = self.max_sv_per_event
        # candidates (soft): at least sv_min_track_hits TPC hits; displaced from the PV: with silicon hits by
        # sPHENIX's impact cuts (sv_min_track_dca_xy / _z), TPC-only by PV compatibility (sv_min_track_pv_chi2)
        q, _ = _perigee_at_points(tracks["perigee"], tracks["bz"], tracks["beam"], pv.nan_to_num().unsqueeze(1))
        cm, sigma = vertex_fit.width("cm"), vertex_fit.width("sigma")
        pv_chi2 = vertex_fit.compatibility(pv_state, pv.nan_to_num().unsqueeze(1)).squeeze(1)
        displaced = torch.where(
            tracks["n_silicon_hits"] > 0,
            _soft_above(q[..., 0].abs(), self.sv_min_track_dca_xy, cm) * _soft_above(q[..., 1].abs(),
                                                                                  self.sv_min_track_dca_z, cm),
            _soft_above(pv_chi2, self.sv_min_track_pv_chi2, sigma * 2 * math.sqrt(self.sv_min_track_pv_chi2)))
        candidate_weight = (displaced * (track_weight > 0) * tracks["input"].to(dtype)
                            * _soft_above(tracks["n_hits"] - tracks["n_silicon_hits"], self.sv_min_track_hits - 0.5,
                                          vertex_fit.width("hits"))
                            * torch.isfinite(pv).all(-1, keepdim=True))
        candidate = candidate_weight > 0.5
        if n_tracks >= 2:
            seeds, pair_dca, score, pairs = self._sv_seeds(tracks, candidate, pv)
        else:   # no pairs: one empty stand-in
            seeds = torch.zeros(n_events, 1, 3, dtype=dtype, device=device)
            pair_dca, score = seeds[..., 0] + torch.inf, seeds[..., 0]
            pairs = torch.zeros(n_events, 1, 2, dtype=torch.long, device=device)
        # the K best pairs (smallest pair DCA)
        best = pair_dca.masked_fill(score <= 0.5, torch.inf).argsort(dim=-1)[:, :K]
        if best.shape[1] < K:
            best = torch.cat([best, best.new_zeros(n_events, K - best.shape[1])], 1)
            valid_slot = torch.arange(K, device=device) < score.shape[1]
        else:
            valid_slot = torch.ones(K, dtype=torch.bool, device=device)
        seed_ok = (score.gather(1, best) > 0.5) & valid_slot
        seed = torch.where(seed_ok.unsqueeze(-1), seeds.gather(1, best.unsqueeze(-1).expand(-1, -1, 3)), torch.nan)
        seed_pair_dca = pair_dca.gather(1, best).masked_fill(~seed_ok, torch.nan)
        pair = pairs.gather(1, best.unsqueeze(-1).expand(-1, -1, 2))
        # the seeds' states (the slots without a seed stay empty)
        state = vertex_fit.empty_vertices(n_events, K, n_tracks, dtype, device)
        at = seed_ok.nonzero(as_tuple=True)
        if len(at[0]):
            fresh = vertex_fit.new_vertices(seed[at].unsqueeze(1), torch.zeros(len(at[0]), 1, n_tracks, dtype=dtype,
                                                                                device=device),
                                            _events(tracks, at[0]), guarded=True)
            state = {key: value.index_put(at, fresh[key][:, 0]) for key, value in state.items()}
        # each seed's tracks: its pair, and candidates compatible with the seed (chi2 <= sv_max_track_chi2, soft)
        compatible = _soft_below(vertex_fit.compatibility(state, seed), self.sv_max_track_chi2,
                                 sigma * 2 * math.sqrt(self.sv_max_track_chi2))
        in_pair = torch.zeros_like(compatible, dtype=torch.bool).scatter_(2, pair, True)
        links = torch.where(in_pair, 1.0, compatible) * candidate_weight.unsqueeze(1) * seed_ok.unsqueeze(-1)
        state["links"] = links
        vertex_fit.fit(state, seed_ok, tracks, shared=False)
        chi2_per_dof = state["chi2"] / state["ndf"].clamp(min=1e-12)
        ok = (seed_ok & (((links > 0.5) & (state["compatibility"] < vertex_fit.max_vertex_chi2)).sum(-1) >= 2)
              & (state["ndf"] > 0) & (chi2_per_dof <= self.sv_max_chi2_per_dof)
              & ((state["position"] - pv.unsqueeze(1)).norm(dim=-1) > self.sv_min_path))
        # one SV per group within sv_merge_distance: the lowest chi2/ndf (ties: the lower slot)
        distance = (state["position"].unsqueeze(2) - state["position"].unsqueeze(1)).nan_to_num().norm(dim=-1)
        quality = chi2_per_dof.masked_fill(~ok, torch.inf)
        beats = (quality.unsqueeze(1) < quality.unsqueeze(2)) | ((quality.unsqueeze(1) == quality.unsqueeze(2))
                                                                  & torch.ones(K, K, dtype=torch.bool, device=device).tril(-1))
        ok = ok & ~(ok.unsqueeze(1) & (distance < self.sv_merge_distance) & beats).any(dim=2)
        return ({"sv_pair_dca": seed_pair_dca.masked_fill(~ok, torch.nan), "displaced_mask": candidate,
                 "displaced_weight": track_weight * candidate_weight}, state, ok)

    def outputs(self, state, ok, pv, pv_covariance):
        """The SV outputs of SV states (B, K) kept where ok, measured from the PV pv (B, 3) with covariance (B, 3, 3):
        the fit, the momentum at the SV (its tracks refitted to it) and the KFParticle-style topology (decay length
        significance with the two vertices' covariances added)."""
        tracks = (state["links"] > 0.5) & (state["weight"] >= 0.5) & ok.unsqueeze(-1)
        momentum = self._vertex_momentum(state, tracks)
        path = state["position"] - pv.unsqueeze(1)
        length = path.norm(dim=-1)
        covariance = state["covariance"] + torch.nan_to_num(pv_covariance).unsqueeze(1)
        unit = path / length.unsqueeze(-1).clamp(min=1e-12)
        sigma = (unit.unsqueeze(-2) @ covariance @ unit.unsqueeze(-1)).squeeze(-1).squeeze(-1).sqrt()
        nan = lambda v: v.masked_fill(~_expand(ok, v), torch.nan)  # noqa: E731
        return {
            "sv_estimate": nan(state["position"]), "sv_is_valid": ok, "sv_covariance": nan(state["covariance"]),
            "sv_chi2": nan(state["chi2"]), "sv_chi2_per_dof": nan(state["chi2"] / state["ndf"].clamp(min=1e-12)),
            "sv_displacement": nan(length), "sv_decay_length_significance": nan(length / sigma),
            "sv_momentum": nan(momentum),
            "sv_dira": nan((unit * momentum).sum(-1) / momentum.norm(dim=-1).clamp(min=1e-12)),
            "sv_n_tracks": tracks.sum(-1), "sv_track_mask": tracks,
            "sv_track_weight": state["weight"] * ok.unsqueeze(-1),
        }

    def _vertex_momentum(self, state, tracks):
        """Sum (B, K, 3) of the momenta (GeV) of each vertex's tracks (B, K, T) refitted to it."""
        A, B, G = state["A"], state["B"], state["G"]
        GB = G @ B
        r = state["q_minus_c"] - (A @ state["position"].nan_to_num().unsqueeze(2).unsqueeze(-1)).squeeze(-1)
        p = (torch.linalg.inv(B.transpose(-1, -2) @ GB) @ GB.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
        phi, theta, qop = p.unbind(-1)
        vector = torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)],
                             -1) / qop.abs().unsqueeze(-1)
        return torch.where(tracks.unsqueeze(-1), vector.nan_to_num(), 0.0).sum(2)
