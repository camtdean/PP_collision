"""
secondaryvertex.py: SecondaryVertices, VertexHead's secondary vertices: sPHENIX's SecondaryVertexFinder pair
selection (circle intersections of displaced tracks), each seed fitted with VertexFit's vertex fit.
"""
import math

import torch

from .helix_tools import _expand, _helix_radius, _soft_above, _soft_below, _transport, _wrap
from .vertexfit import VertexFit



class SecondaryVertices(VertexFit):
    """VertexHead's secondary-vertex search and outputs (on VertexFit's settings and fit)."""
    def _sv_seeds(self, tracks, candidate, pv):
        """
        Pair seeds (B, P, 3) and their pair DCA (B, P) of candidate tracks (B, T): the x-y intersections of their
        circles within sv_max_radius, the helix z there within sv_max_projected_dz, the helix tangents there closer
        than sv_max_pair_dca, the midpoint further than sv_min_path from the PV (soft; score (B, P)); and the pairs'
        tracks (B, P, 2).
        """
        perigee, bz = tracks["perigee"], tracks["bz"]
        beam = perigee.new_tensor(self.beam_position)
        d0, phi = perigee[..., 0], perigee[..., 2]
        rho = _helix_radius(perigee, bz)
        centre = beam + d0.unsqueeze(-1) * torch.stack([-torch.sin(phi), torch.cos(phi)], -1) \
            + rho.unsqueeze(-1) * torch.stack([torch.sin(phi), -torch.cos(phi)], -1)
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

        def at(track, xy):
            """Each track's helix at the x-y point xy: position (z by arc length) and unit direction."""
            q = perigee[:, track].unsqueeze(2).expand(-1, -1, 2, -1)
            p_rho, p_centre = rho[:, track].unsqueeze(2), centre[:, track].unsqueeze(2)
            u = xy - p_centre
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            t = torch.sign(p_rho).unsqueeze(-1) * torch.stack([u[..., 1], -u[..., 0]], -1)
            phi_at = torch.atan2(t[..., 1], t[..., 0])
            z = q[..., 1] - p_rho * _wrap(phi_at - q[..., 2]) / torch.tan(q[..., 3])
            sin_t = torch.sin(q[..., 3]).unsqueeze(-1)
            return (torch.cat([xy, z.unsqueeze(-1)], -1),
                    torch.cat([sin_t * t, torch.cos(q[..., 3]).unsqueeze(-1)], -1))
        x1, u1 = at(i, points)
        x2, u2 = at(j, points)
        # the tangent lines' closest approach
        w0, b = x1 - x2, (u1 * u2).sum(-1)
        d1, d2 = (u1 * w0).sum(-1), (u2 * w0).sum(-1)
        denominator = (1 - b * b).clamp(min=1e-12)
        s1, s2 = (b * d2 - d1) / denominator, (d2 - b * d1) / denominator
        p1, p2 = x1 + s1.unsqueeze(-1) * u1, x2 + s2.unsqueeze(-1) * u2
        pair_dca, midpoint = (p1 - p2).norm(dim=-1), (p1 + p2) / 2
        cm = self._width("cm")
        score = (meets.unsqueeze(-1).to(d.dtype)
                 * _soft_below(points.norm(dim=-1), self.sv_max_radius, cm)
                 * _soft_below((x1[..., 2] - x2[..., 2]).abs(), self.sv_max_projected_dz, cm)
                 * _soft_below(pair_dca, self.sv_max_pair_dca, cm)
                 * _soft_above((midpoint - pv.unsqueeze(1).unsqueeze(1)).norm(dim=-1), self.sv_min_path, cm))
        n_events = perigee.shape[0]
        pairs = torch.stack([i, j], -1).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 2)
        return (midpoint.reshape(n_events, -1, 3), pair_dca.reshape(n_events, -1), score.reshape(n_events, -1),
                pairs.unsqueeze(0).expand(n_events, -1, -1))

    def _find_secondary(self, tracks, track_weight, pv, pv_state):
        """SV finding after the PV (track_weight: without the other primary vertices' tracks): sv_* outputs,
        displaced_mask, displaced_weight; the SV states and which are valid."""
        n_events, n_tracks = track_weight.shape
        dtype, device = track_weight.dtype, track_weight.device
        K = self.max_sv_per_event
        beam = tracks["perigee"].new_tensor(self.beam_position)
        # candidates (soft): at least sv_min_track_hits TPC hits; displaced from the PV: with silicon hits by
        # sPHENIX's impact cuts (sv_min_track_dca_xy / _z), TPC-only by PV compatibility (sv_min_track_pv_chi2)
        q, _, _ = _transport(tracks["perigee"], tracks["covariance"], tracks["bz"], beam, pv.nan_to_num().unsqueeze(1))
        cm, hits, sigma = self._width("cm"), self._width("hits"), self._width("sigma")
        pv_chi2 = self._compatibility(pv_state, pv.nan_to_num().unsqueeze(1)).squeeze(1)
        displaced = torch.where(
            tracks["n_silicon_hits"] > 0,
            _soft_above(q[..., 0].abs(), self.sv_min_track_dca_xy, cm) * _soft_above(q[..., 1].abs(),
                                                                                  self.sv_min_track_dca_z, cm),
            _soft_above(pv_chi2, self.sv_min_track_pv_chi2, sigma * 2 * math.sqrt(self.sv_min_track_pv_chi2)))
        candidate_weight = (displaced * (track_weight > 0) * tracks["input"].to(dtype)
                            * _soft_above(tracks["n_hits"] - tracks["n_silicon_hits"], self.sv_min_track_hits - 0.5, hits)
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
        # each seed's tracks: its pair, and candidates compatible with the seed (chi2 <= sv_max_track_chi2, soft)
        state = self._new_vertices(seed, torch.zeros(n_events, K, n_tracks, dtype=dtype, device=device), tracks,
                                   guarded=True)
        compatible = _soft_below(self._compatibility(state, seed), self.sv_max_track_chi2,
                                 sigma * 2 * math.sqrt(self.sv_max_track_chi2))
        in_pair = torch.zeros_like(compatible, dtype=torch.bool).scatter_(2, pair, True)
        links = torch.where(in_pair, 1.0, compatible) * candidate_weight.unsqueeze(1) * seed_ok.unsqueeze(-1)
        state["links"] = links
        self._fit(state, seed_ok, track_weight, tracks, shared=False)
        linked = links > 0.5
        n_good = (linked & (state["compatibility"] < self.max_vertex_chi2)).sum(-1)
        chi2_per_dof = state["chi2"] / state["ndf"].clamp(min=1e-12)
        length = (state["position"] - pv.unsqueeze(1)).norm(dim=-1)
        ok = (seed_ok & (n_good >= 2) & (state["ndf"] > 0) & (chi2_per_dof <= self.sv_max_chi2_per_dof)
              & (length > self.sv_min_path))
        # one SV per group within sv_merge_distance: the lowest chi2/ndf (ties: the lower slot)
        distance = (state["position"].unsqueeze(2) - state["position"].unsqueeze(1)).nan_to_num().norm(dim=-1)
        quality = chi2_per_dof.masked_fill(~ok, torch.inf)
        beats = (quality.unsqueeze(1) < quality.unsqueeze(2)) | ((quality.unsqueeze(1) == quality.unsqueeze(2))
                                                                  & torch.ones(K, K, dtype=torch.bool, device=device).tril(-1))
        ok = ok & ~(ok.unsqueeze(1) & (distance < self.sv_merge_distance) & beats).any(dim=2)
        result = self._sv_outputs(state, ok, pv, pv_state["covariance"][:, 0])
        result.update({"sv_pair_dca": seed_pair_dca.masked_fill(~ok, torch.nan), "displaced_mask": candidate,
                       "displaced_weight": track_weight * candidate_weight})
        return result, state, ok

    def _sv_outputs(self, state, ok, pv, pv_covariance):
        """The SV outputs of SV states (B, K) kept where ok, measured from the PV pv (B, 3) with covariance (B, 3, 3):
        the fit, the momentum at the SV (its tracks refitted to it) and the KFParticle-style topology (decay length
        significance with the two vertices' covariances added)."""
        linked = state["links"] > 0.5
        momentum = self._vertex_momentum(state, ok)
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
            "sv_n_tracks": ((state["weight"] >= 0.5) & linked).sum(-1).masked_fill(~ok, 0),
            "sv_track_mask": linked & (state["weight"] >= 0.5) & ok.unsqueeze(-1),
            "sv_track_weight": state["weight"] * ok.unsqueeze(-1),
        }

    def _vertex_momentum(self, state, ok):
        """Sum (B, K, 3) of the momenta (GeV) of each vertex's tracks (weight >= 0.5) refitted to it."""
        A, B, G = state["A"], state["B"], state["G"]
        GB = G @ B
        r = state["q_minus_c"] - (A @ state["position"].nan_to_num().unsqueeze(2).unsqueeze(-1)).squeeze(-1)
        p = (torch.linalg.inv(B.transpose(-1, -2) @ GB) @ GB.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
        phi, theta, qop = p.unbind(-1)
        vector = torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)],
                             -1) / qop.abs().unsqueeze(-1)
        use = ((state["weight"] >= 0.5) & (state["links"] > 0.5) & ok.unsqueeze(-1)).unsqueeze(-1)
        return torch.where(use, vector.nan_to_num(), 0.0).sum(2)
