"""
primaryvertex.py: PrimaryVertices, HybridVertexHead's primary vertices: ACTS's adaptive multi-vertex finder (AMVF) with
the settings of its shipped example (AdaptiveMultiVertexFinderAlgorithm, Gaussian track-density seeder): seed,
preselect tracks, fit all connected vertices together, keep or delete the new vertex, repeat on the tracks left.
"""
import math

import torch
import torch.nn as nn

from .helix_tools import _soft_below, _transport
from .vertexfit import _events


def _density_seed(d0, z0, covariance, weight, d0_significance, z0_significance):
    """
    z of the highest track density on the beam line (ACTS's Gaussian track density with the d0-z0 correlation:
    each track a Gaussian in (d0, z0 - z) at d0 = 0, within its z0_significance window; searched from each track's
    z0 with two Gaussian-shaped Newton steps, the first maximum in track order kept). NaN without one.

    d0, z0, weight: (B, T); covariance: (B, T, 2, 2) of (d0, z0); weight 0 = not used.
    """
    c_dd, c_zz, c_dz = covariance[..., 0, 0], covariance[..., 1, 1], covariance[..., 0, 1]
    det = c_dd * c_zz - c_dz * c_dz
    use = (weight > 0) & (c_dd > 0) & (d0 * d0 / c_dd <= d0_significance ** 2) & (c_zz > 0) & (det > 0)
    det = torch.where(use, det, 1.0)
    c0 = -(d0 * d0 * c_zz + z0 * z0 * c_dd + 2 * d0 * z0 * c_dz) / (2 * det)
    c1 = (d0 * c_dz + z0 * c_dd) / det
    c2 = -c_dd / (2 * det)
    disc = c1 * c1 - 4 * c2 * (c0 + 2 * z0_significance ** 2)
    use = use & (disc >= 0)
    root = disc.clamp(min=0).sqrt()
    upper, lower = (-c1 - root) / (2 * c2), (-c1 + root) / (2 * c2)
    c0 = c0 - torch.log(2 * math.pi * det.sqrt()) + torch.log(weight.clamp(min=1e-300))
    c0, c1, c2 = (torch.where(use, v, 0.0).unsqueeze(1) for v in (c0, c1, c2))
    lower, upper = (torch.where(use, v, 0.0).unsqueeze(1) for v in (lower, upper))

    def density(z):
        """Density and its first two derivatives at candidates z (B, C)."""
        z = z.unsqueeze(-1)
        inside = (z > lower) & (z < upper)
        delta = torch.where(inside, torch.exp(c0 + z * (c1 + z * c2)), 0.0)
        slope = c1 + 2 * z * c2
        return delta.sum(-1), (delta * slope).sum(-1), (2 * c2 * delta + slope * delta * slope).sum(-1)

    z, alive = z0.nan_to_num(), use.clone()
    values, candidates = [], []
    for step in range(3):
        y, dy, ddy = density(z)
        alive = alive & (ddy < 0) & (y > 0)
        values.append(torch.where(alive, y, -torch.inf))
        candidates.append(z)
        if step < 2:
            z = z + torch.where(alive, y * dy / torch.where(alive, dy * dy - y * ddy, 1.0), 0.0)
    values, candidates = torch.stack(values, -1).flatten(1), torch.stack(candidates, -1).flatten(1)  # track order
    best = values.argmax(dim=-1, keepdim=True)
    return torch.where(torch.isfinite(values.gather(1, best)), candidates.gather(1, best), torch.nan).squeeze(1)


class PrimaryVertices(nn.Module):
    """
    HybridVertexHead's primary-vertex search, on the vertex fit vertex_fit (VertexFit). Defaults: ACTS's
    AdaptiveMultiVertexFinderAlgorithm; cm.

    Args:
        seed_d0_significance, seed_z0_significance: the Gaussian track-density seeder.
        tracks_max_z_interval (cm), tracks_max_significance: a vertex's tracks: |z0 - z_seed| and the impact
            parameter significance at the seed (soft).
        maximum_vertex_contamination, max_merge_vertex_significance: keeping a vertex (>= 2 of its tracks still
            unused with compatibility below VertexFit's max_vertex_chi2; contamination; z separation).
        max_vertices: vertices kept per event at most (ACTS: no limit).
        pv_fallback, fallback_preselection: events where the finder keeps no vertex (ACTS: none) are searched again
            with the tracks' preselection relaxed, step by step, to each (tracks_max_z_interval cm,
            tracks_max_significance) of fallback_preselection (keeping a vertex as ACTS does). Off with hard_cuts.
    """
    def __init__(
        self,
        vertex_fit,
        seed_d0_significance: float = 3.5,
        seed_z0_significance: float = 12.0,
        tracks_max_z_interval: float = 0.1,
        tracks_max_significance: float = 5.0,
        maximum_vertex_contamination: float = 0.5,
        max_merge_vertex_significance: float = 3.0,
        max_vertices: int = 10,
        pv_fallback: bool = True,
        fallback_preselection: tuple = ((0.3, 7.0), (1.0, 10.0)),
    ):
        super().__init__()
        if max_vertices < 1:
            raise ValueError("max_vertices must be at least 1")
        self.vertex_fit = vertex_fit
        self.seed_d0_significance = seed_d0_significance
        self.seed_z0_significance = seed_z0_significance
        self.tracks_max_z_interval = tracks_max_z_interval
        self.tracks_max_significance = tracks_max_significance
        self.maximum_vertex_contamination = maximum_vertex_contamination
        self.max_merge_vertex_significance = max_merge_vertex_significance
        self.max_vertices = max_vertices
        self.pv_fallback = pv_fallback
        self.fallback_preselection = tuple(tuple(float(v) for v in step) for step in fallback_preselection)

    def forward(self, tracks):
        """
        The primary vertices of each event's tracks: the vertex states (B, K = max_vertices) and kept (B, K), the kept
        ones first, in the order found (the PV: slot 0), and step (B,): the fallback step that found them (0: as
        ACTS, -1: no vertex).
        """
        state, kept = self._amvf(tracks, (self.tracks_max_z_interval, self.tracks_max_significance))
        step = torch.where(kept.any(dim=1), 0, -1)
        for k, preselection in enumerate(
                self.fallback_preselection if self.pv_fallback and not self.vertex_fit.hard_cuts else ()):
            # events without a vertex (only those): searched again with a looser preselection
            missing = (step < 0).nonzero().squeeze(1)
            if not len(missing):
                break
            again, again_kept = self._amvf(_events(tracks, missing), preselection)
            use = again_kept.any(dim=1)
            found = (missing[use],)
            state = {key: value.index_put(found, again[key][use]) for key, value in state.items()}
            kept = kept.index_put(found, again_kept[use])
            step = step.index_put(found, step.new_tensor(k + 1))
        return state, kept, step

    def _links(self, seed, tracks, preselection):
        """Each track's membership (B, T) of a vertex seeded at seed (B, 3): |z0 - z_seed| and the impact significance
        at the seed below preselection (z interval cm, significance) (soft); and the tracks transported to the seed
        (_transport's)."""
        q, cov, pca = _transport(tracks["perigee"], tracks["covariance"], tracks["bz"], tracks["beam"],
                                 seed.nan_to_num().unsqueeze(1))
        significance = (q[..., 0] ** 2 / cov[..., 0, 0] + q[..., 1] ** 2 / cov[..., 1, 1]).clamp(min=1e-30).sqrt()
        dz = (tracks["perigee"][..., 1] - seed[:, 2:]).abs()
        return (_soft_below(dz, preselection[0], self.vertex_fit.width("cm"))
                * _soft_below(significance, preselection[1], self.vertex_fit.width("sigma"))
                * tracks["input"].to(dz.dtype)), q, cov, pca

    def _amvf(self, tracks, preselection):
        """ACTS's AMVF over each event's input tracks (weight 0: not used) with the tracks' preselection (z interval
        cm, significance): the vertex states (B, K) and kept (B, K), the kept ones first, in the order found."""
        vertex_fit, track_weight, perigee, cov = self.vertex_fit, tracks["weight"], tracks["perigee"], tracks["covariance"]
        n_events, n_tracks = track_weight.shape
        K = self.max_vertices
        state = vertex_fit.empty_vertices(n_events, K, n_tracks, track_weight.dtype, track_weight.device)
        kept = torch.zeros(n_events, K, dtype=torch.bool, device=track_weight.device)
        pool = tracks["input"].clone()
        alive = pool.sum(-1) >= 2
        events = torch.arange(n_events, device=track_weight.device)
        for _ in range(n_tracks):                   # each pass takes >= 1 track from the pool of alive events
            alive = alive & ~kept.all(dim=1)
            if not alive.any():
                break
            slot = (~kept).to(torch.int8).argmax(dim=1)
            z = _density_seed(perigee[..., 0], perigee[..., 1], cov[..., :2, :2], track_weight * pool,
                              self.seed_d0_significance, self.seed_z0_significance)
            alive = alive & torch.isfinite(z)
            seed = torch.cat([tracks["beam"].expand(n_events, 2), z.nan_to_num().unsqueeze(-1)], -1)
            prior = seed
            links, *transport = self._links(seed, tracks, preselection)
            # no track: move the seed to the nearest unused track's z0 and try again
            empty = alive & ~(links > 0.5).any(dim=1)
            if empty.any():
                dz = torch.where(pool, (perigee[..., 1] - seed[:, 2:]).abs(), torch.inf)
                seed = torch.where(empty.unsqueeze(-1), torch.stack(
                    [seed[:, 0], seed[:, 1], perigee[events, dz.argmin(dim=1), 1]], -1), seed)
                moved = empty.nonzero().squeeze(1)
                links, *transport = (old.index_put((moved,), new) for old, new in zip(
                    (links, *transport), self._links(seed[moved], _events(tracks, moved), preselection)))
                alive = alive & (links > 0.5).any(dim=1)
            if not alive.any():
                break
            new = torch.zeros_like(kept)
            new[events[alive], slot[alive]] = True
            # the new vertex's state (its prior: the density seed, also where the seed moved)
            fresh = vertex_fit.new_vertices(seed[alive].unsqueeze(1), links[alive].unsqueeze(1), _events(tracks, alive),
                                            prior[alive].unsqueeze(1), transport=[v[alive].unsqueeze(1) for v in transport])
            for key, value in fresh.items():
                state[key] = state[key].index_put((events[alive], slot[alive]), value[:, 0])
            # the vertices to fit: the new one and every kept vertex connected to it through shared tracks
            linked = state["links"] > 0.5
            together = new.clone()
            for _ in range(K):
                shares = ((linked & together.unsqueeze(-1)).any(dim=1, keepdim=True) & linked).any(dim=-1)
                together = together | (shares & kept)
            vertex_fit.fit(state, together, tracks)
            # the new vertex: >= 2 of its tracks still unused and compatible; those leave the pool (if none, the
            # least compatible of its unused tracks, or the unused track nearest the fitted vertex in z)
            new_links, compatibility = linked[events, slot], state["compatibility"][events, slot]
            compatible = new_links & pool & (compatibility < vertex_fit.max_vertex_chi2)
            good = alive & (compatible.sum(-1) >= 2)
            remove = compatible & alive.unsqueeze(-1)
            none = alive & ~compatible.any(dim=-1)
            unused = new_links & pool & (compatibility > 0)
            z_new = state["position"][events, slot, 2:]
            nearest = torch.where(pool, (perigee[..., 1] - z_new).abs(), torch.inf).argmin(dim=-1)
            drop = torch.where(unused.any(dim=-1), torch.where(unused, compatibility, -torch.inf).argmax(dim=-1), nearest)
            remove = remove | (torch.nn.functional.one_hot(drop, n_tracks).bool() & none.unsqueeze(-1))
            pool = pool & ~remove
            # kept: good, contamination <= maximum_vertex_contamination, not within max_merge_vertex_significance
            # sigma in z of another vertex
            w = state["weight"][events, slot] * new_links
            contamination = (w * (1 - w)).sum(-1) / (w * w).sum(-1).clamp(min=1e-300)
            merged = (kept & ((state["position"][..., 2] - z_new).abs() < self.max_merge_vertex_significance * (
                state["covariance"][..., 2, 2] + state["covariance"][events, slot, 2, 2].unsqueeze(1)).sqrt())
                      ).any(dim=1)
            keep = good & (contamination <= self.maximum_vertex_contamination) & ~merged
            kept = kept | (new & keep.unsqueeze(-1))
            rejected = new & ~keep.unsqueeze(-1)
            if rejected.any():
                # the vertex goes; the others it was fitted with are refitted without it
                state["links"] = state["links"].masked_fill(rejected.unsqueeze(-1), 0.0)
                state["weight"] = state["weight"].masked_fill(rejected.unsqueeze(-1), 0.0)
                state["compatibility"] = state["compatibility"].masked_fill(rejected.unsqueeze(-1), torch.inf)
                state["seed"] = state["seed"].masked_fill(rejected.unsqueeze(-1), torch.nan)
                vertex_fit.fit(state, together & ~rejected & rejected.any(dim=1, keepdim=True), tracks)
            alive = alive & (pool.sum(-1) >= 2)
        return state, kept
