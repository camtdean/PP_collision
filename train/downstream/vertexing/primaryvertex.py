"""
primaryvertex.py: PrimaryVertices, VertexHead's primary vertices: ACTS's adaptive multi-vertex finder (AMVF) with
the settings of its shipped example (AdaptiveMultiVertexFinderAlgorithm, Gaussian track-density seeder): seed,
preselect tracks, fit all connected vertices together, keep or delete the new vertex, repeat on the tracks left.
"""
import math

import torch

from .helix_tools import _soft_below, _transport
from .vertexfit import VertexFit

_NO_ORDER = 10 ** 9                # creation order of a vertex slot not kept


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


class PrimaryVertices(VertexFit):
    """VertexHead's primary-vertex search (on VertexFit's settings and fit)."""
    def _links(self, seed, tracks, preselection):
        """Each track's membership (B, T) of a vertex seeded at seed (B, 3): |z0 - z_seed| and the impact significance
        at the seed below preselection (z interval cm, significance) (soft)."""
        q, cov, _ = _transport(tracks["perigee"], tracks["covariance"], tracks["bz"],
                               tracks["perigee"].new_tensor(self.beam_position), seed.nan_to_num().unsqueeze(1))
        significance = (q[..., 0] ** 2 / cov[..., 0, 0] + q[..., 1] ** 2 / cov[..., 1, 1]).clamp(min=1e-30).sqrt()
        dz = (tracks["perigee"][..., 1] - seed[:, 2:]).abs()
        return (_soft_below(dz, preselection[0], self._width("cm"))
                * _soft_below(significance, preselection[1], self._width("sigma"))
                * tracks["input"].to(dz.dtype))

    def _find_primary(self, tracks, track_weight, preselection):
        """ACTS's AMVF over each event's input tracks (weight 0: not used) with the tracks' preselection (z interval
        cm, significance): the vertex states (B, K) and kept (B, K) in creation order."""
        n_events, n_tracks = track_weight.shape
        dtype, device = track_weight.dtype, track_weight.device
        K = self.max_vertices + 1                                   # the kept ones and the one being tried
        perigee, cov = tracks["perigee"], tracks["covariance"]
        state = self._empty_vertices(n_events, K, n_tracks, dtype, device)
        kept = torch.zeros(n_events, K, dtype=torch.bool, device=device)
        order = torch.full((n_events, K), _NO_ORDER, dtype=torch.long, device=device)
        pool = tracks["input"].clone()
        alive = pool.sum(-1) >= 2
        events = torch.arange(n_events, device=device)
        for iteration in range(n_tracks):           # each pass takes >= 1 track from the pool of alive events
            alive = alive & (kept.sum(dim=1) < self.max_vertices)
            if not alive.any():
                break
            slot = (~kept).to(torch.int8).argmax(dim=1)
            z = _density_seed(perigee[..., 0], perigee[..., 1], cov[..., :2, :2], track_weight * pool,
                              self.seed_d0_significance, self.seed_z0_significance)
            alive = alive & torch.isfinite(z)
            seed = torch.stack([torch.full_like(z, self.beam_position[0]), torch.full_like(z, self.beam_position[1]),
                                z.nan_to_num()], -1)
            prior = seed
            links = self._links(seed, tracks, preselection)
            # no track: move the seed to the nearest unused track's z0 and try again
            empty = alive & ~(links > 0.5).any(dim=1)
            if empty.any():
                dz = torch.where(pool, (perigee[..., 1] - seed[:, 2:]).abs(), torch.inf)
                nearest = dz.argmin(dim=1)
                seed = torch.where(empty.unsqueeze(-1), torch.stack(
                    [seed[:, 0], seed[:, 1], perigee[events, nearest, 1]], -1), seed)
                links = torch.where(empty.unsqueeze(-1), self._links(seed, tracks, preselection), links)
                alive = alive & (links > 0.5).any(dim=1)
            if not alive.any():
                break
            new = torch.zeros_like(kept)
            new[events[alive], slot[alive]] = True
            # the new vertex's state (its prior: the density seed, also where the seed moved)
            fresh = self._new_vertices(seed.unsqueeze(1), links.unsqueeze(1), tracks, prior.unsqueeze(1))
            for key, value in fresh.items():
                state[key] = state[key].index_put((events[alive], slot[alive]), value[alive, 0])
            order = torch.where(new, iteration, order)
            # the vertices to fit: the new one and every kept vertex connected to it through shared tracks
            linked = state["links"] > 0.5
            fit = new.clone()
            for _ in range(K):
                shares = ((linked & fit.unsqueeze(-1)).any(dim=1, keepdim=True) & linked).any(dim=-1)
                fit = fit | (shares & kept)
            self._fit(state, fit, track_weight, tracks)
            # the new vertex: >= 2 of its tracks still unused and compatible; those leave the pool (if none, the
            # least compatible of its unused tracks, or the unused track nearest the fitted vertex in z)
            new_links = (linked & new.unsqueeze(-1)).any(dim=1)
            compatibility = torch.where(new.unsqueeze(-1), state["compatibility"], 0.0).sum(dim=1)
            compatible = new_links & pool & (compatibility < self.max_vertex_chi2)
            good = alive & (compatible.sum(-1) >= 2)
            remove = compatible & alive.unsqueeze(-1)
            none = alive & ~compatible.any(dim=-1)
            unused = new_links & pool & (compatibility > 0)
            z_new = (state["position"][..., 2] * new).sum(1, keepdim=True)
            nearest = torch.where(pool, (perigee[..., 1] - z_new).abs(), torch.inf).argmin(dim=-1)
            drop = torch.where(unused.any(dim=-1), torch.where(unused, compatibility, -torch.inf).argmax(dim=-1), nearest)
            remove = remove | (torch.nn.functional.one_hot(drop, n_tracks).bool() & none.unsqueeze(-1))
            pool = pool & ~remove
            # kept: good, contamination <= maximum_vertex_contamination, not within max_merge_vertex_significance
            # sigma in z of another vertex
            w = torch.where(new.unsqueeze(-1), state["weight"], 0.0).sum(dim=1) * new_links
            contamination = (w * (1 - w)).sum(-1) / (w * w).sum(-1).clamp(min=1e-300)
            var_new = (state["covariance"][..., 2, 2] * new).sum(1, keepdim=True)
            merged = (kept & ((state["position"][..., 2] - z_new).abs()
                              < self.max_merge_vertex_significance * (state["covariance"][..., 2, 2] + var_new).sqrt())
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
                order = order.masked_fill(rejected, _NO_ORDER)
                self._fit(state, fit & ~rejected & rejected.any(dim=1, keepdim=True), track_weight, tracks)
            alive = alive & (pool.sum(-1) >= 2)
        return state, kept, order
