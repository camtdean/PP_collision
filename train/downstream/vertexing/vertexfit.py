"""
vertexfit.py: VertexFit, the adaptive multi-vertex fit the vertex finders share (ACTS's: Kalman vertex updates on
linearised helices, annealed track weights shared between vertices), its settings and the soft cuts' widths.

Written from the published maths (R. Fruehwirth et al., adaptive vertex fitting, J. Phys. G 34 (2007) N343;
G. Piacquadio, CERN-THESIS-2010-027, the helix linearisation) and checked against ACTS v48.2.0 (MPL-2.0); all
lengths in cm (ACTS: mm).

The tracks are a dict (TrackBuilder's): perigee (B, T, 5), covariance (B, T, 5, 5), bz, n_hits, n_silicon_hits,
input (the fit's tracks), weight (B, T) and beam (2,), the beam line's (x, y).
"""
import torch
import torch.nn as nn

from .helix_tools import _guarded_impact_with_jacobian, _impact_with_jacobian, _linearise

_LINEARISATION = ("A", "B", "q_minus_c", "G", "G_B", "AtG_B")        # the state's keys of a track linearisation


def _events(tracks, index):
    """The tracks dict of the events index (the beam is shared)."""
    return {k: v if k == "beam" else v[index] for k, v in tracks.items()}


def _project(A, B, G):
    """G_B = G - G B (B^T G B)^-1 B^T G (the tracks' weight with their momentum fitted out) and A^T G_B."""
    GB = G @ B
    G_B = G - GB @ torch.linalg.inv(B.transpose(-1, -2) @ GB) @ GB.transpose(-1, -2)
    return G_B, A.transpose(-1, -2) @ G_B


class VertexFit(nn.Module):
    """
    The vertex fit VertexHead's primary and secondary searches share. Defaults: ACTS's
    AdaptiveMultiVertexFinderAlgorithm; cm.

    Args:
        annealing_temperatures, annealing_cut_off: the fit's annealing (ACTS example: one temperature, 1).
        fit_min_weight, fit_max_iterations, max_dist_to_lin_point (cm), max_relative_shift: the multi-vertex fit.
        initial_variance (cm^2): the vertex prior around its seed (x, y, z).
        max_vertex_chi2: a track's compatibility with a vertex below which it counts as the vertex's.
        cut_softness_cm, cut_softness_sigma, cut_softness_hits: sigmoid widths of the soft cuts in cm, in
            significance and in hits; hard_cuts: ACTS's / sPHENIX's hard cuts instead.
    """
    def __init__(
        self,
        annealing_temperatures: tuple = (1.0,),
        annealing_cut_off: float = 9.0,
        fit_min_weight: float = 0.001,
        fit_max_iterations: int = 30,
        max_dist_to_lin_point: float = 0.05,
        max_relative_shift: float = 0.01,
        initial_variance: float = 1.0,
        max_vertex_chi2: float = 18.42,
        cut_softness_cm: float = 0.005,
        cut_softness_sigma: float = 0.25,
        cut_softness_hits: float = 1.0,
        hard_cuts: bool = False,
    ):
        super().__init__()
        if fit_max_iterations < 1 or not annealing_temperatures:
            raise ValueError("fit_max_iterations must be at least 1, annealing_temperatures non-empty")
        self.annealing_temperatures = tuple(float(t) for t in annealing_temperatures)
        self.annealing_cut_off = annealing_cut_off
        self.fit_min_weight = fit_min_weight
        self.fit_max_iterations = fit_max_iterations
        self.max_dist_to_lin_point = max_dist_to_lin_point
        self.max_relative_shift = max_relative_shift
        self.initial_variance = initial_variance
        self.max_vertex_chi2 = max_vertex_chi2
        self.cut_softness_cm = cut_softness_cm
        self.cut_softness_sigma = cut_softness_sigma
        self.cut_softness_hits = cut_softness_hits
        self.hard_cuts = hard_cuts

    def width(self, kind):
        """A soft cut's width (0 with hard_cuts)."""
        return 0.0 if self.hard_cuts else {"cm": self.cut_softness_cm, "sigma": self.cut_softness_sigma,
                                            "hits": self.cut_softness_hits}[kind]

    # --------------------------------------------------------------------------------------------
    # Vertex states: (B, K) vertices, their tracks (B, K, T)
    # --------------------------------------------------------------------------------------------

    def new_vertices(self, seeds, links, tracks, prior=None, guarded=False, transport=None):
        """
        Vertex states for seeds (B, K, 3) with link factors (B, K, T) (a track's membership, 0..1): the 3D impact
        parameters at the seed (for the compatibility), the tracks linearised at the seed, position = seed, the
        prior (default: the seed) with initial_variance; guarded: the guarded 3D impact point (_impact_3d);
        transport: the tracks already transported to the seeds (_transport's, (B, K, T)).
        """
        perigee, cov, bz, beam = tracks["perigee"], tracks["covariance"], tracks["bz"], tracks["beam"]
        finite = seeds.nan_to_num()          # unused slots (NaN seeds) get finite stand-ins (weight 0)
        point = finite.unsqueeze(2)
        expand = lambda v: v.unsqueeze(1).expand(-1, seeds.shape[1], *v.shape[1:])  # noqa: E731
        impact = _guarded_impact_with_jacobian if guarded else _impact_with_jacobian
        loc, axes, J = impact(expand(perigee), expand(bz), beam, point)
        A, B, q_minus_c, G = _linearise(expand(perigee), expand(cov), expand(bz), beam, point, transport)
        return {"seed": seeds, "prior": finite if prior is None else prior.nan_to_num(), "position": finite,
                "lin_point": finite, "links": links, "ip_loc": loc, "ip_axes": axes,
                "ip_weight": torch.linalg.inv(J @ expand(cov) @ J.transpose(-1, -2)),
                "covariance": torch.diag_embed(seeds.new_full(seeds.shape, self.initial_variance)),
                "compatibility": torch.full_like(links, torch.inf), "weight": torch.zeros_like(links),
                "chi2": seeds.new_zeros(seeds.shape[:2]), "ndf": seeds.new_full(seeds.shape[:2], -3.0),
                **dict(zip(_LINEARISATION, (A, B, q_minus_c, G, *_project(A, B, G))))}

    def empty_vertices(self, n_events, n_vertices, n_tracks, dtype, device):
        """Unused vertex slots (B, K) in the layout of new_vertices (finite, invertible stand-ins)."""
        zeros = lambda *shape: torch.zeros(n_events, n_vertices, *shape, dtype=dtype, device=device)  # noqa: E731
        eye = lambda n, *shape: torch.eye(n, dtype=dtype, device=device).expand(  # noqa: E731
            n_events, n_vertices, *shape, n, n)
        A, B, G = zeros(n_tracks, 5, 3), zeros(n_tracks, 5, 3), eye(5, n_tracks)
        B[..., 2:, :] = torch.eye(3, dtype=dtype, device=device)
        return {"seed": zeros(3) + torch.nan, "prior": zeros(3), "position": zeros(3), "lin_point": zeros(3),
                "links": zeros(n_tracks), "ip_loc": zeros(n_tracks, 2), "ip_axes": zeros(n_tracks, 2, 3),
                "ip_weight": eye(2, n_tracks), "covariance": eye(3) * self.initial_variance,
                "compatibility": zeros(n_tracks) + torch.inf, "weight": zeros(n_tracks), "chi2": zeros(),
                "ndf": zeros() - 3.0,
                **dict(zip(_LINEARISATION, (A, B, zeros(n_tracks, 5), G, *_project(A, B, G))))}

    def compatibility(self, state, position):
        """Each track's chi2 (..., T) to the vertices at position (..., 3), on the 2D frame of its 3D impact
        parameters at the seed (ACTS's 3D vertex compatibility)."""
        r = state["ip_loc"] - ((position - state["seed"]).nan_to_num().unsqueeze(-2).unsqueeze(-2)
                               * state["ip_axes"]).sum(-1)
        return (r.unsqueeze(-2) @ state["ip_weight"] @ r.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def _relinearise(self, state, at, tracks):
        """Relinearise the tracks of the vertices at (events, slots) at their lin_point."""
        if not len(at[0]):
            return
        A, B, q_minus_c, G = _linearise(tracks["perigee"][at[0]], tracks["covariance"][at[0]], tracks["bz"][at[0]],
                                        tracks["beam"], state["lin_point"][at].unsqueeze(1))
        for key, value in zip(_LINEARISATION, (A, B, q_minus_c, G, *_project(A, B, G))):
            state[key] = state[key].index_put(at, value)

    def _kalman(self, state, weight):
        """
        The vertices (...) from their prior and the tracks with weights (..., T): ACTS's Kalman vertex update (all
        tracks at once; the same as one by one): position, covariance, chi2 and ndf.
        """
        A, G_B, AtG_B = state["A"], state["G_B"], state["AtG_B"]
        w = weight[..., None, None]
        prior_info = torch.eye(3, dtype=A.dtype, device=A.device) / self.initial_variance
        rhs = (prior_info @ state["prior"].unsqueeze(-1)).squeeze(-1) + (
            w.squeeze(-1) * (AtG_B @ state["q_minus_c"].unsqueeze(-1)).squeeze(-1)).sum(-2)
        covariance = torch.linalg.inv(prior_info + (w * AtG_B @ A).sum(-3))
        position = (covariance @ rhs.unsqueeze(-1)).squeeze(-1)
        r = state["q_minus_c"] - (A @ position.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
        d = position - state["prior"]
        chi2 = (d.unsqueeze(-2) @ prior_info @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1) + (
            weight * (r.unsqueeze(-2) @ G_B @ r.unsqueeze(-1)).squeeze(-1).squeeze(-1)).sum(-1)
        return position, covariance, chi2, 2 * weight.sum(-1) - 3

    def fit(self, state, which, tracks, shared=True):
        """
        ACTS's adaptive multi-vertex fit of the vertices which (B, K) (the others keep their state; their stored
        compatibilities still share the tracks): annealed weights, Kalman update, relinearisation beyond
        max_dist_to_lin_point, until the last temperature and a small shift (or fit_max_iterations). shared=False:
        each vertex fitted on its own (its tracks not shared with the others).
        """
        temperatures = self.annealing_temperatures
        running = which.any(dim=1)
        for iteration in range(self.fit_max_iterations):
            if not running.any():
                break
            # the vertices still fitted (n,): only their rows are computed
            active = which & running.unsqueeze(1)
            at = active.nonzero(as_tuple=True)
            old = state["position"][at]
            moved = (old[..., :2] - state["lin_point"][at][..., :2]).norm(dim=-1) > self.max_dist_to_lin_point
            state["lin_point"] = state["lin_point"].index_put((at[0][moved], at[1][moved]), old[moved])
            self._relinearise(state, (at[0][moved], at[1][moved]), tracks)
            fitted = {key: state[key][at] for key in ("seed", "prior", "ip_loc", "ip_axes", "ip_weight", "A",
                                                      "q_minus_c", "G_B", "AtG_B")}
            state["compatibility"] = state["compatibility"].index_put(at, self.compatibility(fitted, old))
            # weight = f exp(-chi2/2T) / (exp(-cut/2T) + sum over the track's vertices f_v exp(-chi2_v/2T))
            t2 = 2.0 * temperatures[min(iteration, len(temperatures) - 1)]
            f = state["links"]
            # (unlinked: a large negative stand-in, not -inf, so that the gradient stays finite)
            log_terms = torch.where(f > 0, torch.log(f.clamp(min=1e-300)) - state["compatibility"].clamp(max=1e30) / t2,
                                    -1e30)
            log_denominator = torch.logaddexp(torch.logsumexp(log_terms, dim=1, keepdim=True) if shared else log_terms,
                                              log_terms.new_tensor(-self.annealing_cut_off / t2))
            state["weight"] = torch.where(active.unsqueeze(-1), torch.exp(log_terms - log_denominator).nan_to_num(0.0),
                                          state["weight"])
            weight = state["weight"][at]
            position, covariance, chi2, ndf = self._kalman(fitted, weight * (weight > self.fit_min_weight)
                                                           * tracks["weight"][at[0]])
            for key, value in (("position", position), ("covariance", covariance), ("chi2", chi2), ("ndf", ndf)):
                state[key] = state[key].index_put(at, value)
            d = (old - position).unsqueeze(-1)
            large = torch.zeros_like(active).index_put(at, ~(
                (d.transpose(-1, -2) @ torch.linalg.solve(covariance, d)).squeeze(-1).squeeze(-1) <= self.max_relative_shift))
            running = running & ~(~large.any(dim=1) & (iteration >= len(temperatures) - 1))
