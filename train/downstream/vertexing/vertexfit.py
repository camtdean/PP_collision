"""
vertexfit.py: VertexFit, the vertex finders' settings and the adaptive multi-vertex fit they share (ACTS's: Kalman
vertex updates on linearised helices, annealed track weights shared between vertices).

Written from the published maths (R. Fruehwirth et al., adaptive vertex fitting, J. Phys. G 34 (2007) N343;
G. Piacquadio, CERN-THESIS-2010-027, the helix linearisation) and checked against ACTS v48.2.0 (MPL-2.0); all
lengths in cm (ACTS: mm).
"""
import torch
import torch.nn as nn

from .helix_tools import _expand, _guarded_impact_with_jacobian, _impact_with_jacobian, _linearise


class VertexFit(nn.Module):
    """
    VertexHead's vertex-finding settings and the vertex fit its primary and secondary searches share. Detector
    values are arguments (defaults: ACTS's AdaptiveMultiVertexFinderAlgorithm, sPHENIX's SecondaryVertexFinder), in cm.

    Args:
        beam_position: (x, y), cm (as TrackBuilder's); the seeds' x, y.
        annealing_temperatures, annealing_cut_off: the fit's annealing (ACTS example: one temperature, 1).
        fit_min_weight, fit_max_iterations, max_dist_to_lin_point (cm), max_relative_shift: the multi-vertex fit.
        initial_variance (cm^2): the vertex prior around its seed (x, y, z).
        seed_d0_significance, seed_z0_significance: the Gaussian track-density seeder.
        tracks_max_z_interval (cm), tracks_max_significance: a vertex's tracks: |z0 - z_seed| and the impact
            parameter significance at the seed (soft).
        max_vertex_chi2, maximum_vertex_contamination, max_merge_vertex_significance: keeping a vertex (>= 2 of
            its tracks still unused with compatibility below max_vertex_chi2; contamination; z separation).
        max_vertices: vertices kept per event at most (ACTS: no limit).
        pv_fallback, fallback_preselection: events where the finder keeps no vertex (ACTS: none) are searched again
            with the tracks' preselection relaxed, step by step, to each (tracks_max_z_interval cm,
            tracks_max_significance) of fallback_preselection (keeping a vertex as ACTS does); vertex_from_fallback
            marks them, vertex_fallback_step gives the step (0: as ACTS). Off with hard_cuts.
        cut_softness_cm, cut_softness_sigma, cut_softness_hits: sigmoid widths of the soft cuts in cm, in
            significance and in hits; hard_cuts: ACTS's / sPHENIX's hard cuts instead.
        find_secondary, refit_pv: SVs, and the PV refitted together with them (tracks shared).
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
        beam_position: tuple = (0.0, 0.0),
        annealing_temperatures: tuple = (1.0,),
        annealing_cut_off: float = 9.0,
        fit_min_weight: float = 0.001,
        fit_max_iterations: int = 30,
        max_dist_to_lin_point: float = 0.05,
        max_relative_shift: float = 0.01,
        initial_variance: float = 1.0,
        seed_d0_significance: float = 3.5,
        seed_z0_significance: float = 12.0,
        tracks_max_z_interval: float = 0.1,
        tracks_max_significance: float = 5.0,
        max_vertex_chi2: float = 18.42,
        maximum_vertex_contamination: float = 0.5,
        max_merge_vertex_significance: float = 3.0,
        max_vertices: int = 10,
        pv_fallback: bool = True,
        fallback_preselection: tuple = ((0.3, 7.0), (1.0, 10.0)),
        cut_softness_cm: float = 0.005,
        cut_softness_sigma: float = 0.25,
        cut_softness_hits: float = 1.0,
        hard_cuts: bool = False,
        find_secondary: bool = True,
        refit_pv: bool = True,
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
        if fit_max_iterations < 1 or max_vertices < 1 or not annealing_temperatures:
            raise ValueError("fit_max_iterations and max_vertices must be at least 1, annealing_temperatures non-empty")
        self.beam_position = tuple(beam_position)
        self.annealing_temperatures = tuple(float(t) for t in annealing_temperatures)
        self.annealing_cut_off = annealing_cut_off
        self.fit_min_weight = fit_min_weight
        self.fit_max_iterations = fit_max_iterations
        self.max_dist_to_lin_point = max_dist_to_lin_point
        self.max_relative_shift = max_relative_shift
        self.initial_variance = initial_variance
        self.seed_d0_significance = seed_d0_significance
        self.seed_z0_significance = seed_z0_significance
        self.tracks_max_z_interval = tracks_max_z_interval
        self.tracks_max_significance = tracks_max_significance
        self.max_vertex_chi2 = max_vertex_chi2
        self.maximum_vertex_contamination = maximum_vertex_contamination
        self.max_merge_vertex_significance = max_merge_vertex_significance
        self.max_vertices = max_vertices
        self.pv_fallback = pv_fallback
        self.fallback_preselection = tuple(tuple(float(v) for v in step) for step in fallback_preselection)
        self.cut_softness_cm = cut_softness_cm
        self.cut_softness_sigma = cut_softness_sigma
        self.cut_softness_hits = cut_softness_hits
        self.hard_cuts = hard_cuts
        self.find_secondary = find_secondary
        self.refit_pv = refit_pv
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

    def _width(self, kind):
        """A soft cut's width (0 with hard_cuts)."""
        return 0.0 if self.hard_cuts else {"cm": self.cut_softness_cm, "sigma": self.cut_softness_sigma,
                                            "hits": self.cut_softness_hits}[kind]

    # --------------------------------------------------------------------------------------------
    # Vertex states: (B, K) vertices, their tracks (B, K, T)
    # --------------------------------------------------------------------------------------------

    def _new_vertices(self, seeds, links, tracks, prior=None, guarded=False):
        """
        Vertex states for seeds (B, K, 3) with link factors (B, K, T) (a track's membership, 0..1): the 3D impact
        parameters at the seed (for the compatibility), the tracks linearised at the seed, position = seed, the
        prior (default: the seed) with initial_variance; guarded: the guarded 3D impact point (_impact_3d).
        """
        perigee, cov, bz = tracks["perigee"], tracks["covariance"], tracks["bz"]
        beam = perigee.new_tensor(self.beam_position)
        finite = seeds.nan_to_num()          # unused slots (NaN seeds) get finite stand-ins (weight 0)
        point = finite.unsqueeze(2)
        expand = lambda v: v.unsqueeze(1).expand(-1, seeds.shape[1], *v.shape[1:])  # noqa: E731
        impact = _guarded_impact_with_jacobian if guarded else _impact_with_jacobian
        loc, axes, J = impact(expand(perigee), expand(bz), beam, point)
        loc_cov = J @ expand(cov) @ J.transpose(-1, -2)
        state = {"seed": seeds, "prior": finite if prior is None else prior.nan_to_num(), "position": finite.clone(),
                 "lin_point": finite.clone(), "links": links, "ip_loc": loc, "ip_axes": axes,
                 "ip_weight": torch.linalg.inv(loc_cov),
                 "covariance": torch.diag_embed(seeds.new_full(seeds.shape, self.initial_variance)),
                 "compatibility": torch.full_like(links, torch.inf), "weight": torch.zeros_like(links),
                 "chi2": seeds.new_zeros(seeds.shape[:2]), "ndf": seeds.new_full(seeds.shape[:2], -3.0)}
        state.update(zip(("A", "B", "q_minus_c", "G"), _linearise(expand(perigee), expand(cov), expand(bz), beam, point)))
        return state

    def _empty_vertices(self, n_events, n_vertices, n_tracks, dtype, device):
        """Unused vertex slots (B, K) in the layout of _new_vertices (finite, invertible stand-ins)."""
        zeros = lambda *shape: torch.zeros(n_events, n_vertices, *shape, dtype=dtype, device=device)  # noqa: E731
        eye = lambda n, *shape: torch.eye(n, dtype=dtype, device=device).expand(  # noqa: E731
            n_events, n_vertices, *shape, n, n).clone()
        B = zeros(n_tracks, 5, 3)
        B[..., 2:, :] = torch.eye(3, dtype=dtype, device=device)
        return {"seed": zeros(3) + torch.nan, "prior": zeros(3), "position": zeros(3), "lin_point": zeros(3),
                "links": zeros(n_tracks), "ip_loc": zeros(n_tracks, 2), "ip_axes": zeros(n_tracks, 2, 3),
                "ip_weight": eye(2, n_tracks), "covariance": eye(3) * self.initial_variance,
                "compatibility": zeros(n_tracks) + torch.inf, "weight": zeros(n_tracks), "chi2": zeros(),
                "ndf": zeros() - 3.0, "A": zeros(n_tracks, 5, 3), "B": B, "q_minus_c": zeros(n_tracks, 5),
                "G": eye(5, n_tracks)}

    def _compatibility(self, state, position):
        """Each track's chi2 (B, K, T) to the vertex at position (B, K, 3), on the 2D frame of its 3D impact
        parameters at the seed (ACTS's 3D vertex compatibility)."""
        shift = ((position - state["seed"]).nan_to_num().unsqueeze(2).unsqueeze(-2) * state["ip_axes"]).sum(-1)
        r = state["ip_loc"] - shift
        return (r.unsqueeze(-2) @ state["ip_weight"] @ r.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def _relinearise(self, state, which, tracks):
        """Relinearise the tracks of vertices which (B, K) at their lin_point."""
        if not which.any():
            return
        perigee, cov, bz = tracks["perigee"], tracks["covariance"], tracks["bz"]
        beam = perigee.new_tensor(self.beam_position)
        b, k = which.nonzero(as_tuple=True)
        new = _linearise(perigee[b], cov[b], bz[b], beam, state["lin_point"][b, k].unsqueeze(1))
        for key, value in zip(("A", "B", "q_minus_c", "G"), new):
            state[key] = state[key].index_put((b, k), value)

    def _kalman(self, state, weight):
        """
        The vertices (B, K) from their prior and the tracks with weights (B, K, T): ACTS's Kalman vertex update (all
        tracks at once; the same as one by one): position, covariance, chi2 and ndf.
        """
        A, B, G = state["A"], state["B"], state["G"]
        GB = G @ B
        G_B = G - GB @ torch.linalg.inv(B.transpose(-1, -2) @ GB) @ GB.transpose(-1, -2)
        AtGB = A.transpose(-1, -2) @ G_B
        w = weight[..., None, None]
        prior_info = torch.eye(3, dtype=A.dtype, device=A.device) / self.initial_variance
        information = prior_info + (w * AtGB @ A).sum(2)
        rhs = (prior_info @ state["prior"].unsqueeze(-1)).squeeze(-1) + (
            w.squeeze(-1) * (AtGB @ state["q_minus_c"].unsqueeze(-1)).squeeze(-1)).sum(2)
        covariance = torch.linalg.inv(information)
        position = (covariance @ rhs.unsqueeze(-1)).squeeze(-1)
        r = state["q_minus_c"] - (A @ position.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
        track_chi2 = (r.unsqueeze(-2) @ G_B @ r.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        d = position - state["prior"]
        chi2 = (d.unsqueeze(-2) @ prior_info @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1) + (weight * track_chi2).sum(-1)
        return position, covariance, chi2, 2 * weight.sum(-1) - 3

    def _fit(self, state, which, track_weight, tracks, shared=True):
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
            active = which & running.unsqueeze(1)
            old = state["position"]
            moved = (old[..., :2] - state["lin_point"][..., :2]).norm(dim=-1) > self.max_dist_to_lin_point
            state["lin_point"] = torch.where((active & moved).unsqueeze(-1), old, state["lin_point"])
            state["compatibility"] = torch.where(active.unsqueeze(-1), self._compatibility(state, old),
                                                 state["compatibility"])
            self._relinearise(state, active & moved, tracks)
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
            for key, value in zip(("position", "covariance", "chi2", "ndf"), self._kalman(state, state["weight"] * (
                    state["weight"] > self.fit_min_weight) * track_weight.unsqueeze(1))):
                state[key] = torch.where(_expand(active, value), value, state[key])
            d = (old - state["position"]).unsqueeze(-1)
            shift = (d.transpose(-1, -2) @ torch.linalg.solve(state["covariance"], d)).squeeze(-1).squeeze(-1)
            small = ((shift <= self.max_relative_shift) | ~active).all(dim=1)
            running = running & ~(small & (iteration >= len(temperatures) - 1))

