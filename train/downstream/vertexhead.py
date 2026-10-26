"""
vertexhead.py
=============
Primary-vertex fit from track-finder output.

Track parameters and quality labels (noise, beam-line validity) are fitted from each track's
assigned hits. VertexHead finds the point minimising the weighted sum of squared distances of
closest approach to every usable track, anchoring each track at its innermost assigned hit and
weighting it by (soft hit count) x P(real). With use_helix and Bz != 0 that straight-line vertex
seeds an iterative helix fit (uniform Bz along z in Tesla; scaling_factor converts metres to cm).

Inputs (from MambaAttentionHead.forward and TPCBatchDataset):
    class_probs      (n_events, n_tracks, 2)        P(slot empty), P(slot real)
    mask_probs       (n_events, n_hits, n_tracks)   hit-to-track assignment
    points           (n_events, n_hits, 4)          normalized (E, eta, phi, r), as the model sees them
    padding_mask     (n_events, n_hits)             True = real hit

VertexHead.forward inverts the normalization to (E, x, y, z) in cm; every fit below works in cm.
"""
import torch
import torch.nn as nn

def build_track_quality_mask(noise_labels=None, valid_tracks=None):
    """
    Combine per-track quality labels into one mask of usable tracks.

    Args:
        noise_labels: (n_events, n_tracks) int or None. 1 = noise (pT < threshold), 0 = good.
        valid_tracks: (n_events, n_tracks) int or None. 1 = production vertex within 1 cm, 0 = not.

    Returns:
        (n_events, n_tracks) bool, True = track is used in the fit; None if both inputs are None.
    """
    ref = noise_labels if noise_labels is not None else valid_tracks
    if ref is None:
        return None
    mask = torch.ones(ref.shape, dtype=torch.bool, device=ref.device)

    if noise_labels is not None:
        mask = mask & (noise_labels == 0)

    if valid_tracks is not None:
        mask = mask & (valid_tracks == 1)

    return mask

def track_flight_direction(track_reg_result, epsilon=1e-8):
    """
    Unit flight direction of each track.

    Args:
        track_reg_result: (..., 4) (q/(pT+1), theta, sin phi, cos phi).
        epsilon: floor on the norm before normalising.

    Returns:
        (..., 3) unit direction vectors.
    """
    theta = track_reg_result[..., 1]
    sin_phi = track_reg_result[..., 2]
    cos_phi = track_reg_result[..., 3]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    direction = torch.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dim=-1)
    direction_length = direction.norm(dim=-1, keepdim=True).clamp(min=epsilon)
    return direction / direction_length

def fit_vertex_by_closest_approach(
    track_position,
    track_direction,
    track_weight,
    minimum_total_weight=1e-6,
    regularization=1e-9,
    minimum_eigenvalue_ratio=1e-4,
):
    """
    Weighted least-squares point of closest approach to straight tracks, per event.

    Args:
        track_position: (n_events, n_tracks, 3) a point on each track's line.
        track_direction: (n_events, n_tracks, 3) unit direction of each line.
        track_weight: (n_events, n_tracks) non-negative weight; 0 = track ignored.
        minimum_total_weight: an event needs more total weight than this to be valid.
        regularization: added to the normal matrix's diagonal so the solve stays well-posed.
        minimum_eigenvalue_ratio: smallest/largest eigenvalue ratio below which the event is degenerate.

    Returns:
        dict:
            vertex_estimate: (n_events, 3) fitted vertex; NaN where fit_is_valid is False.
            chi_square: (n_events,) weighted sum of squared DCAs at the vertex; no ndf applied.
            fit_is_valid: (n_events,) bool; False without >= 2 non-parallel weighted tracks.
            track_closest_approach_point: (n_events, n_tracks, 3) point on each line nearest the vertex; NaN where invalid.
            track_dca: (n_events, n_tracks) each track's distance of closest approach; NaN where invalid.
    """
    device = track_position.device
    dtype = track_position.dtype

    weight = track_weight.clamp(min=0.0)

    direction_as_column = track_direction.unsqueeze(-1)
    direction_as_row = track_direction.unsqueeze(-2)
    direction_outer_product = torch.matmul(direction_as_column, direction_as_row)
    I3 = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    P = I3 - direction_outer_product

    weighted_P = weight.unsqueeze(-1).unsqueeze(-1) * P
    normal_matrix = weighted_P.sum(dim=1)

    P_track_position = torch.matmul(P, track_position.unsqueeze(-1)).squeeze(-1)
    weighted_position_sum = (weight.unsqueeze(-1) * P_track_position).sum(dim=1)

    # degeneracy check before solving: smallest/largest eigenvalue ratio (0 tracks: all ~0;
    # 1 track or parallel tracks: one ~0). A rank-deficient matrix can make the solve raise.
    normal_matrix_eigenvalues = torch.linalg.eigvalsh(normal_matrix)
    largest_eigenvalue = normal_matrix_eigenvalues[:, -1].clamp(min=regularization)
    eigenvalue_ratio = normal_matrix_eigenvalues[:, 0] / largest_eigenvalue
    well_determined = eigenvalue_ratio > minimum_eigenvalue_ratio
    enough_total_weight = weight.sum(dim=-1) > minimum_total_weight
    fit_is_valid = well_determined & enough_total_weight

    # invalid events solve against the identity instead; their results are set to NaN below
    regularized_normal_matrix = torch.where(fit_is_valid.view(-1, 1, 1),
                                            normal_matrix + regularization * I3.squeeze(0),
                                            I3.squeeze(0).expand_as(normal_matrix))
    PV = torch.linalg.solve(regularized_normal_matrix, weighted_position_sum)

    track_position_minus_PV = track_position - PV.unsqueeze(1)
    dca_vector = torch.matmul(P, track_position_minus_PV.unsqueeze(-1)).squeeze(-1)
    track_dca = dca_vector.norm(dim=-1)
    chi_square = (weight * track_dca ** 2).sum(dim=-1)

    track_closest_approach_point = PV.unsqueeze(1) + dca_vector

    PV = PV.clone()
    PV[~fit_is_valid] = float("nan")
    track_closest_approach_point = track_closest_approach_point.clone()
    track_closest_approach_point[~fit_is_valid] = float("nan")
    track_dca = track_dca.clone()
    track_dca[~fit_is_valid] = float("nan")

    return {
        "vertex_estimate": PV,
        "chi_square": chi_square,
        "fit_is_valid": fit_is_valid,
        "track_closest_approach_point": track_closest_approach_point,
        "track_dca": track_dca,
    }

def fit_vertex_by_helix_closest_approach(
    circle_centre,
    R_s,
    zslope,
    z0,
    track_weight,
    n_iterations=100,
    seed_vertex=None,
    d_angle=0.005,
    is_cosmics = False,
    **linear_fit_kwargs,
):
    """
    Helix-aware vertex fit: iterates fit_vertex_by_closest_approach on each track's local
    tangent at its point of closest approach to the current vertex.

    Args:
        circle_centre: (n_events, n_tracks, 2) centre of each track's x-y circle, cm.
        R_s: (n_events, n_tracks) signed circle radius, cm; sign(R_s) = sign(q Bz).
        zslope, z0: (n_events, n_tracks) z line, z = zslope * r + z0 (r -> x if is_cosmics).
            The circle and z line come from _track_params_from_hits.
        track_weight: (n_events, n_tracks) non-negative weight; 0 = track ignored.
        n_iterations: maximum iterations; stops early once every event moves < 1e-4.
        seed_vertex: (n_events, 3) starting vertex; origin if None.
        d_angle: angular step (radians) along the circle used to form the tangent.
        is_cosmics: the z line is z against x instead of z against transverse radius r.
        **linear_fit_kwargs: passed to fit_vertex_by_closest_approach.

    Returns:
        dict, same keys as fit_vertex_by_closest_approach, from the last iteration.
        Tracks without a fitted circle and z line (NaN) are excluded from it and get NaN
        track_dca and track_closest_approach_point.
    """
    n_events, n_tracks = R_s.shape
    device = R_s.device
    dtype = R_s.dtype

    if seed_vertex is None:
        print("WARNING: no seed_vertex provided, starting helix iteration from the origin.")
        PV = torch.zeros(n_events, 3, device=device, dtype=dtype)
    else:
        PV = seed_vertex.clone()

    still_moving = torch.ones(n_events, dtype=torch.bool, device=device)
    n_iterations_used = torch.zeros(n_events, dtype=torch.long, device=device)
    result = None

    radius = R_s.abs()
    x0, y0 = circle_centre.unbind(-1)
    origin = circle_centre

    for iteration in range(n_iterations):
        PV_per_track = PV.unsqueeze(1).expand(-1, n_tracks, -1)

        point_xy = PV_per_track[..., 0:2]

        diff = point_xy - origin
        norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pca_circle = origin + radius.unsqueeze(-1) * diff / norm

        angle_pca = torch.atan2(pca_circle[...,1] - y0, pca_circle[...,0] - x0)

        if is_cosmics:
            pca_z = pca_circle[..., 0] * zslope + z0
        else:
            pca_z = pca_circle.norm(dim=-1) * zslope + z0
        pca = torch.cat([pca_circle, pca_z.unsqueeze(-1)], dim=-1)

        # second point a small step along the circle gives the local tangent
        new_angle = angle_pca - torch.sign(R_s) * d_angle
        newx = radius * torch.cos(new_angle) + x0
        newy = radius * torch.sin(new_angle) + y0
        new_xy = torch.stack([newx, newy], dim=-1)

        if is_cosmics:
            new_z = newx * zslope + z0
        else:
            new_z = new_xy.norm(dim=-1) * zslope + z0

        second_point_pca = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)

        raw_tangent = second_point_pca - pca
        tangent = raw_tangent / raw_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        # NaN guard: near-zero-pT tracks (e.g. noise) produce a helix radius ≈ 0
        nan_track = torch.isnan(pca).any(dim=-1) | torch.isnan(tangent).any(dim=-1)
        if nan_track.any():
            pca     = torch.where(nan_track.unsqueeze(-1), torch.zeros_like(pca), pca)
            tangent = torch.where(nan_track.unsqueeze(-1), pca.new_tensor([0.0, 0.0, 1.0]).expand_as(tangent), tangent)

        # NaN tracks are excluded from this fit, not left pulling as a stand-in beam-axis line
        iter_weight = torch.where(nan_track, torch.zeros_like(track_weight), track_weight)
        result = fit_vertex_by_closest_approach(pca, tangent, iter_weight, **linear_fit_kwargs)
        result["track_dca"] = torch.where(nan_track, torch.full_like(result["track_dca"], float("nan")), result["track_dca"])
        result["track_closest_approach_point"] = torch.where(
            nan_track.unsqueeze(-1), torch.full_like(result["track_closest_approach_point"], float("nan")),
            result["track_closest_approach_point"])

        new_PV = result["vertex_estimate"]
        fit_is_valid = result["fit_is_valid"]

        # invalid-fit events never count as still moving
        step = (new_PV - PV).norm(dim=-1)
        step = torch.where(fit_is_valid, step, torch.zeros_like(step))

        n_iterations_used = torch.where(
            still_moving & fit_is_valid, torch.full_like(n_iterations_used, iteration + 1), n_iterations_used
        )

        # advance PV only where the fit is valid, so one bad event can't spread NaN
        PV = torch.where(fit_is_valid.unsqueeze(-1), new_PV, PV)

        still_moving = still_moving & fit_is_valid & (step > 1e-4)
        if not still_moving.any():
            break

    return result

def _masked_mean(value, mask, denom, eps):
    """sum(value * mask) over hits, divided by denom (floored at eps)."""
    return (value * mask).sum(dim=1) / denom.clamp(min=eps)

def _line_fit(u, v, mask, min_hits=2, eps=1e-12):
    """
    Deming (orthogonal-distance) line fit v = slope * u + intercept, per track.

    Args:
        u, v: (n_events, n_hits, n_tracks) coordinates to fit.
        mask: (n_events, n_hits, n_tracks) bool/float, which hits belong to each track.
        min_hits: tracks with fewer hits get NaN.
        eps: numerical floor for divisions.

    Returns:
        slope, intercept: (n_events, n_tracks) each; NaN where fewer than min_hits hits.
    """
    mask = mask.to(u.dtype)
    n_valid = mask.sum(dim=1)
    valid = n_valid >= min_hits

    mean_u = _masked_mean(u, mask, n_valid, eps)
    mean_v = _masked_mean(v, mask, n_valid, eps)

    du = u - mean_u.unsqueeze(1)
    dv = v - mean_v.unsqueeze(1)

    ssd_u = _masked_mean(du * du, mask, n_valid, eps) * n_valid
    ssd_v = _masked_mean(dv * dv, mask, n_valid, eps) * n_valid
    ssd_uv = _masked_mean(du * dv, mask, n_valid, eps) * n_valid

    ssd_uv_safe = torch.where(ssd_uv.abs() < eps, torch.full_like(ssd_uv, eps), ssd_uv)
    slope = (ssd_v - ssd_u + torch.sqrt((ssd_v - ssd_u) ** 2 + 4 * ssd_uv ** 2)) / 2.0 / ssd_uv_safe
    intercept = mean_v - slope * mean_u

    nan = torch.full_like(slope, float("nan"))
    slope = torch.where(valid, slope, nan)
    intercept = torch.where(valid, intercept, nan)
    return slope, intercept

def _circle_fit(x, y, mask, min_hits=3, newton_iters=20, eps=1e-12):
    """
    Taubin algebraic circle fit per track, computed in float64.

    Args:
        x, y: (n_events, n_hits, n_tracks) hit coordinates.
        mask: (n_events, n_hits, n_tracks) bool/float, which hits belong to each track.
        min_hits: tracks with fewer hits get NaN.
        newton_iters: fixed Newton iterations for the Taubin root.
        eps: numerical floor for divisions.

    Returns:
        radius, x_centre, y_centre: (n_events, n_tracks) float64 each; NaN where fewer than min_hits hits.
    """
    x, y, mask = x.double(), y.double(), mask.double()
    n_valid = mask.sum(dim=1)
    valid = n_valid >= min_hits
    mean_x = _masked_mean(x, mask, n_valid, eps)
    mean_y = _masked_mean(y, mask, n_valid, eps)
    Xi, Yi = x - mean_x.unsqueeze(1), y - mean_y.unsqueeze(1)
    Zi = Xi * Xi + Yi * Yi
    Mxy, Mxx, Myy = (_masked_mean(v, mask, n_valid, eps) for v in (Xi * Yi, Xi * Xi, Yi * Yi))
    Mxz, Myz, Mzz = (_masked_mean(v, mask, n_valid, eps) for v in (Xi * Zi, Yi * Zi, Zi * Zi))

    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz
    A3, A2 = 4 * Mz, -3 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy

    xk, yk = torch.zeros_like(Mz), A0.clone()
    improving_any = torch.ones_like(Mz, dtype=torch.bool)
    for _ in range(newton_iters):
        Dy = A1 + xk * (2 * A2 + 3 * A3 * xk)
        xnew = xk - yk / torch.where(Dy.abs() < eps, torch.full_like(Dy, eps), Dy)
        ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3))
        better = (ynew.abs() < yk.abs()) & torch.isfinite(xnew) & improving_any
        xk, yk = torch.where(better, xnew, xk), torch.where(better, ynew, yk)
        improving_any = improving_any & better

    DET = xk * xk - xk * Mz + Cov_xy
    DET = torch.where(DET.abs() < eps, torch.full_like(DET, eps), DET)
    Xc = (Mxz * (Myy - xk) - Myz * Mxy) / DET / 2
    Yc = (Myz * (Mxx - xk) - Mxz * Mxy) / DET / 2
    radius = torch.sqrt((Xc * Xc + Yc * Yc + Mz).clamp(min=0.0))

    nan = torch.full_like(radius, float("nan"))
    return (torch.where(valid, radius, nan), torch.where(valid, Xc + mean_x, nan),
            torch.where(valid, Yc + mean_y, nan))

def _track_params_from_hits(points, hit_r, is_assigned, noise_pt_threshold, Bz=1.4,
                           scaling_factor=100.0, valid_radius=1.0, min_hits=3, is_cosmics=False):
    """
    Track parameters and quality labels fitted from each track's assigned hits -- no truth needed.

    Circle fit in x-y gives pT and the circle; the direction of travel (innermost -> outermost hit)
    gives the charge and phi at the innermost hit, in VertexHead's convention
    (centre = innermost hit + R_s (sin phi, -cos phi), sign(R_s) = sign(q Bz)).
    A z-against-r line fit (z against x for cosmics) gives theta.

    Args:
        points: (n_events, n_hits, 4) hits (E, x, y, z), cm.
        hit_r: (n_events, n_hits) transverse radius of each hit, cm.
        is_assigned: (n_events, n_hits, n_tracks) bool, hard hit-to-track assignment
            (from VertexHead.track_position_from_hits).
        noise_pt_threshold: fitted pT below this (GeV) -> noise_labels = 1.
        Bz: solenoid field along z, Tesla.
        scaling_factor: converts pT / (0.3 Bz), in metres, to the units of points.
        valid_radius: fitted circle's closest approach to the beam line below this -> valid_tracks = 1.
        min_hits: tracks with fewer assigned hits are not fitted.
        is_cosmics: fit z against x instead of z against transverse radius r.

    Returns:
        dict:
            track_reg_result: (n_events, n_tracks, 4) (q/(pT+1), theta, sin phi, cos phi), phi at the
                innermost hit; finite filler where fit_ok is False.
            noise_labels: (n_events, n_tracks) long, 1 = noise; 1 where fit_ok is False.
            valid_tracks: (n_events, n_tracks) long, 1 = circle passes within valid_radius of the beam
                line; 0 where fit_ok is False.
            pT, radius, beam_dca: (n_events, n_tracks); NaN where fit_ok is False.
            fit_ok: (n_events, n_tracks) bool, enough hits and finite fit.
            circle_centre (n_events, n_tracks, 2); R_s, zslope, z0 (n_events, n_tracks): the fitted
                circle (sign(R_s) = sign(q Bz)) and z line, in the dtype of points, for
                fit_vertex_by_helix_closest_approach; NaN where fit_ok is False.
    """
    n_tracks = is_assigned.shape[-1]
    x = points[..., 1].unsqueeze(-1).expand(-1, -1, n_tracks).double()
    y = points[..., 2].unsqueeze(-1).expand(-1, -1, n_tracks).double()
    z = points[..., 3].unsqueeze(-1).expand(-1, -1, n_tracks).double()
    r = hit_r.unsqueeze(-1).expand(-1, -1, n_tracks).double()

    radius, xc, yc = _circle_fit(x, y, is_assigned, min_hits=min_hits)
    zslope, z0 = _line_fit(x if is_cosmics else r, z, is_assigned, min_hits=min_hits)

    # innermost and outermost assigned hit of each track
    inf = torch.full_like(r, float("inf"))
    i_in = torch.where(is_assigned, r, inf).argmin(dim=1, keepdim=True)
    i_out = torch.where(is_assigned, r, -inf).argmax(dim=1, keepdim=True)
    xin, yin = x.gather(1, i_in).squeeze(1), y.gather(1, i_in).squeeze(1)
    xout, yout = x.gather(1, i_out).squeeze(1), y.gather(1, i_out).squeeze(1)

    # travel inner -> outer: anticlockwise about the centre <=> R_s < 0
    anticlockwise = ((xin - xc) * (yout - yc) - (yin - yc) * (xout - xc)) > 0
    sign_Rs = torch.where(anticlockwise, -torch.ones_like(radius), torch.ones_like(radius))
    ux, uy = (xc - xin) / radius, (yc - yin) / radius
    sin_phi, cos_phi = sign_Rs * ux, -sign_Rs * uy
    q = sign_Rs * (1.0 if Bz >= 0 else -1.0)
    pT = 0.3 * abs(Bz) * radius / scaling_factor
    # cot(theta) = dz/ds: dz/dr for tracks from the beam line, dz/dx * dx/ds = zslope cos(phi) for cosmics
    cot_theta = zslope * cos_phi if is_cosmics else zslope
    theta = torch.atan2(torch.ones_like(cot_theta), cot_theta)
    beam_dca = (torch.sqrt(xc * xc + yc * yc) - radius).abs()

    params = torch.stack([q / (pT + 1.0), theta, sin_phi, cos_phi], dim=-1)
    fit_ok = torch.isfinite(params).all(dim=-1)
    filler = params.new_tensor([0.5, torch.pi / 2, 0.0, 1.0]).expand_as(params)
    params = torch.where(fit_ok.unsqueeze(-1), params, filler)
    nan = torch.full_like(pT, float("nan"))
    pT, radius, beam_dca = (torch.where(fit_ok, v, nan) for v in (pT, radius, beam_dca))

    circle_centre = torch.where(fit_ok.unsqueeze(-1), torch.stack([xc, yc], dim=-1), nan.unsqueeze(-1))
    R_s, zslope, z0 = (torch.where(fit_ok, v, nan) for v in (sign_Rs * radius, zslope, z0))

    return {
        "track_reg_result": params.to(points.dtype),
        "noise_labels": torch.where(fit_ok, (pT < noise_pt_threshold).long(), torch.ones_like(fit_ok, dtype=torch.long)),
        "valid_tracks": torch.where(fit_ok, (beam_dca < valid_radius).long(), torch.zeros_like(fit_ok, dtype=torch.long)),
        "pT": pT, "radius": radius, "beam_dca": beam_dca, "fit_ok": fit_ok,
        "circle_centre": circle_centre.to(points.dtype), "R_s": R_s.to(points.dtype),
        "zslope": zslope.to(points.dtype), "z0": z0.to(points.dtype),
    }

class VertexHead(nn.Module):

    """
    Primary-vertex fit from track-finder output. No learnable parameters unless learn_weights=True.

    Args:
        learn_weights: if True, a small network rescales each track's weight.
        weight_hidden_dim: hidden width of that network.
        use_helix: refine the straight-line vertex with the helix fit (when Bz != 0).
        helix_iterations: maximum helix-fit iterations.
        Bz: solenoid field along z, Tesla.
        is_cosmics: fit z against x instead of z against transverse radius r.
        scaling_factor: converts pT / (0.3 Bz), in metres, to the units of points.
        noise_pt_threshold: tracks with fitted pT below this (GeV) are labelled noise and excluded.
    """
    def __init__(
        self,
        learn_weights: bool = False,
        weight_hidden_dim: int = 32,
        use_helix: bool = True,
        helix_iterations: int = 100,
        Bz: float = 1.4,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
    ):
        super().__init__()
        self.learn_weights = learn_weights
        self.use_helix = use_helix
        self.helix_iterations = helix_iterations
        self.Bz = Bz
        self.is_cosmics = is_cosmics
        self.scaling_factor = scaling_factor
        self.noise_pt_threshold = noise_pt_threshold
        if learn_weights:
            # 6 inputs: class_probs (2) + track_reg_result (4); position deliberately excluded
            self.track_weight_network = nn.Sequential(
                nn.Linear(6, weight_hidden_dim),
                nn.GELU(),
                nn.Linear(weight_hidden_dim, 1),
                nn.Softplus(),  # weights must be >= 0
            )

    def dontUseHelix(self, use_helix: bool = False):
        self.use_helix = use_helix

    def isCosmics(self, is_cosmics: bool = True):
        self.is_cosmics = is_cosmics

    def setScalingFactor(self, scaling_factor: float = 100.0):
        self.scaling_factor = scaling_factor

    def setNoisePtThreshold(self, noise_pt_threshold: float = 0.25):
        self.noise_pt_threshold = noise_pt_threshold
    
    @staticmethod
    def _unnormalize_hits(points):
        """
        Invert TPCBatchDataset's normalization and polar transform.

        Args:
            points: (..., 4) normalized (E, eta, phi, r): z-scored E; eta, phi and transverse r
                min-max scaled to [0, 1]. eta is the hit position's, seen from the origin.

        Returns:
            points: (..., 4) (E, x, y, z), cm.
            r: (...) transverse radius, cm.
        """
        # TPCBatchDataset's normalization (fm4npp/datasets/dataset.py, TPCBatchDataset.__init__);
        # keep these in line with it
        eta_lim = (-2.0, 2.0)
        phi_lim = (-torch.pi, torch.pi)
        r_lim = (31.371997833251953, 75.38493347167969)    # transverse radius, cm
        E_mean, E_std = 253.0982, 268.7093

        E = points[..., 0] * E_std + E_mean
        eta = points[..., 1] * (eta_lim[1] - eta_lim[0]) + eta_lim[0]
        phi = points[..., 2] * (phi_lim[1] - phi_lim[0]) + phi_lim[0]
        r = points[..., 3] * (r_lim[1] - r_lim[0]) + r_lim[0]
        return torch.stack([E, r * torch.cos(phi), r * torch.sin(phi), r * torch.sinh(eta)], dim=-1), r

    @staticmethod
    def track_position_from_hits(points, hit_r, mask_probs, padding_mask):
        """
        Per-track reference point, hit weight and hard hit assignment.

        Args:
            points: (n_events, n_hits, 4) hits (E, x, y, z), cm.
            hit_r: (n_events, n_hits) transverse radius of each hit, cm.
            mask_probs: (n_events, n_hits, n_tracks) hit-to-track assignment.
            padding_mask: (n_events, n_hits) True = real hit.

        Returns:
            track_position: (n_events, n_tracks, 3) innermost assigned hit by transverse radius
                (the hit _track_params_from_hits gives phi at); (0, 0, 0) for a track with none.
            total_hit_weight: (n_events, n_tracks) sum of mask_probs over real hits.
            is_assigned: (n_events, n_hits, n_tracks) bool, each real hit assigned to its argmax track;
            hits with zero probability for every track are unassigned.
        """
        hit_position = points[..., 1:4]
        hit_weight = mask_probs * padding_mask.unsqueeze(-1).to(mask_probs.dtype)
        total_hit_weight = hit_weight.sum(dim=1)

        n_events, n_hits, n_tracks = mask_probs.shape

        # each hit belongs to the track slot with the largest mask_probs
        hard_assigned_track = mask_probs.argmax(dim=-1)

        is_assigned = torch.nn.functional.one_hot(hard_assigned_track, num_classes=n_tracks).bool()
        # Exclude hits with zero assignment probability for every track.
        has_assignment = mask_probs.max(dim=-1).values > 0
        is_assigned = (is_assigned & padding_mask.unsqueeze(-1) & has_assignment.unsqueeze(-1))

        radius_if_assigned = torch.where(is_assigned
                                       , hit_r.unsqueeze(-1).expand(-1, -1, n_tracks)
                                       , torch.full((n_events, n_hits, n_tracks), float("inf"), device=points.device, dtype=hit_r.dtype),)
        innermost_hit_index = radius_if_assigned.argmin(dim=1)
        has_any_assigned_hit = is_assigned.any(dim=1)

        event_index = torch.arange(n_events, device=points.device).unsqueeze(-1).expand(-1, n_tracks)
        innermost_hit_position = hit_position[event_index, innermost_hit_index]

        track_position = torch.where(has_any_assigned_hit.unsqueeze(-1), innermost_hit_position, torch.zeros_like(innermost_hit_position))
        return track_position, total_hit_weight, is_assigned

    def forward(
        self,
        class_probs,
        mask_probs,
        points,
        padding_mask,
    ):
        """
        Args:
            class_probs: (n_events, n_tracks, 2) P(slot empty), P(slot real).
            mask_probs: (n_events, n_hits, n_tracks) hit-to-track assignment.
            points: (n_events, n_hits, 4) hits as TPCBatchDataset yields them, normalized
                (E, eta, phi, r); inverted to (E, x, y, z) in cm before fitting.
            padding_mask: (n_events, n_hits) True = real hit.

        Returns:
            dict, positions in cm:
                vertex_estimate: (n_events, 3) fitted vertex; NaN where fit_is_valid is False.
                chi_square: (n_events,) weighted sum of squared DCAs at the vertex; no ndf applied.
                fit_is_valid: (n_events,) bool; False with fewer than 2 usable, non-parallel tracks.
                track_position: (n_events, n_tracks, 3) innermost assigned hit of each track.
                track_direction: (n_events, n_tracks, 3) straight-line flight direction.
                track_weight: (n_events, n_tracks) weight after quality masking; tracks the helix fit
                    later excludes for too few hits still show their weight here.
                track_quality_mask: (n_events, n_tracks) bool, or None if no labels were given.
                track_closest_approach_point: (n_events, n_tracks, 3) point on each track nearest the vertex.
                track_dca: (n_events, n_tracks) each track's distance of closest approach.
        """
        # padded hits are -100 in every column, which overflows sinh(eta); any finite value will do
        points, hit_r = self._unnormalize_hits(points.masked_fill(~padding_mask.unsqueeze(-1), 0.0))
        track_position, total_hit_weight, is_assigned = self.track_position_from_hits(
            points, hit_r, mask_probs, padding_mask)
        params = _track_params_from_hits(points, hit_r, is_assigned, self.noise_pt_threshold,
                                         Bz=self.Bz, scaling_factor=self.scaling_factor,
                                         is_cosmics=self.is_cosmics)
        track_reg_result = params["track_reg_result"]
        noise_labels, valid_tracks = params["noise_labels"], params["valid_tracks"]
        track_direction = track_flight_direction(track_reg_result)

        probability_track_is_real = class_probs[..., 1]

        if self.learn_weights:
            track_quality_features = torch.cat([class_probs, track_reg_result], dim=-1)
            learned_weight_scale = self.track_weight_network(track_quality_features).squeeze(-1)
            track_weight = probability_track_is_real * total_hit_weight * learned_weight_scale
        else:
            track_weight = probability_track_is_real * total_hit_weight

        quality_mask = build_track_quality_mask(noise_labels, valid_tracks)

        if quality_mask is not None:
            if quality_mask.shape != track_weight.shape:
                raise ValueError(
                    f"Track-quality mask shape {quality_mask.shape} does not match "
                    f"track_weight shape {track_weight.shape}.  "
                    f"noise_labels and valid_tracks must be per-TRACK tensors with "
                    f"shape (n_events, n_tracks) = {track_weight.shape}."
                )
            track_weight = track_weight * quality_mask.to(dtype=track_weight.dtype)

        fit = fit_vertex_by_closest_approach(track_position, track_direction, track_weight)

        if (self.use_helix and self.Bz != 0.0):
            fit = fit_vertex_by_helix_closest_approach(
                params["circle_centre"],
                params["R_s"],
                params["zslope"],
                params["z0"],
                track_weight,               # already zeroed for bad tracks
                n_iterations=self.helix_iterations,
                seed_vertex=fit["vertex_estimate"],
                is_cosmics=self.is_cosmics,
            )
        elif self.Bz == 0.0:
            print("WARNING: b_z=0.0, using straight-line fit instead of helix-aware fit.")
        
        if fit is None:
            return None
        
        return {
            "vertex_estimate": fit["vertex_estimate"],
            "chi_square": fit["chi_square"],
            "fit_is_valid": fit["fit_is_valid"],
            "track_position": track_position,
            "track_direction": track_direction,
            "track_weight": track_weight,
            "track_quality_mask": quality_mask,
            "track_closest_approach_point": fit["track_closest_approach_point"],
            "track_dca": fit["track_dca"],
        }