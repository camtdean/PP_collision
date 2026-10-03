"""
Each reconstructed track is a straight line in 3D -- a reference
point on the track plus its direction of flight. The primary vertex is
taken to be the single point that minimizes the weighted sum of squared
perpendicular distances to every track's line, all at once. No
covariance matrix is used; each track is weighted by how many hits
are on it and how confident the track finder is that it's real which is
output by the track finder. This is a pure linear algebra solution, there
is no machine learning calculation (yet). Everything it needs comes straight
out of the already-trained track-finding head (MambaAttentionHead in model.py).

Per-track quantities used, and where they come from:

    track_position  (n_events, n_tracks, 3)
        A point the track's line of flight passes through: specifically,
        the position of the single assigned hit closest to the origin --
        the innermost measurement on that track (see track_position_from_hits
        below). This is a real spatial position from this track's own hits.

    track_direction (n_events, n_tracks, 3)
        A unit vector along the track's flight direction, reconstructed from
        the track-finding head's regressed (theta, sin(phi), cos(phi))

    track_weight (n_events, n_tracks)
        How much this track counts in the fit: the number of hits
        on the track times the track finder's own confidence that
        this slot is a real track and not an empty one.
        
Inputs, all produced by MambaAttentionHead.forward() (model.py) with no
modifications needed there:

    class_probs      (n_events, n_tracks, 2)   P(track slot is real / empty)
    track_reg_result (n_events, n_tracks, 4)   (q/(pT+1), theta, sin(phi), cos(phi))
    mask_probs       (n_events, n_hits, n_tracks)  soft hit-to-track assignment
    points           (n_events, n_hits, 4)     raw (E, x, y, z) hits fed to the backbone
    padding_mask     (n_events, n_hits)        True when a hit is real (i.e. not padding)

theta is the polar angle from the beam axis; phi is the azimuthal angle

For a single track with reference point X and slope S, and a
candidate vertex PV,

    DCA = | X - PV - [S.(X - PV) / S.S] S |

where S = (p_x/p, p_y/p, p_z/p), i.e. the unit momentum vector. 
fit_vertex_by_closest_approach below is exactly this DCA, squared,
weighted, and summed over every track in the event -- a chi-square
as a function of a candidate PV -- solved for the PV that minimizes it,
generalized from a single track to an arbitrary number N.

Deliberately NOT implemented here: The helical fit 
below assumes a uniform magnetic field with the scaling factor 
for the Magnetic field strength and units for this dataset's vtx_x/y/z and momentum 
(see dataset.py's `data_scaler`, currently a placeholder value of 1) still
being undefined, current defualt assumes a meter to cm conversion with Bz being in Tesla.
"""

import torch
import torch.nn as nn

def track_flight_direction(track_reg_result, epsilon=1e-8):
    """
    Reconstruct each track's unit flight direction from the track-finding
    head's regressed parameters.

    Args:
        track_reg_result: (..., 4) tensor, last dimension = (q/(pT+1), theta, sin(phi), cos(phi))

    Returns:
        (..., 3) unit direction vectors, one per track.
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
    Weighted least-squares point of closest approach to a set of straight
    tracks, solved independently for each event.

    PyTorch: squeeze removes dimensions of size 1, while unsqueeze adds a dimension of size 1

    Args:
        track_position: (n_events, n_tracks, 3) reference point on each track's line (eq. 8.3's X)
        track_direction: (n_events, n_tracks, 3) unit direction of each track's line (eq. 8.3's S, unit-normalized)
        track_weight: (n_events, n_tracks) nonnegative per-track weight (0 = ignore)
        minimum_total_weight: an event needs at least this much total
            weight, spread across at least two non-parallel tracks, before
            its fit is considered meaningful rather than degenerate
        regularization: small value added to normal_matrix's diagonal so
            the linear solve stays numerically well-posed even in
            degenerate (too-few-track) events
        minimum_eigenvalue_ratio: degeneracy threshold, expressed as a
            RATIO of normal_matrix's smallest to largest eigenvalue

    Returns:
        {
          "vertex_estimate": (n_events, 3)  the fitted PV
              (NaN in any event where fit_is_valid is False)
          "chi_square": (n_events,)  chi_square(PV) at the solution -- the
              weighted sum of squared distances of closest approach,
              generalized to N tracks, evaluated at the fitted vertex.
              NO NUMBER OF DEGREES OF FREEDOM APPLIED!!!
          "fit_is_valid": (n_events,) bool, False wherever the event did
              not contain enough independent track information to define
              a vertex at all
          "track_closest_approach_point": (n_events, n_tracks, 3)  the
              point ON each track's own line closest to the fitted PV
              (NaN wherever fit_is_valid is False). Plot this alongside
              vertex_estimate and track_position: if the fit is sensible,
              each track's closest_approach_point should sit close to
              vertex_estimate, and roughly along the line from
              track_position through track_direction.
          "track_dca": (n_events, n_tracks)  each track's OWN distance of
              closest approach to the fitted PV (NaN wherever
              fit_is_valid is False) -- the per-track quantity that
              chi_square sums (weighted, squared) over. A track with a
              much larger track_dca than the others in its event is an
              outlier the fit did not actually agree with, even if the
              overall chi_square looks acceptable.
        }
    """
    n_events, n_tracks, _ = track_position.shape
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

    regularized_normal_matrix = normal_matrix + regularization * I3.squeeze(0)
    PV = torch.linalg.solve(regularized_normal_matrix, weighted_position_sum)

    track_position_minus_PV = track_position - PV.unsqueeze(1)
    dca_vector = torch.matmul(P, track_position_minus_PV.unsqueeze(-1)).squeeze(-1)
    track_dca = dca_vector.norm(dim=-1)
    chi_square = (weight * track_dca ** 2).sum(dim=-1)

    track_closest_approach_point = PV.unsqueeze(1) + dca_vector

    # Degeneracy check -- see the minimum_eigenvalue_ratio argument doc
    # above for why this must be a ratio of eigenvalues, not an absolute
    # cutoff. 0 tracks: all eigenvalues ~0. 1 track: exactly one ~0
    # eigenvalue (the singular direction along that track). >=2
    # non-parallel weighted tracks: all three eigenvalues clearly positive.
    normal_matrix_eigenvalues = torch.linalg.eigvalsh(normal_matrix)  # ascending, (n_events, 3)
    largest_eigenvalue = normal_matrix_eigenvalues[:, -1].clamp(min=regularization)
    eigenvalue_ratio = normal_matrix_eigenvalues[:, 0] / largest_eigenvalue
    well_determined = eigenvalue_ratio > minimum_eigenvalue_ratio
    enough_total_weight = weight.sum(dim=-1) > minimum_total_weight
    fit_is_valid = well_determined & enough_total_weight

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
    points,
    mask,
    track_reg_result,
    track_position,
    Bz,
    track_weight,
    n_iterations=100,
    seed_vertex=None,
    d_angle=0.005,
    scaling_factor = 100.0,  # placeholder for dataset.py's data_scaler, which is not yet implemented
    is_cosmics = False,
    **linear_fit_kwargs,
):
    """
    Helix-aware primary vertex fit, built on top of
    fit_vertex_by_closest_approach.

    Args:
        points: (n_events, n_hits, 4) raw hits; spatial position is points[..., 1:4]
        mask: (n_events, n_hits, n_tracks) bool/float, True where a hit
            is assigned to a track
        track_reg_result: (n_events, n_tracks, 4) -- (q/(pT+1), theta, sin(phi), cos(phi))
        track_position: (n_events, n_tracks, 3) -- reference point on each
            track's line (the innermost assigned hit, see track_position_from_hits)
        Bz: solenoid field, Tesla. Scalar (python float) or (...)-dimensional tensor
        track_weight: (n_events, n_tracks) -- same meaning as in
            fit_vertex_by_closest_approach
        n_iterations: max outer iterations
        seed_vertex: (n_events, 3) starting point for iteration 0. Defaults
            to the origin.
        d_angle: small angle (radians) to step along each track's helix
        scaling_factor: placeholder for dataset.py's data_scaler, which is
            not yet implemented. Used to convert the track_reg_result's
            q/(pT+1) into a radius in the same units as points[..., 1:4].
        is_cosmics: if True, fit a straight line in z vs. x
        **linear_fit_kwargs: forwarded to fit_vertex_by_closest_approach
            (minimum_total_weight, regularization, minimum_eigenvalue_ratio)

    Returns:
        {
          "vertex_estimate": (n_events, 3)  the fitted PV
              (NaN in any event where fit_is_valid is False)
          "chi_square": (n_events,)  chi_square(PV) at the solution -- the
              weighted sum of squared distances of closest approach,
              generalized to N tracks, evaluated at the fitted vertex.
              NO NUMBER OF DEGREES OF FREEDOM APPLIED!!!
          "fit_is_valid": (n_events,) bool, False wherever the event did
              not contain enough independent track information to define
              a vertex at all
          "track_closest_approach_point": (n_events, n_tracks, 3)  the
              point ON each track's own line closest to the fitted PV
              (NaN wherever fit_is_valid is False). Plot this alongside
              vertex_estimate and track_position: if the fit is sensible,
              each track's closest_approach_point should sit close to
              vertex_estimate, and roughly along the line from
              track_position through track_direction.
          "track_dca": (n_events, n_tracks)  each track's OWN distance of
              closest approach to the fitted PV (NaN wherever
              fit_is_valid is False) -- the per-track quantity that
              chi_square sums (weighted, squared) over. A track with a
              much larger track_dca than the others in its event is an
              outlier the fit did not actually agree with, even if the
              overall chi_square looks acceptable.
        }
    """
    n_events, n_tracks, _ = track_reg_result.shape
    device = track_reg_result.device
    dtype = track_reg_result.dtype

    if seed_vertex is None:
        print("WARNING: no seed_vertex provided, starting helix iteration from the origin.")
        PV = torch.zeros(n_events, 3, device=device, dtype=dtype)
    else:
        PV = seed_vertex.clone()

    still_moving = torch.ones(n_events, dtype=torch.bool, device=device)
    n_iterations_used = torch.zeros(n_events, dtype=torch.long, device=device)
    result = None

    n_tracks = mask.shape[-1]
    hit_x = points[..., 1].unsqueeze(-1).expand(-1, -1, n_tracks)
    hit_y = points[..., 2].unsqueeze(-1).expand(-1, -1, n_tracks)
    hit_z = points[..., 3].unsqueeze(-1).expand(-1, -1, n_tracks)

    rho = track_reg_result[..., 0]
    sin_phi = track_reg_result[..., 2]
    cos_phi = track_reg_result[..., 3]

    x_h = track_position[..., 0]
    y_h = track_position[..., 1]
              
    q = torch.sign(rho)
    q = torch.where(q == 0, torch.ones_like(q), q)  # rho == 0 is a degenerate/invalid slot either way
    pT = (1.0 / rho.abs()) - 1.0
             
    b_z_t = torch.full_like(rho, float(Bz))
    R_s = (pT / (0.3 * q * b_z_t)) * scaling_factor
    radius = R_s.abs()
              
    x0 = x_h + R_s * sin_phi
    y0 = y_h - R_s * cos_phi
    
    angle = torch.atan2(hit_y - y0, hit_x - x0)
    angle_ref = angle[..., 0, :]  # (n_events, n_tracks)

    if is_cosmics:    
        zslope, z0 = _line_fit(hit_x, hit_z, mask)
    else:
        small_angle = torch.atan2(torch.sin(angle - angle_ref.unsqueeze(1)),
                                       torch.cos(angle - angle_ref.unsqueeze(1))) 
        s = radius.unsqueeze(1) * small_angle
        zslope, z0 = _line_fit(s, hit_z, mask)

    for iteration in range(n_iterations):
        PV_per_track = PV.unsqueeze(1).expand(-1, n_tracks, -1)  # (n_events, n_tracks, 3)
            
        point_xy = PV_per_track[..., 0:2]
        
        origin = torch.stack([x0, y0], dim=-1)  # (..., 2)
        diff = point_xy - origin
        norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pca_circle = origin + radius.unsqueeze(-1) * diff / norm
        
        angle_pca = torch.atan2(pca_circle[...,1] - y0, pca_circle[...,0] - x0)
          
        if is_cosmics:
            pca_z = pca_circle[..., 0] * zslope + z0
        else:
            s_pca = radius * torch.atan2(torch.sin(angle_pca - angle_ref), torch.cos(angle_pca - angle_ref))
            pca_z = s_pca * zslope + z0
        pca = torch.cat([pca_circle, pca_z.unsqueeze(-1)], dim=-1)  # (..., 3)
        
        # Second point on the circle, a small angle further along, to define
        # the local tangent direction.
        new_angle = angle_pca - torch.sign(R_s) * d_angle
        newx = radius * torch.cos(new_angle) + x0
        newy = radius * torch.sin(new_angle) + y0
        new_xy = torch.stack([newx, newy], dim=-1)
        new_s_pca = radius * torch.atan2(torch.sin(new_angle - angle_ref), torch.cos(new_angle - angle_ref))
                    
        if is_cosmics:
            new_z = newx * zslope + z0
        else:
            new_z = new_s_pca * zslope + z0
        
        second_point_pca = torch.cat([new_xy, new_z.unsqueeze(-1)], dim=-1)
        
        raw_tangent = second_point_pca - pca
        tangent = raw_tangent / raw_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        result = fit_vertex_by_closest_approach(pca, tangent, track_weight, **linear_fit_kwargs)

        new_PV = result["vertex_estimate"]
        fit_is_valid = result["fit_is_valid"]

        # NaN-safe step size: invalid-fit events never count as "still
        # moving" (there's nothing to converge), so they don't block the
        # early-exit check below.
        step = (new_PV - PV).norm(dim=-1)
        step = torch.where(fit_is_valid, step, torch.zeros_like(step))

        n_iterations_used = torch.where(
            still_moving & fit_is_valid, torch.full_like(n_iterations_used, iteration + 1), n_iterations_used
        )

        # Only advance the vertex where the fit is valid; hold position
        # elsewhere so one bad event's NaNs can't corrupt PV for the
        # remainder of a batched (vectorized-over-events) iteration.
        PV = torch.where(fit_is_valid.unsqueeze(-1), new_PV, PV)

        still_moving = still_moving & fit_is_valid & (step > 1e-4)
        if not still_moving.any():
            break
        
    return result

def _masked_mean(value, mask, denom, eps):
    """sum(value * mask) / denom, broadcasting mask over value's last dims as needed."""
    return (value * mask).sum(dim=1) / denom.clamp(min=eps)


def _line_fit(u, v, mask, min_hits=2, eps=1e-12):
    """
    the Deming/orthogonal distance line fit -- minimizes true perpendicular distance, not
    vertical residual, assuming equal variance in u and v).
 
    Args:
        u, v: (n_events, n_hits, n_tracks), the two coordinates to fit
            v = slope*u + intercept
        mask: (n_events, n_hits, n_tracks) bool/float, same convention as
            circle_fit_by_taubin
        min_hits: 2 is the practical minimum for aline to mean anything
        eps: numerical floor on ssd_xy
 
    Returns:
        slope, intercept: (n_events, n_tracks)
        valid: (n_events, n_tracks) bool
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

class VertexHead(nn.Module):
    """
    Zero learnable parameters by default (learn_weights=False): the
    per-track weight is simply (soft hit count) x (P(real track)), and
    everything downstream of that is the fixed linear solve above --
    nothing here is trained. Set learn_weights=True to instead score each
    track with a small learned network before the same solve; the linear
    solve itself is unchanged and remains fully differentiable with
    respect to that network's parameters, so it can be trained end-to-end
    if you do this.
    """

    def __init__(
        self,
        learn_weights: bool = False,
        weight_hidden_dim: int = 32,
        use_helix: bool = True,
        helix_iterations: int = 100,
        Bz: float = 1.0,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0
    ):
        super().__init__()
        self.learn_weights = learn_weights
        self.use_helix = use_helix
        self.helix_iterations = helix_iterations
        self.Bz = Bz
        self.is_cosmics = is_cosmics
        self.scaling_factor = scaling_factor
        if learn_weights:
            # Input features: class_probs (2) + track_reg_result (4) = 6.
            # Deliberately no PID and no track_position here -- this network
            # scores track QUALITY (how much to trust this track's fitted
            # direction and hit count), which should not depend on where
            # in space the track happens to be.
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

    @staticmethod
    def track_position_from_hits(points, mask_probs, padding_mask):
        """
        For each track slot, the position of its innermost assigned hit --
        the single real hit closest to the origin -- used as its
        reference point in the fit. 

        Args:
            points: (n_events, n_hits, 4) raw hits; spatial position is points[..., 1:4]
            mask_probs: (n_events, n_hits, n_tracks) soft hit-to-track assignment, from MambaAttentionHead
            padding_mask: (n_events, n_hits) True where a hit is real (not padding)

        Returns:
            track_position: (n_events, n_tracks, 3) -- the innermost
                assigned hit's own position; (0, 0, 0) for any track slot
                with no assigned real hits at all (harmless: such a slot
                also has total_hit_weight ~0 and is excluded from the fit)
            total_hit_weight: (n_events, n_tracks) -- unchanged: the soft
                (mask_probs-weighted) hit count per track, still used as
                part of the fit's default track weight (see forward()
                below); no longer used to build track_position itself
            is_assigned: (n_events, n_hits, n_tracks) bool, True where a hit
                is assigned to a track
        """
        hit_position = points[..., 1:4]  # (n_events, n_hits, 3)
        hit_weight = mask_probs * padding_mask.unsqueeze(-1).to(mask_probs.dtype)
        total_hit_weight = hit_weight.sum(dim=1)

        n_events, n_hits, n_tracks = mask_probs.shape
        hit_radius = hit_position.norm(dim=-1)

        # Hard hit-to-track assignment (see docstring above): each hit
        # belongs to whichever track slot its mask_probs is largest for.
        hard_assigned_track = mask_probs.argmax(dim=-1)
        is_assigned = torch.nn.functional.one_hot(hard_assigned_track, num_classes=n_tracks).bool()
        is_assigned = is_assigned & padding_mask.unsqueeze(-1)

        # Within each track's assigned hits, find the smallest hit_radius:
        radius_if_assigned = torch.where(is_assigned
                                       , hit_radius.unsqueeze(-1).expand(-1, -1, n_tracks)
                                       , torch.full((n_events, n_hits, n_tracks), float("inf"), device=points.device, dtype=hit_radius.dtype),) # Note extra ,
        innermost_hit_index = radius_if_assigned.argmin(dim=1)
        has_any_assigned_hit = is_assigned.any(dim=1)

        event_index = torch.arange(n_events, device=points.device).unsqueeze(-1).expand(-1, n_tracks)
        innermost_hit_position = hit_position[event_index, innermost_hit_index]

        track_position = torch.where(has_any_assigned_hit.unsqueeze(-1), innermost_hit_position, torch.zeros_like(innermost_hit_position))
        return track_position, total_hit_weight, is_assigned

    def forward(self, class_probs, track_reg_result, mask_probs, points, padding_mask):
        """
        Args:
            class_probs: (n_events, n_tracks, 2) from MambaAttentionHead
            track_reg_result: (n_events, n_tracks, 4) from MambaAttentionHead --
                still (q/(pT+1), theta, sin(phi), cos(phi)); only used here to
                score track quality (learn_weights=True) and, when use_helix
                is False, to build the straight-line direction as before.
                It is NOT the helix parameterization -- see fitpars.
            mask_probs: (n_events, n_hits, n_tracks) from MambaAttentionHead
            points: (n_events, n_hits, 4) raw hits (same tensor fed to the backbone)
            padding_mask: (n_events, n_hits) True where a hit is real

        Returns:
            {
              "vertex_estimate": (n_events, 3)  fitted primary vertex
                  (NaN in any event where fit_is_valid is False)
              "chi_square": (n_events,)  weighted sum of squared
                  perpendicular residuals at the fitted vertex
              "fit_is_valid": (n_events,) bool
              "track_position": (n_events, n_tracks, 3)   -- for debugging/plotting
              "track_direction": (n_events, n_tracks, 3)
              "track_weight": (n_events, n_tracks)  -- weights actually used in the fit
              "track_closest_approach_point": (n_events, n_tracks, 3) --
                  the point on each track's own line closest to
                  vertex_estimate; plot this against vertex_estimate and
                  track_position to visually check the fit
              "track_dca": (n_events, n_tracks) -- each track's own
                  distance of closest approach to vertex_estimate; a
                  per-track outlier check, unlike chi_square which is
                  summed over the whole event
            }
        """
        track_position, total_hit_weight, is_assigned = self.track_position_from_hits(points, mask_probs, padding_mask)
        track_direction = track_flight_direction(track_reg_result)

        probability_track_is_real = class_probs[..., 1]

        if self.learn_weights:
            track_quality_features = torch.cat([class_probs, track_reg_result], dim=-1)
            learned_weight_scale = self.track_weight_network(track_quality_features).squeeze(-1)  # >= 0
            track_weight = probability_track_is_real * total_hit_weight * learned_weight_scale
        else:
            # Zero-parameter default: trust a track in proportion to how many hits support it
            # and how confident the track finder is that it's real
            track_weight = probability_track_is_real * total_hit_weight

        fit = fit_vertex_by_closest_approach(track_position, track_direction, track_weight)

        if (self.use_helix and self.Bz != 0.0):
            fit = fit_vertex_by_helix_closest_approach(
                points,
                is_assigned,
                track_reg_result,
                track_position,
                self.Bz,
                track_weight,
                n_iterations=self.helix_iterations,
                seed_vertex = fit["vertex_estimate"],
                is_cosmics = self.is_cosmics,
                scaling_factor = self.scaling_factor
            )
        elif self.Bz == 0.0:
            print("WARNING: b_z=0.0, using straight-line fit instead of helix-aware fit.")
       
        return {
            "vertex_estimate": fit["vertex_estimate"],
            "chi_square": fit["chi_square"], 
            "fit_is_valid": fit["fit_is_valid"],
            "track_position": track_position,
            "track_direction": track_direction,
            "track_weight": track_weight,
            "track_closest_approach_point": fit["track_closest_approach_point"],
            "track_dca": fit["track_dca"],
        }