"""
secondaryvertexhead.py
======================
Secondary-vertex finding and fitting, built on top of VertexHead's machinery.
 
Algorithm overview
------------------
1. **Primary vertex** – fitted first with an internal VertexHead instance (or
   supplied externally via *pv_result*).
 
2. **Track selection** – tracks displaced from the PV (DCA > min_track_pv_dca)
   *and* from the beam line (beam_dca > min_track_beam_dca) are candidates for
   SV seeding.  Noise tracks (below the pT threshold) are already excluded by
   the shared quality flag from _track_params_from_hits.
 
3. **All-pairs DOCA seeding** – for every pair of displaced tracks the straight-
   line distance of closest approach (DOCA) and its midpoint are computed.
   Pairs with DOCA < max_pair_doca become seed candidates; the top-K seeds
   (sorted by DOCA) are kept, where K = max_sv_per_event.
 
4. **Per-seed SV fit** – for each of the K seed slots, tracks with a DCA to the
   seed below max_track_sv_doca are collected.  The vertex is fitted with the
   same _fit_vertex_by_closest_approach used for the PV, optionally refined by
   _fit_vertex_by_helix_closest_approach when Bz ≠ 0.
 
5. **Quality filtering** – a fitted SV is accepted only when:
   * the fit converged (fit_is_valid),
   * at least min_sv_tracks tracks contributed,
   * the 3-D displacement from the PV ≥ min_sv_displacement, and
   * χ²/ndf ≤ max_sv_chi2_per_dof.
 
6. **Duplicate suppression** – after fitting, SV slots whose fitted position is
   within seed_merge_distance of a better (smaller χ²/ndf) slot in the same
   event are invalidated.
 
The interface mirrors VertexHead: the same four raw tensors go in; a rich dict
of per-event, per-SV tensors (padded to max_sv_per_event with NaN / False)
comes out.
 
Relationship to vertexhead.py
------------------------------
All geometry helpers (_fit_vertex_by_closest_approach,
_fit_vertex_by_helix_closest_approach, _unnormalize_hits,
_track_position_from_hits, _track_params_from_hits) are imported unchanged
from vertexhead.py.  SecondaryVertexHead adds only:
 
  * _pairwise_doca_and_midpoint  – O(T²) DOCA between every track pair
  * SecondaryVertexHead          – nn.Module orchestrating the SV search
"""
 
from __future__ import annotations
 
from typing import Optional
 
import torch
import torch.nn as nn
 
# All geometry kernels live in vertexhead.py; import them unchanged.
from vertexhead import (
    _fit_vertex_by_closest_approach,
    _fit_vertex_by_helix_closest_approach,
    _unnormalize_hits,
    _track_position_from_hits,
    _track_params_from_hits,
    VertexHead,
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper: all-pairs DOCA
# ─────────────────────────────────────────────────────────────────────────────
 
def _pairwise_doca_and_midpoint(
    track_position: torch.Tensor,
    track_direction: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched all-pairs distance of closest approach (DOCA) for straight tracks.
 
    For two non-parallel straight lines
        r₁(t₁) = P₁ + t₁ d₁,   r₂(t₂) = P₂ + t₂ d₂
    the unique closest-approach parameters satisfy
        t₁ = (δ·d₁ − c δ·d₂) / (1 − c²),
        t₂ = (c δ·d₁ − δ·d₂) / (1 − c²),
    where δ = P₂ − P₁ and c = d₁·d₂.
 
    For nearly parallel pairs (denominator < eps) the two CPAs degenerate and
    the DOCA becomes the perpendicular distance from P₂ to the line through P₁;
    the clamped denominator means these pairs receive a (slightly incorrect but
    finite) DOCA that will typically exceed max_pair_doca and be rejected.
 
    Args:
        track_position:  (n_events, n_tracks, 3)  a point on each track.
        track_direction: (n_events, n_tracks, 3)  unit direction (normalised).
        eps: denominator floor for parallel-track degenerate cases.
 
    Returns:
        doca:      (n_events, n_tracks, n_tracks)    DOCA; diagonal = 0.
        midpoint:  (n_events, n_tracks, n_tracks, 3) midpoint of the CA segment.
        cpa1:      (n_events, n_tracks, n_tracks, 3) closest-approach point on track i.
        cpa2:      (n_events, n_tracks, n_tracks, 3) closest-approach point on track j.
    """
    # Expand to (B, T, T, 3) broadcasting pairs (i, j)
    P1 = track_position.unsqueeze(2)    # (B, T, 1, 3)  — track i reference point
    P2 = track_position.unsqueeze(1)    # (B, 1, T, 3)  — track j reference point
    d1 = track_direction.unsqueeze(2)   # (B, T, 1, 3)
    d2 = track_direction.unsqueeze(1)   # (B, 1, T, 3)
 
    dp    = P2 - P1                           # δ = P₂ − P₁  (B, T, T, 3)
    c     = (d1 * d2).sum(dim=-1)             # d₁·d₂         (B, T, T)
    dp_d1 = (dp * d1).sum(dim=-1)             # δ·d₁           (B, T, T)
    dp_d2 = (dp * d2).sum(dim=-1)             # δ·d₂           (B, T, T)
 
    denom = (1.0 - c ** 2).clamp(min=eps)
    t1    = (dp_d1 - c * dp_d2) / denom       # (B, T, T)
    t2    = (c * dp_d1 - dp_d2) / denom       # (B, T, T)
 
    cpa1     = P1 + t1.unsqueeze(-1) * d1     # closest-approach point on line i  (B, T, T, 3)
    cpa2     = P2 + t2.unsqueeze(-1) * d2     # closest-approach point on line j  (B, T, T, 3)
    midpoint = (cpa1 + cpa2) * 0.5
 
    doca = (cpa2 - cpa1).norm(dim=-1)         # (B, T, T)
    return doca, midpoint, cpa1, cpa2
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SecondaryVertexHead
# ─────────────────────────────────────────────────────────────────────────────
 
class SecondaryVertexHead(nn.Module):
    """
    Secondary-vertex finding from track-finder output.
 
    Uses the same DCA-minimisation machinery as VertexHead but selects
    tracks displaced from the primary vertex and seeds candidates from
    pairs of displaced tracks with small mutual DOCA.
 
    Args:
        Bz:                   Solenoid field along z, Tesla.
        use_helix:            Refine each SV with the helix-aware iterative fit
                              when Bz ≠ 0 (same flag as VertexHead).
        helix_iterations:     Maximum helix-fit iterations per SV candidate.
        is_cosmics:           Fit z against x instead of z against r.
        scaling_factor:       Metre-to-cm conversion (should match VertexHead).
        noise_pt_threshold:   pT (GeV) below which a track is noise-flagged.
        min_track_pv_dca:     cm – minimum DCA from a track to the PV for the
                              track to be considered an SV candidate.
        min_track_beam_dca:   cm – minimum distance from the beam line. Removes
                              beam-constrained primary tracks that pass the PV-
                              DCA cut in events with a poorly reconstructed PV.
        max_pair_doca:        cm – maximum straight-line DOCA between two seed
                              tracks for the pair to generate a seed.
        max_track_sv_doca:    cm – a track is attached to a seed only when its
                              DCA to the seed position is below this threshold.
        seed_merge_distance:  cm – after fitting, SV slots whose position is
                              within this distance of a better slot (lower
                              χ²/ndf) in the same event are invalidated.
        min_sv_tracks:        Minimum number of tracks at a valid SV.
        min_sv_displacement:  cm – minimum 3-D displacement of the SV from the PV.
        max_sv_chi2_per_dof:  Upper cut on χ²/ndf (ndf = 2·n_tracks − 3).
        max_sv_per_event:     Maximum SV candidates retained per event (padded
                              with NaN / False where fewer are found).
        **pv_kwargs:          Extra keyword arguments forwarded to the internal
                              VertexHead instance.
    """
 
    def __init__(
        self,
        Bz: float = 1.4,
        use_helix: bool = True,
        helix_iterations: int = 100,
        is_cosmics: bool = False,
        scaling_factor: float = 100.0,
        noise_pt_threshold: float = 0.25,
        min_track_pv_dca: float = 0.05,
        min_track_beam_dca: float = 0.05,
        max_pair_doca: float = 1.0,
        max_track_sv_doca: float = 0.5,
        seed_merge_distance: float = 0.1,
        min_sv_tracks: int = 2,
        min_sv_displacement: float = 0.02,
        max_sv_chi2_per_dof: float = 10.0,
        max_sv_per_event: int = 6,
        **pv_kwargs,
    ):
        super().__init__()
 
        self.Bz                  = Bz
        self.use_helix           = use_helix
        self.helix_iterations    = helix_iterations
        self.is_cosmics          = is_cosmics
        self.scaling_factor      = scaling_factor
        self.noise_pt_threshold  = noise_pt_threshold
        self.min_track_pv_dca    = min_track_pv_dca
        self.min_track_beam_dca  = min_track_beam_dca
        self.max_pair_doca       = max_pair_doca
        self.max_track_sv_doca   = max_track_sv_doca
        self.seed_merge_distance = seed_merge_distance
        self.min_sv_tracks       = min_sv_tracks
        self.min_sv_displacement = min_sv_displacement
        self.max_sv_chi2_per_dof = max_sv_chi2_per_dof
        self.max_sv_per_event    = max_sv_per_event
 
        # Internal PV fitter – same physics settings, no learned weights
        self.pv_head = VertexHead(
            learn_weights=False,
            use_helix=use_helix,
            helix_iterations=helix_iterations,
            Bz=Bz,
            is_cosmics=is_cosmics,
            scaling_factor=scaling_factor,
            noise_pt_threshold=noise_pt_threshold,
            **pv_kwargs,
        )
 
    # ── internal helpers ──────────────────────────────────────────────────────
 
    def _seed_sv_candidates(
        self,
        track_position: torch.Tensor,
        track_direction: torch.Tensor,
        displaced_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate up to max_sv_per_event seed positions from pairs of displaced
        tracks with DOCA < max_pair_doca.
 
        Seeds are sorted by ascending DOCA so the tightest (most likely real)
        vertex candidates come first.
 
        Args:
            track_position:  (n_events, n_tracks, 3)
            track_direction: (n_events, n_tracks, 3)
            displaced_mask:  (n_events, n_tracks)  True = track is displaced & usable
 
        Returns:
            seeds:      (n_events, K, 3)  seed positions; NaN where the slot is empty.
            seed_valid: (n_events, K)     bool, True = seed slot is occupied.
        """
        n_events, n_tracks, _ = track_position.shape
        device = track_position.device
        K      = self.max_sv_per_event
 
        # All-pairs straight-line DOCA and midpoints: (B, T, T) and (B, T, T, 3)
        doca, midpoint, _, _ = _pairwise_doca_and_midpoint(track_position, track_direction)
 
        # Valid pair mask: upper-triangle (i < j), both tracks displaced, DOCA within cut
        both_displaced = displaced_mask.unsqueeze(2) & displaced_mask.unsqueeze(1)  # (B, T, T)
        upper_tri = torch.triu(
            torch.ones(n_tracks, n_tracks, device=device, dtype=torch.bool), diagonal=1
        )
        pair_valid = both_displaced & upper_tri.unsqueeze(0) & (doca < self.max_pair_doca)
 
        # Flatten the (T, T) grid for easy sorting
        flat_doca  = doca.view(n_events, -1)           # (B, T²)
        flat_mid   = midpoint.view(n_events, -1, 3)    # (B, T², 3)
        flat_valid = pair_valid.view(n_events, -1)      # (B, T²)
 
        # Push invalid pairs to the back by giving them infinite DOCA
        sortable = flat_doca.masked_fill(~flat_valid, float("inf"))
        top_idx  = sortable.argsort(dim=-1)[:, :K]     # (B, K) indices of the K best pairs
 
        seeds      = flat_mid.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, K, 3)
        seed_valid = flat_valid.gather(1, top_idx)                                  # (B, K)
 
        # NaN-fill unused slots so callers can distinguish them easily
        seeds = seeds.masked_fill(~seed_valid.unsqueeze(-1), float("nan"))
        return seeds, seed_valid
 
    def _track_dca_to_point(
        self,
        point: torch.Tensor,
        track_position: torch.Tensor,
        track_direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Straight-line DCA of each track to an arbitrary 3-D point, per event.
 
        Args:
            point:           (n_events, 3)
            track_position:  (n_events, n_tracks, 3)
            track_direction: (n_events, n_tracks, 3)
 
        Returns:
            dca: (n_events, n_tracks)
        """
        # Vector from the track reference point to the query point
        diff  = point.unsqueeze(1) - track_position               # (B, T, 3)
        # Remove the component along the track direction
        along = (diff * track_direction).sum(-1, keepdim=True) * track_direction
        perp  = diff - along
        return perp.norm(dim=-1)                                    # (B, T)
 
    def _fit_one_sv(
        self,
        seed_pos: torch.Tensor,
        track_position: torch.Tensor,
        track_direction: torch.Tensor,
        base_weight: torch.Tensor,
        track_params: dict,
    ) -> dict:
        """
        Fit a single SV candidate slot (one position per event in the batch).
 
        Only tracks whose straight-line DCA to *seed_pos* is below
        max_track_sv_doca receive non-zero weight; the underlying
        _fit_vertex_by_closest_approach (and optional helix refinement) then
        finds the optimal vertex position for those tracks, exactly as the PV
        fitter does.
 
        Args:
            seed_pos:        (n_events, 3)  initial seed (NaN → use origin as seed).
            track_position:  (n_events, n_tracks, 3)
            track_direction: (n_events, n_tracks, 3)
            base_weight:     (n_events, n_tracks)  displaced & quality weight; 0 = excluded.
            track_params:    output dict of _track_params_from_hits (for helix fit).
 
        Returns:
            dict with all keys of _fit_vertex_by_closest_approach, plus:
                n_tracks_used: (n_events,)           long – tracks with weight > 0.
                chi2_per_dof:  (n_events,)           NaN where fit is invalid.
                sv_track_mask: (n_events, n_tracks)  bool – tracks that contributed.
        """
        seed_safe = seed_pos.nan_to_num()
 
        # Gate: include only tracks close enough to the seed
        dca_to_seed  = self._track_dca_to_point(seed_safe, track_position, track_direction)
        sv_track_mask = (dca_to_seed < self.max_track_sv_doca) & (base_weight > 0)
        sv_weight     = base_weight * sv_track_mask.to(base_weight.dtype)
 
        fit = _fit_vertex_by_closest_approach(track_position, track_direction, sv_weight)
 
        if self.use_helix and self.Bz != 0.0:
            fit = _fit_vertex_by_helix_closest_approach(
                track_params["circle_centre"],
                track_params["R_s"],
                track_params["zslope"],
                track_params["z0"],
                sv_weight,
                n_iterations=self.helix_iterations,
                linear_fit=fit,
                is_cosmics=self.is_cosmics,
            )
        elif self.Bz == 0.0:
            # Mirror the warning from VertexHead
            print("WARNING [SecondaryVertexHead]: Bz=0 – using straight-line SV fit.")
 
        n_used = sv_weight.gt(0).sum(dim=-1)                                     # (B,)  long
        # ndf = 2 * n_tracks − 3  (5 perigee params per track, 3 vertex coords, 3 momenta per track)
        ndf    = (2 * n_used.clamp(min=1) - 3).clamp(min=1).to(fit["chi_square"].dtype)
        chi2pd = fit["chi_square"] / ndf
        chi2pd = chi2pd.masked_fill(~fit["fit_is_valid"], float("nan"))
 
        return {
            **fit,
            "n_tracks_used": n_used,
            "chi2_per_dof":  chi2pd,
            "sv_track_mask": sv_track_mask,
        }
 
    def _suppress_duplicates(
        self,
        sv_estimates: torch.Tensor,
        sv_chi2pd:    torch.Tensor,
        sv_valid:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Invalidate SV slots whose fitted position is within seed_merge_distance
        of a better (lower χ²/ndf) slot in the same event.
 
        Args:
            sv_estimates: (n_events, K, 3)  NaN where not valid.
            sv_chi2pd:    (n_events, K)     NaN where not valid.
            sv_valid:     (n_events, K)     bool.
 
        Returns:
            keep: (n_events, K) bool – True = slot survives duplicate suppression.
        """
        K      = sv_estimates.shape[1]
        device = sv_estimates.device
 
        # Pairwise distances between all SV slots
        pos1 = sv_estimates.unsqueeze(2)  # (B, K, 1, 3)
        pos2 = sv_estimates.unsqueeze(1)  # (B, 1, K, 3)
        dist = (pos1 - pos2).nan_to_num().norm(dim=-1)  # (B, K, K)
 
        # Broadcast χ²/ndf: is slot j strictly better than slot i?
        q1 = sv_chi2pd.unsqueeze(2).nan_to_num(nan=float("inf"))   # (B, K, 1)
        q2 = sv_chi2pd.unsqueeze(1).nan_to_num(nan=float("inf"))   # (B, 1, K)
 
        # Slot i is dominated by slot j if: j is valid, j is closer to i than
        # seed_merge_distance, j has strictly lower χ²/ndf, and i ≠ j.
        eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
        dominated = (
            sv_valid.unsqueeze(1) &          # j is valid
            (dist < self.seed_merge_distance) &
            (q2 < q1) &                      # j is better
            ~eye                             # not the same slot
        ).any(dim=2)                         # (B, K) — dominated by any j
 
        return sv_valid & ~dominated
 
    # ── public forward ────────────────────────────────────────────────────────
 
    def forward(
        self,
        class_probs:   torch.Tensor,
        mask_probs:    torch.Tensor,
        points:        torch.Tensor,
        padding_mask:  torch.Tensor,
        pv_result:     Optional[dict] = None,
    ) -> dict:
        """
        Find and fit secondary vertices.
 
        Inputs are identical to VertexHead.forward.  Optionally supply a
        pre-computed *pv_result* (the dict returned by VertexHead.forward) to
        avoid running the PV fit twice.
 
        Args:
            class_probs:   (n_events, n_tracks, 2)       P(slot empty), P(slot real).
            mask_probs:    (n_events, n_hits, n_tracks)  hit-to-track assignment.
            points:        (n_events, n_hits, 4)         normalised (E, η, φ, r) hits.
            padding_mask:  (n_events, n_hits)            True = real hit.
            pv_result:     optional pre-computed VertexHead output; fitted here if None.
 
        Returns:
            dict — all positions in cm:
 
            sv_estimate      (n_events, K, 3)
                Fitted SV position; NaN where sv_is_valid is False.
 
            sv_is_valid      (n_events, K)  bool
                True = SV passed all quality cuts and survived duplicate suppression.
 
            sv_chi2          (n_events, K)
                Weighted sum of squared DCAs at the fitted SV (no ndf correction).
 
            sv_chi2_per_dof  (n_events, K)
                chi² / (2 n_tracks − 3); NaN where sv_is_valid is False.
 
            sv_displacement  (n_events, K)
                3-D distance from the fitted SV to the primary vertex.
 
            sv_n_tracks      (n_events, K)  long
                Number of tracks that contributed to the SV fit.
 
            sv_track_mask    (n_events, K, n_tracks)  bool
                True where a track contributed to the SV in slot k.
 
            sv_track_dca     (n_events, K, n_tracks)
                Per-track DCA to the fitted SV; NaN where sv_is_valid is False.
 
            pv_result        dict
                The primary vertex result (either supplied or fitted internally).
 
            track_position   (n_events, n_tracks, 3)
                Innermost assigned hit of each track (same as VertexHead output).
 
            track_direction  (n_events, n_tracks, 3)
                Straight-line flight direction (same as VertexHead output).
 
            displaced_mask   (n_events, n_tracks)  bool
                Tracks used for SV seeding (displaced & usable & above pT cut).
 
            displaced_weight (n_events, n_tracks)
                Weight actually used for the SV fits (0 where displaced_mask is False).
        """
        n_events, _, n_tracks = mask_probs.shape
        device = points.device
        K      = self.max_sv_per_event
 
        # ── Step 1: shared track extraction ───────────────────────────────────
        # Replicates the first block of VertexHead.forward so that track_params
        # (including circle_centre, R_s, etc.) are available for the helix fit.
        pts, hit_r = _unnormalize_hits(
            points.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        )
        track_position, total_hit_weight, is_assigned = _track_position_from_hits(
            pts, hit_r, mask_probs, padding_mask
        )
        track_params = _track_params_from_hits(
            pts, hit_r, is_assigned,
            self.noise_pt_threshold,
            Bz=self.Bz,
            scaling_factor=self.scaling_factor,
            is_cosmics=self.is_cosmics,
        )
 
        # Reconstruct track direction from fitted circle parameters
        track_reg = track_params["track_reg_result"]
        sin_t     = torch.sin(track_reg[..., 1])
        cos_t     = torch.cos(track_reg[..., 1])
        track_direction = torch.stack(
            [sin_t * track_reg[..., 3],   # sin(θ) cos(φ)
             sin_t * track_reg[..., 2],   # sin(θ) sin(φ)
             cos_t],                       # cos(θ)
            dim=-1,
        )
        track_direction = (
            track_direction
            / track_direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        )
 
        p_real     = class_probs[..., 1]
        base_weight = p_real * total_hit_weight * track_params["usable"].to(p_real.dtype)
 
        # ── Step 2: primary vertex ─────────────────────────────────────────────
        if pv_result is None:
            pv_result = self.pv_head(class_probs, mask_probs, points, padding_mask)
 
        pv_pos      = pv_result["vertex_estimate"]   # (B, 3); NaN where PV invalid
        pv_pos_safe = pv_pos.nan_to_num()
 
        # ── Step 3: displaced-track selection ─────────────────────────────────
        # Prefer the per-track DCA already computed by the PV fitter (more
        # accurate than a fresh straight-line estimate) when it is available.
        if "track_dca" in pv_result:
            pv_track_dca = pv_result["track_dca"].nan_to_num()     # (B, T)
        else:
            pv_track_dca = self._track_dca_to_point(
                pv_pos_safe, track_position, track_direction
            )
 
        displaced_mask = (
            (pv_track_dca > self.min_track_pv_dca) &
            (track_params["beam_dca"].nan_to_num() > self.min_track_beam_dca) &
            track_params["usable"]
        )
        displaced_weight = base_weight * displaced_mask.to(base_weight.dtype)
 
        # ── Step 4: all-pairs DOCA seeding ────────────────────────────────────
        seeds, seed_valid = self._seed_sv_candidates(
            track_position, track_direction, displaced_mask
        )
        # seeds: (B, K, 3),  seed_valid: (B, K)
 
        # ── Step 5: fit each candidate SV slot ────────────────────────────────
        dtype = pts.dtype
        sv_estimates  = torch.full((n_events, K, 3),          float("nan"), device=device, dtype=dtype)
        sv_chi2       = torch.full((n_events, K),             float("nan"), device=device, dtype=dtype)
        sv_chi2pd     = torch.full((n_events, K),             float("nan"), device=device, dtype=dtype)
        sv_valid      = torch.zeros((n_events, K),            device=device, dtype=torch.bool)
        sv_n_tracks   = torch.zeros((n_events, K),            device=device, dtype=torch.long)
        sv_displace   = torch.full((n_events, K),             float("nan"), device=device, dtype=dtype)
        sv_track_mask = torch.zeros((n_events, K, n_tracks),  device=device, dtype=torch.bool)
        sv_track_dca  = torch.full((n_events, K, n_tracks),   float("nan"), device=device, dtype=dtype)
 
        for k in range(K):
            seed_k = seeds[:, k, :]          # (B, 3); NaN where slot is empty
 
            fit_k = self._fit_one_sv(
                seed_k, track_position, track_direction,
                displaced_weight, track_params,
            )
 
            pos_k    = fit_k["vertex_estimate"]                            # (B, 3)
            disp_k   = (pos_k - pv_pos_safe).norm(dim=-1)                 # (B,)
            n_used_k = fit_k["n_tracks_used"]                             # (B,) long
            c2pd_k   = fit_k["chi2_per_dof"]                             # (B,)
 
            is_ok = (
                seed_valid[:, k] &
                fit_k["fit_is_valid"] &
                (n_used_k >= self.min_sv_tracks) &
                (disp_k   >= self.min_sv_displacement) &
                (c2pd_k.nan_to_num(nan=float("inf")) <= self.max_sv_chi2_per_dof)
            )
 
            sv_estimates[:, k]    = pos_k.masked_fill(~is_ok.unsqueeze(-1), float("nan"))
            sv_chi2[:, k]         = fit_k["chi_square"].masked_fill(~is_ok, float("nan"))
            sv_chi2pd[:, k]       = c2pd_k.masked_fill(~is_ok, float("nan"))
            sv_valid[:, k]        = is_ok
            sv_n_tracks[:, k]     = n_used_k.masked_fill(~is_ok, 0)
            sv_displace[:, k]     = disp_k.masked_fill(~is_ok, float("nan"))
            sv_track_mask[:, k]   = fit_k["sv_track_mask"] & is_ok.unsqueeze(-1)
            sv_track_dca[:, k]    = fit_k["track_dca"].masked_fill(~is_ok.unsqueeze(-1), float("nan"))
 
        # ── Step 6: duplicate suppression ─────────────────────────────────────
        keep      = self._suppress_duplicates(sv_estimates, sv_chi2pd, sv_valid)
        sv_valid  = sv_valid  & keep
        sv_estimates  = sv_estimates.masked_fill(~sv_valid.unsqueeze(-1), float("nan"))
        sv_chi2       = sv_chi2.masked_fill(~sv_valid, float("nan"))
        sv_chi2pd     = sv_chi2pd.masked_fill(~sv_valid, float("nan"))
        sv_n_tracks   = sv_n_tracks.masked_fill(~sv_valid, 0)
        sv_displace   = sv_displace.masked_fill(~sv_valid, float("nan"))
        sv_track_mask = sv_track_mask & sv_valid.unsqueeze(-1)
        sv_track_dca  = sv_track_dca.masked_fill(~sv_valid.unsqueeze(-1), float("nan"))
 
        return {
            "sv_estimate":       sv_estimates,    # (B, K, 3)
            "sv_is_valid":       sv_valid,         # (B, K)  bool
            "sv_chi2":           sv_chi2,          # (B, K)
            "sv_chi2_per_dof":   sv_chi2pd,        # (B, K)
            "sv_displacement":   sv_displace,      # (B, K)
            "sv_n_tracks":       sv_n_tracks,      # (B, K)  long
            "sv_track_mask":     sv_track_mask,    # (B, K, T)  bool
            "sv_track_dca":      sv_track_dca,     # (B, K, T)
            "pv_result":         pv_result,
            "track_position":    track_position,   # (B, T, 3)
            "track_direction":   track_direction,  # (B, T, 3)
            "track_params":      track_params,     # dict from _track_params_from_hits
            "displaced_mask":    displaced_mask,   # (B, T)  bool
            "displaced_weight":  displaced_weight, # (B, T)
        }