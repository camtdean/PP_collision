#!/usr/bin/env python3
"""
test_event_display.py
=====================
3-D event display for the FM4NPP dataset.

Draws
-----
  • Detector hits   – scatter, coloured by truth track ID
                      (noise hits drawn in grey)
  • Track helices   – parametric 3-D curves from VertexHead track parameters
                      (good/selected tracks in colour, filtered-out tracks in
                       light grey)
  • Truth vertex    – black star  ★
  • Reco vertex     – red cross   ✕

Side panels
-----------
  • Transverse (X–Y) projection
  • Longitudinal (R–Z) projection where R = sqrt(x²+y²)

Usage
-----
    python test_event_display.py \\
        [--data_dir PATH]    # default: ~/Storage/PP_collision/.../combined_0
        [--event_idx N]      # default: 1
        [--bz TESLA]         # default: 1.4
        [--no_mask]          # disable track-quality masking
        [--out FILE]         # save to file instead of showing interactively
        [--dpi N]            # output resolution (default 150)
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # safe default; overridden below if interactive
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for projection='3d')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in [_HERE, os.path.join(_HERE, "train", "downstream")]:
    if os.path.isfile(os.path.join(_candidate, "vertex_head.py")):
        sys.path.insert(0, _candidate)
        break

from vertex_head import VertexHead  # noqa: E402
from downstream_util import (get_trackinfo_noiselabel, get_silicon_tpc_match_mask,)

# ---------------------------------------------------------------------------
# Re-use helpers from test_vertex_head (same directory)
# ---------------------------------------------------------------------------
try:
    from test_vertex_head import (
        RaggedMmap,
        load_event,
        build_truth_tracks,
        DEFAULT_DATA_DIR,
        DEFAULT_EVENT_IDX,
        DEFAULT_BZ,
        NOISE_PT_THRESHOLD,
    )
except ImportError:
    # If the file is not on sys.path, inject it now
    _tvh = os.path.join(_HERE, "test_vertex_head.py")
    if not os.path.isfile(_tvh):
        sys.exit(
            f"ERROR: cannot find test_vertex_head.py beside this file ({_HERE}).\n"
            "Make sure both scripts are in the same directory."
        )
    import importlib.util
    _spec = importlib.util.spec_from_file_location("test_vertex_head", _tvh)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    RaggedMmap        = _mod.RaggedMmap
    load_event        = _mod.load_event
    build_truth_tracks = _mod.build_truth_tracks
    DEFAULT_DATA_DIR  = _mod.DEFAULT_DATA_DIR
    DEFAULT_EVENT_IDX = _mod.DEFAULT_EVENT_IDX
    DEFAULT_BZ        = _mod.DEFAULT_BZ
    NOISE_PT_THRESHOLD = _mod.NOISE_PT_THRESHOLD


# ===========================================================================
# Helix utilities
# ===========================================================================


def helix_points(
    rho, theta, sin_phi, cos_phi,
    vtx, Bz, scaling_factor=100.0,
    n_pts=1200, R_max_cm=100.0,
):
    """Generate a continuous helix from the vertex to R_max_cm."""

    if abs(rho) < 1e-9 or abs(Bz) < 1e-12:
        return None

    # VertexHead parameter convention
    q = np.sign(rho)
    pT = 1.0 / abs(rho) - 1.0

    R_s = pT / (0.3 * q * Bz) * scaling_factor
    radius = abs(R_s)

    if not np.isfinite(radius) or radius < 1e-9:
        return None

    phi0 = float(np.arctan2(sin_phi, cos_phi))
    cot_th = float(np.cos(theta) / (np.sin(theta) + 1e-12))
    xv, yv, zv = map(float, vtx)

    # Same circle-center convention as vertex_head.py
    xc = xv + R_s * np.sin(phi0)
    yc = yv - R_s * np.cos(phi0)

    # Begin exactly at the vertex.
    phi_start = np.arctan2(yv - yc, xv - xc)

    # Search forward for the first crossing of the outer radius.
    # One full turn is sufficient to find any reachable radius.
    alpha = np.linspace(0.0, 2.0 * np.pi, n_pts)
    phi_arc = phi_start - np.sign(R_s) * alpha

    xs = xc + radius * np.cos(phi_arc)
    ys = yc + radius * np.sin(phi_arc)

    # Positive alpha follows the forward transverse tangent.
    # z advances according to the polar angle theta.
    zs = zv + radius * alpha * cot_th

    rr = np.hypot(xs, ys)
    crossings = np.flatnonzero(rr >= R_max_cm)

    if crossings.size:
        # Stop at the first outer-radius crossing.
        end = int(crossings[0])
        if end > 0:
            # Interpolate the crossing for a more accurate endpoint.
            frac = (
                (R_max_cm - rr[end - 1])
                / max(rr[end] - rr[end - 1], 1e-12)
            )
            xs[end] = xs[end - 1] + frac * (xs[end] - xs[end - 1])
            ys[end] = ys[end - 1] + frac * (ys[end] - ys[end - 1])
            zs[end] = zs[end - 1] + frac * (zs[end] - zs[end - 1])
        xs, ys, zs = xs[:end + 1], ys[:end + 1], zs[:end + 1]

    return np.column_stack((xs, ys, zs))


def straight_line_points(theta, sin_phi, cos_phi, vtx,
                         half_len_cm=35.0, n_pts=2):
    """Fallback: straight-line segment from vtx in the track direction."""
    phi0 = float(np.arctan2(sin_phi, cos_phi))
    dx   = float(np.cos(phi0) * np.sin(theta))
    dy   = float(np.sin(phi0) * np.sin(theta))
    dz   = float(np.cos(theta))
    t    = np.linspace(-half_len_cm, half_len_cm, n_pts)
    x0, y0, z0 = float(vtx[0]), float(vtx[1]), float(vtx[2])
    pts = np.stack([x0 + t * dx,
                    y0 + t * dy,
                    z0 + t * dz], axis=-1)
    return pts


def track_helix_or_line(rho, theta, sin_phi, cos_phi, vtx,
                        Bz, scaling_factor=100.0,
                        arc_half_cm=35.0, n_pts=120):
    """Try helix first; fall back to straight line."""
    pts = helix_points(rho, theta, sin_phi, cos_phi, vtx,
                       Bz, scaling_factor, n_pts, arc_half_cm)
    if pts is None:
        pts = straight_line_points(theta, sin_phi, cos_phi, vtx,
                                   half_len_cm=arc_half_cm, n_pts=n_pts)
    return pts


# ===========================================================================
# Colour helpers
# ===========================================================================

def track_colour_map(n_tracks, cmap_name="tab20"):
    """Return a list of n_tracks distinct colours."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N / cmap.N) for i in range(n_tracks)]


# ===========================================================================
# Main display function
# ===========================================================================

def make_event_display(
    features, seg_target, reg_target,
    args
):
    """
    Build truth tracks, run VertexHead, and draw the event display.

    Returns the matplotlib Figure.
    """
    print("Building truth tracks...", flush=True)
    td = build_truth_tracks(features, seg_target, reg_target)
    
    # ------------------------------------------------------------------
    # Silicon/TPC matching mask
    #
    # Silicon: 0 < R < 15 cm
    # TPC:     25 < R < 100 cm
    #
    # Silicon hits belonging to truth tracks with no TPC hit are
    # excluded from the VertexHead fit, but are kept for visualization.
    # ------------------------------------------------------------------
    silicon_info = get_silicon_tpc_match_mask(features, seg_target,)
    hit_keep_mask = silicon_info["keep_mask"]
    silicon_mask = silicon_info["silicon_mask"]
    unmatched_silicon_mask = silicon_info["unmatched_silicon_mask"]
    
    
    print("\n=== Silicon/TPC masking diagnostic ===")
    print(f"Total hits: {len(features)}")
    print(f"Silicon hits: {int(silicon_mask.sum())}")
    print(f"Unmatched Silicon hits: {int(unmatched_silicon_mask.sum())}")
    print(f"Unmatched Silicon hits kept by keep_mask: "
        f"{int((unmatched_silicon_mask & hit_keep_mask).sum())}")

    if not args.no_silicon_tpc_mask:
        td["mask_probs"][0, ~hit_keep_mask, :] = 0.0

        # Verify that masked hits have zero assignment probabilities.
        masked_probs = td["mask_probs"][0, unmatched_silicon_mask, :]
        print(f"Unmatched Silicon hits with nonzero assignment probability: "
            f"{int((masked_probs.abs().sum(dim=-1) > 0).sum().item())}")
    else:
        print("WARNING: Silicon/TPC masking is DISABLED (--no_silicon_tpc_mask)")
    
    
    bad_indices = np.flatnonzero(unmatched_silicon_mask)

    for hit_idx in bad_indices:
        probs = td["mask_probs"][0, hit_idx, :]
        if probs.abs().sum().item() > 0:
            print(
                f"ERROR: hit {hit_idx}, "
                f"track_id={seg_target[hit_idx]}, "
                f"xyz={features[hit_idx, 1:4]}, "
                f"assignment_sum={probs.sum().item():.6g}"
            )


    
    if not args.no_silicon_tpc_mask:
        td["mask_probs"][0, ~hit_keep_mask, :] = 0.0
    else:
        unmatched_silicon_mask = np.zeros(len(features), dtype=bool)


    print(f"  Silicon hits       = {int(silicon_mask.sum())}")
    print(f"  unmatched Silicon  = {int(unmatched_silicon_mask.sum())}")
    print(f"  masked hits        = {int((~hit_keep_mask).sum())}")

    n_tracks = td["track_reg_result"].shape[1]
    track_ids = td["track_ids"]                  # (n_tracks,)  int32
    
    r = np.sqrt(features[:, 1]**2 + features[:, 2]**2)

    tpc_mask = (r > 25.0) & (r < 100.0)

    tpc_track_ids = np.unique(seg_target[tpc_mask])

    track_has_tpc = np.isin(track_ids, tpc_track_ids)

    # ------------------------------------------------------------------
    # Run VertexHead
    # ------------------------------------------------------------------
    print("Running VertexHead...", flush=True)
    head = VertexHead(
        learn_weights=False,
        use_helix=True,
        Bz=args.bz,
        scaling_factor=100.0,
    )
    head.eval()
    
    # Get track parameters + quality labels directly from reg_target
    reg_tensor = torch.from_numpy(reg_target).float()
    info = get_trackinfo_noiselabel(reg_tensor, noise_pt_threshold=NOISE_PT_THRESHOLD,)
    
    # Convert per-hit truth information to per-track information
    track_info = torch.zeros((1, n_tracks, 4), dtype=torch.float32,)
    noise_labels = torch.zeros((1, n_tracks), dtype=torch.long,)
    valid_tracks = torch.zeros((1, n_tracks), dtype=torch.long,)
    
    id_to_col = {int(tid): col for col, tid in enumerate(track_ids)}
    track_seen = np.zeros(n_tracks, dtype=bool)
    
    for hit_idx, tid in enumerate(seg_target):
        col = id_to_col[int(tid)]
        
        if not track_seen[col]:
            track_info[0, col] = info["track_info"][hit_idx]
            noise_labels[0, col] = info["noise_labels"][hit_idx]
            valid_tracks[0, col] = info["valid_tracks"][hit_idx]
            track_seen[col] = True


    with torch.no_grad():
        result = head.forward(
            class_probs      = td["class_probs"],
            track_reg_result = td["track_reg_result"],
            mask_probs       = td["mask_probs"],
            points           = td["points"],
            padding_mask     = td["padding_mask"],
            noise_labels     = None if args.no_mask else noise_labels,
            valid_tracks     = None if args.no_mask else valid_tracks,
            track_info       = None if args.no_mask else track_info,
        )

    fit_valid  = result["fit_is_valid"][0].item()
    reco_vtx   = result["vertex_estimate"][0].numpy()   # (3,)
    qm         = result["track_quality_mask"]           # (1, n_tracks) bool | None

    # Quality mask on tracks
    if qm is not None:
        good_mask_np = qm[0].numpy().astype(bool)       # (n_tracks,)
    else:
        good_mask_np = np.ones(n_tracks, dtype=bool)    # all good

    # ------------------------------------------------------------------
    # Truth plurality vertex
    # ------------------------------------------------------------------
    tvtx_all = td["truth_vtx_per_track"].numpy()        # (n_tracks, 3)
    tvtx_rounded = np.round(tvtx_all, 3)
    unique_vtx, counts = np.unique(tvtx_rounded, axis=0, return_counts=True)
    order = np.argsort(-counts)
    truth_vtx = unique_vtx[order[0]]                    # plurality (proxy PV)

    # Print summary
    print(f"  n_tracks        = {n_tracks}")
    print(f"  good tracks     = {good_mask_np.sum()}")
    print(f"  truth vtx (PV)  = ({truth_vtx[0]:.3f}, {truth_vtx[1]:.3f}, {truth_vtx[2]:.3f}) cm")
    if fit_valid:
        print(f"  reco  vtx       = ({reco_vtx[0]:.3f}, {reco_vtx[1]:.3f}, {reco_vtx[2]:.3f}) cm")
        dist = np.linalg.norm(reco_vtx - truth_vtx)
        print(f"  |reco - truth|  = {dist:.3f} cm")
    else:
        print("  reco  vtx       = DEGENERATE FIT")

    # ------------------------------------------------------------------
    # Hit arrays (N_hits, 3)  --  features columns 1,2,3 are x,y,z
    # ------------------------------------------------------------------
    hits_x = features[:, 1].astype(np.float32)
    hits_y = features[:, 2].astype(np.float32)
    hits_z = features[:, 3].astype(np.float32)
    hits_e = features[:, 0].astype(np.float32)         # energy (col 0) for sizing

    # ------------------------------------------------------------------
    # Map each hit to its truth-track colour
    # ------------------------------------------------------------------
    # Build track_id → column-index map
    id_to_col = {int(tid): col for col, tid in enumerate(track_ids)}

    # Track colours (n_tracks distinct hues)
    colours = track_colour_map(n_tracks)

    hit_colours   = []
    hit_is_good   = []        # True if hit belongs to a quality-selected track
    for tid in seg_target:
        col = id_to_col.get(int(tid), 0)
        good = bool(good_mask_np[col])
        hit_colours.append(colours[col] if good else (0.75, 0.75, 0.75, 0.4))
        hit_is_good.append(good)
        
    hit_colours  = np.array(hit_colours)
    hit_is_good  = np.array(hit_is_good, dtype=bool)
    

    # ------------------------------------------------------------------
    # Track parameters for helix drawing  (n_tracks, 4)
    # ------------------------------------------------------------------
    trk_params = track_info[0].numpy()   # (n_tracks, 4)
    #  columns: rho, theta, sin_phi, cos_phi

    # Vertex to use as helix starting point per track.
    # Use the reconstructed vertex for reco tracks; for truth display we
    # use the per-track truth vertex.
    if fit_valid:
        vtx_for_helix = reco_vtx
    else:
        vtx_for_helix = truth_vtx

    # ------------------------------------------------------------------
    # Layout: 1 large 3-D panel + 2 small 2-D projections
    # ------------------------------------------------------------------
    SURFACE = "#fcfcfb"
    INK_PRIMARY = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    AXIS_LINE = "#c3c2b7"

    TRUE_VTX_COLOR = INK_PRIMARY
    RECO_VTX_COLOR = "#d03b3b"
    def _style_ax(ax, xlabel="", ylabel="", title=""):
        ax.set_facecolor(SURFACE)

        for spine in ax.spines.values():
            spine.set_color(AXIS_LINE)
            spine.set_linewidth(0.8)

        ax.tick_params(
            colors=INK_MUTED,
            labelsize=8,
        )

        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        ax.set_title(
            title,
            fontsize=10,
            color=INK_PRIMARY,
            pad=6,
        )

        ax.grid(
            True,
            color=GRIDLINE,
            linewidth=0.5,
            linestyle="-",
            alpha=0.9,
            zorder=0,
        )
    def _style_ax3d(ax, xlabel="", ylabel="", zlabel="", title=""):
        # Background
        ax.set_facecolor(SURFACE)

        # Axis labels
        ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY, labelpad=4,)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY, labelpad=4,)
        ax.set_zlabel(zlabel, fontsize=9, color=INK_SECONDARY, labelpad=4,)

        # Tick labels
        ax.tick_params(colors=INK_MUTED, labelsize=8,)

        # 3D panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor(AXIS_LINE)
        ax.yaxis.pane.set_edgecolor(AXIS_LINE)
        ax.zaxis.pane.set_edgecolor(AXIS_LINE)

        ax.grid(True, color=GRIDLINE, linewidth=0.5, linestyle="-", alpha=0.9,)

        if title:
            ax.set_title(title, fontsize=10, color=INK_PRIMARY,pad=8,)
    
    fig = plt.figure(figsize=(13, 10), facecolor=SURFACE)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_rz = fig.add_subplot(2, 2, 4)
    
    ax3d.set_xlabel("x [cm]", color="white", labelpad=4)
    ax3d.set_ylabel("y [cm]", color="white", labelpad=4)
    ax3d.set_zlabel("z [cm]", color="white", labelpad=4)

    ax3d.tick_params(colors="white", labelsize=7)

    ax3d.xaxis.label.set_color("white")
    ax3d.yaxis.label.set_color("white")
    ax3d.zaxis.label.set_color("white")
    
    ax3d.set_xlim(-100, 100)   # x [cm]
    ax3d.set_ylim(-100, 100)   # y [cm]
    ax3d.set_zlim(-100, 100)   # z [cm]

    # ------------------------------------------------------------------
    # Helper: shared scatter + track drawing
    # ------------------------------------------------------------------
    _HIT_S = np.clip(hits_e * 12 + 1.5, 1.0, 20.0)   # marker size ~ energy

    def _scatter_hits(ax, xs, ys, is_3d=False, zs=None):
        good = hit_is_good & (~unmatched_silicon_mask)
        bad = (~hit_is_good) & (~unmatched_silicon_mask)
        
        print(
            "Unmatched Silicon hits drawn:",
            int((unmatched_silicon_mask & (good | bad)).sum())
        )

        if is_3d:
            ax.scatter(
                xs[good],
                ys[good],
                zs[good],
                s=_HIT_S[good],
                c=hit_colours[good],
                linewidths=0,
                zorder=2,
                alpha=0.85,
            )
            """
            ax.scatter(
                xs[bad],
                ys[bad],
                zs[bad],
                c="#5a5a5a",
                s=1,
                linewidths=0,
                alpha=0.25,
                zorder=1,
            )
            """

        else:
            ax.scatter(
                xs[good],
                ys[good],
                s=_HIT_S[good],
                c=hit_colours[good],
                linewidths=0,
                zorder=2,
                alpha=0.85,
            )
            """
            ax.scatter(
                xs[bad],
                ys[bad],
                c="#444444",
                s=1,
                linewidths=0,
                alpha=0.3,
                zorder=1,
            )
            """

    def _draw_tracks(ax, is_3d=False):
        """Draw one helix arc per track from the vertex."""
        for t_idx in range(n_tracks):
            
            # Do not draw tracks that have Silicon hits but no TPC hits.
            if not track_has_tpc[t_idx]:
                continue

            rho, theta, sp, cp = trk_params[t_idx]
            good = bool(good_mask_np[t_idx])
            
            """
            print(
                f"track={track_ids[t_idx]}, "
                f"rho={rho:.6g}, "
                f"charge_sign_from_rho={np.sign(rho):+.0f}, "
                f"phi={np.arctan2(sp, cp):.4f}, "
                f"Bz={args.bz}"
            )
            """
            
            # Pick vertex starting point:
            # Use per-track truth vertex so tracks visually originate from
            # their true production point (more informative than a single vtx).
            vtx_t = td["truth_vtx_per_track"][t_idx].numpy()
            
            pts = helix_points(
                rho, theta, sp, cp, vtx_t,
                Bz=args.bz, n_pts=1000,
                R_max_cm=100.0,
            )
            
            # Actual hits assigned to this truth track
            track_hit_mask = (seg_target == track_ids[t_idx])
            track_hit_xyz = features[track_hit_mask, 1:4]

            if len(track_hit_xyz) and len(pts):
                # Distance from each measured hit to its nearest generated curve point
                distances = np.linalg.norm(
                    track_hit_xyz[:, None, :] - pts[None, :, :],
                    axis=2,
                )
                nearest_dist = distances.min(axis=1)
                """
                print(
                    f"track={track_ids[t_idx]:6d} "
                    f"rho={rho:+.4f} "
                    f"median hit-to-curve distance="
                    f"{np.median(nearest_dist):.2f} cm"
                )
                """

            if pts is None:
                pts = straight_line_points(theta, sp, cp, vtx_t, half_len_cm=100.0, n_pts=1000,)
            
            if pts is None:
                continue

            if not good:
                continue

            col = (*colours[t_idx][:3], 0.85)
            lw  = 1.1
            zo  = 5

            if is_3d:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=col, lw=lw, zorder=zo)
            else:
                if ax is ax_xy:
                    # X-Y
                    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, zorder=zo,)
                elif ax is ax_yz:
                    # Y-Z
                    ax.plot(pts[:, 2], pts[:, 1], color=col, lw=lw, zorder=zo,)
                else:
                    # R-Z
                    R_t = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
                    ax.plot(pts[:, 2], R_t, color=col, lw=lw, zorder=zo,)

    # ------------------------------------------------------------------
    # 3-D panel
    # ------------------------------------------------------------------
    _scatter_hits(ax3d, hits_x, hits_y, is_3d=True, zs=hits_z)
    _draw_tracks(ax3d, is_3d=True)

    # Truth vertex
    ax3d.scatter(*truth_vtx, marker="*", s=180, color="black",
                 edgecolor="white", linewidths=1.0, zorder=1000,
                 label=f"Truth vtx  ({truth_vtx[0]:.2f}, {truth_vtx[1]:.2f}, {truth_vtx[2]:.2f}) cm")

    # Reco vertex
    if fit_valid:
        ax3d.scatter(*reco_vtx, marker="X", s=120, color="red",
                     edgecolors="white", linewidths=1.0, zorder=1000,
                     label=f"Reco vtx  ({reco_vtx[0]:.2f}, {reco_vtx[1]:.2f}, {reco_vtx[2]:.2f}) cm")

    _style_ax3d(ax3d, "x (cm)", "y (cm)", "z (cm)", "3D event display",)
    _legend_3d = ax3d.legend(
        loc="upper left",
        fontsize=7.5,
        labelcolor="black",
        markerscale=0.8,
        facecolor="white",
        edgecolor="white",
        framealpha=1.0,
    )
    

    # ------------------------------------------------------------------
    # XY projection
    # ------------------------------------------------------------------
    _scatter_hits(ax_xy, hits_x, hits_y)
    _draw_tracks(ax_xy)

    ax_xy.scatter(truth_vtx[0], truth_vtx[1],
                marker="*", s=180, color="black", edgecolors="white",
                linewidths=1.0, zorder=10, label="Truth vtx")

    if fit_valid:
        ax_xy.scatter(reco_vtx[0], reco_vtx[1],
                    marker="X", s=120, color="red", edgecolors="white",
                    linewidths=1.0, zorder=10, label="Reco vtx")

    _style_ax(ax_xy, "y (cm)", "x (cm)", "x-y plane",)
    ax_xy.set_xlim(-100, 100)   
    ax_xy.set_ylim(-100, 100)
    
    # ── YZ projection ─────────────────────────────────────────────
    _scatter_hits(ax_yz, hits_z, hits_y)
    _draw_tracks(ax_yz)

    ax_yz.scatter(
        truth_vtx[2],
        truth_vtx[1],
        marker="*",
        s=180,
        color="black",
        edgecolor="white",
        linewidth=1.0,
        zorder=10,
        label="True PV",
    )

    if fit_valid:
        ax_yz.scatter(
            reco_vtx[2],
            reco_vtx[1],
            marker="X",
            s=120,
            color="red",
            edgecolor="white",
            linewidth=1.0,
            zorder=10,
            label="Reco PV",
        )
    _style_ax(ax_yz, "z (cm)", "y (cm)", "y-z plane",)
    
    ax_yz.set_xlim(-100, 100)   
    ax_yz.set_ylim(-100, 100)     

    # ------------------------------------------------------------------
    # R-Z projection
    # ------------------------------------------------------------------
    hits_R = np.sqrt(hits_x**2 + hits_y**2)

    _scatter_hits(ax_rz, hits_z, hits_R)
    _draw_tracks(ax_rz)

    truth_R = float(np.sqrt(truth_vtx[0]**2 + truth_vtx[1]**2))
    ax_rz.scatter(truth_vtx[2], truth_R, marker="*", s=180, color="black", edgecolors="white", linewidths=1.0, zorder=10, label="Truth vtx",)

    if fit_valid:
        reco_R = float(np.sqrt(reco_vtx[0]**2 + reco_vtx[1]**2))
        ax_rz.scatter(reco_vtx[2],reco_R, marker="X", s=120, color="red", edgecolor="white", linewidth=1.0, zorder=10, label="Reco PV",)
    _style_ax(ax_rz, "z (cm)", "R (cm)", "R-z plane",)
    
    ax_rz.set_xlim(-100, 100)   # z [cm]
    ax_rz.set_ylim(0, 100)      # R [cm]

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------
    n_good  = int(good_mask_np.sum())
    mask_tag = "no mask" if args.no_mask else f"mask ON → {n_good}/{n_tracks} good"
    dist_tag = ""
    if fit_valid:
        res_cm = float(np.linalg.norm(reco_vtx - truth_vtx))
        res_um = res_cm * 1e4  # cm -> micrometers
        dist_tag = f"  |reco−truth| = {res_cm:.2f} cm"
    title = (
        f"FM4NPP Event Display   event {args.event_idx}   "
        #f"{len(features)} hits  •  {n_tracks} truth tracks"
    )
    fig.suptitle(title, color=INK_PRIMARY, fontsize=11, y=0.97,)
    
    if fit_valid:
        fig.text(
            0.5,
            0.48,
            f"|residual| = {res_um:.0f} µm",
            ha="center",
            va="bottom",
            fontsize=9,
            color=RECO_VTX_COLOR,
            fontstyle="italic",
        )

    return fig


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="FM4NPP 3-D event display")
    parser.add_argument("--data_dir",  default=DEFAULT_DATA_DIR,
                        help="Path to combined_N directory with *.mmap files")
    parser.add_argument("--event_idx", type=int, default=DEFAULT_EVENT_IDX,
                        help="Event index to display (default: 1)")
    parser.add_argument("--bz",        type=float, default=DEFAULT_BZ,
                        help="Solenoid field strength in Tesla (default: 1.4)")
    parser.add_argument("--no_mask",   action="store_true",
                        help="Disable track-quality masking")
    parser.add_argument("--out",       default=None,
                        help="Save figure to this file path (PNG / PDF / SVG). "
                             "If omitted, display interactively.")
    parser.add_argument("--dpi",       type=int, default=150,
                        help="Output DPI (default: 150)")
    parser.add_argument("--no_silicon_tpc_mask",
                        action="store_true",
                        help="Disable masking of Silicon hits whose truth track has no TPC hit",)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  FM4NPP Event Display")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  event_idx : {args.event_idx}")
    print(f"  Bz        : {args.bz} T")
    print(f"  masking   : {'DISABLED' if args.no_mask else 'ENABLED'}")
    print(f"{'='*65}\n")

    # Load event
    print("Loading event from disk...", flush=True)
    features, seg_target, reg_target = load_event(args.data_dir, args.event_idx)
    print(f"  {len(features)} hits loaded")
    
    # Build display
    fig = make_event_display(features, seg_target, reg_target, args)
    
    # Output
    """
    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nFigure saved → {args.out}")
    else:
        # Try interactive show; fall back to saving beside the script
        try:
            matplotlib.use("TkAgg")
            plt.show()
        except Exception:
            default_out = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"event_display_ev{args.event_idx}.png",
            )
            fig.savefig(default_out, dpi=args.dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"\nInteractive display unavailable; figure saved → {default_out}")

    plt.close(fig)
    print("Done.\n")
    """
    
    # Output: save event displays in the event_displays directory
    output_dir = os.path.join(_HERE, "event_displays")
    os.makedirs(output_dir, exist_ok=True)

    if args.out:
        # Use the explicitly supplied output path if --out is provided.
        output_path = args.out
        output_parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_parent, exist_ok=True)
    else:
        # Otherwise, save in event_displays using the event index.
        output_path = os.path.join(output_dir, f"event_display_ev{args.event_idx}.png",)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor(),)
    print(f"\nFigure saved → {output_path}")

    plt.close(fig)
    print("Done.\n")


if __name__ == "__main__":
    main()
