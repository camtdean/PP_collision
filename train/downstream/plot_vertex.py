#!/usr/bin/env python3
"""
plot_vertex.py

Figure 1 — Event display
    • Uses the EXACT event saved by test_vertex_simple.py in vertex_test_inputs.pt.
    • Reconstructed vertex is obtained by running VertexHead on the SAVED inputs.

Figure 2 — Truth vs reco scatter
    • Residual mean mu and standard deviation sigma (in microns) shown in each title.

Figure 1 does NOT regenerate the event. It loads the tensors produced by
test_vertex_simple.py:

    vertex_test_inputs.pt

Therefore, the Figure 1 hits, track kinematics, and true vertex are exactly
the ones saved by test_vertex_simple.py.
"""

import os
import sys
import math

import numpy as np
import torch

import matplotlib
import matplotlib.cm as cm
matplotlib.use("Agg")  # headless rendering — no display window needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3-D projection
import argparse

# ── import VertexHead from the same directory as this script ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from vertex_head import VertexHead
    print(f"Loaded VertexHead from: {os.path.join(_HERE, 'vertex_head.py')}\n")
except ImportError as e:
    sys.exit(
        f"ImportError: {e}\n\n"
        f"Fix:\n  cd /Users/nieto/Storage/PP_collision\n"
        f"       git checkout train/downstream/vertex_head.py"
    )
except SyntaxError as e:
    sys.exit(
        f"SyntaxError in vertex_head.py at line {e.lineno}: {e.msg}\n\n"
        f"Fix:\n  cd /Users/nieto/Storage/PP_collision\n"
        f"       git checkout train/downstream/vertex_head.py"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PALETTE. Thanks Claude
# ══════════════════════════════════════════════════════════════════════════════

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

TRUE_VTX_COLOR = INK_PRIMARY
RECO_VTX_COLOR = "#d03b3b"

# ══════════════════════════════════════════════════════════════════════════════
#  Helix configuration
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-helix",
    action="store_true",
    help="Generate tracks as helices instead of straight lines"
)

args = parser.parse_args()

USE_HELIX = args.use_helix

print(f"USE_HELIX = {USE_HELIX}")

B_FIELD_Z = 1.4  # Tesla
HELIX_CONST_CM = 100.0 / 0.3

R_TPC_INNER = 30.0
R_TPC_OUTER = 78.0

def helix_radius_cm(pT, B_z=B_FIELD_Z, q_abs=1.0):
    """Transverse radius of curvature [cm]."""
    return pT * HELIX_CONST_CM / (q_abs * B_z)


def helix_points(
    vertex,
    pT,
    theta,
    phi,
    charge,
    B_z=B_FIELD_Z,
    R_max=R_TPC_OUTER,
    n_points=300,
):
    """
    Generate the helical trajectory using the SAME parameterization
    as test_vertex_simple.py.

    Parameters
    ----------
    vertex : array-like
        Starting vertex [x, y, z] in cm.
    pT : float
        Transverse momentum in GeV.
    theta : float
        Polar angle.
    phi : float
        Initial azimuthal angle.
    charge : float
        Particle charge (+1/-1).
    B_z : float
        Magnetic field in Tesla.
    R_max : float
        Maximum transverse radius to display.
    """

    vx, vy, vz = np.asarray(vertex, dtype=float)

    q = float(charge)
    pT = float(pT)
    theta = float(theta)
    phi = float(phi)

    rho = helix_radius_cm(pT, B_z, abs(q),)

    # Same geometry as test_vertex_simple.py:
    #
    # R = 2 rho sin(|alpha|/2)
    #
    ratio = R_max / (2.0 * rho)

    # Do not go beyond the maximum transverse displacement 2*rho.
    ratio = min(ratio, 1.0)

    alpha_abs = 2.0 * np.arcsin(ratio)

    # SAME sign convention as test_vertex_simple.py
    alpha_max = q * np.copysign(alpha_abs, B_z)

    alpha = np.linspace(0.0, alpha_max, n_points,)

    x = (vx + rho * (np.sin(phi + alpha) - np.sin(phi)))
    y = (vy - rho * (np.cos(phi + alpha) - np.cos(phi)))

    # Arc length along the helix
    s = rho * np.abs(alpha)

    # z advances along the trajectory
    z = vz + s * np.cos(theta)

    return np.column_stack([x, y, z])

# ══════════════════════════════════════════════════════════════════════════════
#  Line configuration
# ══════════════════════════════════════════════════════════════════════════════

def linear_track_points(
    vertex,
    theta,
    phi,
    R_max=R_TPC_OUTER,
    n_points=100,
):
    """
    Generate a straight-line trajectory starting at the true vertex.

    The trajectory is extended until approximately R_max in the
    transverse plane.
    """
    vx, vy, vz = np.asarray(vertex, dtype=float)

    theta = float(theta)
    phi = float(phi)

    # Unit direction vector
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    # For a vertex at (0,0), transverse distance is
    #
    #   R = s * sin(theta)
    #
    # so solve for the path length s that reaches R_max.
    sin_theta = np.sin(theta)

    if abs(sin_theta) < 1e-8:
        s_max = R_max
    else:
        s_max = R_max / abs(sin_theta)

    s = np.linspace(0.0, s_max, n_points)

    x = vx + s * dx
    y = vy + s * dy
    z = vz + s * dz

    return np.column_stack([x, y, z])

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC EVENT GENERATOR
# Used ONLY for Figure 2.
#
# Figure 1 does not use this function; it loads the event saved by
# test_vertex_simple.py.
# ══════════════════════════════════════════════════════════════════════════════
def make_event(
    vtx_true,
    n_tracks: int = 5,
    n_hits_per_track: int = 30,
    hit_smearing: float = 0.01,
    tpc_inner_r: float = 30.0,
    tpc_hit_spacing: float = 1.0,
    seed=None,
):
    """
    Build the five tensors VertexHead.forward() expects for a single event.

    This generator is used for the NEW events in Figure 2.

    All tracks originate at vtx_true.
    TPC geometry: hit k is placed where the straight track crosses
    r_xy = tpc_inner_r + k * tpc_hit_spacing.

    Returns:
        class_probs
        track_reg_result
        mask_probs
        points
        padding_mask
        direction
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_hits = n_tracks * n_hits_per_track

    # Random track kinematics
    p_total = torch.FloatTensor(n_tracks).uniform_(0.2, 5.0)
    theta = torch.FloatTensor(n_tracks).uniform_(0.4, math.pi - 0.4)
    phi = torch.FloatTensor(n_tracks).uniform_(0.0, 2.0 * math.pi)
    charge = torch.randint(0, 2, (n_tracks,)).float() * 2 - 1

    pT = p_total * torch.sin(theta)

    # Unit flight directions
    direction = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)

    # ── class_probs  (1, n_tracks, 2) ────────────────────────────────────────
    class_probs = torch.tensor([[[0.05, 0.95]] * n_tracks])

    # ── track_reg_result  (1, n_tracks, 4) ──────────────────────────────────
    # columns: [q/(pT+1), theta, sin(phi), cos(phi)]
    track_reg_result = torch.stack([
        charge / (pT + 1.0),
        theta,
        torch.sin(phi),
        torch.cos(phi),
    ], dim=-1).unsqueeze(0)

    # ── points  (1, n_hits, 4) ──────────────────────────────────────────────
    # columns: [E, x, y, z]
    vx, vy = vtx_true[0].item(), vtx_true[1].item()
    rows = []

    for t in range(n_tracks):
        dx = direction[t, 0].item()
        dy = direction[t, 1].item()

        a = dx * dx + dy * dy
        b = 2.0 * (vx * dx + vy * dy)

        for k in range(n_hits_per_track):
            R_k = tpc_inner_r + k * tpc_hit_spacing
            c_k = vx * vx + vy * vy - R_k * R_k
            disc = b * b - 4.0 * a * c_k
            s_k = (-b + math.sqrt(disc)) / (2.0 * a)

            pos = (
                vtx_true
                + s_k * direction[t]
                + torch.randn(3) * hit_smearing
            )

            rows.append([
                p_total[t].item(),
                pos[0].item(),
                pos[1].item(),
                pos[2].item(),
            ])

    points = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)

    # ── mask_probs  (1, n_hits, n_tracks) ───────────────────────────────────
    mask_probs = torch.full((1, n_hits, n_tracks), 0.02)

    for t in range(n_tracks):
        s = t * n_hits_per_track
        e = (t + 1) * n_hits_per_track
        mask_probs[0, s:e, t] = 0.90

    # ── padding_mask  (1, n_hits) ───────────────────────────────────────────
    padding_mask = torch.ones(1, n_hits, dtype=torch.bool)

    return dict(
        class_probs=class_probs,
        track_reg_result=track_reg_result,
        mask_probs=mask_probs,
        points=points,
        padding_mask=padding_mask,
        direction=direction,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD EXACT EVENT FROM test_vertex_simple.py
# ══════════════════════════════════════════════════════════════════════════════
def load_saved_test_event(path):
    """
    Load the exact event saved by test_vertex_simple.py.

    The saved file contains:
        class_probs
        track_reg_result
        mask_probs
        points
        padding_mask
        true_vertex
        theta_rad
        phi_rad
        charge
        p_total_GeV
        R_TPC_INNER_cm
        TPC_HIT_SPACING_cm

    No hit positions or track kinematics are regenerated here.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find saved test input file:\n  {path}\n\n"
            "Run test_vertex_simple.py first so it creates "
            "vertex_test_inputs.pt."
        )

    saved = torch.load(path, map_location="cpu")

    required = [
        "class_probs",
        "track_reg_result",
        "mask_probs",
        "points",
        "padding_mask",
        "true_vertex",
        "theta_rad",
        "phi_rad",
    ]

    missing = [key for key in required if key not in saved]
    if missing:
        raise KeyError(
            "vertex_test_inputs.pt is missing required entries: "
            + ", ".join(missing)
        )

    # Reconstruct the ORIGINAL truth directions from the saved theta/phi.
    # This does not regenerate the random angles; it uses the exact saved
    # angles from test_vertex_simple.py.
    theta = saved["theta_rad"]
    phi = saved["phi_rad"]

    direction = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)

    event = {
        "class_probs": saved["class_probs"],
        "track_reg_result": saved["track_reg_result"],
        "mask_probs": saved["mask_probs"],
        "points": saved["points"],
        "padding_mask": saved["padding_mask"],
        "direction": direction,
    }

    return event, saved["true_vertex"], saved


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STYLE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply the reference palette to a 2-D matplotlib Axes."""
    ax.set_facecolor(SURFACE)

    for spine in ax.spines.values():
        spine.set_color(AXIS_LINE)
        spine.set_linewidth(0.8)

    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY, pad=6)

    ax.grid(
        True,
        color=GRIDLINE,
        linewidth=0.5,
        linestyle="-",
        alpha=0.9,
        zorder=0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — EVENT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
def plot_event_display(event, result, vtx_true, save_path):
    
    track_reg = event["track_reg_result"][0].numpy()

    # Actual track parameters stored by test_vertex_simple.py:
    q_over_pt_plus_one = track_reg[:, 0]
    theta = track_reg[:, 1]
    sin_phi = track_reg[:, 2]
    cos_phi = track_reg[:, 3]

    # Recover the physical parameters
    charge = np.sign(q_over_pt_plus_one)

    pT = 1.0 / np.abs(q_over_pt_plus_one) - 1.0

    phi = np.arctan2(sin_phi, cos_phi,)
    
    """
    4-panel event display for the EXACT event loaded from
    test_vertex_simple.py.

    The hit coordinates come directly from event["points"].

    The truth track direction lines come directly from event["direction"],
    reconstructed from the theta_rad and phi_rad values saved by
    test_vertex_simple.py.

    The reconstructed vertex comes from the VertexHead result.
    """
    # ── unpack tensors to numpy ─────────────────────────────────────────────
    pts_xyz = event["points"][0, :, 1:4].numpy()
    mprobs = event["mask_probs"][0].numpy()
    n_tracks = mprobs.shape[1]

    # Generate enough colors for however many tracks this event contains
    track_colors = [
        cm.tab20(i % 20)
        for i in range(n_tracks)
    ]

    track_markers = ["o", "s", "^", "D", "P", "v", "<", ">", "*", "X"]

    hit_track = mprobs.argmax(axis=-1)

    vtx_true_np = vtx_true.numpy()
    vtx_reco_np = result["vertex_estimate"][0].numpy()

    # IMPORTANT:
    # Use the ORIGINAL truth direction saved by test_vertex_simple.py,
    # not VertexHead's reconstructed direction.
    track_dir = event["direction"].numpy()
    #track_dir = result["vertex_direction"][0].numpy()

    # VertexHead's fitted reference positions are still useful for displaying
    # where its track references are located.
    track_pos = result["track_position"][0].numpy()

    is_valid = result["fit_is_valid"][0].item()

    # ── layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10), facecolor=SURFACE)

    fig.suptitle(
        "Primary Vertex — Event Display (saved test event)",
        fontsize=14,
        color=INK_PRIMARY,
        y=0.98,
    )

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    # ── helper: draw one truth track in any projection ──────────────────────
    def draw_track(ax_2d, t, xi, yi, line_len=30.0):
        mask = hit_track == t

        ax_2d.scatter(
            pts_xyz[mask, xi],
            pts_xyz[mask, yi],
            color=track_colors[t],
            marker=track_markers[t % len(track_markers)],
            s=9,
            alpha=0.60,
            linewidths=0,
            zorder=2,
        )

        # Start the truth direction line at the SAVED first hit.
        #
        # The first hit is at index t * N_HITS_PER_TRACK because
        # test_vertex_simple.py stores hits track-by-track.
        if USE_HELIX:

            helix_xyz = helix_points(
                vertex=vtx_true_np,
                pT=pT[t],
                theta=theta[t],
                phi=phi[t],
                charge=charge[t],
                B_z=B_FIELD_Z,
                R_max=R_TPC_OUTER,
                n_points=300,
            )

            ax_2d.plot(
                helix_xyz[:, xi],
                helix_xyz[:, yi],
                color=track_colors[t],
                lw=1.2,
                alpha=0.55,
                zorder=3,
            )

        else:

            trajectory = linear_track_points(
                vertex=vtx_true_np,
                theta=theta[t],
                phi=phi[t],
                R_max=R_TPC_OUTER,
                n_points=100,
            )

            ax_2d.plot(
                trajectory[:, xi],
                trajectory[:, yi],
                color=track_colors[t],
                lw=1.2,
                alpha=0.55,
                zorder=3,
            )

    # ── 3-D panel ────────────────────────────────────────────────────────────
    ax3d.set_facecolor(SURFACE)

    LINE_LEN = 30.0

    for t in range(n_tracks):
        mask = hit_track == t

        ax3d.scatter(
            pts_xyz[mask, 0],
            pts_xyz[mask, 1],
            pts_xyz[mask, 2],
            color=track_colors[t],
            marker=track_markers[t % len(track_markers)],
            s=12,
            alpha=0.60,
            linewidths=0,
        )

        # Start at the exact saved innermost hit and use the exact truth
        # direction saved by test_vertex_simple.py.
        n_hits_per_track = pts_xyz.shape[0] // n_tracks
        
        if USE_HELIX:

            helix_xyz = helix_points(
                vertex=vtx_true_np,
                pT=pT[t],
                theta=theta[t],
                phi=phi[t],
                charge=charge[t],
                B_z=B_FIELD_Z,
                R_max=R_TPC_OUTER,
                n_points=300,
            )

            ax3d.plot(
                helix_xyz[:, 0],
                helix_xyz[:, 1],
                helix_xyz[:, 2],
                color=track_colors[t],
                lw=1.2,
                alpha=0.55,
            )

        else:

            linear_xyz = linear_track_points(
                vertex=vtx_true_np,
                theta=theta[t],
                phi=phi[t],
                R_max=R_TPC_OUTER,
                n_points=100,
            )

            ax3d.plot(
                linear_xyz[:, 0],
                linear_xyz[:, 1],
                linear_xyz[:, 2],
                color=track_colors[t],
                lw=1.2,
                alpha=0.55,
            )

    ax3d.scatter(
        *vtx_true_np,
        marker="*",
        s=260,
        c=TRUE_VTX_COLOR,
        zorder=6,
        depthshade=False,
        label=(
            f"True PV: "
            f"({vtx_true[0]:.4f}, {vtx_true[1]:.4f}, {vtx_true[2]:.4f}) cm"
        ),
    )

    if is_valid:
        ax3d.scatter(
            *vtx_reco_np,
            marker="X",
            s=180,
            c=RECO_VTX_COLOR,
            zorder=7,
            depthshade=False,
            label=(
                f"Reco PV: "
                f"({vtx_reco_np[0]:.4f}, {vtx_reco_np[1]:.4f}, {vtx_reco_np[2]:.4f}) cm"
            ),
        )

    ax3d.set_xlabel("x (cm)", fontsize=8, color=INK_SECONDARY, labelpad=2)
    ax3d.set_ylabel("y (cm)", fontsize=8, color=INK_SECONDARY, labelpad=2)
    ax3d.set_zlabel("z (cm)", fontsize=8, color=INK_SECONDARY, labelpad=2)
    ax3d.set_title("3-D view", fontsize=11, color=INK_PRIMARY, pad=4)

    ax3d.legend(
        fontsize=7.5,
        loc="upper left",
        framealpha=0.92,
        markerscale=1.4,
        handlelength=1.6,
    )

    # ── 2-D projections ─────────────────────────────────────────────────────
    projections = [
        (ax_xy, 0, 1, "x (cm)", "y (cm)", "Transverse  x-y"),
        (ax_xz, 0, 2, "x (cm)", "z (cm)", "x-z plane"),
        (ax_yz, 1, 2, "y (cm)", "z (cm)", "y-z plane"),
    ]

    for ax2, xi, yi, xl, yl, title in projections:
        for t in range(n_tracks):
            draw_track(ax2, t, xi, yi)

        ax2.scatter(
            vtx_true_np[xi],
            vtx_true_np[yi],
            marker="*",
            s=220,
            c=TRUE_VTX_COLOR,
            zorder=8,
            linewidths=0.5,
            edgecolors=SURFACE,
        )

        if is_valid:
            ax2.scatter(
                vtx_reco_np[xi],
                vtx_reco_np[yi],
                marker="X",
                s=150,
                c=RECO_VTX_COLOR,
                zorder=9,
                linewidths=0.5,
            )

        _style_ax(ax2, xl, yl, title)

    # ── shared vertex-marker legend ─────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=TRUE_VTX_COLOR, label="True PV  (★)"),
        mpatches.Patch(color=RECO_VTX_COLOR, label="Reco PV  (✕)"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.005),
    )

    # ── residual annotation ─────────────────────────────────────────────────
    if is_valid:
        res_cm = float(np.linalg.norm(vtx_reco_np - vtx_true_np))
        res_um = res_cm * 1e4

        fig.text(
            0.5,
            0.50,
            f"|residual| = {res_um:.0f} µm",
            ha="center",
            va="bottom",
            fontsize=9,
            color=RECO_VTX_COLOR,
            fontstyle="italic",
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print(f"  Saved event display   →  {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — TRUTH VS RECO
# ══════════════════════════════════════════════════════════════════════════════
def plot_truth_vs_reco(vtx_true_list, vtx_reco_list, save_path):
    """
    Three-panel scatter:
        x_reco vs x_true
        y_reco vs y_true
        z_reco vs z_true

    These values come from the independent 200-event Figure 2 study.
    """
    arr_true = np.array(vtx_true_list)
    arr_reco = np.array(vtx_reco_list)
    residuals = arr_reco - arr_true

    N = len(arr_true)
    coord_labels = ["x", "y", "z"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.8),
        facecolor=SURFACE,
    )

    fig.suptitle(
    f"Reco vs True Primary Vertex  "
    f"({N} valid events, Poisson($\\lambda$={MEAN_N_TRACKS:.0f}) tracks, "
    f"{N_HITS_PER_TRACK} hits/track, "
    r"$r_{xy}$ = 30–59 cm, $\sigma_\mathrm{hit}$ = 100 µm)",
    fontsize=12,
    color=INK_PRIMARY,
    y=1.02,
)

    for i, (ax, coord) in enumerate(zip(axes, coord_labels)):
        t_vals = arr_true[:, i]
        r_vals = arr_reco[:, i]

        mu = float(residuals[:, i].mean())
        sigma = float(residuals[:, i].std())

        lo = min(t_vals.min(), r_vals.min()) - 0.15
        hi = max(t_vals.max(), r_vals.max()) + 0.15

        # Identity line
        ax.plot(
            [lo, hi],
            [lo, hi],
            color=AXIS_LINE,
            lw=1.8,
            zorder=1,
        )

        # Scatter
        ax.scatter(
            t_vals,
            r_vals,
            color=cm.tab20(0),
            marker="o",
            s=20,
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )

        title = (
            f"{coord}:   "
            f"$\\mu$ = {mu * 1e4:+.1f} µm,   "
            f"$\\sigma$ = {sigma * 1e4:.1f} µm"
        )

        _style_ax(
            ax,
            xlabel=f"True {coord}  (cm)",
            ylabel=f"Reco {coord}  (cm)",
            title=title,
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(
        save_path,
        dpi=150,
        facecolor=SURFACE,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"  Saved truth-vs-reco   →  {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
MEAN_N_TRACKS = 10.0
N_HITS_PER_TRACK = 30
N_MULTI_EVENTS = 200

VTX_Z_SIGMA = 3.0
VTX_Z_MIN = -30.0
VTX_Z_MAX = 30.0


def main():
    vertex_head = VertexHead(learn_weights=False,  b_z=1.4)
    vertex_head.eval()

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1: EXACT event generated by test_vertex_simple.py
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("Figure 1: loading exact event from test_vertex_simple.py")

    input_path = os.path.join(_HERE, "vertex_test_inputs.pt")

    try:
        event, vtx_true_saved, saved = load_saved_test_event(input_path)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        sys.exit(f"\nERROR loading saved test event:\n{e}\n")

    # Use the saved true vertex, not a newly constructed vertex.
    VTX_TRUE_SINGLE = vtx_true_saved

    print(f"  Loaded saved inputs → {input_path}")
    print(
        f"  Saved true vertex  : "
        f"({VTX_TRUE_SINGLE[0]:.4f}, "
        f"{VTX_TRUE_SINGLE[1]:.4f}, "
        f"{VTX_TRUE_SINGLE[2]:.4f}) cm"
    )
    print(f"  Saved points shape : {tuple(event['points'].shape)}")
    print(f"  Saved tracks       : {event['track_reg_result'].shape[1]}")
    print(f"  Saved hits         : {event['points'].shape[1]}")

    with torch.no_grad():
        result = vertex_head(
            class_probs=event["class_probs"],
            track_reg_result=event["track_reg_result"],
            mask_probs=event["mask_probs"],
            points=event["points"],
            padding_mask=event["padding_mask"],
        )

    is_valid = result["fit_is_valid"][0].item()
    vtx_est = result["vertex_estimate"][0]

    if is_valid:
        res_um = (
            vtx_est - VTX_TRUE_SINGLE
        ).norm().item() * 1e4

        print(
            f"  True  vertex: "
            f"({VTX_TRUE_SINGLE[0]:.4f}, "
            f"{VTX_TRUE_SINGLE[1]:.4f}, "
            f"{VTX_TRUE_SINGLE[2]:.4f}) cm"
        )
        print(
            f"  Reco  vertex: "
            f"({vtx_est[0]:.4f}, "
            f"{vtx_est[1]:.4f}, "
            f"{vtx_est[2]:.4f}) cm"
        )
        print(f"  |residual|  : {res_um:.1f} µm")
    else:
        print("  Fit invalid — not enough independent tracks.")

    save_display = os.path.join(_HERE, "event_display.png")
    plot_event_display(
        event,
        result,
        VTX_TRUE_SINGLE,
        save_display,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2: 200 NEW synthetic events
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print(f"Figure 2: truth-vs-reco over {N_MULTI_EVENTS} NEW events …")

    vtx_true_list = []
    vtx_reco_list = []

    for ev in range(N_MULTI_EVENTS):

        # ─────────────────────────────────────────────
        # Random track multiplicity
        # N_tracks ~ Poisson(10)
        # ─────────────────────────────────────────────
        n_tracks = int(
            torch.poisson(torch.tensor(MEAN_N_TRACKS)).item()
        )

        # Skip zero-track events
        if n_tracks == 0:
            continue

        # ─────────────────────────────────────────────
        # Random primary vertex
        # x = 0
        # y = 0
        # z = Gaussian(0, VTX_Z_SIGMA)
        # restricted to [-30, +30] cm
        # ─────────────────────────────────────────────
        while True:
            vz = torch.randn(1).item() * VTX_Z_SIGMA

            if VTX_Z_MIN <= vz <= VTX_Z_MAX:
                break

        vtx = torch.tensor([
            0.0,
            0.0,
            vz
        ])

        # ─────────────────────────────────────────────
        # Generate event
        # ─────────────────────────────────────────────
        ev_data = make_event(
            vtx,
            n_tracks=n_tracks,
            n_hits_per_track=N_HITS_PER_TRACK,
        )

        # ─────────────────────────────────────────────
        # Run VertexHead
        # ─────────────────────────────────────────────
        with torch.no_grad():
            res = vertex_head(
                class_probs=ev_data["class_probs"],
                track_reg_result=ev_data["track_reg_result"],
                mask_probs=ev_data["mask_probs"],
                points=ev_data["points"],
                padding_mask=ev_data["padding_mask"],
            )

        # ─────────────────────────────────────────────
        # Save valid reconstruction
        # ─────────────────────────────────────────────
        if res["fit_is_valid"][0].item():
            vtx_true_list.append(vtx.numpy())
            vtx_reco_list.append(
                res["vertex_estimate"][0].numpy()
            )

    n_valid = len(vtx_true_list)

    print(
        f"  Valid fits: {n_valid} / {N_MULTI_EVENTS}"
    )

    if n_valid > 0:
        save_tvr = os.path.join(
            _HERE,
            "truth_vs_reco.png",
        )

        plot_truth_vs_reco(
            vtx_true_list,
            vtx_reco_list,
            save_tvr,
        )
    else:
        print(
            "  No valid fits — skipping Figure 2."
        )

    print("─" * 60)
    print("Done.")


if __name__ == "__main__":
    main()