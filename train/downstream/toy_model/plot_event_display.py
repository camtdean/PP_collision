#!/usr/bin/env python3
"""
plot_vertex.py

Figure — Event display
    • Uses the EXACT event saved by test_vertex_simple.py in vertex_test_inputs.pt.
    • Reconstructed vertex is obtained by running VertexHead on the SAVED inputs.

Figure does NOT regenerate the event. It loads the tensors produced by
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
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from train.downstream.vertex_head import VertexHead
    print(f"Loaded VertexHead from: {os.path.join(_REPO_ROOT, 'train', 'downstream', 'vertex_head.py')}\n")
except ImportError as e:
    sys.exit(f"ImportError: {e}\n\nFix:\n  cd /Users/nieto/Storage/PP_collision\n  git checkout train/downstream/vertex_head.py")
except SyntaxError as e:
    sys.exit(
        f"SyntaxError in vertex_head.py at line {e.lineno}: {e.msg}\n\n"
        "Fix:\n  cd /Users/nieto/Storage/PP_collision\n  git checkout train/downstream/vertex_head.py"
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

parser.add_argument(
    "--input",
    type=str,
    default="vertex_test_inputs.pt",
    help="Saved vertex test event to display",
)

args = parser.parse_args()

USE_HELIX = args.use_helix

print(f"USE_HELIX = {USE_HELIX}")

B_FIELD_Z = 1.4  # Tesla
HELIX_CONST_CM = 100.0 / 0.3

R_TPC_INNER = 30.0
R_TPC_OUTER = 76.0

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
    R_inner=R_TPC_INNER,
    R_outer=R_TPC_OUTER,
    n_points=300,
):
    """
    Generate the helical trajectory using the SAME parameterization
    as vertex_head.py and test_vertex_simple.py.

    phi is the momentum azimuth at the PRIMARY VERTEX.

    The trajectory is generated from the signed curvature radius:

        R_s = pT / (0.3 * q * Bz)

    and the circle-center convention used by VertexHead:

        x0 = x + R_s sin(phi)
        y0 = y - R_s cos(phi)
    """

    vx, vy, vz = np.asarray(vertex, dtype=float)

    q = float(charge)
    pT = float(pT)
    theta = float(theta)
    phi = float(phi)

    rho = helix_radius_cm(pT, B_z, abs(q),)
    radii = np.linspace(0, R_outer, n_points,)
    ratio = np.clip(radii / (2.0 * rho), 0.0, 1.0,)

    alpha_abs = 2.0 * np.arcsin(ratio)
    alpha = (-q * np.sign(B_z) * alpha_abs)

    R_s = rho / q
    
    # Circle center corresponding to the momentum direction
    # phi at the vertex.
    x0 = vx + R_s * np.sin(phi)
    y0 = vy - R_s * np.cos(phi)

    # Position along the helix.
    x = x0 - R_s * np.sin(phi + alpha)
    y = y0 + R_s * np.cos(phi + alpha)

    # Transverse arc length.
    s_xy = rho * np.abs(alpha)

    # Convert transverse arc length to 3-D path length.
    #
    # ds_xy = ds * sin(theta)
    # therefore
    # ds = ds_xy / sin(theta)
    sin_theta = np.sin(theta)

    if abs(sin_theta) < 1e-8:
        s_3d = np.zeros_like(s_xy)
    else:
        s_3d = s_xy / abs(sin_theta)

    # z propagation.
    z = vz + s_3d * np.cos(theta)

    return np.column_stack([x, y, z,])

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
        # Truth kinematics at the PRIMARY VERTEX
        "theta_vertex": saved["theta_rad"],
        "phi_vertex": saved["phi_rad"],
    }

    return event, saved["true_vertex"], saved

def load_saved_event(path):
    """
    Load one event produced by test_vertex_simple.py.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    saved = torch.load(path, map_location="cpu")

    event = {
        "class_probs": saved["class_probs"],
        "track_reg_result": saved["track_reg_result"],
        "mask_probs": saved["mask_probs"],
        "points": saved["points"],
        "padding_mask": saved["padding_mask"],
    }

    true_vertex = saved["true_vertex"]

    return event, true_vertex


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
    charge = np.sign(q_over_pt_plus_one)
    pT = 1.0 / np.abs(q_over_pt_plus_one) - 1.0

    phi_vertex = event["phi_vertex"].numpy()
    theta_vertex = event["theta_vertex"].numpy()
    
    if USE_HELIX:
        phi = phi_vertex
        theta = theta_vertex
    else:
        phi = phi_vertex
        theta = theta_vertex
    
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
    pts_r = np.sqrt(pts_xyz[:, 0]**2 + pts_xyz[:, 1]**2)
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
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_rz = fig.add_subplot(2, 2, 4)

    # ── helper: draw one truth track in any projection ──────────────────────
    def draw_track(ax_2d, t, xi, yi, use_r=False, line_len=30.0):

        mask = hit_track == t

        if use_r:
            x_data = pts_xyz[mask, 2]   # Z
            y_data = pts_r[mask]        # R
        else:
            x_data = pts_xyz[mask, xi]
            y_data = pts_xyz[mask, yi]

        ax_2d.scatter(
            x_data,
            y_data,
            color=track_colors[t],
            marker=track_markers[t % len(track_markers)],
            s=9,
            alpha=0.60,
            linewidths=0,
            zorder=2,
        )

        if USE_HELIX:

            helix_xyz = helix_points(
                vertex=vtx_true_np,
                pT=pT[t],
                theta=theta[t],
                phi=phi[t],
                charge=charge[t],
                B_z=B_FIELD_Z,
                R_inner=R_TPC_INNER,
                R_outer=R_TPC_OUTER,
                n_points=300,
            )

            if use_r:
                helix_r = np.sqrt(helix_xyz[:, 0]**2 + helix_xyz[:, 1]**2)

                ax_2d.plot(
                    helix_xyz[:, 2],   # Z
                    helix_r,           # R
                    color=track_colors[t],
                    lw=1.2,
                    alpha=0.55,
                    zorder=3,
                )
            else:
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

            if use_r:
                trajectory_r = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)

                ax_2d.plot(
                    trajectory[:, 2],   # Z
                    trajectory_r,       # R
                    color=track_colors[t],
                    lw=1.2,
                    alpha=0.55,
                    zorder=3,
                )
            else:
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
                R_inner=R_TPC_INNER,
                R_outer=R_TPC_OUTER,
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
        (ax_xy, 0, 1, "x (cm)", "y (cm)", "Transverse  x-y", False),
        (ax_yz, 2, 1, "z (cm)", "y (cm)", "y-z plane", False),
        (ax_rz, 2, 0, "z (cm)", "R (cm)", "R-z plane", True),
    ]

    for ax2, xi, yi, xl, yl, title, use_r in projections:
        for t in range(n_tracks):
            draw_track(ax2, t, xi, yi, use_r=use_r)
            
        if use_r:
            # Z-R projection
            true_x = vtx_true_np[2]  # Z
            true_y = np.sqrt(vtx_true_np[0]**2 + vtx_true_np[1]**2)

            ax2.scatter(
                true_x,
                true_y,
                marker="*",
                s=180,
                color="black",
                edgecolor="white",
                linewidth=1.0,
                label="True PV",
                zorder=10,
            )

            if is_valid:
                reco_x = vtx_reco_np[2]  # Z
                reco_y = np.sqrt(vtx_reco_np[0]**2 + vtx_reco_np[1]**2)

                ax2.scatter(
                    reco_x,
                    reco_y,
                    marker="X",
                    s=120,
                    color="red",
                    edgecolor="white",
                    linewidth=1.0,
                    label="Reco PV",
                    zorder=10,
                )

        else:
            # X-Y or Z-Y projection
            true_x = vtx_true_np[xi]
            true_y = vtx_true_np[yi]

            ax2.scatter(
                true_x,
                true_y,
                marker="*",
                s=180,
                color="black",
                edgecolor="white",
                linewidth=1.0,
                label="True PV",
                zorder=10,
            )

            if is_valid:
                reco_x = vtx_reco_np[xi]
                reco_y = vtx_reco_np[yi]

                ax2.scatter(
                    reco_x,
                    reco_y,
                    marker="X",
                    s=120,
                    color="red",
                    edgecolor="white",
                    linewidth=1.0,
                    label="Reco PV",
                    zorder=10,
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
MEAN_N_TRACKS = 10.0
N_HITS_PER_TRACK = 47
N_MULTI_EVENTS = 200

VTX_Z_SIGMA = 3.0
VTX_Z_MIN = -30.0
VTX_Z_MAX = 30.0


def main():
    vertex_head = VertexHead(learn_weights=False, use_helix=USE_HELIX, Bz=B_FIELD_Z,)
    vertex_head.eval()

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1: EXACT event generated by test_vertex_simple.py
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("Figure 1: loading exact event from test_vertex_simple.py")

    input_path = os.path.join(_HERE, args.input)

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


if __name__ == "__main__":
    main()