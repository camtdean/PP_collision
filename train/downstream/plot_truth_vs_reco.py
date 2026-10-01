#!/usr/bin/env python3
"""
plot_vertex.py

Figure — Truth vs reco scatter
    • Residual mean mu and standard deviation sigma (in microns) shown in each title.
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

parser.add_argument(
    "--input",
    type=str,
    default="vertex_test_events/vertex_test_inputs_0000.pt",
    help="Saved vertex test event to display",
)

args = parser.parse_args()

USE_HELIX = args.use_helix

print(f"USE_HELIX = {USE_HELIX}")

B_FIELD_Z = 1.4  # Tesla
HELIX_CONST_CM = 100.0 / 0.3

R_TPC_INNER = 30.0
R_TPC_OUTER = 76.0

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
    r"$r_{xy}$ = 30–76 cm, $\sigma_{\mathrm{hit}}$ = 100 µm)",
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
N_HITS_PER_TRACK = 47
N_MULTI_EVENTS = 200

VTX_Z_SIGMA = 3.0
VTX_Z_MIN = -30.0
VTX_Z_MAX = 30.0


def main():
    vertex_head = VertexHead(learn_weights=False, use_helix=USE_HELIX, b_z=B_FIELD_Z,)
    vertex_head.eval()

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2: 200 events generated by test_vertex_simple.py
    # ══════════════════════════════════════════════════════════════════════════

    print("─" * 60)
    print(
        f"Figure 2: truth-vs-reco over {N_MULTI_EVENTS} "
        f"events from test_vertex_simple.py …"
    )

    vtx_true_list = []
    vtx_reco_list = []

    event_dir = os.path.join(_HERE, "vertex_test_events")

    if not os.path.isdir(event_dir):
        sys.exit(
            f"\nERROR: Could not find event directory:\n"
            f"  {event_dir}\n\n"
            f"Run test_vertex_simple.py 200 times first.\n"
        )

    for ev in range(N_MULTI_EVENTS):

        event_path = os.path.join(
            event_dir,
            f"vertex_test_inputs_{ev:04d}.pt",
        )

        if not os.path.exists(event_path):
            print(f"  Missing event {ev}: {event_path}")
            continue

        try:
            event, vtx_true = load_saved_event(event_path)

        except Exception as e:
            print(f"  Failed to load event {ev}: {e}")
            continue

        with torch.no_grad():

            res = vertex_head(
                class_probs=event["class_probs"],
                track_reg_result=event["track_reg_result"],
                mask_probs=event["mask_probs"],
                points=event["points"],
                padding_mask=event["padding_mask"],
            )

        if res["fit_is_valid"][0].item():

            vtx_true_list.append(vtx_true.numpy())
            vtx_reco_list.append(res["vertex_estimate"][0].numpy())

    n_valid = len(vtx_true_list)

    print(f"  Valid fits: {n_valid} / {N_MULTI_EVENTS}")

    if n_valid > 0:
        save_tvr = os.path.join(_HERE, "truth_vs_reco.png",)
        plot_truth_vs_reco(vtx_true_list, vtx_reco_list, save_tvr,)

    else:
        print("  No valid fits — skipping Figure 2.")

if __name__ == "__main__":
    main()