#!/usr/bin/env python3
"""
test_vertex_simple.py
=====================
Minimal standalone test for vertex_head.py.
 
1 event · 5 tracks · 30 hits each (150 total)
Momenta drawn uniformly from 0.2 – 5 GeV.
All tracks originate at the known true primary vertex VTX_TRUE.
 
TPC geometry (straight-line approximation — vertex_head.py uses no B-field):
  The first hit on each track is placed where the straight track crosses
  r_xy = R_TPC_INNER = 30 cm.  Subsequent hits step outward one TPC layer
  at a time (r_xy = 31, 32, … cm).  Each hit is smeared by Gaussian noise
  with std = HIT_SMEARING.  The arc-length s_k along the track to the k-th
  layer is the positive root of the quadratic
 
      (dx² + dy²) s² + 2(Vx·dx + Vy·dy) s + (Vx² + Vy² − R_k²) = 0
 
  where d = (dx, dy, dz) is the unit flight direction and V is the vertex.
  The smallest 3-D radius hit on each track is always the innermost layer
  (k = 0, r_xy ≈ R_TPC_INNER), which is exactly what VertexHead uses as
  its per-track reference point.
 
HOW TO RUN
----------
Place this script in the same directory as vertex_head.py:
    /Users/nieto/Storage/PP_collision/train/downstream/
 
Then run:
    python test_vertex_simple.py
 
If you get an ImportError, first restore vertex_head.py from git:
    cd /Users/nieto/Storage/PP_collision
    git checkout train/downstream/vertex_head.py
    python train/downstream/test_vertex_simple.py
"""
 
import os, sys, math
import torch
import numpy as np
 
# ── import vertex_head from the same directory as this script ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
 
try:
    from vertex_head import VertexHead
    print(f"Loaded VertexHead from: {os.path.join(_HERE, 'vertex_head.py')}\n")
except ImportError as e:
    sys.exit(f"ImportError: {e}\n\nFix:\n  cd /Users/nieto/Storage/PP_collision\n  git checkout train/downstream/vertex_head.py")
except SyntaxError as e:
    sys.exit(
        f"SyntaxError in vertex_head.py at line {e.lineno}: {e.msg}\n\n"
        "Fix:\n  cd /Users/nieto/Storage/PP_collision\n  git checkout train/downstream/vertex_head.py"
    )
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  Simulation parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# torch.manual_seed(0)
 
MEAN_N_TRACKS = 10.0
N_TRACKS = int(torch.poisson(torch.tensor(MEAN_N_TRACKS)).item())

N_HITS_PER_TRACK = 30
N_HITS           = N_TRACKS * N_HITS_PER_TRACK   # 150
N_EVENTS         = 1
 
# True primary vertex (cm) — what VertexHead should recover
VTX_Z_SIGMA = 3.0  # cm

while True:
    vz = torch.randn(1).item() * VTX_Z_SIGMA
    if -30.0 <= vz <= 30.0:
        break

VTX_TRUE = torch.tensor([0.0, 0.0, vz])
 
HIT_SMEARING    = 0.01   # cm  (100 µm detector resolution)
 
# TPC geometry: hits at fixed r_xy layers stepping outward from R_TPC_INNER
R_TPC_INNER     = 30.0   # cm  — innermost TPC layer radius
TPC_HIT_SPACING =  1.0   # cm  — r_xy gap between successive hit layers
# hit k lands at r_xy = R_TPC_INNER + k * TPC_HIT_SPACING
# (k = 0, 1, …, N_HITS_PER_TRACK-1)
# With the defaults this spans 30 – 59 cm.
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  Track kinematics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Thermal-like RHIC pT spectrum ───────────────────────────────────────────
# dN/dpT ∝ pT * exp(-mT/T)
# mT = sqrt(pT^2 + m^2)

PION_MASS = 0.13957   # GeV/c^2
PT_TEMPERATURE = 0.22 # GeV
PT_MIN = 0.0          # GeV/c
PT_MAX = 5.0          # GeV/c


def sample_pt(n):
    """
    Sample n transverse momenta from

        f(pT) ∝ pT * exp(-mT / T)

    using rejection sampling, matching the C++ rhic_pt() code.
    """

    # Find maximum of the spectrum
    pt_grid = torch.linspace(PT_MIN, PT_MAX, 5001)

    mt_grid = torch.sqrt(pt_grid**2 + PION_MASS**2)

    spectrum_grid = (pt_grid * torch.exp(-mt_grid / PT_TEMPERATURE))

    ymax = spectrum_grid.max().item()

    samples = []

    while len(samples) < n:

        # Candidate pT
        pt = torch.empty(1).uniform_(PT_MIN,PT_MAX).item()

        # Random height
        y = torch.empty(1).uniform_(0.0, ymax).item()

        # Spectrum value
        mt = math.sqrt(pt**2 + PION_MASS**2)

        f = (pt * math.exp(-mt / PT_TEMPERATURE))

        # Accept/reject
        if y < f:
            samples.append(pt)

    return torch.tensor(
        samples,
        dtype=torch.float32
    )

pT = sample_pt(N_TRACKS)
theta   = torch.FloatTensor(N_TRACKS).uniform_(0.4, math.pi-0.4)  # polar angle (rad)
phi     = torch.FloatTensor(N_TRACKS).uniform_(0, 2*math.pi)      # azimuthal angle (rad)
charge  = (torch.randint(0, 2, (N_TRACKS,)).float() * 2 - 1)      # ±1
p_total = pT / torch.sin(theta)

direction = torch.stack([               # unit flight direction (N_TRACKS, 3)
    torch.sin(theta) * torch.cos(phi),
    torch.sin(theta) * torch.sin(phi),
    torch.cos(theta),
], dim=-1)
 
print("=" * 66)
print(f"  True primary vertex (cm): ({VTX_TRUE[0]:.3f}, {VTX_TRUE[1]:.3f}, {VTX_TRUE[2]:.3f})")
print(f"  TPC hits: r_xy = {R_TPC_INNER:.0f} … "
      f"{R_TPC_INNER + (N_HITS_PER_TRACK-1)*TPC_HIT_SPACING:.0f} cm "
      f"({N_HITS_PER_TRACK} layers, spacing {TPC_HIT_SPACING:.1f} cm)")
print("=" * 66)
header = (f"  {'Trk':>3}  {'|p| GeV':>8}  {'pT GeV':>7}  "
          f"{'theta':>7}  {'phi':>7}  {'q':>3}  {'s_inner cm':>10}")
print(header)
print("  " + "-" * (len(header) - 2))
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  Build the five tensors VertexHead.forward() expects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
# ── class_probs  (1, 5, 2) ─────────────────────────────────────────────────
#    col 0 = P(empty slot),  col 1 = P(real track)
class_probs = torch.tensor([[[0.05, 0.95]] * N_TRACKS])
 
# ── track_reg_result  (1, 5, 4) ────────────────────────────────────────────
#    columns: [ q/(pT+1),  theta,  sin(phi),  cos(phi) ]
track_reg_result = torch.stack([
    charge / (pT + 1.0),
    theta,
    torch.sin(phi),
    torch.cos(phi),
], dim=-1).unsqueeze(0)   # → (1, 5, 4)
 
# ── points  (1, 150, 4) ────────────────────────────────────────────────────
#    columns: [ E,  x,  y,  z ]
#
#    TPC geometry: hit k on track t is placed where the straight track line
#    crosses the cylindrical surface r_xy = R_TPC_INNER + k * TPC_HIT_SPACING.
#
#    For track t with unit direction d = (dx, dy, dz) starting from vertex V:
#       P(s) = V + s · d
#       r_xy(s)² = (Vx + s·dx)² + (Vy + s·dy)² = R_k²
#
#    Expanding and collecting in s:
#       a · s² + b · s + c_k = 0
#       a   = dx² + dy²               (= sin²θ; constant per track)
#       b   = 2(Vx·dx + Vy·dy)        (constant per track)
#       c_k = Vx² + Vy² − R_k²        (changes with layer k)
#
#    Take the positive (outgoing) root:
#       s_k = (−b + √(b² − 4·a·c_k)) / (2·a)
#
#    The vertex sits at r_xy ≈ 0.58 cm << R_TPC_INNER = 30 cm, so the
#    discriminant is always positive for any R_k ≥ R_TPC_INNER.
#
rows = []
for t in range(N_TRACKS):
    vx, vy, vz = VTX_TRUE.tolist()
    dx, dy, dz = direction[t].tolist()
 
    # Quadratic coefficients (a and b are constant for this track)
    a = dx**2 + dy**2                  # = sin²(theta_t); > 0 for theta in (0, π)
    b = 2.0 * (vx * dx + vy * dy)
 
    # Arc-length to the innermost layer — for the header printout only
    c_inner = vx**2 + vy**2 - R_TPC_INNER**2
    disc_inner = b**2 - 4.0 * a * c_inner
    s_inner = (-b + math.sqrt(max(disc_inner, 0.0))) / (2.0 * a)
 
    print(f"  {t:>3}  {p_total[t]:>8.3f}  {pT[t]:>7.3f}  "
          f"{theta[t]:>7.4f}  {phi[t]:>7.4f}  {int(charge[t]):>3}  {s_inner:>10.2f}")
 
    for k in range(N_HITS_PER_TRACK):
        R_k  = R_TPC_INNER + k * TPC_HIT_SPACING
        c_k  = vx**2 + vy**2 - R_k**2
        disc = b**2 - 4.0 * a * c_k          # always > 0 (vertex inside TPC)
        s_k  = (-b + math.sqrt(disc)) / (2.0 * a)
 
        # ideal hit position on the straight track + Gaussian smearing
        pos = VTX_TRUE + s_k * direction[t] + torch.randn(3) * HIT_SMEARING
        rows.append([p_total[t].item(), pos[0].item(), pos[1].item(), pos[2].item()])
 
points = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)  # (1, 150, 4)
 
# ── mask_probs  (1, 150, 5) ────────────────────────────────────────────────
#    hits 0–29 → track 0 at prob 0.90, etc.
mask_probs = torch.full((1, N_HITS, N_TRACKS), 0.02)
for t in range(N_TRACKS):
    s, e = t * N_HITS_PER_TRACK, (t + 1) * N_HITS_PER_TRACK
    mask_probs[0, s:e, t] = 0.90
 
# ── padding_mask  (1, 150) ─────────────────────────────────────────────────
#    all True — no padding, every hit is real
padding_mask = torch.ones(1, N_HITS, dtype=torch.bool)
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  Print shapes + sanity-check first-hit radii
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n  Input shapes:")
print(f"    class_probs      {tuple(class_probs.shape)}   (events, tracks, 2)")
print(f"    track_reg_result {tuple(track_reg_result.shape)}   (events, tracks, 4)")
print(f"    mask_probs       {tuple(mask_probs.shape)} (events, hits, tracks)")
print(f"    points           {tuple(points.shape)} (events, hits, 4) [E,x,y,z]")
print(f"    padding_mask     {tuple(padding_mask.shape)} (events, hits)  all True")

print(f"  Number of tracks: {N_TRACKS}")
print(f"  Mean track multiplicity: {MEAN_N_TRACKS:.1f}")
 
print(f"\n  First-hit r_xy per track  (should be ≈ {R_TPC_INNER:.0f} cm):")
for t in range(N_TRACKS):
    xy = points[0, t * N_HITS_PER_TRACK, 1:3]
    print(f"    Track {t}: r_xy = {xy.norm().item():.4f} cm")
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  Save inputs so you can reload them later
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
save_path = os.path.join(_HERE, "vertex_test_inputs.pt")
torch.save({
    "class_probs":        class_probs,
    "track_reg_result":   track_reg_result,
    "mask_probs":         mask_probs,
    "points":             points,
    "padding_mask":       padding_mask,
    "true_vertex":        VTX_TRUE,
    "p_total_GeV":        p_total,
    "theta_rad":          theta,
    "phi_rad":            phi,
    "charge":             charge,
    "R_TPC_INNER_cm":     torch.tensor(R_TPC_INNER),
    "TPC_HIT_SPACING_cm": torch.tensor(TPC_HIT_SPACING),
}, save_path)
print(f"\n  Inputs saved → {save_path}")
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  Run VertexHead
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nRunning VertexHead …")
vertex_head = VertexHead(learn_weights=False)
 
with torch.no_grad():
    result = vertex_head(
        class_probs      = class_probs,
        track_reg_result = track_reg_result,
        mask_probs       = mask_probs,
        points           = points,
        padding_mask     = padding_mask,
    )
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.  Print results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vtx_est  = result["vertex_estimate"][0]    # (3,)
chi2     = result["chi_square"][0].item()
is_valid = result["fit_is_valid"][0].item()
t_weight = result["track_weight"][0]       # (5,)
t_dca    = result["track_dca"][0]          # (5,)
t_pos    = result["track_position"][0]     # (5, 3) — innermost hits ≈ r_xy 30 cm
 
print("\n" + "=" * 66)
print("  VertexHead output")
print("=" * 66)
print(f"  fit_is_valid : {is_valid}")
print(f"  chi²         : {chi2:.5f}")
print(f"\n  True   vertex (x, y, z) cm : "
      f"({VTX_TRUE[0]:.4f},  {VTX_TRUE[1]:.4f},  {VTX_TRUE[2]:.4f})")
if is_valid:
    print(f"  Fitted vertex (x, y, z) cm : "
          f"({vtx_est[0]:.4f},  {vtx_est[1]:.4f},  {vtx_est[2]:.4f})")
    res = vtx_est - VTX_TRUE
    print(f"  Residual      (fit-true) cm : "
          f"({res[0]:.4f},  {res[1]:.4f},  {res[2]:.4f})")
    print(f"  |residual|               cm : {res.norm():.4f}  "
          f"({res.norm().item()*1e4:.1f} µm)")
else:
    print("  Fit FAILED — not enough independent tracks.")
 
print(f"\n  Per-track breakdown:")
print(f"  {'Trk':>3}  {'weight':>8}  {'DCA (cm)':>10}  "
      f"{'ref point (x,y,z)':>30}  {'r_xy ref (cm)':>13}")
print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*30}  {'-'*13}")
for t in range(N_TRACKS):
    dca  = t_dca[t].item() if is_valid else float("nan")
    rp   = t_pos[t]
    r_xy = math.sqrt(rp[0].item()**2 + rp[1].item()**2)
    print(f"  {t:>3}  {t_weight[t].item():>8.3f}  {dca:>10.5f}  "
          f"({rp[0]:.2f}, {rp[1]:.2f}, {rp[2]:.2f})  {r_xy:>13.3f}")
 
print("\nDone.\n")
