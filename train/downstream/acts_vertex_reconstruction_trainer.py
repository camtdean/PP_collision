"""
acts_vertex_reconstruction_trainer.py
--------------------------------------
Extends VertexTrainer to use ACTS Kalman Filter + AMVF for vertex
reconstruction instead of (or in addition to) the pure-PyTorch VertexHead.

Usage
-----
    trainer = ActsVertexTrainer(params)
    trainer.setup_vertexing(track_ckpt="path/to/track.pth")
    trainer.run_acts_vertexing(
        max_events=500,
        save_csv=True,
        n_geometry_layers=48,
    )

The output files are compatible with train_vertex_reconstruction.py:
    - vertex_results_acts.npz   (vertex_estimate, vertex_target, fit_is_valid)
    - vertex_metrics_acts.json
    - vertex_per_point_data_acts.csv  (optional)
"""

from __future__ import annotations

import os as _os
import sys as _sys
# Ensure PP_collision/ is on sys.path so 'from train.downstream import ...' works
_here = _os.path.dirname(_os.path.abspath(__file__))
_repo = _os.path.normpath(_os.path.join(_here, '..', '..'))
if _repo not in _sys.path:
    _sys.path.insert(0, _repo)
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import json
import os
import warnings
from typing import Optional

import numpy as np
import torch

# Local imports
from vertex_reconstruction_trainer import VertexTrainer
from downstream_util import get_vertex_label
from loss import compute_vertex_metrics

# ACTS modules (optional)
try:
    from acts_geometry import build_tpc_geometry
    from acts_bridge   import track_params_to_bound, hits_to_measurements
    from acts_fitter   import run_acts_for_event
    _ACTS_AVAILABLE = True
except ImportError:
    _ACTS_AVAILABLE = False

# vertexhead helpers we need for the bridge
from vertexhead import (
    _unnormalize_hits,
    _track_params_from_hits,
    _track_position_from_hits,
)


class ActsVertexTrainer(VertexTrainer):
    """
    VertexTrainer subclass that replaces / augments the PyTorch vertex
    fit with ACTS Kalman Filter + AMVF.

    The pretrained backbone + MambaAttentionHead remain frozen and are
    used to produce cluster assignments (is_assigned).  Those assignments
    are then handed to ACTS for refined track fitting and vertex finding.
    """

    # ------------------------------------------------------------------
    # Geometry (built once, shared across events)
    # ------------------------------------------------------------------

    def _build_geometry(self, n_layers: int = 48):
        """Build and cache the ACTS TPC geometry."""
        if not _ACTS_AVAILABLE:
            warnings.warn(
                "ACTS Python bindings not found – cannot build geometry. "
                "Install ACTS with pybind11 support.",
                ImportWarning,
            )
            self._acts_geometry    = None
            self._acts_field       = None
            self._acts_surfaces    = None
            return

        result = build_tpc_geometry(n_layers=n_layers, Bz=self.Bz)
        if result is None:
            self._acts_geometry    = None
            self._acts_field       = None
            self._acts_surfaces    = None
        else:
            self._acts_geometry, self._acts_field, self._acts_surfaces = result

    # ------------------------------------------------------------------
    # Per-event ACTS inference
    # ------------------------------------------------------------------

    def _acts_event(
        self,
        points_norm: torch.Tensor,   # (N, 4) normalised hits on CPU
        mask: torch.Tensor,          # (N,)   bool padding mask
        class_probs: torch.Tensor,   # (C, 2)
        mask_probs: torch.Tensor,    # (N, C)
        reg: torch.Tensor,           # (N, 8)
    ) -> dict:
        """
        Run ACTS for one event (all tensors single-event, no batch dim).

        Returns
        -------
        dict with vertex_estimate (3,) cm, n_fitted_tracks, fit_is_valid.
        """
        if not _ACTS_AVAILABLE or self._acts_geometry is None:
            return {"vertex_estimate": None, "n_fitted_tracks": 0, "fit_is_valid": False}

        # ---- un-normalise hits ------------------------------------------
        points_np  = points_norm.cpu().numpy()   # (N, 4): E, eta, phi, r (norm)
        mask_np    = mask.cpu().numpy()           # (N,)

        un = _unnormalize_hits(torch.from_numpy(points_np).unsqueeze(0)).squeeze(0)
        # un shape: (N, 4): E_mev, x_cm, y_cm, z_cm
        # (see vertexhead._unnormalize_hits: output columns are E, x, y, z)
        x_cm   = un[:, 1].numpy()
        y_cm   = un[:, 2].numpy()
        z_cm   = un[:, 3].numpy()
        hit_r  = torch.from_numpy(
            np.sqrt(x_cm ** 2 + y_cm ** 2)
        )  # (N,)

        # ---- cluster assignment ------------------------------------------
        mp  = mask_probs.unsqueeze(0)           # (1, N, C)
        cp  = class_probs.unsqueeze(0)          # (1, C, 2)
        pm  = mask.unsqueeze(0)                 # (1, N)
        hr  = hit_r.unsqueeze(0)                # (1, N)

        _, _, is_assigned = _track_position_from_hits(
            torch.from_numpy(points_np).unsqueeze(0), hr, mp, pm
        )
        # is_assigned: (1, N, T)
        is_assigned_np = is_assigned[0].cpu().numpy()   # (N, T)

        # ---- track parameters (Taubin + Deming seeds) -------------------
        tp = _track_params_from_hits(
            torch.from_numpy(points_np).unsqueeze(0),
            hr,
            is_assigned,
            noise_pt_threshold=0.25,
            Bz=self.Bz,
        )
        # Strip batch dimension
        tp_single = {
            k: (v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in tp.items()
        }
        # Add track_position for bridge
        # _track_params_from_hits does not return track_position directly;
        # compute it again via _track_position_from_hits output
        tp_single["track_position"] = (
            _track_position_from_hits(
                torch.from_numpy(points_np).unsqueeze(0), hr, mp, pm
            )[0][0].cpu().numpy()   # (T, 3)
        )

        # ---- call ACTS full pipeline ------------------------------------
        valid_r = hit_r.numpy()[mask_np]
        return run_acts_for_event(
            x_cm       = x_cm[mask_np],
            y_cm       = y_cm[mask_np],
            z_cm       = z_cm[mask_np],
            r_cm       = valid_r,
            track_params_dict = tp_single,
            is_assigned       = is_assigned_np[mask_np],
            geometry          = self._acts_geometry,
            magnetic_field    = self._acts_field,
            surfaces          = self._acts_surfaces,
            Bz                = self.Bz,
        )

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run_acts_vertexing(
        self,
        max_events:        int  = -1,
        save_csv:          bool = False,
        n_geometry_layers: int  = 48,
        output_suffix:     str  = "_acts",
    ):
        """
        Evaluate ACTS-based vertex reconstruction on the test set.

        Parameters
        ----------
        max_events : int
            Maximum number of events to process (-1 = all).
        save_csv : bool
            If True, write per-point CSV (analogous to run_vertexing).
        n_geometry_layers : int
            Number of TPC cylindrical layers for ACTS geometry.
        output_suffix : str
            Appended to output filenames to avoid overwriting PyTorch results.
        """
        if not _ACTS_AVAILABLE:
            print("[ActsVertexTrainer] ACTS not available – aborting.")
            return

        self._build_geometry(n_layers=n_geometry_layers)
        if self._acts_geometry is None:
            print("[ActsVertexTrainer] Geometry build failed – aborting.")
            return

        # Make sure neural network models are in eval mode
        self.model.eval()
        self.track_model.eval()
        if hasattr(self, "vertex_head"):
            self.vertex_head.eval()

        all_vertex_estimates = []
        all_vertex_targets   = []
        all_fit_valid        = []
        csv_rows             = []
        n_processed          = 0

        with torch.no_grad():
            for batch in self.test_loader:
                if max_events > 0 and n_processed >= max_events:
                    break

                grouped, label, mask, reg = batch
                # grouped: (B, N, 4)  mask: (B, N)  reg: (B, N, 8)
                grouped = grouped.to(self.device)
                mask    = mask.to(self.device)
                reg     = reg.cpu()

                B = grouped.shape[0]

                # --- backbone + track head ---
                emb = self.model(grouped, padding_mask=~mask, pretrain=True)
                out = self.track_model(
                    emb["embedding_post_projection"],
                    feature      = None,
                    padding_mask = ~mask,
                )
                class_probs = out["class_probs"].cpu()   # (B, C, 2)
                mask_probs  = out["mask_probs"].cpu()    # (B, N, C)
                grouped_cpu = grouped.cpu()
                mask_cpu    = mask.cpu()

                # --- vertex truth ---
                vtx_target, vtx_valid, _ = get_vertex_label(reg)
                all_vertex_targets.append(vtx_target.numpy())
                all_fit_valid.append(vtx_valid.numpy())

                # --- per-event ACTS pipeline ---
                batch_estimates = []
                for b in range(B):
                    if max_events > 0 and n_processed >= max_events:
                        break

                    result = self._acts_event(
                        points_norm = grouped_cpu[b],     # (N, 4)
                        mask        = mask_cpu[b],        # (N,)
                        class_probs = class_probs[b],     # (C, 2)
                        mask_probs  = mask_probs[b],      # (N, C)
                        reg         = reg[b],             # (N, 8)
                    )

                    if result["vertex_estimate"] is not None:
                        batch_estimates.append(result["vertex_estimate"])
                    else:
                        batch_estimates.append(np.array([0.0, 0.0, 0.0]))

                    n_processed += 1

                all_vertex_estimates.append(np.stack(batch_estimates, axis=0))

        # ------------------------------------------------------------------
        # Aggregate and save
        # ------------------------------------------------------------------
        vertex_estimate = np.concatenate(all_vertex_estimates, axis=0)   # (N_ev, 3)
        vertex_target   = np.concatenate(all_vertex_targets,   axis=0)   # (N_ev, 3)
        fit_is_valid    = np.concatenate(all_fit_valid,         axis=0)   # (N_ev,)

        # Compute metrics using existing helper
        ve_t = torch.from_numpy(vertex_estimate).float()
        vt_t = torch.from_numpy(vertex_target  ).float()
        vv_t = torch.from_numpy(fit_is_valid   ).bool()
        metrics = compute_vertex_metrics(ve_t, vt_t, vv_t)
        metrics = {k: float(v) for k, v in metrics.items()}

        # Output directory (same as standard results)
        out_dir = getattr(self, "output_dir",
                          os.path.join(os.path.dirname(__file__), "results"))
        os.makedirs(out_dir, exist_ok=True)

        npz_path = os.path.join(out_dir, f"vertex_results{output_suffix}.npz")
        np.savez(
            npz_path,
            vertex_estimate = vertex_estimate,
            vertex_target   = vertex_target,
            fit_is_valid    = fit_is_valid,
        )
        print(f"[ActsVertexTrainer] Saved results: {npz_path}")

        json_path = os.path.join(out_dir, f"vertex_metrics{output_suffix}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[ActsVertexTrainer] Saved metrics: {json_path}")
        print(json.dumps(metrics, indent=2))
