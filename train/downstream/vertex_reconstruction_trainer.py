"""
vertex_reconstruction_trainer.py
================================
Runs VertexHead after an already-trained track-finding head.

    frozen backbone -> frozen track finder (train_track_finding.py checkpoint)
        -> hit clusters -> VertexHead (whole batch)

Config, backbone, pretrained weights and data loaders all come from
track_finding_trainer.DownstreamTrainer.launch(). VertexHead(learn_weights=False) has no
parameters, so this evaluates; nothing is trained.

The track finder outputs hit clusters but no track parameters or quality labels; VertexHead
fits both from each predicted cluster's hits. Truth is used only to score the result.

With save_csv, run_vertexing also rebuilds track_finding_trainer.inference()'s per-point CSV
row for row (same columns, order and values), except vtx_x, vtx_y, vtx_z hold the event's
reconstructed vertex (NaN where the fit is invalid) instead of the truth production vertex.
TESTING ONLY truth columns follow, prefixed test_ (see TEST_COLUMNS); they are not
reconstruction output and must not be used as such.
"""

import numpy as np
import torch
from contextlib import nullcontext
from torch.amp import autocast
from tqdm import tqdm

from train.downstream import track_finding_trainer
from train.downstream.vertexhead import VertexHead, OUTLIER_DCA_CUT
from trackinghead import MambaAttentionHead
from downstream_util import get_vertex_label
from loss import compute_vertex_metrics, assign_points_to_masks

PAD_VALUE = -100

# track_finding_trainer.inference()'s per-point CSV columns, in its order
PER_POINT_COLUMNS = ("batch_idx", "point_idx", "E_norm", "eta_norm", "phi_norm", "r_norm",
                     "px", "py", "pz", "vtx_x", "vtx_y", "vtx_z", "charge", "energy",
                     "seg_target", "pred_assignment", "confidence")

# TESTING ONLY, appended after PER_POINT_COLUMNS: the event's truth vertex (get_vertex_label, the
# one the metrics score against) and the reconstructed-minus-truth residual, all cm; NaN where
# either vertex is undefined
TEST_COLUMNS = ("test_truth_vtx_x", "test_truth_vtx_y", "test_truth_vtx_z",
                "test_residual_x", "test_residual_y", "test_residual_z", "test_residual_3d")

CSV_COLUMNS = PER_POINT_COLUMNS + TEST_COLUMNS

def outlier_cut(value):
    """vertex_outlier_dca_cut as VertexHead takes it: None, or <= 0, disables outlier rejection."""
    return None if value is None or float(value) <= 0 else float(value)

def per_point_columns(step, grouped, label, reg, mask, class_probs, mask_probs, vertex, truth_vertex):
    """
    One batch of track_finding_trainer.inference()'s per-point CSV, with the reconstructed vertex.

    Args:
        step: data-loader step, written as batch_idx (as inference() does).
        grouped: (B, N, 4) normalized (E, eta, phi, r) hits.
        label: (B, N) truth track ID per hit.
        reg: (B, N, 8) truth (px, py, pz, vx, vy, vz, q, e) per hit.
        mask: (B, N) bool, True = real hit.
        class_probs, mask_probs: track head outputs exactly as it returned them (bf16 under
            autocast); inference() assigns hits from these, so rounding matches.
        vertex: (B, 3) reconstructed vertex per event.
        truth_vertex: (B, 3) truth vertex per event, NaN where undefined; TESTING ONLY.

    Returns:
        dict of column name -> 1-D numpy array in CSV_COLUMNS order, one entry per real hit,
        ordered by event then hit.
    """
    assign = assign_points_to_masks({"pred_probs": class_probs, "pred_masks": mask_probs.permute(0, 2, 1)},
                                    option=2)["assignments"]
    confidence = mask_probs.max(dim=-1).values
    b, n = mask.nonzero(as_tuple=True)

    def col(t):
        # float64 (exact for bf16 and float32) so the CSV prints each value as inference()'s .item() does
        t = t[b, n] if t.dim() == 2 else t[b]
        return t.double().cpu().numpy()

    point_cols = {name: col(grouped[..., k]) for k, name in enumerate(("E_norm", "eta_norm", "phi_norm", "r_norm"))}
    reg_cols = {name: col(reg[..., k]) for k, name in enumerate(("px", "py", "pz", "vx", "vy", "vz", "charge", "energy"))}
    columns = {
        "batch_idx": np.full(len(b), step, dtype=np.int64),
        "point_idx": n.cpu().numpy(),
        **point_cols,
        "px": reg_cols["px"], "py": reg_cols["py"], "pz": reg_cols["pz"],
        "vtx_x": col(vertex[:, 0]), "vtx_y": col(vertex[:, 1]), "vtx_z": col(vertex[:, 2]),
        "charge": reg_cols["charge"], "energy": reg_cols["energy"],
        "seg_target": label.to(mask.device)[b, n].cpu().numpy(),
        "pred_assignment": assign[b, n].cpu().numpy(),
        "confidence": col(confidence),
    }
    # TESTING ONLY
    residual = vertex.double() - truth_vertex.double()
    for k, axis in enumerate("xyz"):
        columns[f"test_truth_vtx_{axis}"] = col(truth_vertex[:, k])
        columns[f"test_residual_{axis}"] = col(residual[:, k])
    columns["test_residual_3d"] = col(residual.norm(dim=-1))
    return {k: columns[k] for k in CSV_COLUMNS}

class VertexTrainer(track_finding_trainer.DownstreamTrainer):

    def setup_vertexing(self, track_ckpt):
        """
        Build the track head exactly as track_finding_trainer does, load it strictly, build VertexHead.
        Call after launch().

        Args:
            track_ckpt: path to a train_track_finding.py checkpoint.
        """
        n_feat = 1 if getattr(self, "cache_combined", False) else self.params.num_layers_backbone
        self.down_model = MambaAttentionHead(
            input_dim=self.params.embed_dim, num_layers=0,
            num_embedder_layers=self.params.num_embedder_layers,
            d_state=64, d_conv=4, expand=2, num_feature_layers=n_feat,
            num_prototypes=self.params.max_gt_classes,
            embed_method=getattr(self.params, "embed_method", "add")).to(self.device)

        # base load_checkpoint uses strict=False, which would accept a mismatched head silently
        sd = torch.load(track_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = self.down_model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"track head checkpoint does not match the built head: "
                               f"missing={missing[:5]} unexpected={unexpected[:5]}")
        self.down_model.eval()
        for p in self.down_model.parameters():
            p.requires_grad_(False)

        self.vertex_head = VertexHead(
            learn_weights=False,
            use_helix=bool(getattr(self.params, "vertex_use_helix", True)),
            Bz=float(getattr(self.params, "vertex_bz", 1.4)),
            scaling_factor=float(getattr(self.params, "vertex_scaling_factor", 100.0)),
            noise_pt_threshold=float(getattr(self.params, "vertex_noise_pt_threshold", 0.25)),
            outlier_dca_cut=outlier_cut(getattr(self.params, "vertex_outlier_dca_cut", OUTLIER_DCA_CUT)),
        ).to(self.device).eval()

    @torch.no_grad()
    def run_vertexing(self, pretrain=True, max_events=None, save_csv=False):
        """
        Backbone -> track head -> VertexHead over the validation loader.

        Args:
            pretrain: True = frozen-backbone features (as trained); False = head on raw points.
            max_events: stop after this many events; None = whole loader.
            save_csv: also build the per-point CSV columns (see per_point_columns).

        Returns:
            dict:
                metrics: compute_vertex_metrics output plus n_events and fit_valid_rate.
                vertex_estimate, vertex_target: (n_events, 3).
                fit_is_valid: (n_events,) bool, fit valid and truth vertex defined.
                per_point: dict column -> numpy array, inference()'s per-point CSV with the
                    reconstructed vertex, then the TESTING ONLY truth columns; only with save_csv.
        """
        self.model.eval()
        amp = torch.cuda.is_available() and self.use_amp
        est, tgt, ok, outliers, per_point = [], [], [], [], []

        for step, batch in enumerate(tqdm(self.val_data_loader)):
            if len(batch) != 4:
                raise RuntimeError("loader must yield (grouped, label, knearest, reg); "
                                   "set params.return_reg_test = True before launch()")
            grouped, label, _, reg = batch
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c).to(self.device)
            reg = reg.to(self.device)
            mask = grouped[..., 0] != PAD_VALUE

            with autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                if pretrain:
                    _, pre_embed, _ = self.model(grouped, return_z=True)
                    out = self.down_model(grouped, torch.stack(pre_embed), pretrain=True, padding_mask=mask)
                else:
                    out = self.down_model(grouped, feature=None)
            class_probs, mask_probs = out["class_probs"].float(), out["mask_probs"].float()
            mask_probs = mask_probs * mask.unsqueeze(-1)

            result = self.vertex_head(
                class_probs=class_probs, mask_probs=mask_probs,
                points=grouped, padding_mask=mask)

            # padded rows are zeros and would pass get_vertex_label's 1 cm cut; exclude them
            vtx_r = torch.hypot(reg[..., 3], reg[..., 4])
            truth = get_vertex_label(reg, valid_tracks=(vtx_r < 1.0) & mask)
            if save_csv:
                truth_vertex = torch.where(truth["vertex_valid"].unsqueeze(-1), truth["vertex_target"],
                                           torch.full_like(truth["vertex_target"], float("nan")))
                per_point.append(per_point_columns(step, grouped, label, reg, mask, out["class_probs"],
                                                   out["mask_probs"], result["vertex_estimate"], truth_vertex))
            est.append(result["vertex_estimate"])
            tgt.append(truth["vertex_target"])
            ok.append(result["fit_is_valid"] & truth["vertex_valid"])
            outliers.append(result["track_is_outlier"].sum(dim=-1))
            if max_events and sum(len(x) for x in est) >= max_events:
                break

        est, tgt, ok, outliers = torch.cat(est), torch.cat(tgt), torch.cat(ok), torch.cat(outliers)
        metrics = compute_vertex_metrics(est, tgt, ok)
        metrics.update(n_events=len(ok), fit_valid_rate=ok.float().mean().item(),
                       outliers_per_event=outliers.float().mean().item())
        if self.log_to_screen:
            print("[vertex] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items()))
        output = {"metrics": metrics, "vertex_estimate": est.cpu(), "vertex_target": tgt.cpu(),
                  "fit_is_valid": ok.cpu()}
        if save_csv:
            output["per_point"] = {k: np.concatenate([p[k] for p in per_point]) for k in CSV_COLUMNS}
        return output
