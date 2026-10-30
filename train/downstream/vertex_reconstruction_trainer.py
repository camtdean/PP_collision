"""
vertex_reconstruction_trainer.py
================================
Runs VertexHead (and optionally SecondaryVertexHead) after an already-trained
track-finding head.
 
    frozen backbone -> frozen track finder (train_track_finding.py checkpoint)
        -> hit clusters -> VertexHead / SecondaryVertexHead (whole batch)
 
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
 
Secondary vertex support
------------------------
setup_vertexing() also constructs self.sv_head (a SecondaryVertexHead).
run_sv_vertexing() runs the full pipeline:
    1. sv_head.forward() – PV fit + SV seeding/fitting in one pass
    2. Identify tracks assigned to any SV (sv_track_mask)
    3. Refit the PV with those tracks zeroed out (refined PV)
    4. Report metrics for both the original and refined PV, plus SV summary stats
 
The refined PV is what gets written to vertex_estimate in the output (and to the CSV when
save_csv=True); the original PV is available under pv_original in the returned dict.
 
run_vertexing() is unchanged – it still calls self.vertex_head and does not touch sv_head.
"""
 
import numpy as np
import torch
from contextlib import nullcontext
from torch.amp import autocast
from tqdm import tqdm
 
from train.downstream import track_finding_trainer
from train.downstream.vertexhead import (
    VertexHead,
    _fit_vertex_by_closest_approach,
    _fit_vertex_by_helix_closest_approach,
)
from train.downstream.secondaryvertexhead import SecondaryVertexHead
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
 
# SV columns appended by run_sv_vertexing when save_csv=True.
# is_sv_track: the track that owns this hit was assigned to at least one SV.
# sv_x, sv_y, sv_z: position of the best (lowest chi²/ndf) SV the track belongs to; NaN otherwise.
SV_COLUMNS = ("is_sv_track", "sv_x", "sv_y", "sv_z")
 
CSV_COLUMNS        = PER_POINT_COLUMNS + TEST_COLUMNS
CSV_COLUMNS_SV     = PER_POINT_COLUMNS + TEST_COLUMNS + SV_COLUMNS
 
 
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
        "confidence": col(mask_probs.max(dim=-1).values),
    }
    # TESTING ONLY
    residual = vertex.double() - truth_vertex.double()
    for k, axis in enumerate("xyz"):
        columns[f"test_truth_vtx_{axis}"] = col(truth_vertex[:, k])
        columns[f"test_residual_{axis}"] = col(residual[:, k])
    columns["test_residual_3d"] = col(residual.norm(dim=-1))
    return {k: columns[k] for k in CSV_COLUMNS}
 
 
def per_point_columns_sv(
    step, grouped, label, reg, mask, class_probs, mask_probs,
    vertex, truth_vertex, sv_result,
):
    """
    Like per_point_columns but also writes SV_COLUMNS.
 
    For each real hit the owning track's SV membership is looked up via
    sv_result["sv_track_mask"] and the best-SV position (lowest chi²/ndf)
    for that track is written to sv_x / sv_y / sv_z.
 
    Args:
        vertex: (B, 3) refined PV position per event.
        sv_result: dict returned by SecondaryVertexHead.forward.
        (all other args same as per_point_columns)
 
    Returns:
        dict of column name -> 1-D numpy array in CSV_COLUMNS_SV order.
    """
    base = per_point_columns(step, grouped, label, reg, mask, class_probs, mask_probs,
                              vertex, truth_vertex)
 
    assign = assign_points_to_masks(
        {"pred_probs": class_probs, "pred_masks": mask_probs.permute(0, 2, 1)},
        option=2,
    )["assignments"]                           # (B, N)  long – track index per hit
    b, n = mask.nonzero(as_tuple=True)
 
    # sv_track_mask: (B, K, T) bool; sv_chi2_per_dof: (B, K)
    sv_mask  = sv_result["sv_track_mask"]      # (B, K, T)
    sv_chi2  = sv_result["sv_chi2_per_dof"]    # (B, K)  NaN where invalid
    sv_pos   = sv_result["sv_estimate"]        # (B, K, 3)
 
    # For each track, find which SV it belongs to (the one with lowest chi²/ndf)
    # sv_track_any: (B, T) – True if track i belongs to any SV
    sv_track_any = sv_mask.any(dim=1)          # (B, T)
 
    # Best SV index per track: among the K slots that include this track,
    # pick the one with lowest chi²/ndf.  Slots not including the track get +inf.
    # sv_q_per_track: (B, T, K) – chi²/ndf or +inf
    K = sv_mask.shape[1]
    sv_q_per_track = sv_chi2.unsqueeze(1).expand(-1, sv_mask.shape[2], -1)   # (B, T, K)
    sv_q_per_track = sv_q_per_track.nan_to_num(nan=float("inf"))
    sv_q_per_track = sv_q_per_track.masked_fill(~sv_mask.permute(0, 2, 1), float("inf"))
    best_sv_idx = sv_q_per_track.argmin(dim=-1)    # (B, T)  – 0 where not in any SV
 
    # Look up the best SV position for each track
    ev_idx  = torch.arange(sv_pos.shape[0], device=sv_pos.device).unsqueeze(-1).expand_as(best_sv_idx)
    best_sv_pos = sv_pos[ev_idx, best_sv_idx]      # (B, T, 3)
    # Zero out tracks not in any SV
    best_sv_pos = best_sv_pos.masked_fill(~sv_track_any.unsqueeze(-1), float("nan"))
 
    # Look up per-hit values: the track each hit belongs to
    track_idx  = assign[b, n]                       # (n_real_hits,) long
    is_sv      = sv_track_any[b, track_idx]         # (n_real_hits,) bool
    hit_sv_pos = best_sv_pos[b, track_idx]          # (n_real_hits, 3)
 
    columns = dict(base)
    columns["is_sv_track"] = is_sv.cpu().numpy().astype(np.float32)
    for k, axis in enumerate("xyz"):
        columns[f"sv_{axis}"] = hit_sv_pos[:, k].double().cpu().numpy()
 
    return {k: columns[k] for k in CSV_COLUMNS_SV}
 
 
class VertexTrainer(track_finding_trainer.DownstreamTrainer):
 
    def setup_vertexing(self, track_ckpt):
        """
        Build the track head exactly as track_finding_trainer does, load it strictly,
        then build both VertexHead (standalone PV) and SecondaryVertexHead (PV + SV).
 
        SV-specific thresholds can be set via params (all optional, defaults shown):
            params.sv_min_track_pv_dca    = 0.05   cm
            params.sv_min_track_beam_dca  = 0.05   cm
            params.sv_max_pair_doca       = 1.0    cm
            params.sv_max_track_sv_doca   = 0.5    cm
            params.sv_seed_merge_distance = 0.1    cm
            params.sv_min_sv_tracks       = 2
            params.sv_min_sv_displacement = 0.02   cm
            params.sv_max_sv_chi2_per_dof = 10.0
            params.sv_max_sv_per_event    = 6
 
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
 
        sd = torch.load(track_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
 
        # checks for missing keys and rejects mismatched checkpoints
        self.down_model.load_state_dict(sd)
        self.down_model.eval()
 
        bz = float(getattr(self.params, "vertex_bz", 1.4))
 
        # Standalone PV head (unchanged from the original trainer)
        self.vertex_head = VertexHead(Bz=bz).to(self.device).eval()
 
        # Combined PV + SV head – all thresholds can be tuned via params
        def _p(name, default):
            return type(default)(getattr(self.params, name, default))
 
        self.sv_head = SecondaryVertexHead(
            Bz=bz,
            min_track_pv_dca    = _p("sv_min_track_pv_dca",    0.05),
            min_track_beam_dca  = _p("sv_min_track_beam_dca",  0.05),
            max_pair_doca       = _p("sv_max_pair_doca",        1.0),
            max_track_sv_doca   = _p("sv_max_track_sv_doca",   0.5),
            seed_merge_distance = _p("sv_seed_merge_distance",  0.1),
            min_sv_tracks       = _p("sv_min_sv_tracks",        2),
            min_sv_displacement = _p("sv_min_sv_displacement",  0.02),
            max_sv_chi2_per_dof = _p("sv_max_sv_chi2_per_dof", 10.0),
            max_sv_per_event    = _p("sv_max_sv_per_event",     6),
        ).to(self.device).eval()
 
    # ─────────────────────────────────────────────────────────────────────────
    # Original PV-only evaluation (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
 
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
        est, tgt, ok, per_point = [], [], [], []
 
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
 
            result = self.vertex_head(
                class_probs=out["class_probs"].float(), mask_probs=out["mask_probs"].float(),
                points=grouped, padding_mask=mask)
 
            # padded rows are zeros and would pass get_vertex_label's 1 cm cut; exclude them
            truth = get_vertex_label(reg, valid_tracks=(torch.hypot(reg[..., 3], reg[..., 4]) < 1.0) & mask)
            if save_csv:
                truth_vertex = truth["vertex_target"].masked_fill(~truth["vertex_valid"][:, None], float("nan"))
                per_point.append(per_point_columns(step, grouped, label, reg, mask, out["class_probs"],
                                                   out["mask_probs"], result["vertex_estimate"], truth_vertex))
            est.append(result["vertex_estimate"])
            tgt.append(truth["vertex_target"])
            ok.append(result["fit_is_valid"] & truth["vertex_valid"])
            if max_events and sum(len(x) for x in est) >= max_events:
                break
 
        est, tgt, ok = torch.cat(est), torch.cat(tgt), torch.cat(ok)
        metrics = compute_vertex_metrics(est, tgt, ok)
        metrics.update(n_events=len(ok), fit_valid_rate=ok.float().mean().item())
        if self.log_to_screen:
            print("[vertex] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items()))
        output = {"metrics": metrics, "vertex_estimate": est.cpu(), "vertex_target": tgt.cpu(),
                  "fit_is_valid": ok.cpu()}
        if save_csv:
            output["per_point"] = {k: np.concatenate([p[k] for p in per_point]) for k in CSV_COLUMNS}
        return output
 
    # ─────────────────────────────────────────────────────────────────────────
    # Combined PV + SV evaluation with refined PV refit
    # ─────────────────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def run_sv_vertexing(self, pretrain=True, max_events=None, save_csv=False, refit_pv=True):
        """
        Backbone -> track head -> SecondaryVertexHead (PV + SVs) over the validation loader.
 
        After finding secondary vertices, tracks that are assigned to any SV are removed
        from the primary vertex fit (the "refined PV").  The refined PV is reported as
        vertex_estimate; the original (pre-SV-removal) PV is available as pv_original.
 
        Args:
            pretrain: True = frozen-backbone features (as trained).
            max_events: stop after this many events; None = whole loader.
            save_csv: also build the per-point CSV.  vertex_estimate (the refined PV) is
                written to vtx_x/y/z; SV membership and SV position are written to
                is_sv_track and sv_x/y/z.
            refit_pv: if True (default), refit the PV after removing SV tracks.  The helix
                refinement is used when sv_head.use_helix is True and sv_head.Bz != 0.
                Set to False to use the original PV from sv_head directly as vertex_estimate.
 
        Returns:
            dict:
                metrics:          compute_vertex_metrics for the refined PV (or original if
                                  refit_pv=False), plus n_events and fit_valid_rate.
                pv_original_metrics: metrics for the unrefined PV from sv_head.
                sv_summary:       dict of simple SV aggregate stats across all events:
                                      n_sv_mean, n_sv_max, sv_valid_rate,
                                      sv_displacement_mean, sv_n_tracks_mean.
                vertex_estimate:  (n_events, 3) refined PV (or original PV if refit_pv=False).
                vertex_target:    (n_events, 3) truth PV.
                pv_original:      (n_events, 3) PV before SV-track removal.
                fit_is_valid:     (n_events,) bool, fit valid and truth vertex defined.
                per_point:        dict column -> numpy array (only with save_csv=True).
        """
        self.model.eval()
        amp = torch.cuda.is_available() and self.use_amp
 
        est, est_orig, tgt, ok, ok_orig = [], [], [], [], []
        sv_valid_counts  = []   # (B,) number of valid SVs per event
        sv_displace_list = []   # flat list of valid SV displacements
        sv_ntracks_list  = []   # flat list of valid SV n_tracks
        per_point_rows   = []
 
        for step, batch in enumerate(tqdm(self.val_data_loader)):
            if len(batch) != 4:
                raise RuntimeError("loader must yield (grouped, label, knearest, reg); "
                                   "set params.return_reg_test = True before launch()")
            grouped, label, _, reg = batch
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c).to(self.device)
            reg     = reg.to(self.device)
            mask    = grouped[..., 0] != PAD_VALUE
 
            with autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                if pretrain:
                    _, pre_embed, _ = self.model(grouped, return_z=True)
                    out = self.down_model(grouped, torch.stack(pre_embed), pretrain=True, padding_mask=mask)
                else:
                    out = self.down_model(grouped, feature=None)
 
            class_probs = out["class_probs"].float()
            mask_probs  = out["mask_probs"].float()
 
            # ── Step 1: PV + SV in one forward pass ──────────────────────────
            sv_result = self.sv_head(
                class_probs=class_probs,
                mask_probs=mask_probs,
                points=grouped,
                padding_mask=mask,
            )
            pv_result    = sv_result["pv_result"]
            pv_estimate  = pv_result["vertex_estimate"]    # (B, 3)
            pv_valid     = pv_result["fit_is_valid"]       # (B,)
 
            # ── Step 2: identify SV tracks ────────────────────────────────────
            # sv_track_mask: (B, K, T) bool – which tracks contributed to each SV
            sv_track_any = sv_result["sv_track_mask"].any(dim=1)   # (B, T)
 
            # ── Step 3: refined PV (SV tracks removed) ───────────────────────
            if refit_pv:
                pv_weight = pv_result["track_weight"]              # (B, T)
                pv_weight_r = pv_weight * (~sv_track_any).to(pv_weight.dtype)
 
                track_pos = sv_result["track_position"]            # (B, T, 3)
                track_dir = sv_result["track_direction"]           # (B, T, 3)
 
                refined_fit = _fit_vertex_by_closest_approach(track_pos, track_dir, pv_weight_r)
 
                if self.sv_head.use_helix and self.sv_head.Bz != 0.0:
                    tp = sv_result["track_params"]
                    refined_fit = _fit_vertex_by_helix_closest_approach(
                        tp["circle_centre"],
                        tp["R_s"],
                        tp["zslope"],
                        tp["z0"],
                        pv_weight_r,
                        n_iterations=self.sv_head.helix_iterations,
                        linear_fit=refined_fit,
                        is_cosmics=self.sv_head.is_cosmics,
                    )
 
                refined_estimate = refined_fit["vertex_estimate"]  # (B, 3)
                refined_valid    = refined_fit["fit_is_valid"]     # (B,)
            else:
                refined_estimate = pv_estimate
                refined_valid    = pv_valid
 
            # ── Step 4: truth + metrics accumulation ─────────────────────────
            truth = get_vertex_label(
                reg,
                valid_tracks=(torch.hypot(reg[..., 3], reg[..., 4]) < 1.0) & mask,
            )
            truth_valid  = truth["vertex_valid"]
            truth_vertex = truth["vertex_target"]
 
            ok_r   = refined_valid   & truth_valid
            ok_pvo = pv_valid        & truth_valid
 
            est.append(refined_estimate)
            est_orig.append(pv_estimate)
            tgt.append(truth_vertex)
            ok.append(ok_r)
            ok_orig.append(ok_pvo)
 
            # ── Step 5: SV summary stats ──────────────────────────────────────
            sv_is_valid  = sv_result["sv_is_valid"]      # (B, K)
            sv_displace  = sv_result["sv_displacement"]  # (B, K)
            sv_n_tracks  = sv_result["sv_n_tracks"]      # (B, K) long
 
            sv_valid_counts.append(sv_is_valid.sum(dim=-1).float().cpu())   # (B,)
            if sv_is_valid.any():
                sv_displace_list.append(sv_displace[sv_is_valid].cpu())
                sv_ntracks_list.append(sv_n_tracks[sv_is_valid].float().cpu())
 
            # ── Step 6: optional per-point CSV ───────────────────────────────
            if save_csv:
                tv_nan = truth_vertex.masked_fill(~truth_valid[:, None], float("nan"))
                per_point_rows.append(
                    per_point_columns_sv(
                        step, grouped, label, reg, mask,
                        class_probs, mask_probs,
                        refined_estimate, tv_nan,
                        sv_result,
                    )
                )
 
            if max_events and sum(len(x) for x in est) >= max_events:
                break
 
        # ── Aggregate over all batches ────────────────────────────────────────
        est      = torch.cat(est)
        est_orig = torch.cat(est_orig)
        tgt      = torch.cat(tgt)
        ok       = torch.cat(ok)
        ok_orig  = torch.cat(ok_orig)
 
        metrics = compute_vertex_metrics(est, tgt, ok)
        metrics.update(n_events=len(ok), fit_valid_rate=ok.float().mean().item())
 
        pv_orig_metrics = compute_vertex_metrics(est_orig, tgt, ok_orig)
        pv_orig_metrics.update(n_events=len(ok_orig), fit_valid_rate=ok_orig.float().mean().item())
 
        sv_counts  = torch.cat(sv_valid_counts)
        sv_summary = {
            "n_sv_mean":          sv_counts.mean().item(),
            "n_sv_max":           int(sv_counts.max().item()),
            "sv_valid_rate":      (sv_counts > 0).float().mean().item(),
        }
        if sv_displace_list:
            all_disp   = torch.cat(sv_displace_list)
            all_ntracks = torch.cat(sv_ntracks_list)
            sv_summary["sv_displacement_mean"] = all_disp.mean().item()
            sv_summary["sv_n_tracks_mean"]     = all_ntracks.mean().item()
        else:
            sv_summary["sv_displacement_mean"] = float("nan")
            sv_summary["sv_n_tracks_mean"]     = float("nan")
 
        if self.log_to_screen:
            print("[vertex-refined] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()))
            print("[vertex-original] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in pv_orig_metrics.items()))
            print("[sv] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sv_summary.items()))
 
        output = {
            "metrics":              metrics,
            "pv_original_metrics":  pv_orig_metrics,
            "sv_summary":           sv_summary,
            "vertex_estimate":      est.cpu(),          # refined PV
            "pv_original":          est_orig.cpu(),     # PV before SV removal
            "vertex_target":        tgt.cpu(),
            "fit_is_valid":         ok.cpu(),
        }
        if save_csv:
            output["per_point"] = {
                k: np.concatenate([p[k] for p in per_point_rows])
                for k in CSV_COLUMNS_SV
            }
        return output