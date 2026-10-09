"""
vertex_reconstruction_trainer.py
================================
Runs VertexHead after an already-trained track-finding head.

    frozen backbone -> frozen track finder (train_track_finding.py checkpoint)
        -> per-prototype track parameters + quality labels -> VertexHead (whole batch)

Config, backbone, pretrained weights and data loaders all come from
track_finding_trainer.DownstreamTrainer.launch(). VertexHead(learn_weights=False) has no
parameters, so this evaluates; nothing is trained.

The track finder outputs hit clusters but no track parameters or quality labels.
params.vertex_track_params chooses where they come from:
    "hits"  (default) fitted from each predicted cluster's hits (vertexhead.track_params_from_hits);
            no truth used anywhere except to score the result
    "truth" PLACEHOLDER for comparison: from the truth track holding the plurality of each
            predicted cluster's hits (truth_track_params)
"""
import math

import numpy as np
import torch
from contextlib import nullcontext
from torch.amp import autocast
from tqdm import tqdm

from train.downstream import track_finding_trainer
from train.downstream.vertexhead import (VertexHead, build_tracks, apply_silicon_tpc_mask,
                                         track_params_from_hits, NOISE_PT_THRESHOLD)
from trackinghead import MambaAttentionHead
from downstream_util import get_vertex_label
from loss import compute_vertex_metrics

PAD_VALUE = -100
EMPTY_SLOT_PARAMS = (0.5, math.pi / 2, 0.0, 1.0)   # finite filler; empty slots are labelled noise anyway


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

        self.noise_pt_threshold = float(getattr(self.params, "noise_pt_threshold", NOISE_PT_THRESHOLD))
        self.use_quality_mask = bool(getattr(self.params, "vertex_use_quality_mask", True))
        self.use_silicon_tpc_mask = bool(getattr(self.params, "vertex_silicon_tpc_mask", False))
        self.track_params_source = getattr(self.params, "vertex_track_params", "hits")
        if self.track_params_source not in ("hits", "truth"):
            raise ValueError(f"vertex_track_params must be 'hits' or 'truth', got {self.track_params_source!r}")
        self.vertex_head = VertexHead(
            learn_weights=False,
            use_helix=bool(getattr(self.params, "vertex_use_helix", True)),
            Bz=float(getattr(self.params, "vertex_bz", 1.4)),
            scaling_factor=float(getattr(self.params, "vertex_scaling_factor", 100.0)),
        ).to(self.device).eval()

    def truth_track_params(self, assign, mask, labels, reg):
        """
        PLACEHOLDER for a reconstructed estimate: per-prototype parameters and quality labels
        taken from the truth track holding the plurality of that prototype's hits.

        Args:
            assign: (B, N) long, predicted prototype per hit; -1 = unassigned.
            mask: (B, N) bool, True = real hit.
            labels: (B, N) truth track ID per hit.
            reg: (B, N, 8) truth (px, py, pz, vx, vy, vz, q, e) per hit.

        Returns:
            track_reg_result: (B, C, 4) (q/(pT+1), theta, sin phi, cos phi).
            noise_labels: (B, C) long, 1 = noise; empty prototypes are labelled noise.
            valid_tracks: (B, C) long, 1 = production vertex within 1 cm.
        """
        B, C = assign.size(0), self.params.max_gt_classes
        trr = torch.tensor(EMPTY_SLOT_PARAMS).repeat(B, C, 1)
        noise = torch.ones(B, C, dtype=torch.long)
        valid = torch.zeros(B, C, dtype=torch.long)
        for e in range(B):
            real = mask[e].cpu()
            seg = labels[e].cpu()[real].numpy().astype(np.int32)
            proto = assign[e].cpu()[real].numpy()
            if len(seg) == 0:
                continue
            # build_tracks reads reg only for these outputs; features feed points, unused here
            td = build_tracks(np.zeros((len(seg), 4), np.float32), seg,
                              reg[e].cpu()[real].numpy().astype(np.float64), self.noise_pt_threshold)
            col_of = {int(t): k for k, t in enumerate(td["track_ids"])}
            for c in np.unique(proto[proto >= 0]):
                ids, counts = np.unique(seg[proto == c], return_counts=True)
                k = col_of[int(ids[counts.argmax()])]
                trr[e, c] = td["track_reg_result"][0, k]
                noise[e, c] = td["noise_labels"][0, k]
                valid[e, c] = td["valid_tracks"][0, k]
        return trr.to(self.device), noise.to(self.device), valid.to(self.device)

    @torch.no_grad()
    def run_vertexing(self, pretrain=True, max_events=None):
        """
        Backbone -> track head -> VertexHead over the validation loader.

        Args:
            pretrain: True = frozen-backbone features (as trained); False = head on raw points.
            max_events: stop after this many events; None = whole loader.

        Returns:
            dict:
                metrics: compute_vertex_metrics output plus n_events and fit_valid_rate.
                vertex_estimate, vertex_target: (n_events, 3).
                fit_is_valid: (n_events,) bool, fit valid and truth vertex defined.
        """
        self.model.eval()
        amp = torch.cuda.is_available() and self.use_amp
        est, tgt, ok = [], [], []

        for batch in tqdm(self.val_data_loader):
            if len(batch) != 4:
                raise RuntimeError("loader must yield (grouped, label, knearest, reg); "
                                   "set params.return_reg_test = True before launch()")
            grouped, label, _, reg = batch
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c).to(self.device)
            labels, reg = label.to(self.device), reg.to(self.device)
            mask = grouped[..., 0] != PAD_VALUE

            with autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                if pretrain:
                    _, pre_embed, _ = self.model(grouped, return_z=True)
                    out = self.down_model(grouped, torch.stack(pre_embed), pretrain=True, padding_mask=mask)
                else:
                    out = self.down_model(grouped, feature=None)
            class_probs, mask_probs = out["class_probs"].float(), out["mask_probs"].float()
            mask_probs = mask_probs * mask.unsqueeze(-1)

            if self.use_silicon_tpc_mask:
                # reco-level: TPC membership judged from the PREDICTED clusters, not truth
                for e in range(b):
                    real = mask[e]
                    td = {"mask_probs": mask_probs[e:e+1, real]}
                    apply_silicon_tpc_mask(td, grouped[e, real].cpu().numpy(),
                                           mask_probs[e, real].argmax(-1).cpu().numpy())
                    mask_probs[e, real] = td["mask_probs"][0]

            assign = torch.where(mask_probs.amax(-1) > 0, mask_probs.argmax(-1),
                                 torch.full_like(mask, -1, dtype=torch.long))
            if self.track_params_source == "hits":
                hp = track_params_from_hits(grouped, mask_probs, mask, Bz=self.vertex_head.Bz,
                                            scaling_factor=self.vertex_head.scaling_factor,
                                            noise_pt_threshold=self.noise_pt_threshold)
                track_reg_result, noise_labels, valid_tracks = hp["track_reg_result"], hp["noise_labels"], hp["valid_tracks"]
            else:
                track_reg_result, noise_labels, valid_tracks = self.truth_track_params(assign, mask, labels, reg)

            result = self.vertex_head(
                class_probs=class_probs, track_reg_result=track_reg_result, mask_probs=mask_probs,
                points=grouped, padding_mask=mask,
                noise_labels=noise_labels if self.use_quality_mask else None,
                valid_tracks=valid_tracks if self.use_quality_mask else None)

            # padded rows are zeros and would pass get_vertex_label's 1 cm cut; exclude them
            vtx_r = torch.hypot(reg[..., 3], reg[..., 4])
            truth = get_vertex_label(reg, valid_tracks=(vtx_r < 1.0) & mask)
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
        return {"metrics": metrics, "vertex_estimate": est.cpu(), "vertex_target": tgt.cpu(),
                "fit_is_valid": ok.cpu()}
