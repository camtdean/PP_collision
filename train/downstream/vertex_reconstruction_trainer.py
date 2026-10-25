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
"""

import torch
from contextlib import nullcontext
from torch.amp import autocast
from tqdm import tqdm

from train.downstream import track_finding_trainer
from train.downstream.vertexhead import VertexHead
from trackinghead import MambaAttentionHead
from downstream_util import get_vertex_label, get_silicon_tpc_match_mask
from loss import compute_vertex_metrics

PAD_VALUE = -100

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

        self.use_silicon_tpc_mask = bool(getattr(self.params, "vertex_silicon_tpc_mask", False))
        self.vertex_head = VertexHead(
            learn_weights=False,
            use_helix=bool(getattr(self.params, "vertex_use_helix", True)),
            Bz=float(getattr(self.params, "vertex_bz", 1.4)),
            scaling_factor=float(getattr(self.params, "vertex_scaling_factor", 100.0)),
            noise_pt_threshold=float(getattr(self.params, "vertex_noise_pt_threshold", 0.25)),
        ).to(self.device).eval()

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
            grouped, _, _, reg = batch
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

            if self.use_silicon_tpc_mask:
                # reco-level: TPC membership judged from the PREDICTED clusters, not truth
                for e in range(b):
                    real = mask[e]
                    info = get_silicon_tpc_match_mask(grouped[e, real].cpu().numpy(),
                                                      mask_probs[e, real].argmax(-1).cpu().numpy())
                    drop = ~torch.as_tensor(info["keep_mask"], dtype=torch.bool, device=self.device)
                    hit_index = real.nonzero(as_tuple=True)[0][drop]
                    mask_probs[e, hit_index] = 0.0

            result = self.vertex_head(
                class_probs=class_probs, mask_probs=mask_probs,
                points=grouped, padding_mask=mask)

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
