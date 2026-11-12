"""
Learned vertex head: a set-prediction head on the backbone features, with vertices as the slots.

Shapes: B events, N hits, V vertex slots, L backbone feature layers, D feature width.
"""
import torch
import torch.nn as nn

from fm4npp.models.embed import EmbedderAdd
from trackinghead import FFNBlock, RefinementLayer


class LearnedVertexHead(nn.Module):
    """
    Vertex queries refined against the per-hit backbone features; each slot predicts whether it is a
    vertex, which hits belong to it, and its position with an uncertainty.

    Args:
        input_dim: backbone feature width D.
        num_feature_layers: number of backbone feature layers L combined by learned weights.
        pe_method: coordinate embedding of the points ('none', 'ff', 'nerf', 'cpe').
        n_vertex_slots: number of vertex queries V.
        embed_dim: working width of the head.
        num_heads: attention heads of the refinement layers.
        ffn_dim: hidden width of the feed-forward blocks.
        num_refinement_layers: number of query refinement layers.
        position_scale_xy: cm, multiplies the raw x and y position outputs.
        position_scale_z: cm, multiplies the raw z position output.
        min_sigma_xy: cm, floor of the predicted x and y position uncertainty, added in quadrature (0: none).
        min_sigma_z: cm, the same for z.
        softmax_mask: hit-to-slot probabilities by softmax over slots instead of a per-slot sigmoid.
        do_masked_attn: bias each slot's cross-attention towards the hits of its current mask.
        attn_mask_eps: added to the mask probability before the log of the attention bias.
        dropout: dropout of the attention and prediction layers.
    """

    def __init__(self, input_dim, num_feature_layers, pe_method="nerf", n_vertex_slots=10, embed_dim=256, num_heads=4,
                 ffn_dim=512, num_refinement_layers=2, position_scale_xy=1.0, position_scale_z=10.0,
                 min_sigma_xy=0.0, min_sigma_z=0.0, softmax_mask=False, do_masked_attn=True, attn_mask_eps=1e-6, dropout=0.0):
        super().__init__()
        self.n_vertex_slots, self.num_heads = n_vertex_slots, num_heads
        self.softmax_mask, self.do_masked_attn, self.attn_mask_eps = softmax_mask, do_masked_attn, attn_mask_eps

        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))
        self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, embed_dim))
        self.embedder = EmbedderAdd(pe_method=pe_method, embed_dim=embed_dim)
        self.output_layer = FFNBlock(embed_dim, ffn_dim)

        self.query_embed = nn.Embedding(n_vertex_slots, embed_dim)
        self.query_pos_embed = nn.Embedding(n_vertex_slots, embed_dim)
        self.refinement_layers = nn.ModuleList(
            RefinementLayer(embed_dim, num_heads, ffn_dim, dropout=dropout) for _ in range(num_refinement_layers))

        def mlp(n_out):
            return nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(embed_dim, n_out))
        self.class_mlp, self.mask_mlp = mlp(2), mlp(embed_dim)
        self.position_mlp, self.log_variance_mlp = mlp(3), mlp(3)
        self.register_buffer("position_scale", torch.tensor([position_scale_xy, position_scale_xy, position_scale_z]))
        # a setting, not a weight: not in checkpoints
        self.register_buffer("log_min_variance", 2 * torch.tensor([min_sigma_xy, min_sigma_xy, min_sigma_z]).log(),
                             persistent=False)

    def predict(self, queries, hit_features):
        """
        queries (V, B, E), hit_features (B, N, E) -> the output dict of forward without aux_list.
        """
        queries = queries.transpose(0, 1)
        similarity = torch.einsum("bne,bve->bnv", hit_features, self.mask_mlp(queries))
        return {
            "vertex_class_probs": torch.softmax(self.class_mlp(queries), dim=-1),
            "vertex_mask_probs": torch.softmax(similarity, dim=-1) if self.softmax_mask else torch.sigmoid(similarity),
            "vertex_position": self.position_mlp(queries) * self.position_scale,
            "vertex_log_variance": torch.logaddexp(
                self.log_variance_mlp(queries) + 2 * torch.log(self.position_scale), self.log_min_variance),
        }

    def forward(self, x, feature, padding_mask, pretrain=True):
        """
        Args:
            x: (B, N, 4) normalized points (energy, x, y, z).
            feature: (L, B, N, D) backbone layer features.
            padding_mask: (B, N) bool, True for real hits.
            pretrain: must be True; the head has no path without backbone features.

        Returns a dict:
            vertex_class_probs: (B, V, 2) probabilities (empty, real).
            vertex_mask_probs: (B, N, V) probability that the hit belongs to the slot.
            vertex_position: (B, V, 3) cm.
            vertex_log_variance: (B, V, 3) log of the position variance (with the floor), cm^2.
            aux_list: the same four keys before each refinement layer.
        """
        if not pretrain:
            raise NotImplementedError("LearnedVertexHead needs the backbone features (pretrain=True)")
        hit_features = self.output_layer(
            self.input_proj(torch.einsum("lbnd,l->bnd", feature,
                                         torch.softmax(self.weighted_avg_weights, dim=0).to(feature.dtype)))
            + self.embedder(x.masked_fill(~padding_mask.unsqueeze(-1), 0.0))[0])
        context = hit_features.transpose(0, 1)

        slots = torch.arange(self.n_vertex_slots, device=x.device)
        queries = self.query_embed(slots).unsqueeze(1).expand(-1, x.size(0), -1)
        query_pos = self.query_pos_embed(slots).unsqueeze(1).expand(-1, x.size(0), -1)

        out, aux_list = self.predict(queries, hit_features), []
        for layer in self.refinement_layers:
            aux_list.append(out)
            queries = layer(queries, context, query_pos, padding_mask, attn_mask=(
                torch.log(out["vertex_mask_probs"].transpose(1, 2) + self.attn_mask_eps)
                .masked_fill(~padding_mask.unsqueeze(1), -1e9)
                .repeat_interleave(self.num_heads, dim=0)) if self.do_masked_attn else None)
            out = self.predict(queries, hit_features)
        return {**out, "aux_list": aux_list}


_F, _I, _B, _NAN = torch.float64, torch.int64, torch.bool, float("nan")
# VertexHead.forward's keys: (shape after B, dtype, value when not filled); T track slots, V vertices, K SVs
_EMPTY = {
    "track_parameters": (("T", 6), _F, _NAN), "track_position": (("T", 3), _F, _NAN),
    "track_direction": (("T", 3), _F, _NAN), "track_weight": (("T",), _F, 0.0),
    "track_quality_mask": (("T",), _B, False), "track_sigma": (("T", 2), _F, _NAN),
    "track_input_mask": (("T",), _B, False),
    "sv_estimate": (("K", 3), _F, _NAN), "sv_is_valid": (("K",), _B, False),
    "sv_covariance": (("K", 3, 3), _F, _NAN), "sv_chi2": (("K",), _F, _NAN), "sv_chi2_per_dof": (("K",), _F, _NAN),
    "sv_displacement": (("K",), _F, _NAN), "sv_decay_length_significance": (("K",), _F, _NAN),
    "sv_momentum": (("K", 3), _F, _NAN), "sv_dira": (("K",), _F, _NAN), "sv_n_tracks": (("K",), _I, 0),
    "sv_track_mask": (("K", "T"), _B, False), "sv_track_weight": (("K", "T"), _F, 0.0),
    "sv_pair_dca": (("K",), _F, _NAN), "displaced_mask": (("T",), _B, False), "displaced_weight": (("T",), _F, 0.0),
    "vertex_estimate": ((3,), _F, _NAN), "fit_is_valid": ((), _B, False), "vertex_from_fallback": ((), _B, False),
    "vertex_fallback_step": ((), _I, -1), "vertex_covariance": ((3, 3), _F, _NAN), "chi_square": ((), _F, _NAN),
    "vertex_ndf": ((), _F, _NAN), "track_vertex_weight": (("T",), _F, 0.0), "vertex_n_tracks": ((), _I, 0),
    "track_chi2": (("T",), _F, _NAN), "track_dca": (("T",), _F, _NAN),
    "track_closest_approach_point": (("T", 3), _F, _NAN),
    "primary_vertices": (("V", 3), _F, _NAN), "primary_vertex_valid": (("V",), _B, False),
    "primary_vertex_n_tracks": (("V",), _I, 0), "track_primary_vertex": (("T",), _I, -1),
}


def learned_outputs(raw, padding_mask, max_vertices, n_track_slots, max_sv_per_event, real_threshold):
    """
    LearnedVertexHead's output as VertexHead.forward's dict (same keys, shapes and dtypes).

    A slot is a vertex when P(real) >= real_threshold. The vertices are ordered by P(real) x the slot's hits (the
    sum of its mask probabilities over the real hits), highest first; the PV is the first. The first max_vertices
    are kept. vertex_covariance is diagonal, exp(vertex_log_variance); vertex_fallback_step is 0 with a PV, else
    -1. The track, SV, chi-square and ndf keys are not filled (NaN, False, 0, track_primary_vertex -1).

    Args:
        raw: LearnedVertexHead.forward's dict.
        padding_mask: (B, N) bool, True for real hits.
        max_vertices: V of primary_vertices.
        n_track_slots: T of the track keys.
        max_sv_per_event: K of the sv keys.
        real_threshold: least P(real) of a vertex.
    """
    sizes = {"T": n_track_slots, "V": max_vertices, "K": max_sv_per_event}
    out = {key: torch.full((padding_mask.shape[0], *(sizes.get(n, n) for n in shape)), fill, dtype=dtype,
                           device=padding_mask.device) for key, (shape, dtype, fill) in _EMPTY.items()}
    p_real = raw["vertex_class_probs"][..., 1]
    real = p_real >= real_threshold
    order = (p_real * (raw["vertex_mask_probs"] * padding_mask.unsqueeze(-1)).sum(dim=1)).masked_fill(
        ~real, -1.0).argsort(dim=1, descending=True)[:, :max_vertices]
    kept, n = real.gather(1, order), order.shape[1]
    ordered = lambda v: v.double().gather(1, order.unsqueeze(-1).expand(-1, -1, 3)).masked_fill(  # noqa: E731
        ~kept.unsqueeze(-1), torch.nan)
    out["primary_vertices"][:, :n], out["primary_vertex_valid"][:, :n] = ordered(raw["vertex_position"]), kept
    out["vertex_estimate"], out["fit_is_valid"] = out["primary_vertices"][:, 0], kept[:, 0]
    out["vertex_covariance"] = torch.diag_embed(ordered(raw["vertex_log_variance"])[:, 0].exp()).masked_fill(
        ~kept[:, :1].unsqueeze(-1), torch.nan)
    out["vertex_fallback_step"] = kept[:, 0].long() - 1
    return out


class LearnedVertexing(nn.Module):
    """
    The learned vertexing method: LearnedVertexHead, its output as VertexHead.forward's dict.

    Args:
        head: the LearnedVertexHead.
        n_track_slots: T, the track finder's slots (the shape of the track keys).
        max_vertices: V, the most primary vertices returned.
        max_sv_per_event: K, the shape of the sv keys.
        real_threshold: least P(real) of a vertex.
    """

    def __init__(self, head, n_track_slots, max_vertices=10, max_sv_per_event=6, real_threshold=0.5):
        super().__init__()
        self.head = head
        self.n_track_slots, self.max_vertices, self.max_sv_per_event = n_track_slots, max_vertices, max_sv_per_event
        self.real_threshold = real_threshold

    def forward(self, x, feature, padding_mask):
        """
        x (B, N, 4), feature (L, B, N, D), padding_mask (B, N) as LearnedVertexHead.forward.

        Returns (outputs, raw): VertexHead.forward's dict (learned_outputs) and the head's own dict (for the loss).
        """
        raw = self.head(x, feature, padding_mask)
        return learned_outputs(raw, padding_mask, self.max_vertices, self.n_track_slots, self.max_sv_per_event,
                               self.real_threshold), raw
