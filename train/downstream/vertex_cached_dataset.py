"""
The track finder's feature cache (scripts/cache_features.py, cached_dataset.py; neither modified) for vertex
training: each cached event with its truth reg from the live dataset.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cached_dataset import CachedFeatureDataset, CachedCollator


class CachedVertexDataset(Dataset):
    """
    (points, seg_target, features, reg) per event: the first three from a feature cache, reg from the live dataset
    at the same index. Raises where the two disagree on the event's points.

    Args:
        cache_dir: directory written by scripts/cache_features.py.
        live: the TPCBatchDataset the cache was built from (same split and settings) with return_reg; its length
            limits the events used.
    """

    def __init__(self, cache_dir, live):
        self.cache, self.live = CachedFeatureDataset(cache_dir, limit_size=len(live)), live

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        points, target, features = self.cache[i]
        live_points, _, _, reg = self.live[i]
        live_points = live_points.reshape(-1, live_points.shape[-1])
        if live_points.shape != points.shape or not torch.equal(live_points.to(points.dtype), points):
            raise ValueError(f"event {i}: the points of {self.cache.cache_dir} are not the live dataset's (another "
                             "split, event selection or preprocessing, or random chunking): not this data's cache")
        return points, target, features, reg.reshape(-1, reg.shape[-1])


class CachedVertexCollator(CachedCollator):
    """CachedCollator with reg (B, N, 8), padded as the points: (points, targets, features, reg)."""

    def __call__(self, batch):
        points, targets, features = super().__call__([b[:3] for b in batch])
        return points, targets, features, torch.stack(
            [F.pad(b[3], (0, 0, 0, points.size(1) - b[3].size(0)), value=self.pad_val) for b in batch])
