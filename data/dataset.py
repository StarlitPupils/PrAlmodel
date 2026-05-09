"""
E:\PrAlmodel\data\dataset.py
Sliding-window trajectory dataset with balanced sampling.
Standalone test: python data/dataset.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import sys
import os
sys.path.insert(0, "E:/PrAlmodel")
from config.task_config import TaskConfig, get_task_config
from data.trajectory_generator import LissajousGenerator


class TrajectoryWindowDataset(Dataset):
    """
    Balanced sliding-window dataset for Lissajous mutation detection.

    Design rationale:
      - Lissajous mutations are LOCAL events (instantaneous parameter jumps).
        A sliding window decomposes the global detection problem into local
        anomaly classification: "does this 32-step snippet contain a jump?"
      - Feature engineering uses velocity (dx,dy) and acceleration (ddx,ddy)
        because:
          * amplitude jump → coordinate discontinuity → dx/dy spike
          * frequency jump → angular velocity step → ddx/ddy δ-like spike (strongest)
          * phase jump     → trajectory kink/flip    → dx/dy direction change
      - Severe class imbalance (~0.8% positive) is handled by:
          a) oversampling positive windows (repeat pos_oversample times)
          b) stochastic negative subsampling (keep only neg_keep_ratio)
          c) weighted loss in training (handled by the model, not here)
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        mutation_flags_list: List[np.ndarray],
        window_size: int = 32,
        pos_oversample: int = 10,
        neg_keep_ratio: float = 0.20,
        seed: Optional[int] = None,
    ):
        self.window_size = window_size
        self.feature_dim = 6  # x, y, dx, dy, ddx, ddy
        self.pos_oversample = pos_oversample
        self.neg_keep_ratio = neg_keep_ratio
        self.rng = np.random.default_rng(seed)

        self.X = []
        self.y = []

        for traj, flags in zip(trajectories, mutation_flags_list):
            features = self._compute_features(traj)
            self._extract_windows(features, flags)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        n_pos = int(self.y.sum())
        n_neg = len(self.y) - n_pos
        total = len(self.y)
        print(f"  Dataset: {total} samples | "
              f"pos={n_pos} ({100*n_pos/total:.1f}%) | "
              f"neg={n_neg} ({100*n_neg/total:.1f}%)")

    def _compute_features(self, traj: np.ndarray) -> np.ndarray:
        """
        Compute 6-dim kinematics features from 2-D normalised trajectory.

        Dim | Name | Computation        | Physical meaning
        ----|------|--------------------|------------------------
          0 | x    | traj[:,0]          | normalised x-position
          1 | y    | traj[:,1]          | normalised y-position
          2 | dx   | x[t] - x[t-1]      | x-velocity
          3 | dy   | y[t] - y[t-1]      | y-velocity
          4 | ddx  | dx[t] - dx[t-1]    | x-acceleration (key for frequency jumps)
          5 | ddy  | dy[t] - dy[t-1]    | y-acceleration (key for frequency jumps)
        """
        n = len(traj)
        F = np.zeros((n, 6), dtype=np.float32)
        F[:, 0] = traj[:, 0]
        F[:, 1] = traj[:, 1]
        F[1:, 2] = traj[1:, 0] - traj[:-1, 0]
        F[1:, 3] = traj[1:, 1] - traj[:-1, 1]
        F[2:, 4] = F[2:, 2] - F[1:-1, 2]
        F[2:, 5] = F[2:, 3] - F[1:-1, 3]
        return F

    def _extract_windows(self, features: np.ndarray, flags: np.ndarray):
        """
        Slide a window across the trajectory and collect balanced samples.

        Window layout (window_size = 32, even):
            indices  0  1  ...  15  [16]  17  ...  30  31
                                  ^-- centre step = the step we predict

        For a centre step t, the window is features[t-16 : t+16].
        """
        n = len(features)
        half = self.window_size // 2
        valid_centres = np.arange(half, n - half)

        for centre in valid_centres:
            window = features[centre - half : centre + half]   # (32, 6)
            label = int(flags[centre])

            if label == 1:
                for _ in range(self.pos_oversample):
                    self.X.append(window)
                    self.y.append(1)
            else:
                if self.rng.random() < self.neg_keep_ratio:
                    self.X.append(window)
                    self.y.append(0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def create_dataloaders(
    n_train: int = 200,
    n_val: int = 50,
    n_test: int = 50,
    steps_per_traj: int = 2000,
    window_size: int = 32,
    batch_size: int = 64,
    pos_oversample: int = 10,
    neg_keep_ratio: float = 0.20,
    seed: int = 42,
    config: Optional[TaskConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    One-call factory: generate trajectories → build datasets → wrap DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = get_task_config()

    gen = LissajousGenerator(config, seed=seed)

    def _make_set(n_traj: int, tag: str, seed_off: int):
        trajs, flags_list = [], []
        for i in range(n_traj):
            traj, fl, _ = gen.generate_trajectory(steps_per_traj)
            traj_norm, _ = gen.normalize_trajectory(traj)
            trajs.append(traj_norm)
            flags_list.append(fl)
        ds = TrajectoryWindowDataset(
            trajs, flags_list,
            window_size=window_size,
            pos_oversample=pos_oversample,
            neg_keep_ratio=neg_keep_ratio,
            seed=seed + seed_off,
        )
        shuffle = (tag == "train")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        print(f"  [{tag.upper():>5}] {len(ds):>6} samples → {len(dl):>4} batches "
              f"(batch_size={batch_size})")
        return dl

    print("=" * 60)
    print("  CREATING DATALOADERS")
    print("=" * 60)
    print(f"  config : {n_train} train / {n_val} val / {n_test} test trajectories")
    print(f"  window : {window_size}  |  batch : {batch_size}")
    print(f"  pos_oversample : {pos_oversample}  |  neg_keep : {neg_keep_ratio}")
    print("-" * 60)

    train_dl = _make_set(n_train, "train", seed_off=0)
    val_dl   = _make_set(n_val,   "val",   seed_off=1000)
    test_dl  = _make_set(n_test,  "test",  seed_off=2000)

    print("=" * 60)
    return train_dl, val_dl, test_dl


# ==================== SELF-TEST ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  DATASET MODULE — SELF-TEST")
    print("=" * 60)

    config = get_task_config()
    gen = LissajousGenerator(config, seed=42)

    # ── Generate 3 mini-trajectories ──
    print("\n  Generating 3 trajectories ...")
    trajs, flags_list = [], []
    for i in range(3):
        traj, fl, info = gen.generate_trajectory(2000)
        traj_norm, _ = gen.normalize_trajectory(traj)
        trajs.append(traj_norm)
        flags_list.append(fl)
        print(f"    traj[{i}]  shape={traj.shape}  mutations={fl.sum()}")

    # ── Build dataset ──
    print("\n  Building dataset (window=32, oversample=5, neg_keep=0.30) ...")
    ds = TrajectoryWindowDataset(
        trajs, flags_list,
        window_size=32,
        pos_oversample=5,
        neg_keep_ratio=0.30,
        seed=42,
    )

    print(f"\n  X shape : {ds.X.shape}")
    print(f"  y shape : {ds.y.shape}")
    counts = np.bincount(ds.y.astype(int))
    print(f"  y dist  : neg={counts[0]}  pos={counts[1]}")

    # ── Inspect one positive sample ──
    pos_idx = np.where(ds.y == 1)[0]
    if len(pos_idx):
        i = pos_idx[0]
        print(f"\n  Positive sample [idx={i}]")
        print(f"    X shape       : {ds.X[i].shape}")
        print(f"    centre (t=16) : x={ds.X[i,16,0]:.4f}  y={ds.X[i,16,1]:.4f}")
        print(f"    dx peak       : {np.abs(ds.X[i,:,2]).max():.4f}")
        print(f"    ddx peak      : {np.abs(ds.X[i,:,4]).max():.4f}")

    # ── Inspect one negative sample ──
    neg_idx = np.where(ds.y == 0)[0]
    if len(neg_idx):
        i = neg_idx[0]
        print(f"\n  Negative sample [idx={i}]")
        print(f"    centre (t=16) : x={ds.X[i,16,0]:.4f}  y={ds.X[i,16,1]:.4f}")
        print(f"    dx peak       : {np.abs(ds.X[i,:,2]).max():.4f}")
        print(f"    ddx peak      : {np.abs(ds.X[i,:,4]).max():.4f}")

    # ── Test DataLoader ──
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    bx, by = next(iter(dl))
    print(f"\n  Batch X : {bx.shape}   Batch y : {by.shape}")
    print(f"  Batch positive count : {int(by.sum())} / {len(by)}")

    print("\n" + "=" * 60)
    print("  SELF-TEST PASSED")
    print("=" * 60)
