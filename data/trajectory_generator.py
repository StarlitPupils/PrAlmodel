"""
E:\PrAlmodel\data\trajectory_generator.py
Lissajous curve mutation tracking -- data generator
Standalone test: python trajectory_generator.py
Outputs:
    - data_sample.png           (4-panel visualization)
    - trajectory_data.csv       (step, x, y, x_norm, y_norm, is_mutation)
    - mutation_info.txt         (mutation event details)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import os
sys.path.insert(0, "E:/PrAlmodel")
from config.task_config import TaskConfig, get_task_config


@dataclass
class LissajousParams:
    """One set of Lissajous curve parameters"""
    Ax: float
    Ay: float
    ax: int
    ay: int
    phix: float
    phiy: float

    def copy(self) -> "LissajousParams":
        return LissajousParams(self.Ax, self.Ay, self.ax, self.ay, self.phix, self.phiy)


class LissajousGenerator:
    """
    Lissajous curve mutation trajectory generator.

    Usage:
        gen = LissajousGenerator(config)
        trajectory, mutation_flags, mutation_info = gen.generate_trajectory(2000)
    """

    def __init__(self, config: Optional[TaskConfig] = None, seed: Optional[int] = None):
        self.config = config or get_task_config()
        self.rng = np.random.default_rng(seed)

    def _sample_params(self, ood: bool = False) -> LissajousParams:
        cfg = self.config
        if ood:
            amp_lo, amp_hi = cfg.ood_amplitude_range
            freq_opts = cfg.ood_frequency_options
        else:
            amp_lo, amp_hi = cfg.amplitude_range
            freq_opts = cfg.frequency_options

        return LissajousParams(
            Ax=self.rng.uniform(amp_lo, amp_hi),
            Ay=self.rng.uniform(amp_lo, amp_hi),
            ax=int(self.rng.choice(freq_opts)),
            ay=int(self.rng.choice(freq_opts)),
            phix=self.rng.uniform(*cfg.phase_range),
            phiy=self.rng.uniform(*cfg.phase_range),
        )

    def _mutate_params(self, params: LissajousParams, ood: bool = False) -> Tuple[LissajousParams, dict]:
        cfg = self.config
        new = params.copy()
        mutation_type = self.rng.choice(["frequency", "amplitude", "phase"])

        if mutation_type == "frequency":
            if ood:
                freq_opts = cfg.ood_frequency_options
            else:
                freq_opts = cfg.frequency_options
            if self.rng.random() < 0.5:
                old_val = new.ax
                candidates = [f for f in freq_opts if f != old_val]
                new.ax = int(self.rng.choice(candidates)) if candidates else old_val
                detail = {"type": "frequency", "param": "ax", "old": old_val, "new": new.ax}
            else:
                old_val = new.ay
                candidates = [f for f in freq_opts if f != old_val]
                new.ay = int(self.rng.choice(candidates)) if candidates else old_val
                detail = {"type": "frequency", "param": "ay", "old": old_val, "new": new.ay}

        elif mutation_type == "amplitude":
            scale = self.rng.uniform(*cfg.amplitude_mutation_scale)
            if self.rng.random() < 0.5:
                old_val = new.Ax
                new.Ax = old_val * scale
                detail = {"type": "amplitude", "param": "Ax", "old": old_val, "new": new.Ax, "scale": scale}
            else:
                old_val = new.Ay
                new.Ay = old_val * scale
                detail = {"type": "amplitude", "param": "Ay", "old": old_val, "new": new.Ay, "scale": scale}

        else:  # phase
            offset = self.rng.uniform(-cfg.phase_mutation_max, cfg.phase_mutation_max)
            if self.rng.random() < 0.5:
                old_val = new.phix
                new.phix = (old_val + offset) % (2 * np.pi)
                detail = {"type": "phase", "param": "phix", "old": old_val, "new": new.phix, "offset": offset}
            else:
                old_val = new.phiy
                new.phiy = (old_val + offset) % (2 * np.pi)
                detail = {"type": "phase", "param": "phiy", "old": old_val, "new": new.phiy, "offset": offset}

        return new, detail

    def _step(self, t: int, params: LissajousParams) -> np.ndarray:
        dt = self.config.delta_t
        x = params.Ax * np.sin(params.ax * t * dt + params.phix)
        y = params.Ay * np.sin(params.ay * t * dt + params.phiy)
        return np.array([x, y], dtype=np.float32)

    def _schedule_mutations(self, n_steps: int) -> List[int]:
        cfg = self.config
        mutations = []
        t = self.rng.integers(*cfg.mutation_interval)
        while t < n_steps - 10:
            mutations.append(t)
            t += self.rng.integers(*cfg.mutation_interval)
        return mutations

    def generate_trajectory(
        self, n_steps: int, ood: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        params = self._sample_params(ood=ood)
        mutation_schedule = set(self._schedule_mutations(n_steps))

        trajectory = np.zeros((n_steps, 2), dtype=np.float32)
        mutation_flags = np.zeros(n_steps, dtype=bool)
        mutation_info = []

        for t in range(n_steps):
            if t in mutation_schedule:
                params, detail = self._mutate_params(params, ood=ood)
                mutation_flags[t] = True
                mutation_info.append({"step": t, **detail})

            trajectory[t] = self._step(t, params)

        return trajectory, mutation_flags, mutation_info

    def normalize_trajectory(
        self, trajectory: np.ndarray, stats: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if stats is None:
            n_buffer = min(self.config.normalization_buffer, len(trajectory))
            buffer = trajectory[:n_buffer]
            stats = {
                "mean": buffer.mean(axis=0),
                "std": buffer.std(axis=0) + 1e-8,
            }
        normalized = (trajectory - stats["mean"]) / stats["std"]
        return normalized, stats


# ==================== STANDALONE TEST ====================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")  # Use TkAgg for Windows GUI; fallback to Agg if headless
    import matplotlib.pyplot as plt

    OUTPUT_DIR = "E:/PrAlmodel"

    config = TaskConfig()
    gen = LissajousGenerator(config, seed=42)

    # Generate one trajectory
    traj, flags, info = gen.generate_trajectory(2000)
    traj_norm, norm_stats = gen.normalize_trajectory(traj)

    print("=" * 60)
    print("  LISSAJOUS TRAJECTORY GENERATOR -- SELF-TEST")
    print("=" * 60)
    print(f"  Trajectory shape    : {traj.shape}")
    print(f"  Mutation events     : {flags.sum()}")
    print(f"  Normalization mean  : {norm_stats['mean']}")
    print(f"  Normalization std   : {norm_stats['std']}")
    print("-" * 60)
    print("  First 10 mutation events:")
    for i, m in enumerate(info[:10]):
        if "scale" in m:
            print(f"    [{i}] step={m['step']:>6}  type={m['type']:<10}  param={m['param']}  "
                  f"old={m['old']:.3f} -> new={m['new']:.3f}  (scale={m['scale']:.3f})")
        elif "offset" in m:
            print(f"    [{i}] step={m['step']:>6}  type={m['type']:<10}  param={m['param']}  "
                  f"old={m['old']:.3f} -> new={m['new']:.3f}  (offset={m['offset']:.3f})")
        else:
            print(f"    [{i}] step={m['step']:>6}  type={m['type']:<10}  param={m['param']}  "
                  f"old={m['old']} -> new={m['new']}")
    print("=" * 60)

    # ==================== OUTPUT 1: CSV data file ====================
    csv_path = os.path.join(OUTPUT_DIR, "trajectory_data.csv")
    mutation_steps_set = set(info_item["step"] for info_item in info)
    with open(csv_path, "w") as f:
        f.write("step,x_raw,y_raw,x_norm,y_norm,is_mutation\n")
        for t in range(len(traj)):
            f.write(f"{t},{traj[t,0]:.6f},{traj[t,1]:.6f},"
                    f"{traj_norm[t,0]:.6f},{traj_norm[t,1]:.6f},"
                    f"{int(t in mutation_steps_set)}\n")
    print(f"\n  [OK] CSV saved to: {csv_path}")

    # ==================== OUTPUT 2: Mutation info TXT ====================
    txt_path = os.path.join(OUTPUT_DIR, "mutation_info.txt")
    with open(txt_path, "w") as f:
        f.write("Mutation Event Details\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total mutations: {len(info)}\n")
        f.write("-" * 60 + "\n")
        for i, m in enumerate(info):
            f.write(f"[{i:>3}] step={m['step']:>6}  type={m['type']:<10}  param={m['param']}\n")
            f.write(f"      old={m['old']}  ->  new={m['new']}\n")
            if "scale" in m:
                f.write(f"      scale={m['scale']:.4f}\n")
            if "offset" in m:
                f.write(f"      offset={m['offset']:.4f}\n")
    print(f"  [OK] TXT saved to: {txt_path}")

    # ==================== OUTPUT 3: PNG visualization ====================
    mutation_steps = np.where(flags)[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lissajous Mutation Trajectory -- Sample (seed=42)", fontsize=14, fontweight="bold")

    # Panel 1: Full trajectory with mutation markers
    ax = axes[0, 0]
    ax.plot(traj_norm[:, 0], traj_norm[:, 1], linewidth=0.5, alpha=0.7, color="steelblue")
    ax.scatter(traj_norm[mutation_steps, 0], traj_norm[mutation_steps, 1],
               c='red', s=20, alpha=0.8, label=f'Mutations (n={len(mutation_steps)})')
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Full Trajectory (normalized)")
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 2: x coordinate time series
    ax = axes[0, 1]
    ax.plot(traj_norm[:, 0], linewidth=0.5, label="x", color="steelblue")
    ax.plot(traj_norm[:, 1], linewidth=0.5, label="y", color="darkorange", alpha=0.7)
    for step in mutation_steps:
        ax.axvspan(step, min(step + 10, 2000), color='red', alpha=0.08)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized coordinate")
    ax.set_title("x, y Time Series (red zones = 10 steps post-mutation)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Zoom-in around first mutation
    ax = axes[1, 0]
    if len(info) > 0:
        first_mutation = info[0]["step"]
        window_start = max(0, first_mutation - 30)
        window_end = min(2000, first_mutation + 30)
        t_window = np.arange(window_start, window_end)
        ax.plot(t_window, traj_norm[window_start:window_end, 0], label="x", linewidth=1.2, color="steelblue")
        ax.plot(t_window, traj_norm[window_start:window_end, 1], label="y", linewidth=1.2, color="darkorange")
        ax.axvline(first_mutation, color='red', linestyle='--', linewidth=2,
                   label=f"Mutation @ step {first_mutation}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalized coordinate")
        ax.set_title(f"Zoom-in: step {first_mutation} +/- 30")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 4: Mutation type distribution
    ax = axes[1, 1]
    type_counts = {"frequency": 0, "amplitude": 0, "phase": 0}
    for m in info:
        type_counts[m["type"]] += 1
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(type_counts.keys(), type_counts.values(), color=colors, edgecolor='white')
    for bar, count in zip(bars, type_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                str(count), ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Mutation Type Distribution")
    ax.set_ylim(0, max(type_counts.values()) * 1.25)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "data_sample.png")
    plt.savefig(png_path, dpi=150)
    print(f"  [OK] PNG saved to: {png_path}")

    # Try to show; gracefully skip if no display available
    try:
        plt.show()
    except Exception:
        print("  [INFO] plt.show() skipped (no display available). Open the PNG manually.")
