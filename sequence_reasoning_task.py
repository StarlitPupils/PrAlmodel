# sequence_reasoning_task.py —— Level 3 升级版
import numpy as np
from typing import Tuple, List
import torch

class SequenceReasoningGenerator:
    def __init__(self, seg_length: int = 30, n_segments: int = 4, seed: int = None):
        self.seg_length = seg_length
        self.n_segments = n_segments
        self.rng = np.random.default_rng(seed)

    def _apply_rule(self, start_val, rule_type, rule_param, length):
        seq = np.zeros(length, dtype=np.float32)
        seq[0] = start_val
        for i in range(1, length):
            if rule_type == "linear":
                seq[i] = seq[i-1] + rule_param
            elif rule_type == "multiply":
                seq[i] = seq[i-1] * rule_param
            elif rule_type == "fibonacci":
                seq[i] = seq[i-1] + seq[i-2] if i > 1 else seq[i-1] + rule_param
            elif rule_type == "repeat":
                seq[i] = seq[i % 2]
            elif rule_type == "sinusoidal":
                seq[i] = start_val + rule_param * np.sin(i * 0.3)
            elif rule_type == "polynomial":
                seq[i] = start_val + rule_param * i + 0.1 * i**2
        return seq

    def generate(self):
        total_len = self.seg_length * self.n_segments
        seq = np.zeros(total_len, dtype=np.float32)
        switches = []
        rule_types = ["linear", "multiply", "fibonacci", "repeat", "sinusoidal", "polynomial"]
        current_val = self.rng.uniform(0.3, 0.9)

        for seg_idx in range(self.n_segments):
            start = seg_idx * self.seg_length
            rule = self.rng.choice(rule_types)
            param = self.rng.uniform(0.01, 0.2) if rule in ["linear","polynomial"] else self.rng.uniform(0.5, 1.5)
            segment = self._apply_rule(current_val, rule, param, self.seg_length)
            seq[start:start+self.seg_length] = segment
            current_val = segment[-1]
            if seg_idx > 0:
                switches.append(start)
        # 归一化
        mx = np.abs(seq).max()
        if mx > 0:
            seq = seq / (mx + 1e-8)
        return seq, switches

    def generate_batch(self, batch_size):
        seqs, all_switches = [], []
        for _ in range(batch_size):
            s, sw = self.generate()
            seqs.append(s)
            all_switches.append(sw)
        return torch.from_numpy(np.stack(seqs)), all_switches


class SequenceReasoningAdapter:
    def __init__(self, model, proto_config=None, seg_length=30, n_segments=4, seed=42):
        self.model = model
        self.config = proto_config or get_config()
        self.generator = SequenceReasoningGenerator(seg_length=seg_length, n_segments=n_segments, seed=seed)
        self.total_len = seg_length * n_segments
        self.pr_errors = []
        self.al_errors = []

    def train_epoch(self, n_sequences=20):
        device = next(self.model.parameters()).device
        total_pr_err = 0.0
        total_al_err = 0.0
        total_iz_err = 0.0
        for _ in range(n_sequences):
            seq, switches = self.generator.generate()
            seq_t = torch.from_numpy(seq).float().to(device)
            T = len(seq_t)
            k = 20
            positions = torch.zeros(T, 2, device=device)
            positions[:, 0] = seq_t
            histories = torch.zeros(T, k, 2, device=device)
            for t in range(T):
                start = max(0, t - k)
                end = t
                hist_len = end - start
                if hist_len > 0:
                    histories[t, -hist_len:, 0] = seq_t[start:end]
            targets = {}
            targets['pr'] = torch.zeros(T, 2, device=device)
            targets['pr'][:-1, 0] = seq_t[1:]
            targets['al'] = torch.zeros(T, 2, device=device)
            if T > 10:
                targets['al'][:-10, 0] = seq_t[10:]
            targets['iz'] = torch.zeros(T, 2, device=device)
            if T > 5:
                targets['iz'][:-5, 0] = seq_t[5:]
            loss, errors = self.model.forward_epoch(positions, histories, targets)
            total_pr_err += errors['pr_error']
            total_al_err += errors['al_error']
            total_iz_err += errors['iz_error']
        n = n_sequences
        self.pr_errors.append(total_pr_err / n)
        self.al_errors.append(total_al_err / n)
        return {'pr_error': total_pr_err/n, 'al_error': total_al_err/n, 'iz_error': total_iz_err/n}
