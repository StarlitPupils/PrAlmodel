# anomaly_sequence_task.py —— Level 4: 规律中注入反规律异常点
import numpy as np
from typing import Tuple, List
import torch

class AnomalySequenceGenerator:
    """
    生成带异常点的规律序列。
    例: [+2, +2, +2, +2, ×3, +2, +2, +2, +2]
                      ↑ 异常: Pr 被欺骗, Al 从远处识别
    """
    def __init__(self, length: int = 80, anomaly_prob: float = 0.05, seed: int = None):
        self.length = length
        self.anomaly_prob = anomaly_prob
        self.rng = np.random.default_rng(seed)

    def generate(self):
        seq = np.zeros(self.length, dtype=np.float32)
        anomaly_positions = []
        anomalies = []

        # 主规则: 线性 +delta, 每步可能有小波动
        base_delta = self.rng.uniform(0.05, 0.15)
        current = self.rng.uniform(0.3, 0.7)
        seq[0] = current

        for i in range(1, self.length):
            if self.rng.random() < self.anomaly_prob:
                # 注入异常: 跳跃、反向、乘法
                anomaly_type = self.rng.choice(["jump", "reverse", "multiply"])
                if anomaly_type == "jump":
                    current = current + self.rng.uniform(-0.5, 0.5)
                elif anomaly_type == "reverse":
                    current = current - base_delta * self.rng.uniform(5, 10)
                else:
                    current = current * self.rng.uniform(0.3, 2.0)
                anomaly_positions.append(i)
                anomalies.append(anomaly_type)
            else:
                current = current + base_delta + self.rng.uniform(-0.02, 0.02)
            seq[i] = current

        # 归一化
        mx = np.abs(seq).max()
        if mx > 0:
            seq = seq / (mx + 1e-8)
        return seq, anomaly_positions, anomalies


class AnomalySequenceAdapter:
    def __init__(self, model, proto_config=None, length=80, anomaly_prob=0.05, seed=42):
        self.model = model
        self.config = proto_config or get_config()
        self.generator = AnomalySequenceGenerator(length=length, anomaly_prob=anomaly_prob, seed=seed)
        self.length = length
        self.pr_errors = []
        self.al_errors = []

    def train_epoch(self, n_sequences=30):
        device = next(self.model.parameters()).device
        total_pr_err = 0.0
        total_al_err = 0.0
        total_iz_err = 0.0
        total_anomalies = 0

        for _ in range(n_sequences):
            seq, anom_pos, anom_types = self.generator.generate()
            total_anomalies += len(anom_pos)
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
        return {'pr_error': total_pr_err/n, 'al_error': total_al_err/n, 'iz_error': total_iz_err/n,
                'anomalies_per_seq': total_anomalies / n}
