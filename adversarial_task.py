# adversarial_task.py —— 对抗模式切换任务（专门压测 IZ 生长）
import numpy as np
import torch

class AdversarialTaskAdapter:
    """
    前半段: 周期振荡 [+0.1, -0.1, +0.1, -0.1, ...]
    后半段: 突然切换为线性 [+0.05, +0.05, +0.05, ...]
    
    切换后 Pr 快速适应线性, Al 仍锚定振荡 → 持久冲突 → IZ 多次生长
    """
    def __init__(self, model, proto_config=None, osc_len=15, linear_len=25, seed=42):
        self.model = model
        self.config = proto_config or get_config()
        self.osc_len = osc_len
        self.linear_len = linear_len
        self.total_len = osc_len + linear_len
        self.rng = np.random.default_rng(seed)
        self.pr_errors = []
        self.al_errors = []

    def generate(self):
        seq = np.zeros(self.total_len, dtype=np.float32)
        # 振荡段
        for i in range(self.osc_len):
            seq[i] = 0.3 + 0.15 * (1 if i % 2 == 0 else -1)
        # 线性段
        base = seq[self.osc_len - 1]
        for i in range(self.linear_len):
            seq[self.osc_len + i] = base + 0.05 * (i + 1)
        # 归一化
        mx = np.abs(seq).max()
        if mx > 0:
            seq = seq / (mx + 1e-8)
        return seq, self.osc_len

    def train_epoch(self, n_sequences=40):
        device = next(self.model.parameters()).device
        total_pr_err = 0.0
        total_al_err = 0.0
        total_iz_err = 0.0
        for _ in range(n_sequences):
            seq, switch_point = self.generate()
            seq_t = torch.from_numpy(seq).float().to(device)
            T = len(seq_t)
            k = 20
            positions = torch.zeros(T, 2, device=device)
            positions[:, 0] = seq_t
            histories = torch.zeros(T, k, 2, device=device)
            for t in range(T):
                start = max(0, t - k)
                end = t
                if end > start:
                    histories[t, -(end-start):, 0] = seq_t[start:end]
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
