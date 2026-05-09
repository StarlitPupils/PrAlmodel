# connections.py —— 纯函数版：单个融合前向，无副作用
import torch
import torch.nn as nn
from protocortex_config import ProtoConfig, get_config

class GrowingConnections(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or get_config()
        n = self.config.n_neurons
        sp = 1.0 - self.config.reserve_sparsity
        g = torch.Generator().manual_seed(self.config.seed)
        W = torch.randn(n, n, generator=g) * 0.1
        mask = torch.rand(n, n, generator=g) > sp
        W[mask] = 0.0
        radius = torch.linalg.eigvals(W).abs().max()
        if radius > 0:
            W *= self.config.reserve_spectral_radius / radius
        self.register_buffer("W", W)
        self.register_buffer("mask", (W.abs() > 1e-8))
        self.register_buffer("ages", torch.zeros(n, n, dtype=torch.long))
        self.co_activation_buffer = []
        self.register_buffer("co_activation_sum", torch.zeros(n, n))

    def forward(self, activation):
        if activation.dim() == 2:
            return activation @ (self.W * self.mask.float()).T
        return (self.W * self.mask.float()) @ activation

    def n_connections(self):
        return self.mask.sum().item()

    def record_activation(self, act):
        self.co_activation_buffer.append(act.detach().clone())
        self.co_activation_sum += torch.outer(act, act)
        if len(self.co_activation_buffer) > self.config.hebbian_window:
            oldest = self.co_activation_buffer.pop(0)
            self.co_activation_sum -= torch.outer(oldest, oldest)

    def get_hebbian_scores(self):
        n = max(len(self.co_activation_buffer), 1)
        return self.co_activation_sum / n

    def hebbian_grow(self, axial_codes, n_new=5):
        return 0

    def prune_weak(self):
        return 0

    def grow(self, n_new):
        pass
