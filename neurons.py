# neurons.py —— 事件驱动版（移除赫布更新，保留核心功能）
import torch
import torch.nn as nn
from protocortex_config import ProtoConfig, get_config

class AxialNeuronLayer(nn.Module):
    def __init__(self, config=None, d_embed=768):
        super().__init__()
        self.config = config or get_config()
        n = self.config.n_neurons
        self.axial_codes = nn.Parameter(torch.zeros(n, 2))
        self._init_axial()
        g = torch.Generator().manual_seed(self.config.seed + 1)
        self.W_in_pr = nn.Parameter(torch.randn(n, d_embed, generator=g) * 0.02)
        self.W_in_al = nn.Parameter(torch.randn(n, d_embed, generator=g) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n))
        self.register_buffer("activation", torch.zeros(n))
        self.register_buffer("predicted", torch.zeros(n))
        self.register_buffer("surprise_thresholds", torch.full((n,), 0.05))

    def _init_axial(self):
        n = self.config.n_neurons
        self.axial_codes.data[:, 0] = torch.linspace(1.0, 0.0, n)
        self.axial_codes.data[:, 1] = torch.linspace(0.0, 1.0, n)
        self.axial_codes.data += torch.randn(n, 2) * 0.02
        self.axial_codes.data.clamp_(0.0, 1.0)

    def forward_step(self, pr_emb, al_emb, prev_act):
        B = pr_emb.shape[0]
        n = self.config.n_neurons
        
        p_weight = torch.sigmoid((self.axial_codes[:, 0] - 0.5) * 10.0)
        a_weight = torch.sigmoid((self.axial_codes[:, 1] - 0.5) * 10.0)
        pr_drive = (self.W_in_pr @ pr_emb.T).T
        al_drive = (self.W_in_al @ al_emb.T).T
        sensory = (pr_drive * p_weight + al_drive * a_weight) * self.config.reserve_input_scaling
        candidate = torch.tanh(sensory + self.bias)
        
        predicted = prev_act
        surprise = torch.abs(candidate - predicted)
        mask = surprise > self.surprise_thresholds.unsqueeze(0)
        new_act = torch.where(mask, candidate, prev_act)
        
        if B > 0:
            self.activation = new_act.mean(dim=0).detach()
            self.predicted = predicted.mean(dim=0).detach()
        return new_act

    def get_pr_neurons(self, frac=0.5):
        n = self.config.n_neurons; k = int(n * frac)
        _, idx = torch.topk(self.axial_codes[:, 0] - self.axial_codes[:, 1], k)
        return idx
    def get_al_neurons(self, frac=0.5):
        n = self.config.n_neurons; k = int(n * frac)
        _, idx = torch.topk(self.axial_codes[:, 1] - self.axial_codes[:, 0], k)
        return idx
    def get_iz_neurons(self, frac=0.5):
        n = self.config.n_neurons; k = int(n * frac)
        dist = (self.axial_codes[:, 0] - 0.5)**2 + (self.axial_codes[:, 1] - 0.5)**2
        _, idx = torch.topk(dist, k, largest=False)
        return idx
