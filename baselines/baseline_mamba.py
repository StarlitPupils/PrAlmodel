# baselines/baseline_mamba.py —— 简化 SSM 基线（Copy 任务）
import torch
import torch.nn as nn
import numpy as np
from copy_task import CopyTaskGenerator

class SimpleSSM(nn.Module):
    """简化的状态空间模型 —— 不带选择机制的 Mamba 前身 S4 风格。"""
    def __init__(self, input_dim=1, hidden_dim=128, state_dim=64, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # SSM 参数
        self.A = nn.Parameter(torch.randn(hidden_dim, state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.zeros(hidden_dim))
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        batch, seq_len = x.shape[0], x.shape[1]
        x = self.input_proj(x.unsqueeze(-1))
        outputs = []
        state = torch.zeros(batch, self.hidden_dim, self.state_dim, device=x.device)
        
        for t in range(seq_len):
            u = x[:, t, :]
            state = state + torch.einsum('bh,hs->bhs', u, self.B)
            state = torch.einsum('bhs,hsu->bhu', state, self.A)
            y = torch.einsum('bhs,hs->bh', state, self.C) + self.D * u
            outputs.append(y)
        
        out = torch.stack(outputs, dim=1)
        return self.readout(out).squeeze(-1)

def train_ssm_copy(n_epochs=500, seq_len=10, blank_len=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSSM(input_dim=1, hidden_dim=128, state_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = CopyTaskGenerator(seq_len=seq_len, blank_len=blank_len, seed=42)
    
    best_err = float("inf")
    history = []
    
    for epoch in range(1, n_epochs + 1):
        total_err = 0.0
        n_batches = 50
        for _ in range(n_batches):
            inp, target = gen.generate()
            inp_t = torch.from_numpy(inp).float().to(device)
            tgt_t = torch.from_numpy(target).float().to(device)
            
            pred = model(inp_t.unsqueeze(0))
            loss = nn.functional.mse_loss(pred.squeeze(0), tgt_t)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_err += loss.item()
        
        avg_err = total_err / n_batches
        history.append(avg_err)
        if avg_err < best_err:
            best_err = avg_err
        
        if epoch % 50 == 0 or epoch <= 5:
            print(f"  SSM Epoch {epoch:>4} | MSE={avg_err:.6f}")
    
    return best_err, history, sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    best, hist, params = train_ssm_copy(n_epochs=300)
    print(f"SSM Best: {best:.6f} | Params: {params:,}")
