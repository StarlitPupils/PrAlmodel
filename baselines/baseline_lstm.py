# baselines/baseline_lstm.py —— LSTM 基线（Copy 任务）
import torch
import torch.nn as nn
import numpy as np
from copy_task import CopyTaskGenerator

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.readout(out).squeeze(-1)

def train_lstm_copy(n_epochs=500, seq_len=10, blank_len=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_len = seq_len + blank_len + 1 + seq_len
    model = LSTMBaseline(input_dim=1, hidden_dim=256, num_layers=2).to(device)
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
            print(f"  LSTM Epoch {epoch:>4} | MSE={avg_err:.6f}")
    
    return best_err, history, sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    best, hist, params = train_lstm_copy(n_epochs=300)
    print(f"LSTM Best: {best:.6f} | Params: {params:,}")
