# baselines/baseline_transformer.py —— Transformer 基线（Copy 任务）
import torch
import torch.nn as nn
import numpy as np
from copy_task import CopyTaskGenerator

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=2, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=256, dropout=0.1,
                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.embed(x.unsqueeze(-1))
        x = self.pos_enc(x)
        out = self.transformer(x)
        return self.readout(out).squeeze(-1)

def train_transformer_copy(n_epochs=500, seq_len=10, blank_len=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerBaseline(input_dim=1, d_model=128, nhead=4, num_layers=2).to(device)
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
            print(f"  Transformer Epoch {epoch:>4} | MSE={avg_err:.6f}")
    
    return best_err, history, sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    best, hist, params = train_transformer_copy(n_epochs=300)
    print(f"Transformer Best: {best:.6f} | Params: {params:,}")
