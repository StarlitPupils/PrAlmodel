# copy_adapter.py —— 适配三区独立架构
import torch, sys
sys.path.insert(0, "E:/PrAlmodel")
from copy_task import CopyTaskGenerator
from protocortex_config import ProtoConfig, get_config

class CopyAdapter:
    def __init__(self, model, proto_config=None, seq_len=10, blank_len=20, seed=42, cache=None):
        self.model = model
        self.config = proto_config or get_config()
        self.generator = CopyTaskGenerator(seq_len=seq_len, blank_len=blank_len, seed=seed)
        self.total_len = seq_len + blank_len + 1 + seq_len
        self.pr_errors, self.al_errors = [], []
        self.cache = cache

    def train_epoch(self, n_sequences=None):
        if n_sequences is None:
            n_sequences = self.config.batch_sequences
        device = next(self.model.parameters()).device
        B, T, k = n_sequences, self.total_len, 20
        
        all_inp, all_out = [], []
        for _ in range(B):
            inp, out = self.generator.generate()
            all_inp.append(torch.from_numpy(inp).float())
            all_out.append(torch.from_numpy(out).float())
        inp_batch = torch.stack(all_inp).pin_memory().to(device, non_blocking=True)
        out_batch = torch.stack(all_out).pin_memory().to(device, non_blocking=True)
        
        positions = torch.zeros(B, T, 2, device=device)
        positions[:, :, 0] = inp_batch
        histories = torch.zeros(B, T, k, 2, device=device)
        for t in range(T):
            start, end = max(0, t - k), t
            if end > start:
                histories[:, t, -(end-start):, 0] = inp_batch[:, start:end]
        
        cache_key = (B, T)
        if self.cache and cache_key in self.cache:
            pr_encs, al_encs = self.cache[cache_key]
            pr_encs, al_encs = pr_encs.to(device, non_blocking=True), al_encs.to(device, non_blocking=True)
        else:
            pr_encs = torch.zeros(B, T, self.config.pr_encoding_dim, device=device)
            al_encs = torch.zeros(B, T, self.config.al_encoding_dim, device=device)
            for b in range(B):
                for t in range(T):
                    pr_encs[b, t] = self.model.encoder.encode_pr(positions[b, t])
                    al_encs[b, t] = self.model.encoder.encode_al(histories[b, t])
            if self.cache is not None:
                self.cache[cache_key] = (pr_encs.clone().cpu(), al_encs.clone().cpu())
        
        targets = {
            'pr': torch.zeros(B, T, 2, device=device),
            'al': torch.zeros(B, T, 2, device=device),
            'iz': torch.zeros(B, T, 2, device=device),
        }
        targets['pr'][:, :-1, 0] = out_batch[:, 1:]
        if T > 10: targets['al'][:, :-10, 0] = out_batch[:, 10:]
        if T > 5:  targets['iz'][:, :-5, 0]  = out_batch[:, 5:]
        
        loss, errors = self.model.forward_batch(pr_encs, al_encs, targets)
        self.pr_errors.append(errors['pr_error'])
        self.al_errors.append(errors['al_error'])
        return errors
