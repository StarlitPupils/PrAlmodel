# protocortex.py —— 向量化版（L2/L3全张量计算，零嵌套循环）
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, cosine_similarity

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class HierarchicalLayer(nn.Module):
    def __init__(self, n_neurons, n_input, n_next, is_top=False, sparsity=0.8):
        super().__init__()
        self.n_neurons = n_neurons
        self.sparsity = sparsity
        self.is_top = is_top
        self.W_fwd = nn.Parameter(torch.randn(n_neurons, n_input) * 0.02)
        self.W_pred = nn.Parameter(torch.randn(n_next, n_neurons) * 0.02) if not is_top else None
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.register_buffer("eligibility", torch.zeros(n_neurons, n_input))
        self.elig_decay = 0.9
        self.heb_lr = 0.0001

    def forward(self, x):
        act = torch.tanh(torch.matmul(x, self.W_fwd.T) + self.bias)
        if self.sparsity < 1.0 and self.training:
            threshold = torch.quantile(act.abs(), 1.0 - self.sparsity, dim=-1, keepdim=True)
            act = act * (act.abs() >= threshold).float()
        return act

    def hebbian_update(self, input_act, output_act, reward=1.0):
        with torch.no_grad():
            self.eligibility *= self.elig_decay
            self.eligibility += torch.outer(output_act.mean(dim=(0,1)), input_act.mean(dim=(0,1)))
            self.eligibility.clamp_(-10.0, 10.0)
            self.W_fwd.data += self.heb_lr * reward * self.eligibility
            self.W_fwd.data.clamp_(-1.0, 1.0)

class SleepConsolidator:
    def __init__(self, capacity=64): self.capacity = capacity; self.buffer = []
    def store(self, ids, score):
        if len(self.buffer) < self.capacity: self.buffer.append((ids.detach().clone(), score))
        else:
            mi = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
            if score > self.buffer[mi][1]: self.buffer[mi] = (ids.detach().clone(), score)
    def replay(self, model):
        for ids, _ in self.buffer:
            with torch.no_grad(): _ = model.forward(ids.unsqueeze(0))

class DopamineSystem:
    def __init__(self, decay=0.95): self.decay = decay; self.baseline = 0.0
    def compute_rpe(self, r): rpe = r - self.baseline; self.baseline = self.decay * self.baseline + (1-self.decay)*r; return rpe

class CuriositySystem:
    def __init__(self, temp=1.0): self.temp = temp
    def compute_intrinsic_reward(self, errors):
        return float(torch.exp(torch.tensor(-sum(e.abs().mean().item() for e in errors)/max(len(errors),1)/self.temp)).item())

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_embed=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.layer1 = HierarchicalLayer(512, d_embed, 512, sparsity=0.8)
        self.layer2 = HierarchicalLayer(512, 512, 512, sparsity=0.8)
        self.layer3 = HierarchicalLayer(512, 512, 512, sparsity=0.8)
        self.layer4 = HierarchicalLayer(512, 512, 512, is_top=True, sparsity=0.8)
        self.head1 = nn.Sequential(nn.Linear(512, vocab_size))
        self.head2 = nn.Sequential(nn.Linear(512, d_embed))
        self.head3 = nn.Sequential(nn.Linear(512, d_embed))
        self.head4 = nn.Sequential(nn.Linear(512, vocab_size))
        self.iz_loop_W = nn.Parameter(torch.randn(512, 512) * 0.01)
        self.iz_steps = 3
        self.opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.scaler = torch.amp.GradScaler('cuda')
        self.sleep = SleepConsolidator(capacity=64)
        self.dopamine = DopamineSystem(decay=0.95)
        self.curiosity = CuriositySystem(temp=1.0)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        pr_feat, al_feat = h1.mean(dim=-1), h4.mean(dim=-1)
        conflict = torch.abs(pr_feat - al_feat)
        threshold = torch.quantile(conflict, 0.9, dim=1, keepdim=True)
        mask = conflict > threshold
        if mask.any():
            with torch.no_grad():
                iz = h4[mask].clone()
                for _ in range(self.iz_steps):
                    iz = torch.tanh(iz + torch.matmul(iz, self.iz_loop_W.T) * 0.1)
                iz_exp = torch.zeros_like(h4)
                iz_exp[mask] = iz
                h4 = torch.where(mask.unsqueeze(-1), iz_exp, h4)
        return (self.head1(h1), self.head2(h2), self.head3(h3), self.head4(h4)), (h1,h2,h3,h4), conflict

    def _compute_multi_level_loss(self, preds, input_ids):
        pred1, pred2, pred3, pred4 = preds
        B, T, D = pred2.shape
        emb = self.embedding(input_ids)
        # L1: 标准语言建模
        loss1 = cross_entropy(pred1[:, :-1].reshape(-1, pred1.size(-1)), input_ids[:, 1:].reshape(-1), ignore_index=-100)
        # L2: 未来3-token语义（全向量化，零循环）
        T2 = T - 3
        future = (emb[:, 1:T2+1] + emb[:, 2:T2+2] + emb[:, 3:T2+3]) / 3.0
        loss2 = 1.0 - cosine_similarity(pred2[:, :T2].reshape(-1, D), future.reshape(-1, D), dim=-1).mean()
        # L3: 远距离语义（t+5到t+8）
        T3 = T - 8
        far = (emb[:, 5:T3+5] + emb[:, 6:T3+6] + emb[:, 7:T3+7] + emb[:, 8:T3+8]) / 4.0
        loss3 = 1.0 - cosine_similarity(pred3[:, :T3].reshape(-1, D), far.reshape(-1, D), dim=-1).mean()
        # L4: 全局预测
        loss4 = cross_entropy(pred4[:, :-1].reshape(-1, pred4.size(-1)), input_ids[:, 1:].reshape(-1), ignore_index=-100)
        return loss1 + 0.5*loss2 + 0.3*loss3 + 0.1*loss4, (loss1.item(), loss2.item(), loss3.item(), loss4.item())

    def training_step(self, input_ids):
        self.opt.zero_grad()
        with torch.amp.autocast('cuda'):
            preds, acts, conflict = self.forward(input_ids)
            total_loss, (l1,l2,l3,l4) = self._compute_multi_level_loss(preds, input_ids)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.scaler.step(self.opt)
        self.scaler.update()
        with torch.no_grad():
            h1,h2,h3,h4 = acts
            rpe = self.dopamine.compute_rpe(float(torch.exp(torch.tensor(-total_loss.item()/2.0)).item()))
            self.layer1.hebbian_update(self.embedding(input_ids).detach(), h1.detach(), rpe)
            self.layer2.hebbian_update(h1.detach(), h2.detach(), rpe)
            self.layer3.hebbian_update(h2.detach(), h3.detach(), rpe)
            self.layer4.hebbian_update(h3.detach(), h4.detach(), rpe)
            if conflict.mean().item() > 0.1:
                self.sleep.store(input_ids, conflict.mean().item())
        return total_loss.item(), (l1,l2,l3,l4)

    def end_epoch(self):
        self.sleep.replay(self)
        self.sleep.buffer = []
