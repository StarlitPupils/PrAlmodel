# baselines/baseline_lstm_lm.py —— 同等参数 LSTM 基线对比
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import sys, time, os
sys.path.insert(0, "E:/PrAlmodel")
from tqdm import tqdm

# 设置
VOCAB_SIZE = 50257  # GPT-2 vocab
D_EMBED = 768
HIDDEN = 2048        # 使 LSTM 参数量接近我们的 145M
N_LAYERS = 4

class LSTMLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_EMBED)
        self.lstm = nn.LSTM(D_EMBED, HIDDEN, N_LAYERS, batch_first=True)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE)
        self.opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        return self.lm_head(lstm_out)

    def training_step(self, input_ids):
        B, T = input_ids.shape
        self.opt.zero_grad()
        logits = self.forward(input_ids)
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        return loss.item()

# 训练
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
subset = dataset.select(range(max(1, len(dataset) // 200)))  # 0.5%

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

subset = subset.map(tokenize_fn, batched=True, remove_columns=["text"])
subset.set_format(type='torch', columns=['input_ids'])
loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True, num_workers=0)

model = LSTMLanguageModel().cuda()
params = sum(p.numel() for p in model.parameters())
print(f"LSTM 参数: {params:,}")
print(f"训练 (5 epochs)...")

model.train()
for epoch in range(1, 6):
    total_loss = 0
    pbar = tqdm(loader, desc=f"LSTM Epoch {epoch}/5", ncols=70)
    for batch in pbar:
        loss = model.training_step(batch['input_ids'].cuda())
        total_loss += loss
        pbar.set_postfix(loss=f"{loss:.2f}")
    avg_loss = total_loss / len(loader)
    ppl = torch.exp(torch.tensor(min(avg_loss, 10.0))).item()
    print(f"  Loss: {avg_loss:.4f} | PPL: {ppl:.1f}")
