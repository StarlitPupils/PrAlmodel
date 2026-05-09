# train_multiscale.py —— 多时间尺度预测性编码训练
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import torch, sys, time
sys.path.insert(0, "E:/PrAlmodel")
from protocortex_config import ProtoConfig
from protocortex import LanguageModel
from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
subset_size = max(1, len(dataset) // 10)
dataset = dataset.select(range(subset_size))
print(f"数据: {len(dataset)} 行 (10% wikitext-103)")

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=64, padding="max_length")

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
dataset.set_format(type='torch', columns=['input_ids'])
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
print(f"Batch 数: {len(loader)}")

cfg = ProtoConfig()
cfg.n_neurons = 2048
model = LanguageModel(vocab_size=len(tokenizer), d_embed=768, config=cfg).cuda()
print(f"参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"多时间尺度预测: Pr(t+1) + IZ(t+5) + Al(t+10)")

N = 5
print(f"\n多尺度训练 ({N} epochs)...")
model.train()
for epoch in range(1, N + 1):
    t0 = time.time()
    total = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{N}", ncols=70)
    for batch in pbar:
        loss = model.training_step(batch['input_ids'].cuda())
        total += loss
        pbar.set_postfix(loss=f"{loss:.2f}")
    avg = total / len(loader)
    ppl = torch.exp(torch.tensor(min(avg, 10.0))).item()
    print(f"  Loss: {avg:.4f} | PPL: {ppl:.1f} | {time.time()-t0:.0f}s")

print(f"\n多尺度预测 最终 PPL: {ppl:.1f}")
print(f"对比: GPT-2 Small = 6.5, Pr-Al (单尺度 dense) = 7.0")
