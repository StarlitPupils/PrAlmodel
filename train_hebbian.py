# train_hebbian.py —— 纯赫布学习训练（零反向传播）
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch, sys, time
sys.path.insert(0, "E:/PrAlmodel")
from protocortex import LanguageModel
from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train").select(range(max(1, 1801350 // 10)))
print(f"Data: {len(dataset)} rows (10% wikitext-103)")

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.pad_token = tokenizer.eos_token
def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=64, padding="max_length")
dataset = dataset.map(tok, batched=True, remove_columns=["text"]).with_format("torch")
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
print(f"Batches: {len(loader)} (64 token, batch=16)")

model = LanguageModel(vocab_size=len(tokenizer)).cuda()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Opt: {model.opt}, Scaler: {model.scaler}  (should be None/None)")

N = 10
print(f"\nPure Hebbian Training ({N} epochs)...")
for epoch in range(1, N + 1):
    t0 = time.time(); total = 0.0
    for batch in (pbar := tqdm(loader, desc=f"Epoch {epoch}/{N}", ncols=70)):
        loss = model.training_step(batch['input_ids'].cuda(non_blocking=True))
        total += loss; pbar.set_postfix(loss=f"{loss:.2f}")
    avg = total / len(loader); ppl = torch.exp(torch.tensor(min(avg, 10.0))).item()
    print(f"  Loss: {avg:.4f} | PPL: {ppl:.1f} | {time.time()-t0:.0f}s")
print(f"\nFinal PPL: {ppl:.1f} (Pure Hebbian, zero backprop)")
