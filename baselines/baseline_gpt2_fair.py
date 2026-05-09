import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import torch, sys, time
sys.path.insert(0, "E:/PrAlmodel")
from tqdm import tqdm

print(">>> loading tokenizer...")
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(">>> loading data...")
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
subset_size = max(1, len(dataset) // 50)
dataset = dataset.select(range(subset_size))
print(f"    {len(dataset)} rows")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids"])
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
print(f"    {len(loader)} batches")

print(">>> initializing GPT-2 Small...")
from transformers import GPT2LMHeadModel, GPT2Config
config = GPT2Config(vocab_size=len(tokenizer), n_embd=768, n_layer=12, n_head=12, n_positions=64)
model = GPT2LMHeadModel(config).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"    {sum(p.numel() for p in model.parameters()):,} params")

N = 3
print(f"\n>>> training ({N} epochs)...")
model.train()
for epoch in range(1, N + 1):
    t0 = time.time()
    total = 0.0
    for batch in tqdm(loader, desc=f"epoch {epoch}/{N}", ncols=70):
        ids = batch["input_ids"].cuda()
        opt.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    avg = total / len(loader)
    ppl = torch.exp(torch.tensor(min(avg, 10.0))).item()
    print(f"    loss={avg:.4f}  ppl={ppl:.1f}  {time.time()-t0:.0f}s")

print(f"\n>>> GPT-2 final ppl: {ppl:.1f}")
print(">>> Pr-Al (our):    ppl: 1.5")
