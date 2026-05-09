# train_full.py —— 2%数据快速验证
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch, sys, time
sys.path.insert(0, "E:/PrAlmodel")
from protocortex import LanguageModel
from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train").select(range(max(1, 1801350 // 50)))
print(f"Data: {len(dataset)} rows (2%)")

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.pad_token = tokenizer.eos_token
def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
dataset = dataset.map(tok, batched=True, remove_columns=["text"]).with_format("torch")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
print(f"Batches: {len(loader)} (128 token, batch=32)")

model = LanguageModel(vocab_size=len(tokenizer)).cuda()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

N = 20
print(f"\n4-Level World Model ({N} epochs, ~10min each)...")
for epoch in range(1, N + 1):
    t0 = time.time(); total, l1s, l2s, l3s, l4s = 0, 0, 0, 0, 0
    for batch in (pbar := tqdm(loader, desc=f"Epoch {epoch}/{N}", ncols=70)):
        total_loss, (l1,l2,l3,l4) = model.training_step(batch['input_ids'].cuda(non_blocking=True))
        total += total_loss; l1s+=l1; l2s+=l2; l3s+=l3; l4s+=l4
        pbar.set_postfix(loss=f"{total_loss:.2f}")
    model.end_epoch()
    n = len(loader)
    ppl = torch.exp(torch.tensor(min(total/n, 10.0))).item()
    print(f"  Loss: {total/n:.4f} | PPL: {ppl:.1f} | L1={l1s/n:.3f} L2={l2s/n:.3f} L3={l3s/n:.3f} L4={l4s/n:.3f} | {time.time()-t0:.0f}s")
print(f"\nFinal PPL: {ppl:.1f}")
