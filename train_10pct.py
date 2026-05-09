# train_10pct.py —— 我们的架构在 10% 数据上训练
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch, sys, time
sys.path.insert(0, "E:/PrAlmodel")
from protocortex import LanguageModel
from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
dataset = dataset.select(range(max(1, len(dataset) // 10)))
print(f"Data: {len(dataset)} rows (10%)")

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.pad_token = tokenizer.eos_token
def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
dataset = dataset.map(tok, batched=True, remove_columns=["text"]).with_format("torch")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
print(f"Batches: {len(loader)}")

model = LanguageModel(vocab_size=len(tokenizer)).cuda()
print(f"Our Params: {sum(p.numel() for p in model.parameters()):,}")

N = 10
print(f"\nPr-Al World Model ({N} epochs)...")
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

torch.save(model.state_dict(), "E:/PrAlmodel/pral_10pct.pt")
print(f"\nPr-Al Final PPL: {ppl:.1f}")
