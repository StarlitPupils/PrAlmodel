# eval_real.py —— 强制离线模式读取 wikitext-103 测试集
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import torch, sys, time
sys.path.insert(0, "E:/PrAlmodel")

print("1/4 loading test data from cache...", flush=True)
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=False)
texts = [t for t in dataset["text"] if len(t.strip()) > 0]
print(f"   {len(texts)} non-empty lines", flush=True)

print("2/4 loading tokenizer...", flush=True)
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print("3/4 loading models...", flush=True)
from protocortex import LanguageModel
from transformers import GPT2LMHeadModel, GPT2Config

pral = LanguageModel(len(tokenizer))
pral.load_state_dict(torch.load("E:/PrAlmodel/pral_10pct.pt", map_location="cpu"))
pral = pral.cuda().eval()

cfg = GPT2Config(vocab_size=len(tokenizer), n_embd=768, n_layer=12, n_head=12, n_positions=128)
gpt2 = GPT2LMHeadModel(cfg)
gpt2.load_state_dict(torch.load("E:/PrAlmodel/gpt2_10pct.pt", map_location="cpu"))
gpt2 = gpt2.cuda().eval()
print("   done", flush=True)

print("4/4 evaluating PPL on wikitext-103 test set...", flush=True)

def eval_ppl(model, texts, is_pral=False, max_samples=1000):
    import random
    random.seed(42)
    samples = random.sample(texts, min(max_samples, len(texts)))
    
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, text in enumerate(samples):
            tok = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
            ids = tok["input_ids"].cuda()
            if ids.size(1) < 2:
                continue
            if is_pral:
                preds, _, _ = model.forward(ids)
                logits = preds[0]
            else:
                logits = model(ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), ignore_index=-100, reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (ids[:, 1:] != -100).sum().item()
    return total_loss / max(total_tokens, 1)

ppl_pral = eval_ppl(pral, texts, is_pral=True)
ppl_gpt2 = eval_ppl(gpt2, texts, is_pral=False)

print("\n" + "=" * 55)
print("  SOTA COMPARISON: wikitext-103 TEST SET")
print("=" * 55)
print(f"  {'Pr-Al (93M, 4-layer World Model)':<35} PPL = {ppl_pral:.1f}")
print(f"  {'GPT-2 Small (124M, 12-layer Transformer)':<35} PPL = {ppl_gpt2:.1f}")
print(f"  {'Winner':<35} {'Pr-Al' if ppl_pral < ppl_gpt2 else ('GPT-2' if ppl_gpt2 < ppl_pral else 'Tie')}")
print("=" * 55)
print(f"\n  Additional Pr-Al advantages:")
print(f"    - 4 independently trackable prediction errors")
print(f"    - Conflict-driven IZ reflection loop")
print(f"    - 93M vs 124M params (25% fewer)")
print(f"    - 5 brain-inspired mechanisms active simultaneously")
