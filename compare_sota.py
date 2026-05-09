import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import torch, sys, time
print("Loading models...")
sys.path.insert(0, "E:/PrAlmodel")
from protocortex import LanguageModel
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Pr-Al
pral = LanguageModel(vocab_size=len(tokenizer))
pral.load_state_dict(torch.load("E:/PrAlmodel/pral_10pct.pt", map_location="cpu"))
pral = pral.cuda(); pral.eval()
print(f"  Pr-Al: {sum(p.numel() for p in pral.parameters()):,} params")

# GPT-2
gpt2_config = GPT2Config(vocab_size=len(tokenizer), n_embd=768, n_layer=12, n_head=12, n_positions=128)
gpt2 = GPT2LMHeadModel(gpt2_config)
gpt2.load_state_dict(torch.load("E:/PrAlmodel/gpt2_10pct.pt", map_location="cpu"))
gpt2 = gpt2.cuda(); gpt2.eval()
print(f"  GPT-2: {sum(p.numel() for p in gpt2.parameters()):,} params")

# Test data (from wikitext-103 train last 10%)
print("Loading test data...")
from datasets import load_dataset
full = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
test_data = full.select(range(int(len(full)*0.9), len(full)))
def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
test_data = test_data.map(tok, batched=True, remove_columns=["text"]).with_format("torch")
from torch.utils.data import DataLoader
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)
print(f"  {len(test_loader)} batches")

# PPL evaluation
print("\nEvaluating...")
def compute_ppl(model, loader, is_pral=False):
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].cuda()
            if is_pral:
                preds, _, _ = model.forward(ids)
                logits = preds[0]
            else:
                logits = model(ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), ignore_index=-100, reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += (ids[:, 1:] != -100).sum().item()
    return total_loss / max(total_tokens, 1)

ppl_pral = compute_ppl(pral, test_loader, is_pral=True)
ppl_gpt2 = compute_ppl(gpt2, test_loader, is_pral=False)

print("\n" + "=" * 55)
print("  SOTA COMPARISON")
print("=" * 55)
print(f"  Pr-Al  Test PPL: {ppl_pral:.1f}  (93M params)")
print(f"  GPT-2  Test PPL: {ppl_gpt2:.1f}  (124M params)")
print(f"  Delta: {ppl_gpt2 - ppl_pral:.1f}")
print(f"  Winner: {'Pr-Al' if ppl_pral < ppl_gpt2 else 'GPT-2'}")
print(f"\n  Pr-Al unique advantages:")
print(f"    - 4-level predictive errors (L1/L2/L3/L4)")
print(f"    - Conflict-driven IZ loop (system 2)")
print(f"    - Hebbian plasticity + dopamine RPE")
print(f"    - Lateral inhibition (80% sparse)")
print(f"    - Sleep consolidation")
