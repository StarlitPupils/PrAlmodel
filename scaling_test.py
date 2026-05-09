# scaling_test.py —— 2048 vs 4096 神经元对比测试
import torch, numpy as np, sys, time
sys.path.insert(0, "E:/PrAlmodel")
from anomaly_sequence_task import AnomalySequenceGenerator
from protocortex_config import ProtoConfig
from protocortex import Protocortex
from sklearn.metrics import roc_auc_score

def test_auc(n_neurons, pr_zone, iz_zone, al_zone, n_seqs=200):
    cfg = ProtoConfig()
    cfg.n_neurons = n_neurons
    cfg.pr_zone = pr_zone
    cfg.iz_zone = iz_zone
    cfg.al_zone = al_zone
    model = Protocortex(cfg).cuda()
    k = 20
    conflicts, labels = [], []
    t0 = time.time()
    for seed in range(n_seqs):
        gen = AnomalySequenceGenerator(length=80, anomaly_prob=0.05, seed=seed)
        seq, anom_pos, _ = gen.generate()
        seq_t = torch.from_numpy(seq).float().cuda()
        T = len(seq_t)
        positions = seq_t.unsqueeze(-1).expand(-1, 2)
        histories = torch.zeros(T, k, 2, device="cuda")
        for t in range(T):
            start, end = max(0, t - k), t
            if end > start:
                histories[t, -(end-start):, 0] = seq_t[start:end]
        pr_encs = torch.zeros(1, T, cfg.pr_encoding_dim, device="cuda")
        al_encs = torch.zeros(1, T, cfg.al_encoding_dim, device="cuda")
        with torch.no_grad():
            for t in range(T):
                pr_encs[0, t] = model.encoder.encode_pr(positions[t])
                al_encs[0, t] = model.encoder.encode_al(histories[t])
        targets_pr = torch.zeros(1, T, 2, device="cuda")
        targets_pr[:, :-1, 0] = seq_t[1:]
        _, errors = model.forward_batch(pr_encs, al_encs, {
            "pr": targets_pr, "al": targets_pr.clone(), "iz": targets_pr.clone(),
        })
        conflicts.append(abs(errors["pr_error"] - errors["al_error"]))
        labels.append(1.0 if len(anom_pos) > 0 else 0.0)
    t1 = time.time()
    conflicts = np.array(conflicts)
    labels = np.array(labels)
    n_anom = int(labels.sum())
    n_norm = n_seqs - n_anom
    anom_mean = conflicts[labels == 1].mean() if n_anom > 0 else 0
    norm_mean = conflicts[labels == 0].mean() if n_norm > 0 else 0
    auc = roc_auc_score(labels, conflicts) if n_anom > 0 and n_norm > 0 else 0.0
    params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {
        "neurons": n_neurons,
        "anom": anom_mean, "norm": norm_mean,
        "ratio": anom_mean / (norm_mean + 1e-10),
        "auc": auc, "time_s": t1 - t0, "params": params,
    }

if __name__ == "__main__":
    print("=" * 64)
    print("  NEURON SCALING TEST: 2048 vs 4096")
    print("=" * 64)
    r1 = test_auc(2048, (0, 682), (683, 1365), (1366, 2047))
    r2 = test_auc(4096, (0, 1365), (1366, 2730), (2731, 4095))
    print("-" * 64)
    print(f"  {'Metric':<28} {'2048 neurons':>16} {'4096 neurons':>16}")
    print("-" * 64)
    print(f"  {'Learnable params':<28} {r1['params']:>16,} {r2['params']:>16,}")
    print(f"  {'Anomaly mean conflict':<28} {r1['anom']:>16.6f} {r2['anom']:>16.6f}")
    print(f"  {'Normal  mean conflict':<28} {r1['norm']:>16.6f} {r2['norm']:>16.6f}")
    print(f"  {'Anomaly/Normal ratio':<28} {r1['ratio']:>16.1f} {r2['ratio']:>16.1f}")
    print(f"  {'AUC':<28} {r1['auc']:>16.4f} {r2['auc']:>16.4f}")
    print(f"  {'Time (s)':<28} {r1['time_s']:>16.1f} {r2['time_s']:>16.1f}")
    print("=" * 64)
    d_auc = r2["auc"] - r1["auc"]
    d_ratio = r2["ratio"] - r1["ratio"]
    print(f"  AUC delta: {d_auc:+.4f}")
    print(f"  Ratio delta: {d_ratio:+.1f}")
    if d_auc > 0.01:
        print("  >> 4096 significantly better -- scaling works")
    elif d_auc > -0.01:
        print("  >> Similar performance -- 2048 near ceiling")
    else:
        print("  >> 4096 worse -- overscaling hurts")
