# dual_source.py —— Pr/Al 双源固定编码器
"""
Pr 源编码器（感觉-运动通路）:
  - 当前坐标 (x,y) → 固定随机投影 → 64 维
  - 生物学对应：初级感觉皮层，进化固定的感受野
  - 不学习，不更新权重

Al 源编码器（记忆-模式通路）:
  - 历史 20 帧 → 回声状态网络 (ESN) → 64 维
  - 生物学对应：海马体循环回路，进化产物
  - ESN 储备池权重固定随机，不学习

关键设计原则：
  - 编码器是"感觉器官"，不是学习系统
  - 真正的学习发生在原型皮层内部
  - 保证两个源的编码在统计上可区分（不同的随机种子）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from protocortex_config import ProtoConfig, get_config


# ================================================================
# 1. Pr 编码器：固定随机投影
# ================================================================
class PrEncoder(nn.Module):
    """
    当前坐标 → 64 维固定编码。

    使用高斯随机投影矩阵。矩阵在初始化时随机生成，
    之后冻结（requires_grad=False）。

    原理：随机投影近似保持向量间距离（Johnson-Lindenstrauss 引理），
    因此不同位置的编码仍能保留空间关系。
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 64,
                 config: ProtoConfig = None, seed: int = 42):
        super().__init__()
        self.config = config or get_config()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 固定随机投影矩阵（不学习！）
        g = torch.Generator()
        g.manual_seed(seed)
        self.projection = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0 / (input_dim ** 0.5))
        nn.init.zeros_(self.projection.bias)

        # 冻结
        for p in self.projection.parameters():
            p.requires_grad = False

        # 非线性激活（固定：tanh）
        self.activation = nn.Tanh()

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position: (2,) 或 (B, 2) 当前坐标 (x, y)
        Returns:
            encoding: (64,) 或 (B, 64) 固定编码
        """
        if position.dim() == 1:
            position = position.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        encoded = self.activation(self.projection(position))

        if squeeze:
            encoded = encoded.squeeze(0)
        return encoded


# ================================================================
# 2. Al 编码器：回声状态网络 (ESN)
# ================================================================
class EchoStateNetwork(nn.Module):
    """
    历史序列 → 64 维固定编码。

    ESN 原理：
      - 储备池（reservoir）是一组随机连接的循环神经元
      - 输入权重和储备池权重随机生成并固定
      - 输出权重也可固定（我们只用最终状态，不需要训练读出层）
      - 储备池的动力学会自然"回声"输入序列的时序模式

    生物学对应：海马 CA3 回路的循环连接模式
    """

    def __init__(self, input_dim: int = 2, reservoir_size: int = 128,
                 output_dim: int = 64, config: ProtoConfig = None,
                 seed: int = 99):
        super().__init__()
        self.config = config or get_config()
        cfg = self.config
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim

        g = torch.Generator()
        g.manual_seed(seed)

        # 储备池递归权重（固定随机，稀疏）
        W_res = torch.randn(reservoir_size, reservoir_size, generator=g) * 0.1
        # 稀疏化：80% 的连接置零
        mask = torch.rand(reservoir_size, reservoir_size, generator=g) > 0.8
        W_res[mask] = 0.0
        # 调整谱半径
        radius = torch.linalg.eigvals(W_res).abs().max()
        if radius > 0:
            W_res = W_res * (cfg.esn_spectral_radius / radius)
        self.register_buffer('W_reservoir', W_res)

        # 输入权重（固定随机）
        W_in = torch.randn(reservoir_size, input_dim, generator=g) * cfg.esn_input_scaling
        self.register_buffer('W_input', W_in)

        # 读出头 —— 固定随机投影到 64 维
        W_out = torch.randn(output_dim, reservoir_size, generator=g) * 0.1
        self.register_buffer('W_output', W_out)

        # 激活函数
        self.activation = torch.tanh

    def forward(self, history: torch.Tensor,
                initial_state: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            history: (seq_len, 2) 或 (B, seq_len, 2) 历史轨迹
            initial_state: (reservoir_size,)  初始储备池状态
        Returns:
            encoding: (64,) 或 (B, 64)
        """
        if history.dim() == 2:
            # (seq_len, 2) → (1, seq_len, 2)
            history = history.unsqueeze(0)
            single = True
        else:
            single = False

        B, seq_len, _ = history.shape
        device = history.device

        # 储备池状态
        if initial_state is None:
            state = torch.zeros(B, self.reservoir_size, device=device)
        else:
            state = initial_state.unsqueeze(0).expand(B, -1)

        # 逐帧更新储备池
        for t in range(seq_len):
            inp = history[:, t, :]  # (B, 2)
            # state ← tanh(W_in · inp + W_res · state)
            state = self.activation(
                inp @ self.W_input.T + state @ self.W_reservoir.T
            )

        # 最终状态 → 64 维编码
        encoding = state @ self.W_output.T  # (B, 64)

        if single:
            encoding = encoding.squeeze(0)
        return encoding


# ================================================================
# 3. 双源编码器（组合）
# ================================================================
class DualSourceEncoder(nn.Module):
    """
    Pr + Al 双源编码器。

    提供统一接口：
      encode_pr(position) → (64,)
      encode_al(history)  → (64,)
      encode_both(position, history) → ((64,), (64,))
    """

    def __init__(self, config: ProtoConfig = None,
                 pr_seed: int = 42, al_seed: int = 99):
        super().__init__()
        self.config = config or get_config()
        self.pr_encoder = PrEncoder(
            input_dim=2,
            output_dim=self.config.pr_encoding_dim,
            config=self.config,
            seed=pr_seed,
        )
        self.al_encoder = EchoStateNetwork(
            input_dim=2,
            reservoir_size=self.config.esn_reservoir_size,
            output_dim=self.config.al_encoding_dim,
            config=self.config,
            seed=al_seed,
        )

    def encode_pr(self, position: torch.Tensor) -> torch.Tensor:
        return self.pr_encoder(position)

    def encode_al(self, history: torch.Tensor) -> torch.Tensor:
        return self.al_encoder(history)

    def encode_both(self, position: torch.Tensor,
                    history: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pr(position), self.encode_al(history)

    def get_encoding_stats(self, n_samples: int = 100
                           ) -> dict:
        """统计编码的区分性（两个源的编码应该是可区分的）。"""
        device = next(self.parameters()).device
        pr_encs = []
        al_encs = []
        with torch.no_grad():
            for _ in range(n_samples):
                pos = torch.randn(2, device=device)
                hist = torch.randn(20, 2, device=device)
                pr_encs.append(self.encode_pr(pos))
                al_encs.append(self.encode_al(hist))

        pr_stack = torch.stack(pr_encs)  # (N, 64)
        al_stack = torch.stack(al_encs)  # (N, 64)

        # 计算两个源编码之间的余弦相似度
        # 低相似度 = 两个源产生可区分的编码（好）
        cos_sim = F.cosine_similarity(
            pr_stack.mean(dim=0).unsqueeze(0),
            al_stack.mean(dim=0).unsqueeze(0)
        ).item()

        return {
            'pr_mean_norm': pr_stack.norm(dim=1).mean().item(),
            'al_mean_norm': al_stack.norm(dim=1).mean().item(),
            'cross_source_cosine': cos_sim,
            'pr_std': pr_stack.std().item(),
            'al_std': al_stack.std().item(),
        }


# ================================================================
# 单元测试
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('dual_source.py — 单元测试')
    print('=' * 60)

    cfg = ProtoConfig()
    encoder = DualSourceEncoder(cfg)

    # ---- Pr 编码器测试 ----
    print('\n--- Pr 编码器 ---')
    pos_single = torch.tensor([0.5, -0.3])
    pr_enc = encoder.encode_pr(pos_single)
    print(f'  输入:  {pos_single.tolist()}')
    print(f'  编码:  shape={pr_enc.shape}  range=[{pr_enc.min().item():.3f}, {pr_enc.max().item():.3f}]')

    # 批量测试
    pos_batch = torch.randn(8, 2)
    pr_batch = encoder.encode_pr(pos_batch)
    print(f'  批量输入 (8,2):  输出 {pr_batch.shape}')

    # 验证冻结
    for name, p in encoder.pr_encoder.named_parameters():
        if p.requires_grad:
            print(f'  ✗ {name} should NOT require grad!')
            break
    else:
        print(f'  ✓ Pr 编码器全部参数已冻结')

    # ---- Al 编码器测试 ----
    print('\n--- Al 编码器 ---')
    hist_single = torch.randn(20, 2)
    al_enc = encoder.encode_al(hist_single)
    print(f'  输入:  (20, 2) 历史序列')
    print(f'  编码:  shape={al_enc.shape}  range=[{al_enc.min().item():.3f}, {al_enc.max().item():.3f}]')

    # 批量测试
    hist_batch = torch.randn(4, 20, 2)
    al_batch = encoder.encode_al(hist_batch)
    print(f'  批量输入 (4,20,2):  输出 {al_batch.shape}')

    # 验证冻结
    for name, p in encoder.al_encoder.named_parameters():
        if p.requires_grad:
            print(f'  ✗ {name} should NOT require grad!')
            break
    else:
        print(f'  ✓ Al 编码器全部参数已冻结')

    # ---- 编码区分性测试 ----
    print('\n--- 编码区分性 ---')
    stats = encoder.get_encoding_stats(n_samples=200)
    print(f'  Pr 编码平均范数:   {stats["pr_mean_norm"]:.3f}')
    print(f'  Al 编码平均范数:   {stats["al_mean_norm"]:.3f}')
    print(f'  Pr 编码标准差:     {stats["pr_std"]:.3f}')
    print(f'  Al 编码标准差:     {stats["al_std"]:.3f}')
    print(f'  跨源余弦相似度:    {stats["cross_source_cosine"]:.4f}')
    print(f'  → 两个源的编码可区分 ?  '
          f'{"✓" if abs(stats["cross_source_cosine"]) < 0.5 else "✗ (相似度偏高)"}')

    # ---- 确定性与一致性 ----
    print('\n--- 确定性验证 ---')
    pos = torch.tensor([0.1, 0.2])
    enc1 = encoder.encode_pr(pos)
    enc2 = encoder.encode_pr(pos)
    diff = (enc1 - enc2).abs().max().item()
    print(f'  同一输入两次编码的最大差异: {diff:.10f}')
    print(f'  → 确定性 ?  {"✓" if diff < 1e-6 else "✗"}')

    print(f'\n{"=" * 60}')
    print(f'dual_source.py 测试完成 ✓')
    print(f'{"=" * 60}')
