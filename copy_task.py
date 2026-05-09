# copy_task.py —— 数学 Copy 任务生成器
import numpy as np
from typing import Tuple, Dict
import torch

class CopyTaskGenerator:
    """生成 Copy 任务数据：记住一段随机序列，延迟后复现。"""
    
    def __init__(self, seq_len: int = 10, blank_len: int = 20, seed: int = None):
        self.seq_len = seq_len
        self.blank_len = blank_len
        self.rng = np.random.default_rng(seed)
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            input_seq:  (total_len,)  输入序列
            output_seq: (total_len,)  期望输出（前 blank 步为 0）
        """
        total = self.seq_len + self.blank_len + 1 + self.seq_len  # 10 + 20 + 1 + 10 = 41
        seq = self.rng.uniform(-1, 1, self.seq_len).astype(np.float32)
        
        # 输入：随机序列 + 空白 + 标记(2.0) + 空白输出位
        input_seq = np.zeros(total, dtype=np.float32)
        input_seq[:self.seq_len] = seq
        input_seq[self.seq_len + self.blank_len] = 2.0  # 标记：开始输出
        
        # 期望输出：延迟后在最后 10 步复现原序列
        output_seq = np.zeros(total, dtype=np.float32)
        output_seq[-self.seq_len:] = seq
        
        return input_seq, output_seq
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, outputs = [], []
        for _ in range(batch_size):
            inp, out = self.generate()
            inputs.append(inp)
            outputs.append(out)
        return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(outputs))
