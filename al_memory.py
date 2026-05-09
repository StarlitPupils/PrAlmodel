# al_memory.py —— Al 联想记忆层（替代 ESN 编码器）
import torch
import torch.nn as nn
import torch.nn.functional as F

class AssociativeMemory(nn.Module):
    """
    内容寻址联想记忆 —— 模拟海马体的模式完成功能。
    
    给定当前输入，一步跳到最相似的历史模式，
    不用串行扫描整个序列。
    
    原理：Modern Hopfield Network (MHN)
    - 存储：所有历史 hidden state 存入记忆矩阵
    - 检索：新输入与记忆矩阵做 softmax 相似度 → 加权检索
    - 复杂度：O(1) 的检索路径，不受序列长度限制
    
    这解决了"Al 为什么不能一步直达远历史"的问题。
    """
    def __init__(self, input_dim=128, memory_size=256, beta=8.0):
        super().__init__()
        self.beta = beta  # 检索锐度（越大越"硬"检索）
        self.register_buffer("memory", torch.randn(memory_size, input_dim) * 0.01)
        self.memory_ptr = 0
        self.memory_filled = 0
    
    def store(self, x):
        """将当前状态存入记忆。x: (B, D)"""
        B = x.shape[0]
        for i in range(B):
            idx = (self.memory_ptr + i) % len(self.memory)
            self.memory[idx] = x[i].detach()
        self.memory_ptr = (self.memory_ptr + B) % len(self.memory)
        self.memory_filled = min(self.memory_filled + B, len(self.memory))
    
    def retrieve(self, query):
        """
        一步检索最相似的记忆。
        query: (B, D) or (B, T, D)
        returns: (B, D) or (B, T, D) 检索到的记忆向量
        """
        # 确保 memory 与 query 同设备
        if self.memory.device != query.device:
            self.memory = self.memory.to(query.device)
        if query.dim() == 2:
            B, D = query.shape
            # 归一化
            q_norm = F.normalize(query, dim=-1)
            m_norm = F.normalize(self.memory[:max(1, self.memory_filled)], dim=-1)
            # 相似度 → softmax → 加权检索
            sim = q_norm @ m_norm.T * self.beta  # (B, M)
            attn = F.softmax(sim, dim=-1)         # (B, M)
            retrieved = attn @ self.memory[:max(1, self.memory_filled)]  # (B, D)
            return retrieved
        else:
            B, T, D = query.shape
            retrieved_list = []
            for t in range(T):
                retrieved_list.append(self.retrieve(query[:, t, :]))
            return torch.stack(retrieved_list, dim=1)
