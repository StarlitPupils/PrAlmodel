# neurons_hierarchical.py —— 层级预测性编码神经元层
import torch
import torch.nn as nn

class HierarchicalLayer(nn.Module):
    """一层预测性编码神经元。预测下一层的活动，用预测误差更新自身权重。"""
    def __init__(self, n_neurons, n_input, n_next, is_top=False):
        super().__init__()
        self.n_neurons = n_neurons
        self.is_top = is_top
        
        # 前向权重：输入 → 本层
        self.W_fwd = nn.Parameter(torch.randn(n_neurons, n_input) * 0.02)
        # 预测权重：本层 → 上层（预测上层活动）
        self.W_pred = nn.Parameter(torch.randn(n_next, n_neurons) * 0.02) if not is_top else None
        # 偏置
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        # 赫布学习率
        self.heb_lr = 0.0001
        # 资格迹
        self.register_buffer("eligibility", torch.zeros(n_neurons, n_input))
        self.elig_decay = 0.9
    
    def forward(self, input_act, prediction_from_above=None):
        """前向传播。
        input_act: (B, n_input) 来自下一层的活动
        prediction_from_above: (B, n_neurons) 上层对本层的预测（可选）
        返回: (B, n_neurons) 本层活动, (B, n_neurons) 预测误差
        """
        # 计算本层活动
        act = torch.tanh((self.W_fwd @ input_act.T).T + self.bias)
        
        # 预测误差：上层预测 vs 实际活动
        if prediction_from_above is not None:
            error = act - prediction_from_above
        else:
            error = torch.zeros_like(act)
        
        return act, error
    
    def predict_layer_below(self, below_act):
        """用本层活动预测下一层的活动。仅顶层调用。"""
        if self.W_pred is None:
            return None
        return torch.tanh((self.W_pred @ below_act.T).T)
    
    def hebbian_update(self, input_act, error, reward=1.0):
        """用局部预测误差 + 奖励调制更新前向权重。"""
        with torch.no_grad():
            # 资格迹累积
            self.eligibility *= self.elig_decay
            self.eligibility += torch.outer(error.mean(dim=0), input_act.mean(dim=0))
            self.eligibility.clamp_(-10.0, 10.0)
            # 赫布更新
            self.W_fwd.data += self.heb_lr * reward * self.eligibility * error.abs().mean().item()
            self.W_fwd.data.clamp_(-1.0, 1.0)
