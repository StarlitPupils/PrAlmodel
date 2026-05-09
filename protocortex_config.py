# protocortex_config.py —— 极限优化版
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ProtoConfig:
    n_neurons: int = 4096
    pr_zone: Tuple[int, int] = (0, 682)
    iz_zone: Tuple[int, int] = (683, 1365)
    al_zone: Tuple[int, int] = (1366, 2047)
    pr_encoding_dim: int = 128
    al_encoding_dim: int = 128
    iz_encoding_dim: int = 128

    esn_reservoir_size: int = 128
    esn_spectral_radius: float = 0.95
    esn_input_scaling: float = 0.5

    reserve_sparsity: float = 0.10
    reserve_spectral_radius: float = 0.98
    reserve_input_scaling: float = 0.5

    axial_lr: float = 0.005
    axial_update_interval: int = 100

    readout_hidden: int = 512
    readout_layers: int = 4
    readout_lr: float = 0.001
    readout_grad_clip: float = 1.0

    iz_growth_threshold: float = 0.1
    iz_growth_cooldown: int = 25
    iz_max_capacity: int = 1024
    iz_prune_threshold: float = 0.01
    hebbian_window: int = 50
    init_connection_weight: float = 0.01
    prune_threshold: float = 0.001
    prune_patience_steps: int = 100

    phase1_epochs: int = 50
    phase2_epochs: int = 200
    phase3_epochs: int = 300
    reward_temperature: float = 0.5

    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "E:/PrAlmodel/checkpoints"

    # ========== 极限优化参数 ==========
    batch_sequences: int = 50       # 50→120，利用12GB显存
    use_amp: bool = False             # 混合精度训练
    amp_dtype: str = "float16"       # FP16 使用张量核心

_default_config = ProtoConfig()
def get_config() -> ProtoConfig:
    return _default_config
