# helios/adapters.py
import torch
import torch.nn as nn
from typing import Optional

class EdgeAdapter(nn.Module):
    """
    可训练的“边”：把源节点隐藏态映射到目标节点隐藏态空间，并做一次序列变换。
    结构：Linear(src->dst) -> 1层TransformerEncoderLayer -> LayerNorm
    仅处理 dense hidden，不涉及 tokenization 或 lm_head。
    """
    def __init__(
        self,
        src_dim: int,
        dst_dim: int,
        nhead: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.proj = nn.Linear(src_dim, dst_dim, bias=False) if src_dim != dst_dim else nn.Identity()
        self.enc = nn.TransformerEncoderLayer(
            d_model=dst_dim,
            nhead=nhead,
            dim_feedforward=dst_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(dst_dim)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # 形状[T,T]，上三角为 -inf，兼容 nn.Transformer 的 attn_mask 语义
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    @staticmethod
    def _key_padding_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        将 [B,T] 的 1/0 attention_mask 转成 key_padding_mask: True 表示需要mask的pad位。
        nn.TransformerEncoderLayer 的参数是 src_key_padding_mask（pad=True 屏蔽）。
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # [B,1,1,T] -> [B,T]
            attention_mask = attention_mask.squeeze(1).squeeze(1)
        return (attention_mask == 0)

    def forward(
        self,
        src_hidden: torch.Tensor,            # [B, T, C_src]
        attention_mask: Optional[torch.Tensor] = None,  # [B,T] 或 [B,1,1,T]
    ) -> torch.Tensor:
        # 懒移动到输入张量的设备/精度，以兼容 device_map="auto"
        dev, dt = src_hidden.device, src_hidden.dtype
        if next(self.parameters(), None) is not None:
            p = next(self.parameters())
            if p.device != dev or p.dtype != dt:
                self.to(device=dev, dtype=dt)

        x = self.proj(src_hidden)            # [B, T, C_dst]
        B, T, _ = x.shape
        attn_mask = self._causal_mask(T, x.device)
        kpm = self._key_padding_mask(attention_mask)
        x = self.enc(x, mask=attn_mask, src_key_padding_mask=kpm)  # [B,T,C_dst]
        return self.norm(x)
