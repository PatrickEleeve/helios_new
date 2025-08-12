import torch
import pytest

from helios.adapters import EdgeAdapter


@pytest.mark.parametrize("B,T,src,dst", [
    (2, 8, 32, 64),
    (1, 5, 64, 64),
])
def test_edge_adapter_shapes_and_masks(B, T, src, dst):
    torch.manual_seed(0)
    x = torch.randn(B, T, src)
    attn = torch.ones(B, T, dtype=torch.long)
    attn[:, -2:] = 0  # pad last two positions

    ea = EdgeAdapter(src_dim=src, dst_dim=dst, nhead=4, ffn_mult=2, dropout=0.0)

    # no attention mask
    y = ea(x)
    assert y.shape == (B, T, dst)
    assert torch.isfinite(y).all()

    # with [B,T] attention mask
    y2 = ea(x, attention_mask=attn)
    assert y2.shape == (B, T, dst)
    assert torch.isfinite(y2).all()

    # with broadcast-style [B,1,1,T] attention mask
    y3 = ea(x, attention_mask=attn[:, None, None, :])
    assert y3.shape == (B, T, dst)
    assert torch.isfinite(y3).all()

