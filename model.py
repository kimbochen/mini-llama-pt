from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    n_layers: int
    d_embd: int               # D
    n_heads: int              # N
    seq_len: int = 2048       # T
    vocab_size: int = 32000   # V
    rope_base: float = 1e4
    ffn_mult: int = 256
    norm_eps: float = 1e-5

# Source: Pythia https://github.com/EleutherAI/pythia?tab=readme-ov-file#models
MODEL_CONFIGS = {
    '14M': ModelConfig(n_layers=6, d_embd=128, n_heads=4),
    '31M': ModelConfig(n_layers=6, d_embd=256, n_heads=8),
    '70M': ModelConfig(n_layers=6, d_embd=512, n_heads=8),
    '160M': ModelConfig(n_layers=12, d_embd=768, n_heads=12),
    '410M': ModelConfig(n_layers=24, d_embd=1024, n_heads=16)
}


class Transformer(nn.Module):
    def __init__(
        self, n_layers, vocab_size, pad_token_id, d_embd, seq_len, n_heads, rope_base, **kwargs
    ):
        super().__init__()
        self.embd_token = nn.Embedding(vocab_size, d_embd, padding_idx=pad_token_id)
        self.tsfmr_blks = nn.ModuleList([
            TransformerBlock(d_embd=d_embd, seq_len=seq_len, n_heads=n_heads, **kwargs)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_embd=d_embd, **kwargs)
        self.lm_head = nn.Linear(d_embd, vocab_size, bias=False)
        self.register_buffer('attn_mask', torch.ones([seq_len, seq_len], dtype=torch.bool).tril())
        self.register_buffer('freq_cis_TF', precompute_freq_cis(rope_base, d_embd//n_heads, seq_len))

    def forward(self, idx_BT):
        h_BTD = self.embd_token(idx_BT)

        T = idx_BT.size(1)
        attn_mask = self.attn_mask[:T, :T]
        freq_cis_TF = self.freq_cis_TF[:T, :]

        for tsfmr_blk in self.tsfmr_blks:
            h_BTD = tsfmr_blk(h_BTD, attn_mask, freq_cis_TF)
        h_BTD = self.out_norm(h_BTD)
        logits_BTV = self.lm_head(h_BTD)

        return logits_BTV


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attn_norm = RMSNorm(**kwargs)
        self.attn = SelfAttention(**kwargs)
        self.ffn_norm = RMSNorm(**kwargs)
        self.ffn = FeedForwardNet(**kwargs)

    def forward(self, x_BTD, mask_TT, freq_cis_TF):
        h_BTD = self.attn(self.attn_norm(x_BTD), mask_TT, freq_cis_TF)
        out_BTD = h_BTD + self.ffn(self.ffn_norm(h_BTD))
        return out_BTD


class SelfAttention(nn.Module):
    def __init__(self, d_embd, n_heads, **kwargs):
        super().__init__()
        assert d_embd % n_heads == 0
        self.d_head = d_embd // n_heads  # H
        self.attn_proj = nn.Linear(d_embd, 3*d_embd, bias=False)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=False)

    def forward(self, x_BTD, mask_TT, freq_cis_TF):
        qkv = self.attn_proj(x_BTD).split(x_BTD.size(-1), dim=-1)
        q_BNTH, k_BNTH, v_BNTH = map(lambda z: z.unflatten(-1, [-1, self.d_head]).transpose(1, 2), qkv)
        k_BNTH, v_BNTH = apply_rotary_embd(k_BNTH, freq_cis_TF), apply_rotary_embd(v_BNTH, freq_cis_TF)
        o_BNTH = F.scaled_dot_product_attention(q_BNTH, k_BNTH, v_BNTH, attn_mask=mask_TT, dropout_p=0.0)
        o_BTD = o_BNTH.transpose(1, 2).flatten(-2)
        y_BTD = self.out_proj(o_BTD)
        return y_BTD


class RMSNorm(nn.Module):
    def __init__(self, d_embd, norm_eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_embd))
        self.eps = norm_eps
        self.repr_str = f'RMSNorm(dim={d_embd}, eps={norm_eps})'

    def forward(self, x_BTD):
        r_rms = torch.rsqrt(x_BTD.float().pow(2).mean(-1, keepdim=True) + self.eps)
        y_BTD = (x_BTD.float() * r_rms).type_as(x_BTD) * self.weight
        return y_BTD

    def __repr__(self):
        return self.repr_str


class FeedForwardNet(nn.Module):
    def __init__(self, d_embd, ffn_mult, **kwargs):
        super().__init__()
        hid_dim = int((4 * d_embd) * 2 / 3)  # C
        hid_dim = ffn_mult * ((hid_dim + ffn_mult - 1) // ffn_mult)  # Next multiple of ffn_mult

        self.gate_proj = nn.Linear(d_embd, hid_dim, bias=False)
        self.up_proj = nn.Linear(d_embd, hid_dim, bias=False)
        self.down_proj = nn.Linear(hid_dim, d_embd, bias=False)

    def forward(self, x_BTD):
        h_BTC = F.silu(self.gate_proj(x_BTD)) * self.up_proj(x_BTD)  # SwiGLU
        out_BTD = self.down_proj(h_BTC)
        return out_BTD


def precompute_freq_cis(base, dim, seq_len):
    assert dim % 2 == 0
    theta_F = 1 / (base ** (torch.arange(0, dim, 2).float() / dim))  # F = dim // 2
    pos_idx_T = torch.arange(seq_len)
    freq_TF = pos_idx_T.unsqueeze(1) * theta_F.unsqueeze(0)
    freq_cis_TF = torch.polar(torch.ones_like(freq_TF), freq_TF)
    return freq_cis_TF


def apply_rotary_embd(x_BNTH, freq_cis_TF):
    x_BNTF = x_BNTH.unflatten(-1, [-1, 2])
    x_cis_BNTF = torch.view_as_complex(x_BNTF)
    out_BNTF = x_cis_BNTF * freq_cis_TF.expand_as(x_cis_BNTF)
    out_BNTH = torch.view_as_real(out_BNTF).flatten(-2)
    return out_BNTH.type_as(x_BNTH)


if __name__ == '__main__':
    from dataclasses import asdict
    cfg_m = MODEL_CONFIGS['160M']
    B, T = 2, cfg_m.seq_len
    model = Transformer(**asdict(cfg_m), pad_token_id=128002).to('cuda')
    dummy_x = torch.randint(0, cfg_m.vocab_size, [B, T], device='cuda')
    print(model, model(dummy_x).shape, sep='\n')
