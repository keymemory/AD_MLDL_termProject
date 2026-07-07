import torch

from llava.model.llava_arch import (
    _attention_gain_elbow_indices,
    _greedy_cosine_gain_elbow_indices,
    _l2norm,
)


def main():
    attn = torch.cat([
        torch.linspace(0.02, 0.01, 40),
        torch.linspace(0.002, 0.0005, 536),
    ])
    attn = attn / attn.sum()

    feats = _l2norm(torch.randn(576, 32))
    imp_gain, n_imp, attn_gains = _attention_gain_elbow_indices(attn)
    div_gain, n_div, div_gains = _greedy_cosine_gain_elbow_indices(feats, imp_gain)
    assert imp_gain.numel() == n_imp
    assert div_gain.numel() == n_div
    assert attn_gains.numel() == attn.numel() - 1
    assert div_gains.numel() <= attn.numel() - imp_gain.numel()

    print("AGG selection smoke: ok")


if __name__ == "__main__":
    main()
