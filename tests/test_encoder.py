import torch
import torch.nn as nn
import torch.nn.functional as F

from point_and_play.clap import LoRALinear, inject_lora
from point_and_play.encoder import CueGatedStatsPooling, PointAndPlayEncoder


def test_lora_starts_as_identity_update_and_trains_only_adapters():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    lora = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))
    lora(x).sum().backward()
    assert base.weight.grad is None and lora.B.grad is not None
    assert lora.scale == 2.0


def test_inject_lora_wraps_only_target_layers():
    model = nn.Module()
    model.block = nn.Module()
    model.block.qkv = nn.Linear(4, 12)
    model.block.proj = nn.Linear(4, 4)
    model.fc1 = nn.Linear(4, 4)
    assert inject_lora(model, targets=("qkv", "fc1")) == 2
    assert isinstance(model.block.qkv, LoRALinear) and isinstance(model.fc1, LoRALinear)
    assert isinstance(model.block.proj, nn.Linear)


def test_cgsp_returns_normalized_cue_at_initialization():
    torch.manual_seed(0)
    head = CueGatedStatsPooling(d_tok=16, d_q=8, d=8, out=8)
    tokens, cue = torch.randn(2, 5, 16), torch.randn(2, 8)
    assert torch.allclose(head(tokens, cue), F.normalize(cue, dim=-1), atol=1e-6)


def test_cgsp_is_unit_norm_and_token_order_invariant():
    torch.manual_seed(0)
    head = CueGatedStatsPooling(d_tok=16, d_q=8, d=8, out=8)
    nn.init.normal_(head.res[-1].weight)
    tokens, cue = torch.randn(2, 5, 16), torch.randn(2, 8)
    z = head(tokens, cue)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(z, head(tokens[:, torch.randperm(5)], cue), atol=1e-6)


def test_pool_multi_matches_per_cue_pooling():
    torch.manual_seed(0)

    class Stub:
        extractor = CueGatedStatsPooling(d_tok=16, d_q=8, d=8, out=8)

    nn.init.normal_(Stub.extractor.res[-1].weight)
    tokens, cues = torch.randn(2, 5, 16), torch.randn(2, 3, 8)
    pooled = PointAndPlayEncoder.pool_multi(Stub, tokens, cues)
    for k in range(3):
        assert torch.allclose(pooled[:, k], Stub.extractor(tokens, cues[:, k]), atol=1e-6)
