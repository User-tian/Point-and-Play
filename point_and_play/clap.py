"""LAION-CLAP audio tower with token outputs and LoRA adapters."""

from __future__ import annotations

import math

import julius
import torch
import torch.nn as nn
import torch.nn.functional as F
from laion_clap.training.data import get_audio_features

CLAP_SR = 48000
CLAP_MAXLEN = 480000  # CLAP repeat-pads shorter clips to 10 s at 48 kHz.
LORA_TARGETS = ("qkv", "fc1", "fc2")


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update ``(B A) x * alpha / r``."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        if r < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(root: nn.Module, r: int = 8, alpha: int = 16, targets=LORA_TARGETS) -> int:
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets``; return the count."""
    count = 0
    for name, module in list(root.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            parent = root
            *path, leaf = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha))
            count += 1
    return count


class ClapAudio(nn.Module):
    """LAION-CLAP model exposing the 512-d joint embedding and the HTSAT token sequence."""

    def __init__(self, checkpoint, amodel: str = "HTSAT-base", device="cuda"):
        super().__init__()
        import laion_clap

        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=device)
        self.clap.load_ckpt(str(checkpoint))
        self.audio_cfg = self.clap.model_cfg["audio_cfg"]
        self.branch = self.clap.model.audio_branch
        self.proj = self.clap.model.audio_projection
        self._dev = torch.device(device)

    def _batch_dict(self, wav):
        features = []
        for clip in wav:
            item = {}
            get_audio_features(item, clip, CLAP_MAXLEN, data_truncating="rand_trunc",
                               data_filling="repeatpad", audio_cfg=self.audio_cfg,
                               require_grad=clip.requires_grad)
            features.append(item)
        return {"waveform": torch.stack([f["waveform"] for f in features]).to(self._dev),
                "longer": torch.stack([f["longer"] for f in features]).to(self._dev)}

    def forward(self, wav, src_sr: int = CLAP_SR):
        """``wav``: [B, T] mono at ``src_sr``. Returns (embedding [B,512], tokens [B,T',D])."""
        wav = wav.to(self._dev)
        if src_sr != CLAP_SR:
            wav = julius.resample_frac(wav, src_sr, CLAP_SR)
        out = self.branch(self._batch_dict(wav), device=self._dev)
        embedding = F.normalize(self.proj(out["embedding"]), dim=-1)
        return embedding, out["fine_grained_embedding"]

    @torch.no_grad()
    def embed(self, wav, src_sr: int = CLAP_SR):
        return self.forward(wav, src_sr)[0]

    @torch.no_grad()
    def embed_text(self, texts):
        return F.normalize(self.clap.get_text_embedding(list(texts), use_tensor=True).float(), dim=-1)
