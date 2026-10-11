"""Point-and-Play timbre extractor: LoRA music-CLAP mixture branch + Cue-Gated Statistics Pooling."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clap import CLAP_SR, LORA_TARGETS, ClapAudio, inject_lora

CODE_DIM = 512


class CueGatedStatsPooling(nn.Module):
    """CGSP (Eq. 1 of the paper).

    a_i = sigmoid((W_h h_i)^T W_q q / sqrt(d)); mu / sigma are the a-weighted mean / std of the
    mixture tokens; z = norm(norm(q) + R([mu; sigma])). The last layer of R is zero-initialized,
    so z equals the normalized cue embedding at initialization.
    """

    def __init__(self, d_tok: int, d_q: int = CODE_DIM, d: int = 512, out: int = CODE_DIM):
        super().__init__()
        self.tok_proj = nn.Linear(d_tok, d)
        self.q_proj = nn.Linear(d_q, d)
        self.scale = 1.0 / math.sqrt(d)
        self.res = nn.Sequential(nn.Linear(2 * d_tok, out), nn.GELU(), nn.Linear(out, d_q))
        nn.init.zeros_(self.res[-1].weight)
        nn.init.zeros_(self.res[-1].bias)

    def forward(self, tokens, q):
        """tokens: [B, T, d_tok]; q: [B, d_q] cue embedding -> timbre code [B, d_q]."""
        score = (self.tok_proj(tokens) * self.q_proj(q).unsqueeze(1)).sum(-1) * self.scale
        gate = torch.sigmoid(score).unsqueeze(-1)
        weight = gate.sum(dim=1).clamp_min(1e-6)
        mean = (gate * tokens).sum(dim=1) / weight
        second = (gate * tokens.pow(2)).sum(dim=1) / weight
        std = (second - mean.pow(2)).clamp_min(1e-8).sqrt()
        residual = self.res(torch.cat([mean, std], dim=-1))
        return F.normalize(F.normalize(q, dim=-1) + residual, dim=-1)


class PointAndPlayEncoder(nn.Module):
    """Cue-conditioned timbre extractor.

    ``frozen``    : music-CLAP used for audio/text cues and for the isolated-source teacher.
    ``lora``      : the same music-CLAP with LoRA adapters; encodes the mixture.
    ``extractor`` : CGSP head. Only LoRA and CGSP parameters are trained.
    """

    def __init__(self, clap_checkpoint, device="cuda", lora_r: int = 8, lora_alpha: int = 16,
                 lora_targets=LORA_TARGETS, head_dim: int = 512, amodel: str = "HTSAT-base"):
        super().__init__()
        self.dev = torch.device(device)
        self.frozen = ClapAudio(clap_checkpoint, amodel, device)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)
        self.frozen.clap.eval()

        self.lora = ClapAudio(clap_checkpoint, amodel, device)
        for parameter in self.lora.parameters():
            parameter.requires_grad_(False)
        self.n_lora = inject_lora(self.lora.branch, r=lora_r, alpha=lora_alpha,
                                  targets=tuple(lora_targets))
        if self.n_lora == 0:
            raise ValueError(f"no LoRA target layers matched {lora_targets}")
        self.lora.to(device)

        with torch.no_grad():
            _, tokens = self.lora(torch.zeros(1, CLAP_SR, device=device))
        self.d_tok = tokens.shape[-1]
        self.extractor = CueGatedStatsPooling(self.d_tok, d=head_dim).to(device)

    # ---- frozen music-CLAP: cues and teacher ----
    def audio_cue(self, wav, src_sr: int = CLAP_SR):
        return self.frozen.embed(wav, src_sr)

    def text_cue(self, words):
        return self.frozen.embed_text(words).to(self.dev)

    def teacher(self, wav, src_sr: int = CLAP_SR):
        return self.frozen.embed(wav, src_sr)

    # ---- LoRA mixture branch + CGSP ----
    def mixture_tokens(self, mix, src_sr: int = CLAP_SR):
        """One LoRA-CLAP pass: (global embedding g [B,512], tokens H [B,T,d_tok])."""
        return self.lora(mix, src_sr)

    def global_embedding(self, mix, src_sr: int = CLAP_SR):
        return self.lora(mix, src_sr)[0]

    def pool(self, tokens, cue):
        return self.extractor(tokens, cue)

    def pool_multi(self, tokens, cues):
        """Apply CGSP for K cues per mixture, reusing the tokens. cues: [B,K,512] -> [B,K,512]."""
        batch, k, dim = cues.shape
        steps, d_tok = tokens.shape[1], tokens.shape[2]
        expanded = tokens.unsqueeze(1).expand(batch, k, steps, d_tok).reshape(batch * k, steps, d_tok)
        return self.extractor(expanded, cues.reshape(batch * k, dim)).reshape(batch, k, -1)

    def extract(self, mix, cue, src_sr: int = CLAP_SR):
        """Timbre code of the cued source. mix: [B,T] mono at ``src_sr``; cue: [B,512]."""
        _, tokens = self.lora(mix, src_sr)
        return self.extractor(tokens, cue)

    def trainable_parameters(self):
        return [p for p in self.lora.parameters() if p.requires_grad] + list(self.extractor.parameters())

    def set_train(self, mode: bool = True):
        # The CLAP backbones stay in eval mode (no dropout / drop-path); gradients still reach LoRA.
        self.lora.eval()
        self.extractor.train(mode)
        self.frozen.clap.eval()
        return self

    def lora_state_dict(self):
        return {k: v for k, v in self.lora.state_dict().items() if k.endswith((".A", ".B"))}

    def checkpoint(self, config: dict, step: int, **extra) -> dict:
        return {"lora": self.lora_state_dict(), "extractor": self.extractor.state_dict(),
                "args": dict(config), "step": step, **extra}


def load_encoder(checkpoint_path, clap_checkpoint, device="cuda") -> PointAndPlayEncoder:
    """Rebuild a frozen encoder from a checkpoint written by ``train_encoder``."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = payload.get("args", {}) or {}
    if config.get("ext_head", "gatestats") != "gatestats" or config.get("clap_space", "music") != "music":
        raise ValueError("checkpoint is not a Point-and-Play CGSP / music-CLAP encoder")
    targets = config.get("lora_targets", LORA_TARGETS)
    if isinstance(targets, str):
        targets = tuple(t for t in targets.split(",") if t)
    model = PointAndPlayEncoder(
        clap_checkpoint, device=device,
        lora_r=int(config.get("lora_r", 8)), lora_alpha=int(config.get("lora_alpha", 16)),
        lora_targets=tuple(targets), head_dim=int(config.get("ext_dim", 512)),
    )
    expected = set(model.lora_state_dict())
    if set(payload["lora"]) != expected:
        raise ValueError("checkpoint LoRA keys do not match the encoder architecture")
    model.lora.load_state_dict(payload["lora"], strict=False)
    model.extractor.load_state_dict(payload["extractor"])
    model.set_train(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
