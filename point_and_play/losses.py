"""Extractor objectives of Eq. (2)-(3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(a, b, temperature: float = 0.1):
    """Symmetric InfoNCE between row-aligned embeddings."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def _same_label_mask(labels, device):
    """True where two different rows share a label (false negatives excluded from InfoNCE)."""
    index = {}
    ids = torch.tensor([index.setdefault(label, len(index)) for label in labels], device=device)
    same = ids[:, None] == ids[None, :]
    return same & ~torch.eye(len(labels), dtype=torch.bool, device=device)


def _mixture_weights(valid, rows, batch):
    """Weight 1 / (B * K_valid) per valid source so every mixture contributes equally."""
    return (1.0 / (batch * valid.sum(dim=1).clamp_min(1).float()))[rows]


def alignment_loss(codes, targets, valid, labels, temperature: float = 0.1):
    """Per-mixture-normalized symmetric InfoNCE between timbre codes and isolated-source teachers.

    codes, targets: [B,K,512]; valid: [B,K] bool; labels: B lists of K labels (None when padded).
    Same-label pairs are masked out of the negatives.
    """
    batch = codes.shape[0]
    valid = valid.to(codes.device)
    index = valid.nonzero(as_tuple=False)
    if index.shape[0] == 0:
        return codes.sum() * 0.0
    rows, slots = index[:, 0], index[:, 1]
    z = F.normalize(codes[rows, slots], dim=-1)
    t = F.normalize(targets[rows, slots], dim=-1)
    flat_labels = [labels[int(b)][int(k)] for b, k in index.tolist()]
    logits = (z @ t.t() / temperature).masked_fill(_same_label_mask(flat_labels, codes.device),
                                                   float("-inf"))
    target = torch.arange(z.shape[0], device=codes.device)
    weight = _mixture_weights(valid, rows, batch)
    return 0.5 * ((weight * F.cross_entropy(logits, target, reduction="none")).sum()
                  + (weight * F.cross_entropy(logits.t(), target, reduction="none")).sum())


def selectivity_loss(wrong_codes, targets, valid, margin: float = 0.0):
    """Hinge [cos(z_wrong, t) - m]_+ averaged per mixture (L_sel^a)."""
    batch = wrong_codes.shape[0]
    valid = valid.to(wrong_codes.device)
    index = valid.nonzero(as_tuple=False)
    if index.shape[0] == 0:
        return wrong_codes.sum() * 0.0
    rows, slots = index[:, 0], index[:, 1]
    t = F.normalize(targets[rows, slots], dim=-1)
    cosine = (F.normalize(wrong_codes[rows, slots], dim=-1) * t).sum(-1)
    return (_mixture_weights(valid, rows, batch) * F.relu(cosine - margin)).sum()


def wrong_audio_cue_mask(valid, labels):
    """Valid slots for the mismatched audio cue obtained by rolling cues by one mixture.

    Slot (b, k) receives the cue of (b-1, k); it is kept only when both slots are real and the
    two sources have different labels (instrument identity for NSynth/NLakh, stem class for
    Slakh/MoisesDB).
    """
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be a [B,K] boolean tensor")
    batch, slots = valid.shape
    if len(labels) != batch or any(len(row) != slots for row in labels):
        raise ValueError("labels must match the [B,K] validity mask")
    same = torch.tensor(
        [[label is not None and labels[b - 1][k] is not None and label == labels[b - 1][k]
          for k, label in enumerate(row)] for b, row in enumerate(labels)],
        dtype=torch.bool, device=valid.device)
    return valid & torch.roll(valid, 1, dims=0) & ~same
