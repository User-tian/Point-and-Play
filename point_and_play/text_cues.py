"""Category-name text cues ("guitar", "bass", ...) embedded by the frozen music-CLAP text tower."""

from __future__ import annotations

import torch


def family(label: str) -> str:
    """Instrument category of an identity label, e.g. ``guitar_acoustic_010`` -> ``guitar``."""
    return label.split("_")[0]


class TextCueBank:
    """Frozen-CLAP embeddings of the sorted category names occurring in ``labels``."""

    def __init__(self, encoder, labels, device=None):
        self.device = torch.device(device) if device is not None else encoder.dev
        self.families = sorted({family(label) for label in labels})
        self.index = {name: i for i, name in enumerate(self.families)}
        self.embeddings = encoder.text_cue(self.families).to(self.device)
        self.dim = self.embeddings.shape[1]

    def cues(self, labels):
        """[N,512] category cue for each label."""
        return self.embeddings[[self.index[family(label)] for label in labels]]

    def wrong_cues(self, labels):
        """[N,512] cue of the next category in sorted order (always a mismatched category)."""
        n = len(self.families)
        return self.embeddings[[(self.index[family(label)] + 1) % n for label in labels]]

    def batch_cues(self, labels2d, valid):
        """[B,K,512] category cue per source slot; padded slots are zero."""
        batch, slots = len(labels2d), len(labels2d[0])
        out = torch.zeros(batch, slots, self.dim, device=self.device)
        for b in range(batch):
            for k in range(slots):
                label = labels2d[b][k]
                if bool(valid[b, k]) and label is not None:
                    out[b, k] = self.embeddings[self.index[family(label)]]
        return out
