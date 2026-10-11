import math

import torch
import torch.nn.functional as F

from point_and_play import losses


def test_alignment_loss_reduces_to_infonce_for_one_source_per_mixture():
    torch.manual_seed(0)
    z, t = torch.randn(6, 1, 16), torch.randn(6, 1, 16)
    valid = torch.ones(6, 1, dtype=torch.bool)
    labels = [[f"id{i}"] for i in range(6)]
    expected = losses.info_nce(z[:, 0], t[:, 0], 0.1)
    assert torch.allclose(losses.alignment_loss(z, t, valid, labels, 0.1), expected, atol=1e-6)


def test_alignment_loss_masks_same_label_negatives_and_padding():
    torch.manual_seed(0)
    z = torch.randn(2, 2, 8)
    t = z.clone()
    t[0, 1] = t[1, 0]  # identical teachers would be false negatives
    valid = torch.tensor([[True, True], [True, False]])
    labels = [["a", "b"], ["b", None]]
    loss = losses.alignment_loss(z, t, valid, labels, 0.1)
    assert torch.isfinite(loss)
    padded = losses.alignment_loss(torch.cat([z, torch.randn(2, 1, 8)], 1),
                                   torch.cat([t, torch.randn(2, 1, 8)], 1),
                                   torch.cat([valid, torch.zeros(2, 1, dtype=torch.bool)], 1),
                                   [row + [None] for row in labels], 0.1)
    assert torch.allclose(loss, padded)


def test_selectivity_hinge_and_per_mixture_weighting():
    t = F.normalize(torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]), dim=-1)
    wrong = torch.tensor([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True], [True, False]])
    # mixture 0: mean(cos 0.707, 0) / 2 mixtures; mixture 1: cos 1 / 2 mixtures
    expected = 0.5 * (math.sqrt(0.5) / 2) + 0.5 * 1.0
    assert math.isclose(float(losses.selectivity_loss(wrong, t, valid, 0.0)), expected, rel_tol=1e-6)
    assert float(losses.selectivity_loss(wrong, t, valid, 1.0)) == 0.0


def test_wrong_audio_cue_mask_drops_padding_and_same_label_rolls():
    valid = torch.tensor([[True, True], [True, False], [True, True]])
    labels = [["a", "b"], ["c", None], ["c", "d"]]
    mask = losses.wrong_audio_cue_mask(valid, labels)
    # row b receives the cues of row b-1 (row 0 receives row 2)
    assert mask.tolist() == [[True, True], [True, False], [False, False]]
