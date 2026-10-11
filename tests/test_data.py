import json

import numpy as np
import soundfile as sf
import torch

from point_and_play.data.catalog import index_nsynth, nlakh_identity_map, normalize_preset
from point_and_play.data.mixtures import DistinctKeySampler, NSynthMixtures, collate_mixtures

IDENTITIES = ["guitar_acoustic_001", "guitar_electronic_002", "bass_synthetic_003", "keyboard_acoustic_004"]


def _fake_nsynth(root, notes=4, seconds=0.25):
    """Constant-valued notes: amplitude encodes (identity, note) so items can be decoded."""
    root.mkdir(parents=True)
    for i, identity in enumerate(IDENTITIES):
        for j in range(notes):
            value = (i + 1) * 0.1 + j * 0.001
            sf.write(root / f"{identity}-{40 + j:03d}-100.wav",
                     np.full(int(seconds * 16000), value, np.float32), 16000)
    return root


def _identity_of(wav):
    return IDENTITIES[int(round(float(wav[0]) / 0.1)) - 1]


def test_index_nsynth_note_split_partitions_notes(tmp_path):
    root = _fake_nsynth(tmp_path / "audio", notes=12)
    train, test, every = (index_nsynth(root, s) for s in ("train", "test", None))
    for identity in IDENTITIES:
        assert sorted(train.get(identity, []) + test.get(identity, [])) == every[identity]
        assert not set(train.get(identity, [])) & set(test.get(identity, []))


def test_nsynth_mixtures_sources_and_class_cues(tmp_path):
    root = _fake_nsynth(tmp_path / "audio")
    ds = NSynthMixtures(root, seconds=0.25, n_interf=2, n_sources=3, note_split=None, seed=0)
    for index in range(len(ds)):
        item = ds[index]
        assert item["labels"][0] == ds.files[index][1] and len(set(item["labels"])) == 3
        for label, isolated, cue in zip(item["labels"], item["isolated"], item["cues"]):
            assert _identity_of(isolated) == label
            if label.startswith("guitar"):  # the only category with a second instrument
                assert _identity_of(cue) != label and _identity_of(cue).startswith("guitar")
            else:
                assert _identity_of(cue) == label
        assert item["mix"].shape == item["mixB"].shape == (4000,)
    batch = collate_mixtures([ds[0], ds[1]])
    assert batch["cues"].shape == (2, 3, 4000) and batch["valid"].all()


def test_same_seed_gives_same_stream(tmp_path):
    root = _fake_nsynth(tmp_path / "audio")
    a, b = (NSynthMixtures(root, seconds=0.25, n_interf=2, note_split=None, seed=3) for _ in range(2))
    for index in (0, 5, 9):
        assert torch.equal(a[index]["mix"], b[index]["mix"])


def test_collate_pads_missing_sources():
    item = lambda k: {"mix": torch.zeros(4), "mixB": torch.zeros(4), "cues": [torch.ones(4)] * k,
                      "isolated": [torch.ones(4)] * k, "labels": ["x"] * k, "domain": "slakh"}
    batch = collate_mixtures([item(1), item(3)])
    assert batch["valid"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["labels"][0] == ["x", None, None] and batch["cues"][0, 1:].abs().sum() == 0


def test_distinct_key_sampler():
    keys = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    for batch in DistinctKeySampler(keys, batch_size=3, steps=10, seed=0):
        assert len({keys[i] for i in batch}) == 3


def test_nlakh_identity_map(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    presets = {f"synth_lead_synthetic_{i}": i for i in range(53)}
    (maps / "idx_to_fam_inst_valid.json").write_text(json.dumps(presets))
    split = tmp_path / "valid"
    for i in range(1, 54):
        (split / f"{i:02d}").mkdir(parents=True)
    mapping = nlakh_identity_map(split, maps, "valid")
    assert mapping["01"] == "synth_lead_synthetic_000" and mapping["53"] == "synth_lead_synthetic_052"
    assert normalize_preset("guitar_acoustic_7") == "guitar_acoustic_007"
