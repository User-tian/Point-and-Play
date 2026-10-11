import json
import math

import numpy as np
import pytest
import torch

from point_and_play.flowsynth import conditions
from point_and_play.flowsynth import train as train_cli
from point_and_play.flowsynth.config import TINY_MODEL
from point_and_play.flowsynth.latents import LatentDataset, write_cache
from point_and_play.flowsynth.model import FlowSynthDiT
from point_and_play.flowsynth.sampling import sample_euler
from point_and_play.flowsynth.training import GlobalBatchStream


def test_model_shapes_and_zero_initialized_output():
    model = FlowSynthDiT(TINY_MODEL)
    x = torch.randn(3, 64, 86)
    out = model(x, torch.rand(3), torch.randn(3, 512), torch.tensor([21, 60, 108]), torch.tensor([1, 64, 127]))
    assert out.shape == x.shape and torch.count_nonzero(out) == 0
    with pytest.raises(ValueError):
        model(x, torch.rand(3), torch.randn(3, 512), torch.tensor([60, 60, 128]), torch.tensor([1, 1, 1]))


def test_euler_integrates_constant_velocity_exactly():
    class Constant(torch.nn.Module):
        def forward(self, latent, timestep, code, pitch, velocity):
            return torch.ones_like(latent) * 2.0

    x0 = torch.zeros(2, 4, 3)
    out = sample_euler(Constant(), x0=x0, code=torch.zeros(2, 512), pitch=torch.tensor([60, 60]),
                       velocity=torch.tensor([90, 90]), steps=50)
    assert torch.allclose(out, torch.full_like(x0, 2.0), atol=1e-5)


def test_global_batch_stream_partitions_and_resumes():
    streams = [GlobalBatchStream(size=20, batch_per_rank=3, rank=r, world_size=2, seed=7) for r in range(2)]
    seen = [streams[0].next_indices() + streams[1].next_indices() for _ in range(3)]
    assert all(len(set(batch)) == 6 for batch in seen)
    assert len(set(seen[0] + seen[1] + seen[2])) == 18  # one epoch of 20 without repetition
    epoch, offset = streams[0].position
    resumed = GlobalBatchStream(size=20, batch_per_rank=3, rank=0, world_size=2, seed=7, epoch=epoch, offset=offset)
    assert resumed.next_indices() == streams[0].next_indices()


def _tiny_caches(tmp_path, rows=24):
    rng = np.random.default_rng(0)
    arrays = {"lat": rng.standard_normal((rows, 64, 86)).astype(np.float16),
              "pitch": rng.integers(21, 109, rows).astype(np.int16),
              "vel": rng.integers(1, 128, rows).astype(np.int16),
              "row_id": np.arange(rows, dtype=np.int64)}
    latents = write_cache(tmp_path / "latents", arrays, {"test": "1"})
    conditions_path = tmp_path / "conditions.npy"
    np.save(conditions_path, rng.standard_normal((rows, 512)).astype(np.float32))
    return latents, conditions_path


def test_latent_dataset_normalizes(tmp_path):
    latents, _ = _tiny_caches(tmp_path)
    dataset = LatentDataset(latents)
    x, pitch, velocity = dataset.batch(list(range(len(dataset))))
    assert abs(float(x.mean())) < 1e-3 and math.isclose(float(x.std()), 1.0, rel_tol=1e-2)
    assert pitch.dtype == velocity.dtype == torch.long


def test_decoder_training_resumes_bit_exactly(tmp_path):
    latents, conditions_path = _tiny_caches(tmp_path)
    common = ["--latents", str(latents), "--conditions", str(conditions_path), "--device", "cpu",
              "--tiny-model", "--batch-per-gpu", "4", "--steps", "6", "--log-every", "1"]
    train_cli.main(common + ["--out", str(tmp_path / "straight"), "--checkpoint-every", "100"])
    train_cli.main(common + ["--out", str(tmp_path / "resumed"), "--checkpoint-every", "3"])
    # Simulate an interruption after step 3 by resuming from the step-3 checkpoint into a copy.
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "config.json").write_text((tmp_path / "resumed" / "config.json").read_text())
    (interrupted / "checkpoints").mkdir()
    (interrupted / "checkpoints" / "latest.pt").write_bytes(
        (tmp_path / "resumed" / "checkpoints" / "latest.pt").read_bytes())
    train_cli.main(common + ["--out", str(interrupted), "--checkpoint-every", "3", "--resume"])
    straight = torch.load(tmp_path / "straight" / "checkpoints" / "final.pt", weights_only=False)
    resumed = torch.load(interrupted / "checkpoints" / "final.pt", weights_only=False)
    for key in ("model", "ema"):
        for name, value in straight[key].items():
            assert torch.equal(value, resumed[key][name]), (key, name)
    assert resumed["train_state"]["step"] == 6 and resumed["latent_normalization"]["std"] > 0
    log = [json.loads(line) for line in (tmp_path / "straight" / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in log] == [1, 2, 3, 4, 5, 6]


def test_condition_rows_are_deterministic_and_mixing_hits_target_snr():
    first = [conditions.row_rng(1234, "train", row).random() for row in range(3)]
    assert first == [conditions.row_rng(1234, "train", row).random() for row in range(3)]
    rng = np.random.default_rng(0)
    target, interferer = rng.standard_normal(16000), rng.standard_normal(16000) * 3
    mixed = conditions.mix_at_snr(target, interferer, 10.0)
    snr = 20 * math.log10(conditions._rms(target) / conditions._rms(mixed - target))
    assert math.isclose(snr, 10.0, abs_tol=1e-6)
