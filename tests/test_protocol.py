import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_fsor.data import load_rows
from boundary_fsor.episodes import EpisodeSampler, scheduled_rank_weights


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        labels = np.repeat(np.arange(12), 25)
        features = np.arange(len(labels) * 8, dtype=np.float32).reshape(len(labels), 8)
        self.sampler = EpisodeSampler(features, labels, ways=5, shots=5, queries=10,
                                      open_ways=5, seed=17)

    def test_class_disjointness_and_exact_support(self):
        episode = self.sampler.sample(3)
        self.assertFalse(set(episode.known_classes) & set(episode.unknown_classes))
        self.assertEqual(len(episode.support), 25)
        self.assertTrue(all(np.sum(episode.support_labels == label) == 5 for label in range(5)))

    def test_episode_seed_is_local(self):
        first = self.sampler.sample(11)
        np.random.seed(999); np.random.random(10000)
        second = self.sampler.sample(11)
        np.testing.assert_array_equal(first.support, second.support)

    def test_curriculum_moves_from_easy_to_hard(self):
        values = np.array([0.0, 0.5, 1.0])
        early = scheduled_rank_weights(values, 0.0, 0.25)
        late = scheduled_rank_weights(values, 1.0, 0.25)
        self.assertGreater(early[0], early[-1])
        self.assertGreater(late[-1], late[0])

    def test_ns100_validation_classes_are_loaded_from_training_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); metadata = root / "meta"; metadata.mkdir()
            pd.DataFrame([{"instrument": "a", "audio_source": "nsynth-train",
                           "filename": "validation-class"}]).to_csv(
                               metadata / "nsynth-100-fs_train.csv", index=False)
            pd.DataFrame([{"instrument": "b", "audio_source": "nsynth-valid",
                           "filename": "published-base-val"}]).to_csv(
                               metadata / "nsynth-100-fs_val.csv", index=False)
            (metadata / "nsynth-100-fs_vocab.json").write_text(
                json.dumps({"a": 60, "b": 0}), encoding="utf-8")
            cfg = {"dataset": "ns100", "data_root": str(root),
                   "metadata_root": str(metadata), "validation_classes": [60, 79],
                   "meta_train_classes": [0, 59], "test_classes": [80, 99]}
            rows = load_rows(cfg, "val")
            self.assertEqual([row.label for row in rows], [60])
            self.assertTrue(rows[0].path.endswith("validation-class.wav"))


if __name__ == "__main__":
    unittest.main()
