import tempfile
import unittest
from pathlib import Path

from boundary_fsor.audit import rows_manifest, validate_class_disjointness
from boundary_fsor.data import AudioRow


class AuditTest(unittest.TestCase):
    def test_disjoint_complete_partition(self):
        result = validate_class_disjointness({
            "meta_train_classes": [0, 4], "validation_classes": [5, 7],
            "test_classes": [8, 9], "num_classes": 10,
        })
        self.assertEqual(result["final_test"], [8, 9])

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_class_disjointness({
                "meta_train_classes": [0, 5], "validation_classes": [5, 7],
                "test_classes": [8, 9], "num_classes": 10,
            })

    def test_row_manifest_is_order_sensitive(self):
        rows = [AudioRow("a.wav", 0), AudioRow("b.wav", 1)]
        first = rows_manifest(rows)
        second = rows_manifest(list(reversed(rows)))
        self.assertNotEqual(first["ordered_rows_sha256"], second["ordered_rows_sha256"])

    def test_missing_audio_fails_strict_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.wav"
            with self.assertRaises(FileNotFoundError):
                rows_manifest([AudioRow(str(missing), 0)], verify_files=True)


if __name__ == "__main__":
    unittest.main()
