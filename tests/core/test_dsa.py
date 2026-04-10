"""Tests for Daily and Sports Activities (DSA) dataset."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import DSADataset

EXPECTED_MANIFEST_COLUMNS = (
    "subject_id",
    "activity_name",
    "activity_code",
    "segment_path",
)

LOAD_TABLE_COLUMNS = frozenset(
    {
        "patient_id",
        "event_type",
        "timestamp",
        "dsa_segments/segment_path",
        "dsa_segments/activity_name",
        "dsa_segments/activity_code",
    }
)

ACTIVITY_RECORD_KEYS = frozenset({"id", "segments"})

SEGMENT_RECORD_KEYS = frozenset(
    {
        "activity",
        "data",
        "file_path",
        "num_samples",
        "sampling_rate",
        "segment_filename",
        "sensors",
        "subject_id",
        "units",
    }
)

EXPECTED_UNIT_KEYS_IN_ORDER = ("T", "RA", "LA", "RL", "LL")
EXPECTED_SENSOR_KEYS_IN_ORDER = (
    "xacc", "yacc", "zacc",
    "xgyro", "ygyro", "zgyro",
    "xmag", "ymag", "zmag",
)


def _write_segment(path: Path, n_rows: int = 125, n_cols: int = 45) -> None:
    """Write a synthetic DSA segment file."""
    line = ",".join(["0.0"] * n_cols)
    path.write_text("\n".join([line] * n_rows) + "\n", encoding="utf-8")


def _make_minimal_dsa_tree(root: Path) -> Path:
    """Create minimal DSA directory structure with one segment."""
    seg_dir = root / "a01" / "p1"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_path = seg_dir / "s01.txt"
    _write_segment(seg_path)
    return seg_path


class TestDSADataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root_path = cls._tmpdir.name
        # Create minimal tree; DSADataset creates manifest if missing
        seg_dir = Path(cls.root_path) / "a01" / "p1"
        seg_dir.mkdir(parents=True)
        _write_segment(seg_dir / "s01.txt")

        cls.dataset = DSADataset(root=cls.root_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_dataset_initialization(self):
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.dataset_name, "dsa")
        self.assertIsNotNone(self.dataset.config)
        manifest = Path(self.root_path) / "dsa_manifest.csv"
        self.assertTrue(manifest.is_file())

    def test_get_subject_ids(self):
        subject_ids = self.dataset.get_subject_ids()
        self.assertIsInstance(subject_ids, list)
        self.assertEqual(subject_ids, ["p1"])

    def test_get_activity_labels(self):
        activity_labels = self.dataset.get_activity_labels()
        self.assertIsInstance(activity_labels, dict)
        self.assertEqual(len(activity_labels), 19)
        self.assertEqual(activity_labels.get("sitting"), 0)

    def test_subject_data_loading(self):
        subject_ids = self.dataset.get_subject_ids()
        self.assertTrue(subject_ids)
        subject_id = subject_ids[0]
        subject_data = self.dataset.get_subject_data(subject_id)

        self.assertIsInstance(subject_data, dict)
        self.assertEqual(subject_data["id"], subject_id)
        self.assertIn("activities", subject_data)
        self.assertIn("sitting", subject_data["activities"])

        activity_data = subject_data["activities"]["sitting"]
        self.assertIsInstance(activity_data["segments"], list)
        self.assertTrue(activity_data["segments"])

        segment = activity_data["segments"][0]
        self.assertIsInstance(segment["data"], np.ndarray)
        self.assertEqual(segment["sampling_rate"], 25)
        self.assertEqual(segment["data"].shape, (125, 45))

    def test_segment_schema(self):
        """Each segment dict exposes a stable key schema."""
        subject_id = self.dataset.get_subject_ids()[0]
        subject_data = self.dataset.get_subject_data(subject_id)

        for activity_name, activity_data in subject_data["activities"].items():
            self.assertEqual(frozenset(activity_data.keys()), ACTIVITY_RECORD_KEYS)
            self.assertIsInstance(activity_data["id"], str)
            self.assertIsInstance(activity_data["segments"], list)

            for seg in activity_data["segments"]:
                self.assertEqual(frozenset(seg.keys()), SEGMENT_RECORD_KEYS)
                self.assertEqual(seg["subject_id"], subject_id)
                self.assertEqual(seg["activity"], activity_name)
                self.assertEqual(seg["segment_filename"], seg["file_path"].name)
                self.assertIsInstance(seg["file_path"], Path)
                arr = seg["data"]
                self.assertIsInstance(arr, np.ndarray)
                self.assertEqual(arr.ndim, 2)
                self.assertEqual(arr.shape[1], 45, "45 channels per DSA segment row")
                self.assertEqual(arr.shape[0], seg["num_samples"])

    def test_sensor_and_unit_channel_metadata(self):
        """Sensors/units lists match YAML config."""
        ds = self.dataset

        self.assertEqual(len(ds.units), len(EXPECTED_UNIT_KEYS_IN_ORDER))
        self.assertEqual(len(ds.sensors), len(EXPECTED_SENSOR_KEYS_IN_ORDER))

        unit_keys = [list(u.keys())[0] for u in ds.units]
        self.assertEqual(tuple(unit_keys), EXPECTED_UNIT_KEYS_IN_ORDER)

        sensor_keys = [list(s.keys())[0] for s in ds.sensors]
        self.assertEqual(tuple(sensor_keys), EXPECTED_SENSOR_KEYS_IN_ORDER)

    def test_manifest_csv_columns(self):
        manifest = Path(self.root_path) / "dsa_manifest.csv"
        df = pd.read_csv(manifest)
        self.assertEqual(list(df.columns), list(EXPECTED_MANIFEST_COLUMNS))
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["subject_id"], "p1")
        self.assertEqual(row["activity_name"], "sitting")
        self.assertEqual(row["activity_code"], "A1")
        self.assertEqual(row["segment_path"], "a01/p1/s01.txt")

    def test_load_table_manifest_via_base_dataset(self):
        df = self.dataset.load_table("dsa_segments").compute()
        self.assertEqual(frozenset(df.columns), LOAD_TABLE_COLUMNS)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["patient_id"], "p1")
        self.assertEqual(row["event_type"], "dsa_segments")
        self.assertEqual(row["dsa_segments/activity_name"], "sitting")
        self.assertEqual(row["dsa_segments/activity_code"], "A1")
        self.assertEqual(row["dsa_segments/segment_path"], "a01/p1/s01.txt")
        self.assertTrue(pd.isna(row["timestamp"]))

    def test_segment_raises_on_wrong_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            _write_segment(seg_path, n_rows=124, n_cols=45)
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "has 124 rows, expected 125"):
                ds.get_subject_data("p1")

    def test_segment_raises_on_non_numeric_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            bad_line = ",".join(["oops"] + ["0.0"] * 44)
            seg_path.write_text("\n".join([bad_line] * 125) + "\n", encoding="utf-8")
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "Failed to parse DSA segment"):
                ds.get_subject_data("p1")

    def test_segment_raises_on_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            nan_line = ",".join(["nan"] + ["0.0"] * 44)
            seg_path.write_text("\n".join([nan_line] * 125) + "\n", encoding="utf-8")
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "contains non-finite values"):
                ds.get_subject_data("p1")


if __name__ == "__main__":
    unittest.main()
