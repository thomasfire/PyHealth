"""Daily and Sports Activities (DSA) dataset loader."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset metadata (SleepEDF-style: domain facts live in code, YAML is tables only)
# -----------------------------------------------------------------------------

DSA_PYHEALTH_MANIFEST = "dsa-pyhealth.csv"

_LABEL_MAPPING: Dict[str, str] = {
    "A1": "sitting",
    "A2": "standing",
    "A3": "lying_on_back",
    "A4": "lying_on_right_side",
    "A5": "ascending_stairs",
    "A6": "descending_stairs",
    "A7": "standing_in_elevator_still",
    "A8": "moving_around_in_elevator",
    "A9": "walking_in_parking_lot",
    "A10": "walking_on_treadmill_flat",
    "A11": "walking_on_treadmill_inclined",
    "A12": "running_on_treadmill",
    "A13": "exercising_on_stepper",
    "A14": "exercising_on_cross_trainer",
    "A15": "cycling_on_exercise_bike_horizontal",
    "A16": "cycling_on_exercise_bike_vertical",
    "A17": "rowing",
    "A18": "jumping",
    "A19": "playing_basketball",
}

_UNITS: List[Dict[str, str]] = [
    {"T": "Torso"},
    {"RA": "Right Arm"},
    {"LA": "Left Arm"},
    {"RL": "Right Leg"},
    {"LL": "Left Leg"},
]

_SENSORS: List[Dict[str, str]] = [
    {"xacc": "X-axis Accelerometer"},
    {"yacc": "Y-axis Accelerometer"},
    {"zacc": "Z-axis Accelerometer"},
    {"xgyro": "X-axis Gyroscope"},
    {"ygyro": "Y-axis Gyroscope"},
    {"zgyro": "Z-axis Gyroscope"},
    {"xmag": "X-axis Magnetometer"},
    {"ymag": "Y-axis Magnetometer"},
    {"zmag": "Z-axis Magnetometer"},
]

_SAMPLING_FREQUENCY = 25
_NUM_COLUMNS = 45
_NUM_ROWS = 125

_LAYOUT = {
    "activity_dir_pattern": r"^a\d{2}$",
    "subject_dir_pattern": r"^p\d+$",
    "segment_file_pattern": r"^s\d+\.txt$",
    "code_regex_pattern": r"^A(\d+)$",
    "file_extension": ".txt",
}

DSA_TABLE_NAME = "segments"


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) time-series dataset (Barshan & Altun, 2010).

    Structure mirrors :class:`SleepEDFDataset`: YAML lists only the manifest table;
    labels, layout regexes, and channel metadata are defined in this module. If
    ``dsa-pyhealth.csv`` is missing under ``root``, the tree is scanned and the
    manifest is written before :class:`BaseDataset` is initialized.

    Recordings use five on-body IMU units (torso, two arms, two legs); each unit
    contributes nine columns per row (3-axis accelerometer, gyroscope, and
    magnetometer), so each segment row has 45 comma-separated values. The public
    release is sampled at 25 Hz; each ``.txt`` segment is typically 125 lines (about
    five seconds of data).

    On disk, activities live in folders ``a01`` through ``a19``, subjects in ``p1``
    through ``p8``, and segment files ``s01.txt``, ``s02.txt``, … under each
    subject.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Citations:
        If you use this dataset, cite: Barshan, B., & Altun, K. (2010). Daily and
        Sports Activities [Dataset]. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5C59F

    Args:
        root str: Dataset root (activity folders; manifest created if missing).
        dataset_name: Passed to :class:`BaseDataset`. Default ``"dsa"``.
        config_path: Path to ``dsa.yaml`` (default: package ``configs/dsa.yaml``).
        cache_dir: Cache directory for :class:`BaseDataset`.
        num_workers: Parallel workers for base pipelines.
        dev: Passed to :class:`BaseDataset` (limits patients when building events).

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> dataset = DSADataset(root="/path/to/dsa")
        >>> dataset.stat()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "dsa.yaml"
            )

        metadata_path = os.path.join(root, DSA_PYHEALTH_MANIFEST)
        if not os.path.exists(metadata_path):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=[DSA_TABLE_NAME],
            dataset_name=dataset_name or "dsa",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

        self.label_mapping: Dict[str, str] = dict(_LABEL_MAPPING)
        self.units: List[Dict[str, str]] = list(_UNITS)
        self.sensors: List[Dict[str, str]] = list(_SENSORS)
        self.sampling_frequency: int = _SAMPLING_FREQUENCY
        self._num_columns: int = _NUM_COLUMNS
        self._num_rows: int = _NUM_ROWS

        layout = _LAYOUT
        self._activity_dir_pattern = re.compile(layout["activity_dir_pattern"])
        self._subject_dir_pattern = re.compile(layout["subject_dir_pattern"])
        self._segment_file_pattern = re.compile(layout["segment_file_pattern"])
        self._code_regex_pattern = layout["code_regex_pattern"]
        self._file_extension = layout["file_extension"]

    def _manifest_path(self) -> str:
        table_cfg = self.config.tables[DSA_TABLE_NAME]
        return os.path.join(self.root, table_cfg.file_path)

    def _get_label_mapping(self) -> Dict[str, str]:
        return dict(self.label_mapping)

    def prepare_metadata(self, root: str) -> None:
        """Scan ``root`` and write ``dsa-pyhealth.csv`` (``tables.segments``)."""
        layout = _LAYOUT
        a_re = re.compile(layout["activity_dir_pattern"])
        p_re = re.compile(layout["subject_dir_pattern"])
        s_re = re.compile(layout["segment_file_pattern"])

        rows = []
        for a_dir in sorted(os.listdir(root)):
            if not a_re.match(a_dir):
                continue
            activity_code = f"A{int(a_dir[1:])}"
            a_path = os.path.join(root, a_dir)
            if not os.path.isdir(a_path):
                continue

            for p_dir in sorted(os.listdir(a_path)):
                if not p_re.match(p_dir):
                    continue
                p_path = os.path.join(a_path, p_dir)
                if not os.path.isdir(p_path):
                    continue

                for s_file in sorted(os.listdir(p_path)):
                    if not s_re.match(s_file):
                        continue

                    rows.append(
                        {
                            "subject_id": p_dir,
                            "activity_name": _LABEL_MAPPING.get(activity_code, ""),
                            "activity_code": activity_code,
                            "segment_path": f"{a_dir}/{p_dir}/{s_file}",
                        }
                    )

        if not rows:
            raise ValueError(
                f"No DSA segments under {root}; expected aXX/pY/sZZ.txt layout."
            )

        metadata_path = os.path.join(root, DSA_PYHEALTH_MANIFEST)
        df = pd.DataFrame(rows)
        df = df[["subject_id", "activity_name", "activity_code", "segment_path"]]
        df.to_csv(metadata_path, index=False)

    def get_subject_ids(self) -> List[str]:
        """Return sorted subject IDs from the manifest."""
        df = pd.read_csv(self._manifest_path())
        return sorted(df["subject_id"].unique().tolist())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map activity name to class index (ordered by activity code)."""
        code_re = re.compile(self._code_regex_pattern)
        def _code_order(code: str) -> int:
            match = code_re.match(code)
            if match is None:
                raise ValueError(f"Invalid activity code {code!r}")
            return int(match.group(1))

        codes = sorted(self.label_mapping.keys(), key=_code_order)
        return {self.label_mapping[c]: i for i, c in enumerate(codes)}

    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Load all segment arrays for one subject."""
        df = pd.read_csv(self._manifest_path())

        subject_df = df[df["subject_id"] == subject_id]
        if subject_df.empty:
            raise ValueError(f"Subject {subject_id!r} not found in manifest")

        subject_data: Dict[str, Any] = {"id": subject_id, "activities": {}}

        for (activity_name, activity_code), group in subject_df.groupby(
            ["activity_name", "activity_code"]
        ):
            segments = []
            for _, row in group.iterrows():
                segment_path = os.path.join(self.root, row["segment_path"])
                segment_data = self._load_segment(segment_path, subject_id, activity_name)
                segments.append(segment_data)

            subject_data["activities"][activity_name] = {
                "id": activity_code,
                "segments": segments,
            }

        return subject_data

    def _load_segment(
        self,
        file_path: str,
        subject_id: str,
        activity: str,
    ) -> Dict[str, Any]:
        """Load a single segment file and return as dict."""
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        except Exception as e:
            raise ValueError(
                f"Failed to parse DSA segment {file_path}; expected a "
                f"{self._num_rows}x{self._num_columns} comma-separated numeric file."
            ) from e

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_rows, n_cols = data.shape
        if n_rows != self._num_rows:
            raise ValueError(
                f"{file_path} has {n_rows} rows, expected {self._num_rows}"
            )
        if n_cols != self._num_columns:
            raise ValueError(
                f"{file_path} has {n_cols} columns, expected {self._num_columns}"
            )
        if not np.isfinite(data).all():
            raise ValueError(f"{file_path} contains non-finite values (NaN or Inf).")

        return {
            "file_path": Path(file_path),
            "subject_id": subject_id,
            "activity": activity,
            "data": data,
            "num_samples": n_rows,
            "sampling_rate": self.sampling_frequency,
            "units": self.units,
            "sensors": self.sensors,
            "segment_filename": os.path.basename(file_path),
        }
