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


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) time-series dataset (Barshan & Altun, 2010).

    Recordings use five on-body IMU units (torso, two arms, two legs); each unit
    contributes nine columns per row (3-axis accelerometer, gyroscope, and
    magnetometer), so each segment row has 45 comma-separated values. The public
    release is sampled at 25 Hz; each ``.txt`` segment is typically 125 lines (about
    five seconds of data).

    On disk, activities live in folders ``a01`` through ``a19``, subjects in ``p1``
    through ``p8``, and segment files ``s01.txt``, ``s02.txt``, … under each
    subject. PyHealth maps ``aXX`` folder names to activity labels using the
    ``label_mapping`` in ``configs/dsa.yaml``.

    :class:`BaseDataset` reads a single tabular index of segments. The path to that
    CSV (by default ``dsa_manifest.csv`` next to the activity folders) is set in the
    YAML ``tables.dsa_segments.file_path`` entry. If that file is not present under
    ``root`` when you construct this class, the loader walks the tree, matches the
    same layout patterns as in the YAML, and writes the manifest. You can rebuild it
    later with :meth:`prepare_metadata`.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Citations:
        If you use this dataset, cite: Barshan, B., & Altun, K. (2010). Daily and
        Sports Activities [Dataset]. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5C59F

    Args:
        root: Dataset root (activity folders; manifest created if missing).
        dataset_name: Defaults to ``internal_name`` from config or ``"dsa"``.
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
        config_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "dsa.yaml"
            )

        metadata_filename = "dsa_manifest.csv"
        metadata_path = os.path.join(root, metadata_filename)

        if not os.path.exists(metadata_path):
            self.prepare_metadata(root, metadata_path, config_path)

        super().__init__(
            root=root,
            tables=["dsa_segments"],
            dataset_name=dataset_name or "dsa",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

        # Load config for dataset-specific attributes
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        task_cfg = (
            config.get("dataset", {})
            .get("task_configs", {})
            .get("activity_recognition", {})
        )
        self.label_mapping: Dict[str, str] = dict(task_cfg.get("label_mapping", {}))
        self.units: List[Dict[str, str]] = [
            u for u in config.get("dataset", {}).get("units", []) if isinstance(u, dict)
        ]
        self.sensors: List[Dict[str, str]] = [
            s for s in config.get("dataset", {}).get("sensors", []) if isinstance(s, dict)
        ]
        self.sampling_frequency: int = int(config.get("dataset", {}).get("sampling_frequency", 25))
        ds_cfg = config.get("dataset", {}).get("data_structure", {})
        self._num_columns: int = int(ds_cfg.get("num_columns", 45))
        self._num_rows: int = int(ds_cfg.get("num_rows", 125))

    def prepare_metadata(
        self, root: str, metadata_path: str, config_path: str
    ) -> None:
        """Scan ``root`` and overwrite the manifest CSV (``tables.dsa_segments``)."""
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        label_mapping = (
            config.get("dataset", {})
            .get("task_configs", {})
            .get("activity_recognition", {})
            .get("label_mapping", {})
        )

        layout = config.get("dataset", {}).get("layout", {})
        a_re = re.compile(layout.get("activity_dir_pattern", r"^a\d{2}$"))
        p_re = re.compile(layout.get("subject_dir_pattern", r"^p\d+$"))
        s_re = re.compile(layout.get("segment_file_pattern", r"^s\d+\.txt$"))

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
                            "activity_name": label_mapping.get(activity_code, ""),
                            "activity_code": activity_code,
                            "segment_path": f"{a_dir}/{p_dir}/{s_file}",
                        }
                    )

        if not rows:
            raise ValueError(
                f"No DSA segments under {root}; expected aXX/pY/sZZ.txt layout."
            )

        df = pd.DataFrame(rows)
        df = df[["subject_id", "activity_name", "activity_code", "segment_path"]]
        df.to_csv(metadata_path, index=False)

    def get_subject_ids(self) -> List[str]:
        """Return sorted subject IDs from the manifest."""
        manifest_path = os.path.join(self.root, "dsa_manifest.csv")
        df = pd.read_csv(manifest_path)
        return sorted(df["subject_id"].unique().tolist())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map activity name to class index (ordered by activity code)."""
        codes = sorted(
            self.label_mapping.keys(),
            key=lambda c: int(re.match(r"^A(\d+)$", c, re.IGNORECASE).group(1))
        )
        return {self.label_mapping[c]: i for i, c in enumerate(codes)}

    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Load all segment arrays for one subject."""
        manifest_path = os.path.join(self.root, "dsa_manifest.csv")
        df = pd.read_csv(manifest_path)

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
