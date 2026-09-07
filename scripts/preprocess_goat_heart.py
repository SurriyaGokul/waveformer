#!/usr/bin/env python3
"""
Preprocess Goat Heart IQ data into model-ready sparse/velocity/magnitude triplets.

Pipeline:
1. Convert MATLAB .mat IQ files to .npy using the existing converter script
   (run separately): scripts/convert_mat_to_npy.py
2. Reuse scripts/iq_to_velocity.py Loupas pipeline to produce:
   - sparse_<name>.npy   (5, H, W) complex64
   - velocity_<name>.npy (49, H, W) float32 [mm/s]
   - magnitude_<name>.npy (H, W) float32
3. Write metadata in the same schema used by data/fullscale_velocity.

By default, all goat-heart samples are exported to the "inference" split so the
existing model can be tested on them directly.

Usage:
    python scripts/preprocess_goat_heart.py
    python scripts/preprocess_goat_heart.py --split_mode trainval --train_ratio 0.8
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from tqdm import tqdm

# Add project root to path so we can import scripts.iq_to_velocity when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.iq_to_velocity import LoupasConfig, process_single_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_PRF_TO_FREQ = {
    500: "500Hz",
    1000: "1kHz",
    5000: "5kHz",
    10000: "10kHz",
    20000: "20kHz",
}


def _slug(text: str) -> str:
    """Convert a folder/file token to the same underscore style as existing data."""
    return text.replace(" ", "_").replace("(", "").replace(")", "")


def _prf_to_freq_name(prf_hz: float) -> str:
    """Map PRF value to canonical frequency token (e.g., 20000 -> 20kHz)."""
    prf_int = int(round(prf_hz))
    if prf_int in _PRF_TO_FREQ:
        return _PRF_TO_FREQ[prf_int]

    # Fall back to nearest known PRF if exact key is missing.
    nearest = min(_PRF_TO_FREQ.keys(), key=lambda x: abs(x - prf_hz))
    logger.warning(
        "Unknown PRF %.3f Hz, using nearest known %d Hz -> %s",
        prf_hz,
        nearest,
        _PRF_TO_FREQ[nearest],
    )
    return _PRF_TO_FREQ[nearest]


def discover_goat_files(data_dir: Path) -> List[Dict]:
    """
    Discover converted goat-heart .npy files and attach per-file PRF from .mat metadata.

    Expected layout:
        data/Goat Heart Data/Sample X/Plane Y/Plane Y.npy
        data/Goat Heart Data/Sample X/Plane Y/Plane Y.mat
    """
    files: List[Dict] = []

    for sample_dir in sorted(data_dir.glob("Sample *")):
        if not sample_dir.is_dir():
            continue

        for plane_dir in sorted(sample_dir.iterdir()):
            if not plane_dir.is_dir() or not plane_dir.name.lower().startswith("plane"):
                continue

            plane_match = re.search(r"(\d+)", plane_dir.name)
            plane_label = f"Plane {plane_match.group(1)}" if plane_match else plane_dir.name

            # Handle occasional casing typo in source zip (PLane 5).
            npy_files = sorted(plane_dir.glob("*.npy"))
            if not npy_files:
                continue

            for npy_path in npy_files:
                mat_path = npy_path.with_suffix(".mat")
                if not mat_path.exists():
                    raise FileNotFoundError(
                        f"Missing .mat sidecar for PRF extraction: {mat_path}"
                    )

                mat_data = sio.loadmat(str(mat_path))
                if "Framerate" not in mat_data:
                    raise KeyError(f"Framerate not found in {mat_path}")

                prf_hz = float(mat_data["Framerate"][0][0])
                freq_name = _prf_to_freq_name(prf_hz)

                files.append(
                    {
                        "npy_path": npy_path,
                        "mat_path": mat_path,
                        "sample": sample_dir.name,
                        "plane": plane_label,
                        "prf_hz": prf_hz,
                        "frequency": freq_name,
                    }
                )

    logger.info("Discovered %d goat-heart IQ files", len(files))
    return files


def split_files(
    files: List[Dict],
    split_mode: str,
    train_ratio: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    """Split goat-heart files into train/val/inference groups."""
    if split_mode == "inference":
        return {"train": [], "val": [], "inference": files}

    # train/val split by sample to avoid leakage between planes of same sample.
    rng = np.random.RandomState(seed)
    sample_ids = sorted({f["sample"] for f in files})
    rng.shuffle(sample_ids)

    n_train = max(1, int(len(sample_ids) * train_ratio))
    train_samples = set(sample_ids[:n_train])

    train_files = [f for f in files if f["sample"] in train_samples]
    val_files = [f for f in files if f["sample"] not in train_samples]

    return {"train": train_files, "val": val_files, "inference": []}


def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = discover_goat_files(data_dir)
    if not files:
        raise RuntimeError(
            "No goat-heart .npy files found. Run convert_mat_to_npy.py first."
        )

    splits = split_files(
        files=files,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    for split_name, entries in splits.items():
        if entries:
            (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    config = LoupasConfig(
        sparse_step=args.sparse_step,
        center_frequency=args.center_freq,
        do_interpolation=False,
        do_bandpass=False,
    )

    tissue_props = {
        "material": "biological",
        "description": "Ex-vivo goat heart",
        "background_modulus_kpa": None,
        "density_kg_m3": 1000.0,
        "viscosity_pa_s": None,
        "estimated_shear_speed_m_s": None,
    }

    all_vel_abs_max: List[float] = []
    metadata_by_split: Dict[str, List[Dict]] = {"train": [], "val": [], "inference": []}

    for split_name in ["train", "val", "inference"]:
        split_files_list = splits[split_name]
        if not split_files_list:
            continue

        split_out = output_dir / split_name
        logger.info("Processing %d samples for split '%s'", len(split_files_list), split_name)

        for file_info in tqdm(split_files_list, desc=f"{split_name}"):
            sample_slug = _slug(file_info["sample"])   # e.g., Sample_1
            plane_slug = _slug(file_info["plane"])     # e.g., Plane_3
            freq_name = file_info["frequency"]

            name = f"goat_heart_{sample_slug}_{plane_slug}_{freq_name}_w0"

            result = process_single_file(
                file_info["npy_path"],
                freq_name,
                config,
            )
            if result is None:
                continue

            np.save(split_out / f"sparse_{name}.npy", result["sparse_iq"])
            np.save(split_out / f"velocity_{name}.npy", result["velocity"])
            np.save(split_out / f"magnitude_{name}.npy", result["magnitude_mask"])

            vel = result["velocity"]
            vel_abs_max = float(np.abs(vel).max())
            all_vel_abs_max.append(vel_abs_max)

            vel_std = float(vel.std())
            vel_snr = float(vel_abs_max / (vel_std + 1e-8))

            meta_entry = {
                "source": "goat_heart",
                "tissue_type": "biological",
                "frequency": freq_name,
                "prf_hz": float(result["metadata"]["prf_hz"]),
                "frame_time_s": float(result["metadata"]["frame_time_s"]),
                "focus": "unknown",
                "region": file_info["plane"],
                "sample_id": file_info["sample"],
                "original_iq_shape": result["metadata"]["original_iq_shape"],
                "window": [0, 50],
                "roi": result["metadata"]["roi"],
                "sparse_shape": result["metadata"]["sparse_shape"],
                "velocity_shape": result["metadata"]["velocity_shape"],
                "velocity_range_mms": result["metadata"]["velocity_range_mms"],
                "velocity_std_mms": vel_std,
                "velocity_snr": vel_snr,
                "sparse_step": int(args.sparse_step),
                "original_path": str(file_info["npy_path"]),
                "tissue_properties": tissue_props,
                "name": name,
            }
            metadata_by_split[split_name].append(meta_entry)

    velocity_norm = (
        float(np.percentile(all_vel_abs_max, 99)) if all_vel_abs_max else 1.0
    )

    metadata = {
        "config": {
            "speed_of_sound": float(config.speed_of_sound),
            "center_frequency": float(config.center_frequency),
            "M": int(config.M),
            "N": int(config.N),
            "medfilt_kernel": [int(config.medfilt_kernel_ax), int(config.medfilt_kernel_lat)],
            "sparse_step": int(args.sparse_step),
            "temporal_window": 50,
            "train_frequencies": ["20kHz"] if args.split_mode == "trainval" else [],
            "inference_frequencies": ["20kHz"],
            "train_ratio": float(args.train_ratio),
        },
        "normalization": {
            "velocity_abs_max_99pct": velocity_norm,
            "velocity_unit": "mm/s",
        },
        "tissue_properties": {
            "goat_heart": tissue_props,
        },
        "sample_counts": {
            "train": len(metadata_by_split["train"]),
            "val": len(metadata_by_split["val"]),
            "inference": len(metadata_by_split["inference"]),
        },
        "train": metadata_by_split["train"],
        "val": metadata_by_split["val"],
        "inference": metadata_by_split["inference"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Goat-heart preprocessing complete")
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Train samples: %d", metadata["sample_counts"]["train"])
    logger.info("  Val samples: %d", metadata["sample_counts"]["val"])
    logger.info("  Inference samples: %d", metadata["sample_counts"]["inference"])
    logger.info("  Velocity norm (99th pct |v|): %.6f mm/s", velocity_norm)
    logger.info("  Metadata: %s", metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess goat-heart IQ data to sparse/velocity/magnitude files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/Goat Heart Data",
        help="Directory containing converted goat-heart .npy files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/goat_heart_velocity",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["inference", "trainval"],
        default="inference",
        help="inference: put all samples in inference split; trainval: split by sample",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio when split_mode=trainval",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val splitting",
    )
    parser.add_argument(
        "--sparse_step",
        type=int,
        default=12,
        help="Sparse frame step to match model upsampling_factor",
    )
    parser.add_argument(
        "--center_freq",
        type=float,
        default=7.6e6,
        help="Transducer center frequency in Hz for Loupas estimator",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
