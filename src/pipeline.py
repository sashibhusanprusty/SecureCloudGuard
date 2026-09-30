from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import time
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encrypt import aes_encrypt_bytes, logistic_map_key_iv, split_bitplanes
from src.svm_selector import plane_feature_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAFFIC_CSV = PROJECT_ROOT / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "rf_model.pkl"
DEFAULT_SVM_PATH = PROJECT_ROOT / "outputs" / "svm_selector.pkl"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "images"
GAUSSIAN_DP_DELTA = 1e-5


def apply_differential_privacy(true_count: int | float, epsilon: float) -> float:
    """Add Gaussian noise calibrated for (epsilon, 1e-5)-DP at sensitivity 1."""
    try:
        count = float(true_count)
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("true_count and epsilon must be finite numbers") from exc
    if not math.isfinite(count) or count < 0:
        raise ValueError("true_count must be a finite, non-negative number")
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        raise ValueError("epsilon must be a finite number greater than zero")

    normal = NormalDist()
    epsilon_exp = math.exp(epsilon_value)

    def delta_for_sigma(sigma: float) -> float:
        mu = 1.0 / sigma
        first = normal.cdf(mu / 2.0 - epsilon_value / mu)
        second = epsilon_exp * normal.cdf(-mu / 2.0 - epsilon_value / mu)
        return max(0.0, first - second)

    lower_sigma = 0.0
    upper_sigma = 1.0
    while delta_for_sigma(upper_sigma) > GAUSSIAN_DP_DELTA:
        upper_sigma *= 2.0

    for _ in range(80):
        sigma = (lower_sigma + upper_sigma) / 2.0
        if delta_for_sigma(sigma) > GAUSSIAN_DP_DELTA:
            lower_sigma = sigma
        else:
            upper_sigma = sigma

    noise = secrets.SystemRandom().gauss(0.0, upper_sigma)
    return count + noise


def load_feature_columns(csv_path: str | Path = DEFAULT_TRAFFIC_CSV) -> list[str]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.map(lambda col: str(col).strip())
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
    feature_columns = [col for col in df.columns if col not in {"Label", "target"}]
    return feature_columns


def normalize_traffic_row(row: dict | pd.Series, feature_columns: list[str] | None = None) -> pd.DataFrame:
    if feature_columns is None:
        feature_columns = load_feature_columns()

    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    elif isinstance(row, dict):
        row_dict = row.copy()
    else:
        raise TypeError("row must be a pandas Series or a dictionary")

    cleaned = {}
    for key, value in row_dict.items():
        cleaned[str(key).strip()] = value

    prepared = pd.DataFrame([cleaned])
    prepared.columns = prepared.columns.map(lambda col: str(col).strip())
    prepared = prepared.reindex(columns=feature_columns, fill_value=np.nan)
    return prepared


def predict_traffic_row(row: dict | pd.Series, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, float | str | int]:
    model = joblib.load(model_path)
    feature_columns = getattr(model, "feature_names_in_", None)
    if feature_columns is None:
        feature_columns = load_feature_columns()

    sample = normalize_traffic_row(row, list(feature_columns))
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    confidence = float(np.max(probabilities)) * 100.0
    label = "attack" if prediction == 1 else "normal"

    return {
        "prediction": label,
        "score": prediction,
        "confidence": confidence,
        "probabilities": {str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)},
    }


def select_image_planes_svm(image: np.ndarray, model_path: str | Path = DEFAULT_SVM_PATH) -> list[int]:
    model = joblib.load(model_path)
    score_pairs: list[tuple[float, int]] = []
    for bit in range(8):
        row = plane_feature_vector(image, bit).reshape(1, -1)
        score = float(model.decision_function(row)[0])
        score_pairs.append((score, bit))

    selected = [bit for _, bit in sorted(score_pairs, key=lambda pair: pair[0], reverse=True)[:4]]
    return selected


def encrypt_image(image: np.ndarray, selected_planes: list[int] | None = None, seed: float = 0.73) -> dict[str, object]:
    bitplanes = split_bitplanes(image)
    if selected_planes is None:
        selected_planes = list(range(8))

    key, iv = logistic_map_key_iv(seed=seed)
    start = time.perf_counter()
    encrypted = {}
    for bit in selected_planes:
        encrypted[bit] = aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv)
    elapsed = time.perf_counter() - start

    return {
        "selected_planes": selected_planes,
        "encryption_time": elapsed,
        "encrypted_data": encrypted,
        "encrypted_byte_count": sum(len(payload) for payload in encrypted.values()),
        "key": key,
        "iv": iv,
    }


def load_image(image_path: str | Path | None = None) -> np.ndarray:
    if image_path is None:
        candidate_paths = [
            PROJECT_ROOT / "data" / "images" / "im1.jpeg",
            PROJECT_ROOT / "data" / "images" / "im1.jpg",
            PROJECT_ROOT / "data" / "images" / "im2.jpeg",
            PROJECT_ROOT / "data" / "images" / "im2.jpg",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                image_path = candidate
                break

    if image_path is None or not Path(image_path).exists():
        raise FileNotFoundError("No valid image file found in data/images/. Provide --image-path.")

    return np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)


def load_traffic_row_from_csv(csv_path: str | Path = DEFAULT_TRAFFIC_CSV, row_index: int = 0) -> pd.Series:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.map(lambda col: str(col).strip())
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
    return df.iloc[row_index].copy()


def run_system_run(
    traffic_row: dict | pd.Series | None = None,
    image_path: str | Path | None = None,
    csv_path: str | Path = DEFAULT_TRAFFIC_CSV,
    row_index: int = 0,
) -> dict[str, object]:
    if traffic_row is None:
        traffic_row = load_traffic_row_from_csv(csv_path, row_index=row_index)

    traffic_result = predict_traffic_row(traffic_row)
    image = load_image(image_path)
    selected_planes = select_image_planes_svm(image)
    encryption_result = encrypt_image(image, selected_planes)

    system_summary = {
        "traffic": traffic_result,
        "image": {
            "selected_planes": selected_planes,
            "encryption_time": encryption_result["encryption_time"],
            "encrypted_byte_count": encryption_result["encrypted_byte_count"],
        },
    }
    return system_summary


def print_system_summary(summary: dict[str, object]) -> None:
    traffic = summary["traffic"]
    image = summary["image"]
    confidence = float(traffic["confidence"])
    prediction = str(traffic["prediction"]).upper()
    planes = image["selected_planes"]
    encryption_time = float(image["encryption_time"])
    encrypted_bytes = int(image["encrypted_byte_count"])

    print("\n" + "=" * 72)
    print("SYSTEM RUN SUMMARY")
    print("=" * 72)
    print(f"Traffic classification: {prediction} | confidence: {confidence:.2f}%")
    print(f"Image encryption: {planes} | time: {encryption_time:.6f}s | bytes encrypted: {encrypted_bytes}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traffic + image security pipeline in one summary.")
    parser.add_argument("--traffic-csv", type=str, default=str(DEFAULT_TRAFFIC_CSV), help="Path to the CSV containing the traffic feature row.")
    parser.add_argument("--row-index", type=int, default=0, help="Row index to use from the traffic CSV.")
    parser.add_argument("--traffic-row-json", type=str, default=None, help="Optional JSON object describing a single traffic row.")
    parser.add_argument("--image-path", type=str, default=None, help="Optional path to the image to encrypt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.traffic_row_json:
        row = json.loads(args.traffic_row_json)
    else:
        row = load_traffic_row_from_csv(args.traffic_csv, row_index=args.row_index)

    summary = run_system_run(traffic_row=row, image_path=args.image_path, csv_path=args.traffic_csv, row_index=args.row_index)
    print_system_summary(summary)


if __name__ == "__main__":
    main()
