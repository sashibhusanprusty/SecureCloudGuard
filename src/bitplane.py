from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_test_image() -> Path:
    image_dir = Path(__file__).resolve().parents[1] / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    preferred_names = ["im1.jpeg", "im2.jpeg", "im3.jpeg", "im1.jpg", "im2.jpg", "im3.jpg"]
    for name in preferred_names:
        candidate = image_dir / name
        if candidate.exists():
            return candidate

    candidates = [
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]

    if candidates:
        return sorted(candidates, key=lambda p: p.name.lower())[0]

    rng = np.random.default_rng(42)
    height, width = 256, 256
    y, x = np.mgrid[0:height, 0:width]
    base = (x + y) % 256
    noise = rng.integers(0, 64, size=(height, width), dtype=np.uint8)
    image = ((base.astype(np.int16) + noise.astype(np.int16)) % 256).astype(np.uint8)

    placeholder_path = image_dir / "placeholder_grayscale.png"
    Image.fromarray(image, mode="L").save(placeholder_path)
    return placeholder_path


def horizontal_adjacent_correlation(bits: np.ndarray) -> float:
    if bits.size < 2:
        return 0.0

    left = bits[:, :-1].astype(np.float64).ravel()
    right = bits[:, 1:].astype(np.float64).ravel()

    if left.size == 0:
        return 0.0

    left_mean = left.mean()
    right_mean = right.mean()
    numerator = np.sum((left - left_mean) * (right - right_mean))
    left_var = np.sum((left - left_mean) ** 2)
    right_var = np.sum((right - right_mean) ** 2)

    if left_var == 0 or right_var == 0:
        return 0.0

    return float(numerator / math.sqrt(left_var * right_var))


def main() -> None:
    image_path = ensure_test_image()
    image = Image.open(image_path).convert("L")
    grayscale = np.asarray(image, dtype=np.uint8)

    if grayscale.ndim != 2:
        raise ValueError(f"Expected a grayscale image but got shape {grayscale.shape} from {image_path}.")

    bitplanes = []
    info = []

    for bit in range(8):
        plane = ((grayscale >> bit) & 1).astype(np.uint8)
        corr = horizontal_adjacent_correlation(plane)
        bitplanes.append(plane)
        info.append({"bit": bit, "corr": corr})

    print(f"Loaded grayscale image: {image_path}")
    print(f"Image shape: {grayscale.shape}")
    print("\nHorizontal adjacent-pixel correlation by bit-plane:")
    for entry in info:
        print(f"Plane {entry['bit']:>2}: correlation = {entry['corr']:.4f}")

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "bitplanes"
    output_dir.mkdir(parents=True, exist_ok=True)
    for bit, plane in enumerate(bitplanes):
        plane_image = (plane * 255).astype(np.uint8)
        Image.fromarray(plane_image, mode="L").save(output_dir / f"bitplane_{bit}.png")

    ranked = sorted(info, key=lambda item: item["corr"], reverse=True)
    ordered_planes = [entry["bit"] for entry in ranked]
    print("\nPlanes ranked from most structured (highest correlation) to least:")
    print(ordered_planes)

    selected = ordered_planes[:5]
    print(f"\nSuggested encryption planes (top 5): {selected}")
    print(f"Saved bit-plane PNGs to: {output_dir}")


if __name__ == "__main__":
    main()
