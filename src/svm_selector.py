from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def list_grayscale_images() -> list[Path]:
    image_dir = Path(__file__).resolve().parents[1] / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    excluded = {".gitkeep", "placeholder_grayscale.png", "generated_base.png"}
    candidates = []
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in excluded:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and not path.name.startswith("generated_"):
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.name.lower())


def ensure_minimum_images() -> list[Path]:
    images = list_grayscale_images()
    if len(images) >= 3:
        return images[:5]

    base = images[0] if images else None
    if base is None:
        base_img = np.zeros((128, 128), dtype=np.uint8)
        base_path = Path(__file__).resolve().parents[1] / "data" / "images" / "generated_base.png"
        Image.fromarray(base_img, mode="L").save(base_path)
        base = base_path

    generated = []
    for i, transform in enumerate(["original", "flip_lr", "flip_ud", "rotate_90", "transpose"]):
        src = Image.open(base).convert("L")
        arr = np.asarray(src, dtype=np.uint8)
        if transform == "flip_lr":
            arr = np.fliplr(arr)
        elif transform == "flip_ud":
            arr = np.flipud(arr)
        elif transform == "rotate_90":
            arr = np.rot90(arr)
        elif transform == "transpose":
            arr = arr.T

        out = Path(__file__).resolve().parents[1] / "data" / "images" / f"generated_{i}.png"
        Image.fromarray(arr, mode="L").save(out)
        generated.append(out)

    return [p for p in list_grayscale_images() if not p.name.startswith("generated_")] + generated[:5]


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
    denom_left = np.sum((left - left_mean) ** 2)
    denom_right = np.sum((right - right_mean) ** 2)
    if denom_left == 0 or denom_right == 0:
        return 0.0
    return float(numerator / math.sqrt(denom_left * denom_right))


def bit_share(bits: np.ndarray) -> float:
    return float(bits.mean())


def top_four_planes(image: np.ndarray) -> set[int]:
    scores = []
    for bit in range(8):
        plane = ((image >> bit) & 1).astype(np.uint8)
        corr = horizontal_adjacent_correlation(plane)
        scores.append((corr, bit))
    paired = sorted(scores, key=lambda x: x[0], reverse=True)
    return {bit for _, bit in paired[:4]}


def plane_feature_vector(image: np.ndarray, bit: int) -> np.ndarray:
    plane = ((image >> bit) & 1).astype(np.uint8)
    corr = horizontal_adjacent_correlation(plane)
    share = bit_share(plane)
    bit_index = bit / 7.0
    return np.array([bit_index, corr, share], dtype=np.float64)


def make_dataset(images: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for image_path in images:
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        target_planes = top_four_planes(image)
        for bit in range(8):
            row = plane_feature_vector(image, bit)
            label = 1 if bit in target_planes else 0
            features.append(row)
            labels.append(label)
    return np.vstack(features), np.array(labels, dtype=int)


def generate_new_test_image(base_image: np.ndarray) -> np.ndarray:
    img = np.asarray(base_image, dtype=np.uint8)
    transformed = np.rot90(np.fliplr(img.copy()))
    structure = np.linspace(0, 255, num=img.size, dtype=np.uint8).reshape(img.shape)
    blended = ((0.7 * transformed) + (0.3 * structure)).astype(np.uint8)
    return blended


def main() -> None:
    image_paths = ensure_minimum_images()
    X, y = make_dataset(image_paths)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = SVC(kernel="rbf", C=10.0, gamma="scale")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"SVM accuracy on held-out split: {acc:.4f}")

    base_image = np.asarray(Image.open(image_paths[0]).convert("L"), dtype=np.uint8)
    new_image = generate_new_test_image(base_image)
    actual_set = top_four_planes(new_image)

    score_pairs = []
    for bit in range(8):
        row = plane_feature_vector(new_image, bit)
        score = float(model.decision_function(row.reshape(1, -1))[0])
        score_pairs.append((score, bit))

    predicted_set = {bit for _, bit in sorted(score_pairs, key=lambda pair: pair[0], reverse=True)[:4]}

    print(f"Expected top-4 correlation planes for new image: {sorted(actual_set)}")
    print(f"SVM-selected planes for new image: {sorted(predicted_set)}")

    expected_pattern = {4, 5, 6, 7}
    if actual_set == expected_pattern and predicted_set == actual_set:
        print("Confirmation: SVM matches the expected 4-7 plane pattern for the new test image.")
    else:
        print(f"Confirmation: SVM does not match the expected 4-7 pattern. Actual: {sorted(actual_set)}; Predicted: {sorted(predicted_set)}")

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "svm_selector.pkl")
    print(f"Saved trained SVM to: {output_dir / 'svm_selector.pkl'}")


if __name__ == "__main__":
    main()
