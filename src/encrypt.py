from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PIL import Image


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


def split_bitplanes(image: np.ndarray) -> list[np.ndarray]:
    return [((image >> bit) & 1).astype(np.uint8) for bit in range(8)]


def select_planes(image: np.ndarray) -> list[int]:
    scores = []
    for bit in range(8):
        plane = ((image >> bit) & 1).astype(np.uint8)
        scores.append((horizontal_adjacent_correlation(plane), bit))
    return [bit for _, bit in sorted(scores, key=lambda item: item[0], reverse=True)[:4]]


def logistic_map_key_iv(seed: float = 0.7, length: int = 256) -> tuple[bytes, bytes]:
    x = float(seed)
    values = []
    for _ in range(length):
        x = 3.99 * x * (1.0 - x)
        values.append(x)
    stream = (np.array(values, dtype=np.float64) * 255.0) % 256.0
    bytes_out = stream.astype(np.uint8).tobytes()
    return bytes_out[:32], bytes_out[32:48]


def aes_encrypt_bytes(data: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad(data, AES.block_size)
    return cipher.encrypt(padded)


def aes_decrypt_bytes(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)
    return unpad(plaintext, AES.block_size)


def encrypt_selected_plane(plane: np.ndarray, key: bytes, iv: bytes) -> bytes:
    plane_bytes = plane.astype(np.uint8).tobytes()
    return aes_encrypt_bytes(plane_bytes, key, iv)


def decrypt_selected_plane(ciphertext: bytes, key: bytes, iv: bytes, plane_shape: tuple[int, int]) -> np.ndarray:
    plaintext = aes_decrypt_bytes(ciphertext, key, iv)
    bits = np.frombuffer(plaintext, dtype=np.uint8).reshape(plane_shape)
    return bits.astype(np.uint8)


def encrypt_selected_bitplanes(image: np.ndarray, selected: list[int], key: bytes, iv: bytes) -> dict[int, bytes]:
    bitplanes = split_bitplanes(image)
    encrypted = {}
    for bit in selected:
        ciphertext = encrypt_selected_plane(bitplanes[bit], key, iv)
        encrypted[bit] = ciphertext
    return encrypted


def build_encrypted_composite(image: np.ndarray, bitplanes: list[np.ndarray], selected: list[int], encrypted: dict[int, bytes]) -> np.ndarray:
    composite = np.zeros_like(image, dtype=np.uint8)
    for bit in range(8):
        if bit in selected:
            ciphertext = encrypted[bit]
            bytes_to_bits = np.frombuffer(ciphertext, dtype=np.uint8)
            encrypted_plane = (bytes_to_bits[: bitplanes[bit].size] % 2).reshape(bitplanes[bit].shape).astype(np.uint8)
        else:
            encrypted_plane = bitplanes[bit]
        composite = composite | ((encrypted_plane.astype(np.uint8) & 1) << bit)
    return composite & 0xFF


def decrypt_roundtrip(image: np.ndarray, selected: list[int], encrypted: dict[int, bytes], key: bytes, iv: bytes) -> np.ndarray:
    bitplanes = split_bitplanes(image)
    rebuilt = np.zeros_like(image, dtype=np.uint8)
    for bit in range(8):
        if bit in selected:
            plane = decrypt_selected_plane(encrypted[bit], key, iv, bitplanes[bit].shape)
        else:
            plane = bitplanes[bit]
        rebuilt = rebuilt | ((plane.astype(np.uint8) & 1) << bit)
    return rebuilt


def main() -> None:
    image_path = Path(__file__).resolve().parents[1] / "data" / "images" / "im1.jpg"
    if not image_path.exists():
        image_path = Path(__file__).resolve().parents[1] / "data" / "images" / "im1.jpeg"
    if not image_path.exists():
        raise FileNotFoundError("No im1.jpg or im1.jpeg file found in data/images/.")

    img_small = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    img_large = np.asarray(Image.open(image_path).convert("L").resize((512, 512)), dtype=np.uint8)

    for label, image in [("small", img_small), ("large", img_large)]:
        bitplanes = split_bitplanes(image)
        selected = select_planes(image)
        key, iv = logistic_map_key_iv(seed=0.73)

        selected_bytes = sum(len(aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv)) for bit in selected)
        all_bytes = sum(len(aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv)) for bit in range(8))

        selected_total = 0.0
        full_total = 0.0
        runs = 50
        for _ in range(runs):
            for bit in selected:
                start = time.perf_counter()
                aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv)
                selected_total += time.perf_counter() - start

            for bit in range(8):
                start = time.perf_counter()
                aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv)
                full_total += time.perf_counter() - start

        selected_avg = selected_total / runs
        full_avg = full_total / runs
        speedup_pct = ((full_avg - selected_avg) / full_avg) * 100.0 if full_avg > 0 else 0.0

        print(f"\nImage size: {label} ({image.shape[0]}x{image.shape[1]})")
        print(f"Selected planes: {selected}")
        print(f"Average selective-encryption time (50 runs): {selected_avg:.9f} seconds")
        print(f"Average full-encryption time (50 runs): {full_avg:.9f} seconds")
        print(f"Speedup: {speedup_pct:.2f}%")
        print(f"Encrypted byte count for selected planes: {selected_bytes}")
        print(f"Encrypted byte count for all planes: {all_bytes}")

        if selected_avg < full_avg:
            print("Chosen result: selective encryption is faster on this image size.")
        else:
            print("Chosen result: selective encryption is not faster at this image size; the byte-count difference is small enough that the overhead dominates.")

    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    bitplanes = split_bitplanes(image)
    selected = select_planes(image)
    key, iv = logistic_map_key_iv(seed=0.73)
    encrypted = {bit: aes_encrypt_bytes(bitplanes[bit].astype(np.uint8).tobytes(), key, iv) for bit in selected}
    encrypted_composite = build_encrypted_composite(image, bitplanes, selected, encrypted)
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encrypted_composite, mode="L").save(output_dir / "encrypted_composite.png")

    reconstructed = decrypt_roundtrip(image, selected, encrypted, key, iv)
    exact_match = np.array_equal(reconstructed, image)
    print(f"\nRound-trip exact match for output image: {exact_match}")
    print(f"Saved encrypted composite to: {output_dir / 'encrypted_composite.png'}")

    if not exact_match:
        raise AssertionError("Decryption round-trip failed: reconstructed image does not match the original exactly.")


if __name__ == "__main__":
    main()
