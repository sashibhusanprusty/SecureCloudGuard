from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_ddos.csv"
SAMPLE_SIZE = 4000
LABEL_COLUMN = "Label"


def main() -> None:
    dataset = pd.read_csv(SOURCE_PATH)
    dataset.columns = dataset.columns.map(lambda column: str(column).strip())

    if LABEL_COLUMN not in dataset.columns:
        raise ValueError(f"Expected a {LABEL_COLUMN!r} column in {SOURCE_PATH}")

    labels = dataset[LABEL_COLUMN].astype(str).str.strip()
    if set(labels.unique()) != {"BENIGN", "DDoS"}:
        raise ValueError("Expected the full dataset to contain only BENIGN and DDoS labels")

    sample, _ = train_test_split(
        dataset,
        train_size=SAMPLE_SIZE,
        stratify=labels,
        random_state=42,
    )
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(sample):,} stratified rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()