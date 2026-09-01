from pathlib import Path

import pandas as pd


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = pd.read_csv(csv_path)

    df.columns = df.columns.map(lambda col: str(col).strip())
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

    print(df.head())
    print()
    print(df["Label"].value_counts())


if __name__ == "__main__":
    main()
