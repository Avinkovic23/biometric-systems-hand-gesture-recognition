from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

all_csv = [
    p for p in DATA_DIR.glob("hand_landmarks_*.csv")
    if p.name != "hand_landmarks_merged.csv"
]

if not all_csv:
    print("Nema CSV datoteka za merge.")
    exit()

dfs = []
for path in all_csv:
    df = pd.read_csv(path)
    if "source_file" not in df.columns:
        df["source_file"] = path.name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
output_path = DATA_DIR / "hand_landmarks_merged.csv"
df_all.to_csv(output_path, index=False)

print("Spojeno datoteka:", len(all_csv))
print("Ukupno redova:", len(df_all))
print("Spremljeno u:", output_path)
