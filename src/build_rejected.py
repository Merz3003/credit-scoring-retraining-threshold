"""
Выборка отклонённых заявок Lending Club за окно наблюдения.

Файл содержит 27,6 млн записей и всего 9 полей (исхода нет), поэтому
используется только для двух вещей:
  - доля одобрения по месяцам (одобренные / все заявки);
  - PSI по входящему потоку заявок (стратегия S4b).

Вход:  rejected_2007_to_2018Q4_csv.zip
Выход: rej.pkl
"""

import zipfile

import pandas as pd

from paths import REJECTED_ZIP, REJECTED, require

SRC = require(REJECTED_ZIP)
MEMBER = "rejected_2007_to_2018Q4.csv"   # в архиве есть ещё служебная папка __MACOSX
OUT = REJECTED
START, END = "2012-01-01", "2015-12-31"

COLS = ["Amount Requested", "Application Date", "Risk_Score",
        "Debt-To-Income Ratio", "Employment Length"]


def main() -> None:
    parts, total = [], 0
    archive = zipfile.ZipFile(SRC)
    reader = pd.read_csv(archive.open(MEMBER), usecols=COLS, chunksize=500_000,
                         dtype={"Amount Requested": "float32", "Risk_Score": "float32"},
                         low_memory=False)
    for chunk in reader:
        total += len(chunk)
        d = pd.to_datetime(chunk["Application Date"], errors="coerce")
        mask = (d >= START) & (d <= END)
        if not mask.any():
            continue
        sub = chunk[mask].copy()
        sub["ym"] = d[mask].dt.to_period("M").astype(str)
        parts.append(sub[["ym", "Amount Requested", "Risk_Score", "Employment Length"]])

    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(OUT)

    print(f"всего строк в файле: {total:,}")
    print(f"в окне {START[:7]}…{END[:7]}: {len(df):,}")
    print(f"заполненность Risk_Score: {df['Risk_Score'].notna().mean():.1%}")
    print(f"заполненность Amount Requested: {df['Amount Requested'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
