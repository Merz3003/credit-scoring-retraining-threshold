"""
Этап 5b: проверка гипотез H1-H3 на закэшированных предсказаниях.

H1 - точка безубыточности C*.
H2 - триггерная стратегия против лучшей календарной.
H3 - зависимость выгоды от порога триггера.
"""

import sys
import numpy as np
import pandas as pd

from paths import RESULTS, cache, require

TW = int(sys.argv[1]) if len(sys.argv) > 1 else 24
COOLDOWN = 3
N_BOOT = 2000
RNG = np.random.default_rng(42)
AR_IDX = {0.6: 0, 0.7: 1, 0.8: 2, 0.9: 3}

C = np.load(require(cache(TW)), allow_pickle=True)
MONTHS, PRED, CASH = C["months"], C["pred"], C["cash_by_month"]
CALIB, EDGES, QGRID = C["calib"], C["edges"], C["qgrid"]
REF_SCORE, REF_APPL, CUR_APPL = C["ref_score"], C["ref_appl"], C["cur_appl"]
T = len(MONTHS)


def psi(ref, cur, eps=1e-6):
    e = np.clip(ref / max(ref.sum(), 1), eps, None)
    a = np.clip(cur / max(cur.sum(), 1), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def simulate(rule, approval_rate=0.80):
    """Возвращает помесячную прибыль и месяцы переобучения."""
    kind, param = rule
    active, last = 0, -99
    profits, retrains = [], []

    for j in range(T):
        if j > 0 and (j - last) > COOLDOWN:
            fit = False
            if kind == "calendar":
                fit = j % param == 0
            elif kind == "psi_score":
                fit = psi(REF_SCORE[active], np.histogram(PRED[active][j - 1], bins=EDGES[active])[0]) > param
            elif kind == "psi_appl":
                fit = psi(REF_APPL[active], CUR_APPL[j - 1]) > param
            elif kind == "calib":
                fit = CALIB[active][j - 1] > param
            if fit:
                active, last = j, j
                retrains.append(str(MONTHS[j]))

        p = PRED[active][j]
        cut = QGRID[active][AR_IDX[approval_rate]]
        profits.append(float(CASH[j][p <= cut].sum()))

    return np.array(profits), retrains


def boot_ci(d, n=N_BOOT):
    s = [d[RNG.integers(0, len(d), len(d))].sum() for _ in range(n)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    base, _ = simulate(("none", None))
    print(f"S1 без переобучения: ${base.sum()/1e6:.1f} млн\n")

    strategies = {
        "S2 календарь 3 мес.": ("calendar", 3),
        "S3 календарь 6 мес.": ("calendar", 6),
        "S4a PSI скор θ=0.10": ("psi_score", 0.10),
        "S4b PSI заявки θ=0.10": ("psi_appl", 0.10),
        "S4c калибровка δ=0.15": ("calib", 0.15),
    }

    rows = []
    store = {}
    for name, rule in strategies.items():
        pr, rt = simulate(rule)
        store[name] = pr
        d = pr - base
        lo, hi = boot_ci(d)
        rows.append({
            "стратегия": name, "переобуч.": len(rt),
            "выгода_млн": round(d.sum() / 1e6, 2),
            "ДИ_низ": round(lo / 1e6, 2), "ДИ_верх": round(hi / 1e6, 2),
            "значимо": "да" if lo > 0 else "нет",
            "C*_млн": round(d.sum() / len(rt) / 1e6, 2) if rt else np.nan,
            "месяцы": ", ".join(rt[:4]) + ("…" if len(rt) > 4 else ""),
        })
    h1 = pd.DataFrame(rows)
    h1.to_csv(RESULTS / f"by_window/h1_tw{TW}.csv", index=False)
    print("=== H1: выгода против S1, ДИ 95%, точка безубыточности ===")
    print(h1.to_string(index=False))

    # H2: лучшая триггерная против лучшей календарной
    best_cal = max(["S2 календарь 3 мес.", "S3 календарь 6 мес."], key=lambda k: store[k].sum())
    print(f"\n=== H2: против лучшей календарной ({best_cal}) ===")
    for name in ["S4a PSI скор θ=0.10", "S4b PSI заявки θ=0.10", "S4c калибровка δ=0.15"]:
        d = store[name] - store[best_cal]
        lo, hi = boot_ci(d)
        verdict = "не хуже" if hi > 0 and lo < 0 else ("лучше" if lo > 0 else "хуже")
        print(f"  {name:24s} разность {d.sum()/1e6:+6.2f} млн  ДИ [{lo/1e6:+.2f}; {hi/1e6:+.2f}]  {verdict}")

    # H3: перебор порогов
    grid = []
    for th in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for kind, label in [("psi_score", "PSI скор"), ("psi_appl", "PSI заявки"), ("calib", "калибровка")]:
            pr, rt = simulate((kind, th))
            grid.append({"сигнал": label, "порог": th, "переобуч.": len(rt),
                         "выгода_млн": round((pr - base).sum() / 1e6, 2),
                         "C*_млн": round((pr - base).sum() / len(rt) / 1e6, 2) if rt else np.nan})
    h3 = pd.DataFrame(grid)
    h3.to_csv(RESULTS / f"by_window/h3_tw{TW}.csv", index=False)
    print("\n=== H3: выгода, млн $, по порогу срабатывания ===")
    print(h3.pivot(index="порог", columns="сигнал", values="выгода_млн").to_string())
    print("\nчисло переобучений:")
    print(h3.pivot(index="порог", columns="сигнал", values="переобуч.").to_string())

    # Чувствительность по доле одобрения
    sens = []
    for ar in [0.6, 0.7, 0.8, 0.9]:
        b, _ = simulate(("none", None), ar)
        for name, rule in strategies.items():
            pr, rt = simulate(rule, ar)
            sens.append({"доля одобрения": ar, "стратегия": name,
                         "выгода_млн": round((pr - b).sum() / 1e6, 2)})
    s = pd.DataFrame(sens)
    s.to_csv(RESULTS / f"by_window/sens_appr_tw{TW}.csv", index=False)
    print("\n=== Чувствительность к доле одобрения (выгода против S1, млн $) ===")
    print(s.pivot(index="доля одобрения", columns="стратегия", values="выгода_млн").to_string())


if __name__ == "__main__":
    main()
