"""
Этап 6: сводная таблица чувствительности к длине окна обучения.

Собирает результаты прогонов stage5b для окон 12, 18, 24 и 36 месяцев
и проверяет, устойчивы ли выводы H1-H3 к этому параметру.

Вход:  results/h1_tw*.csv, results/h3_tw*.csv
Выход: window_sensitivity.csv, window_thresholds.csv
"""

import pandas as pd

WINDOWS = [12, 18, 24, 36]
from paths import RESULTS

RES = RESULTS / "by_window"


def main() -> None:
    h1 = pd.concat(
        [pd.read_csv(f"{RES}/h1_tw{w}.csv").assign(окно=w) for w in WINDOWS],
        ignore_index=True)
    h1.to_csv(RESULTS / "window_sensitivity.csv", index=False)

    print("=== Выгода против S1, млн $ ===")
    print(h1.pivot(index="стратегия", columns="окно", values="выгода_млн").to_string())

    print("\n=== Число переобучений ===")
    print(h1.pivot(index="стратегия", columns="окно", values="переобуч.").to_string())

    print("\n=== Точка безубыточности C*, млн $ ===")
    print(h1.pivot(index="стратегия", columns="окно", values="C*_млн").to_string())

    print("\n=== Значимость выгоды (нижняя граница ДИ > 0) ===")
    print(h1.pivot(index="стратегия", columns="окно", values="значимо").to_string())

    h3 = pd.concat(
        [pd.read_csv(f"{RES}/h3_tw{w}.csv").assign(окно=w) for w in WINDOWS],
        ignore_index=True)
    h3.to_csv(RESULTS / "window_thresholds.csv", index=False)

    print("\n=== Оптимальный порог по каждому сигналу ===")
    best = h3.loc[h3.groupby(["окно", "сигнал"])["выгода_млн"].idxmax()]
    print(best.pivot(index="сигнал", columns="окно", values="порог").to_string())

    print("\n=== Выгода при оптимальном пороге, млн $ ===")
    print(best.pivot(index="сигнал", columns="окно", values="выгода_млн").to_string())

    # Устойчивость главного вывода
    print("\n=== Проверка устойчивости выводов ===")
    sig = h1.groupby("окно")["значимо"].apply(lambda s: (s == "да").sum())
    print(f"H1: стратегий со значимой выгодой по окнам: {dict(sig)} из 5")
    psi_fires = h1[h1["стратегия"].str.startswith("S4a")].set_index("окно")["переобуч."]
    print(f"H3: срабатываний PSI по скору при θ=0,10: {dict(psi_fires)}")
    never = h3[(h3["сигнал"] == "PSI скор") & (h3["порог"] >= 0.15)]["выгода_млн"].abs().sum()
    print(f"    выгода PSI по скору при θ >= 0,15 во всех окнах: {never:.2f} млн "
          f"(сигнал не срабатывает)")


if __name__ == "__main__":
    main()
