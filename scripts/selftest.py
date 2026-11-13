"""Selbsttest der Bewertungslogik. Läuft vor jedem Datenlauf. Schlägt ein Test fehl, wird nichts veröffentlicht."""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402

TODAY = datetime(2026, 11, 12)
IDX = pd.bdate_range("2005-11-14", "2026-11-11")


def frame(last, growth=0.10, div_yield=0.01):
    """Gleichmäßig steigender Kurs mit leichter Welle, endet bei `last`."""
    t = np.arange(len(IDX)) / 252.0
    c = np.exp(growth * t + 0.01 * np.sin(t * 6))
    c = c / c[-1] * last
    d = np.zeros(len(IDX))
    d[::63] = c[::63] * div_yield / 4
    return pd.DataFrame({"Close": c, "Volume": 1e6, "Dividends": d}, index=IDX)


def base(**kw):
    f = dict(sector="Industrials", industry="Specialty Industrial Machinery", currency="USD", finCurrency="USD",
             mcap=50e9, priceAtFetch=100.0, trailingEps=5.0, forwardEps=5.5, grossMargin=0.45,
             incDates=["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"],
             revenue=[10e9, 9.3e9, 8.7e9, 8.1e9], ebit=[2.0e9, 1.8e9, 1.65e9, 1.5e9],
             netIncome=[2.5e9, 2.3e9, 2.1e9, 1.9e9], dilutedEps=[5.0, 4.6, 4.2, 3.8], ebitda=[2.4e9],
             equity=[8e9, 7.5e9, 7e9, 6.5e9], assets=[20e9, 19e9, 18e9, 17e9], totalDebt=[4e9], cash=[1e9],
             revGrowthEst=0.08, epsGrowthEst=0.11)
    f.update(kw)
    return f


def crit(rec, key):
    return next(c for c in rec["F"] if c[0] == key)


def check(name, cond, detail=""):
    print(("ok   " if cond else "FEHLER ") + name + ("" if cond else "  -> " + str(detail)))
    return cond


def main():
    ok = True
    pm = scoring.price_metrics(frame(100.0))

    # 1. Aon-Beispiel aus der öffentlichen Analyse (Juli 2024): Average Grower, 6 von 10 Punkten
    price = 100.0
    aon = base(sector="Financial Services", industry="Insurance Brokers", mcap=70e9, priceAtFetch=price,
               revenue=[13.376e9, 12.5e9, 12.2e9, 13.376e9 / 1.0498 ** 3], ebit=[3.72e9, 3.5e9, 3.0e9, 2.7e9],
               netIncome=[2.60e9, 2.56e9, 2.1e9, 1.97e9], equity=[-0.01e9], assets=[34e9], totalDebt=[11e9],
               cash=[1e9], ebitda=[4.2e9], trailingEps=price / 24.89, forwardEps=None,
               dilutedEps=[price / 24.89, price / 30, price / 40, price / 45], revGrowthEst=0.1119, epsGrowthEst=0.08)
    r = scoring.score(aon, pm, 70e9, 1.0, TODAY)
    ok &= check("Aon: Average Grower, normale Kriterien", r["cat"] == "Average Grower" and r["fin"] == "", r["cat"])
    ok &= check("Aon: 6 von 10 Punkten", r["f"] == 6.0 and r["fa"] == 10, (r["f"], r["fa"], r["F"]))
    ok &= check("Aon: KGV 24,9", abs(r["pe"] - 24.9) < 0.05, r["pe"])

    # 2. Londoner Aktie: Kurs in Pence, Gewinn in Pfund
    r = scoring.score(base(currency="GBp", finCurrency="USD", mcap=200e9, priceAtFetch=3665.0, trailingEps=4.3,
                           forwardEps=4.5, dilutedEps=[5.6, 5.0, 4.6, 4.2]),
                      scoring.price_metrics(frame(3665.0)), 270e9, 1.3, TODAY)
    ok &= check("Pence: KGV 8,5 statt 852", abs(r["pe"] - 8.5) < 0.1, r["pe"])

    # 3. Bank: eigene Kriterien, zehn auswertbar, niedrigere KGV-Schwelle
    bank = base(sector="Financial Services", industry="Banks - Diversified", ebitda=[], grossMargin=None,
                revenue=[50e9, 47e9, 44e9, 41e9], ebit=[20e9, 18e9, 17e9, 16e9],
                netIncome=[15e9, 14e9, 13e9, 12e9], equity=[110e9, 105e9, 100e9, 95e9],
                assets=[1500e9, 1450e9, 1400e9, 1350e9], totalDebt=[300e9], cash=[200e9], mcap=180e9,
                priceAtFetch=100.0, trailingEps=8.3, forwardEps=9.0, dilutedEps=[8.3, 7.8, 7.2, 6.7])
    r = scoring.score(bank, pm, 180e9, 1.0, TODAY)
    keys = [c[0] for c in r["F"]]
    ok &= check("Bank: Kriterien Eigenkapital/Bilanzsumme, Gesamtkapitalrendite, Nettomarge",
                r["fin"] == "bank" and {"eqA", "roa", "nm"} <= set(keys) and "debt" not in keys, keys)
    ok &= check("Bank: zehn Kriterien auswertbar", r["fa"] == 10, r["fa"])
    ok &= check("Bank: KGV-Schwelle halbiert", crit(r, "pe5")[2] == scoring.TH[r["cat"]]["pe5"] * 0.5, crit(r, "pe5"))
    ok &= check("Bank: Gesamtkapitalrendite 1,0 %", crit(r, "roa")[1] == 1.0, crit(r, "roa"))

    # 4. Falscher Gewinn je Aktie nach Aktiensplit: Schätzung fürs nächste Jahr statt Ausreißer
    r = scoring.score(base(trailingEps=0.12, forwardEps=5.5), pm, 50e9, 1.0, TODAY)
    ok &= check("Split-Ausreißer erkannt", r["es"] == "f" and abs(r["pe"] - 18.2) < 0.1 and r["q"] != "A",
                (r["es"], r["pe"], r["q"]))
    # 5. Echter Gewinnsprung wird nicht verworfen
    r = scoring.score(base(trailingEps=9.0, forwardEps=9.5, dilutedEps=[2.0, 4.6, 4.2, 3.8],
                           netIncome=[1.0e9, 2.3e9, 2.1e9, 1.9e9]), pm, 50e9, 1.0, TODAY)
    ok &= check("Gewinnsprung bleibt", r["es"] == "t" and abs(r["pe"] - 11.1) < 0.1, (r["es"], r["pe"]))
    # 6. Verlust
    r = scoring.score(base(trailingEps=-2.0, forwardEps=None), pm, 50e9, 1.0, TODAY)
    ok &= check("Verlust: kein KGV, Bewertungskriterium verfehlt", r["pe"] is None and crit(r, "pe5")[3] is False, r["pe"])
    # 7. Unplausible Umsatzschätzung
    r = scoring.score(base(revGrowthEst=0.99), pm, 50e9, 1.0, TODAY)
    ok &= check("Umsatzschätzung +99 % verworfen", crit(r, "revE")[1] is None and len(r["qn"]) == 1, r["qn"])
    # 8. Ziel-KGV nie über dem heutigen KGV, Trendbruch gedeckelt
    r = scoring.score(base(trailingEps=10.0, forwardEps=10.5, dilutedEps=[10.0, 9.0, 8.0, 7.0],
                           netIncome=[5e9, 4.5e9, 4e9, 3.5e9]), pm, 50e9, 1.0, TODAY)
    ok &= check("Ziel-KGV höchstens heutiges KGV", r["kgve"] <= r["pe"], (r["kgve"], r["pe"]))
    capped = dict(r, st="Trendbruch")
    ok &= check("Trendbruch höchstens 61 Punkte", scoring.total(capped) <= 61.0, scoring.total(capped))
    # 9. Kurskennzahlen
    ok &= check("Kurs über 200-Tage-Linie, Aufwärtstrend", pm["last"] > pm["sma200"] and pm["tr"][0] == 1, pm["tr"])
    ok &= check("12-Monats-Entwicklung plausibel", 0.0 < pm["p1y"] < 0.5, pm["p1y"])

    print("Selbsttest", "bestanden" if ok else "FEHLGESCHLAGEN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
