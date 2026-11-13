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


def frame(last, growth=0.10, div_yield=0.01, drop=0.0):
    """Gleichmäßig steigender Kurs mit leichter Welle, endet bei `last`. drop: Einbruch in den letzten 120 Tagen."""
    t = np.arange(len(IDX)) / 252.0
    c = np.exp(growth * t + 0.01 * np.sin(t * 6))
    if drop:
        c[-120:] *= np.linspace(1, 1 - drop, 120)
    c = c / c[-1] * last
    d = np.zeros(len(IDX))
    d[::63] = c[::63] * div_yield / 4
    v = np.full(len(IDX), 1e6)
    v[-20:] = 1.4e6
    return pd.DataFrame({"Close": c, "Volume": v, "Dividends": d}, index=IDX)


def base(**kw):
    f = dict(sector="Industrials", industry="Specialty Industrial Machinery", currency="USD", finCurrency="USD",
             mcap=50e9, priceAtFetch=100.0, trailingEps=5.0, forwardEps=5.5, grossMargin=0.45,
             incDates=["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"],
             revenue=[10e9, 9.3e9, 8.7e9, 8.1e9], ebit=[2.0e9, 1.8e9, 1.65e9, 1.5e9],
             netIncome=[2.5e9, 2.3e9, 2.1e9, 1.9e9], dilutedEps=[5.0, 4.6, 4.2, 3.8], ebitda=[2.4e9],
             equity=[8e9, 7.5e9, 7e9, 6.5e9], assets=[20e9, 19e9, 18e9, 17e9], totalDebt=[4e9], cash=[1e9],
             revGrowthEst=0.08, epsGrowthEst=0.11,
             ocf=[2.8e9, 2.5e9, 2.3e9, 2.0e9], fcf=[2.2e9, 2.0e9, 1.8e9, 1.6e9], interest=[0.2e9],
             sharesDil=[500e6, 500e6, 505e6, 510e6], curAssets=[6e9], curLiab=[4e9],
             epsTrend={"+1y": {"cur": 6.2, "d30": 6.1, "d90": 6.0}}, epsRev={"up30": 8, "down30": 1},
             rec={"0m": [10, 12, 6, 1, 0], "-3m": [9, 12, 7, 1, 0]},
             updown=[["2026-10-20", "Testbank", "Hold", "Buy", "up"]], earn="2026-11-20", news=[],
             y=dict(recommendationMean=1.9, numberOfAnalystOpinions=29, targetMeanPrice=125.0, payoutRatio=0.3))
    f.update(kw)
    return f


def one(fund, pm, mcap=50e9, fx=1.0):
    r = scoring.score(fund, pm, mcap, fx, TODAY)
    r["sec"] = fund.get("sector")
    return scoring.finalize([r])[0]


def crit(rec, key):
    return next(c for c in rec["F"] if c[1] == key)


def check(name, cond, detail=""):
    print(("ok   " if cond else "FEHLER ") + name + ("" if cond else "  -> " + str(detail)))
    return bool(cond)


def main():
    ok = True
    pm = scoring.price_metrics(frame(100.0))

    # 1. Aon-Beispiel aus der öffentlichen Analyse (Juli 2024)
    price = 100.0
    aon = base(sector="Financial Services", industry="Insurance Brokers", mcap=70e9, priceAtFetch=price,
               revenue=[13.376e9, 12.5e9, 12.2e9, 13.376e9 / 1.0498 ** 3], ebit=[3.72e9, 3.5e9, 3.0e9, 2.7e9],
               netIncome=[2.60e9, 2.56e9, 2.1e9, 1.97e9], equity=[-0.01e9], assets=[34e9], totalDebt=[11e9],
               cash=[1e9], ebitda=[4.2e9], trailingEps=price / 24.89, forwardEps=None,
               dilutedEps=[price / 24.89, price / 30, price / 40, price / 45], revGrowthEst=0.1119, epsGrowthEst=0.08)
    r = one(aon, pm, 70e9)
    ok &= check("Aon: Average Grower, normale Kriterien", r["cat"] == "Average Grower" and r["fin"] == "", r["cat"])
    exp = dict(revH=False, revE=True, ebitH=True, epsE=False, eq=False, roe=False, debt=True, margin=True, pe5=True,
               dec=True)
    got = {k: crit(r, k)[4] for k in exp}
    ok &= check("Aon: die zehn ursprünglichen Kriterien wie in der Analyse (6 erfüllt)", got == exp, got)
    ok &= check("Aon: KGV 24,9", abs(r["pe"] - 24.9) < 0.05, r["pe"])

    # 2. Londoner Aktie: Kurs in Pence, Gewinn in Pfund
    r = one(base(currency="GBp", finCurrency="USD", mcap=200e9, priceAtFetch=3665.0, trailingEps=4.3,
                 forwardEps=4.5, dilutedEps=[5.6, 5.0, 4.6, 4.2], y={}),
            scoring.price_metrics(frame(3665.0)), 270e9, 1.3)
    ok &= check("Pence: KGV 8,5 statt 852", abs(r["pe"] - 8.5) < 0.1, r["pe"])

    # 3. Bank: eigene Kriterien
    bank = base(sector="Financial Services", industry="Banks - Diversified", ebitda=[], grossMargin=None,
                revenue=[50e9, 47e9, 44e9, 41e9], ebit=[20e9, 18e9, 17e9, 16e9],
                netIncome=[15e9, 14e9, 13e9, 12e9], equity=[110e9, 105e9, 100e9, 95e9],
                assets=[1500e9, 1450e9, 1400e9, 1350e9], totalDebt=[300e9], cash=[200e9], mcap=180e9,
                priceAtFetch=100.0, trailingEps=8.3, forwardEps=9.0, dilutedEps=[8.3, 7.8, 7.2, 6.7],
                ocf=[], fcf=[], interest=[], curAssets=[], curLiab=[])
    r = one(bank, pm, 180e9)
    keys = [c[1] for c in r["F"]]
    ok &= check("Bank: Kriterien Eigenkapital/Bilanzsumme, Gesamtkapitalrendite, Nettomarge",
                r["fin"] == "bank" and {"eqA", "roa", "nm"} <= set(keys) and "debt" not in keys and "fcfM" not in keys,
                keys)
    ok &= check("Bank: 16 Prüfungen, alle auswertbar", r["fn"] == 16 and r["fa"] == 16, (r["fn"], r["fa"]))
    ok &= check("Bank: KGV-Schwelle halbiert", crit(r, "pe5")[3] == scoring.TH[r["cat"]]["pe5"] * 0.5, crit(r, "pe5"))
    ok &= check("Bank: Gesamtkapitalrendite 1,0 %", crit(r, "roa")[2] == 1.0, crit(r, "roa"))

    # 4. Normale Firma: 21 Prüfungen, neue Cashflow- und Bilanzprüfungen
    r = one(base(), pm)
    ok &= check("21 Prüfungen, alle auswertbar", r["fn"] == 21 and r["fa"] == 21, (r["fn"], r["fa"]))
    ok &= check("Free-Cashflow-Marge 22 %", crit(r, "fcfM")[2] == 22.0 and crit(r, "fcfM")[4], crit(r, "fcfM"))
    ok &= check("Zinsdeckung 10", crit(r, "cov")[2] == 10.0 and crit(r, "cov")[4], crit(r, "cov"))
    ok &= check("Cashflow deckt Gewinn (112 %)", crit(r, "cash")[2] == 112.0 and crit(r, "cash")[4], crit(r, "cash"))
    ok &= check("Free-Cashflow-Rendite 4,4 %", crit(r, "fcfY")[2] == 4.4, crit(r, "fcfY"))
    ok &= check("Schätzung in 90 Tagen +3,3 %", crit(r, "rev90")[2] == 3.3, crit(r, "rev90"))
    weak = one(base(ocf=[1.0e9, 2.5e9, 2.3e9, 2.0e9], fcf=[-0.2e9, 2.0e9, 1.8e9, 1.6e9], interest=[1.0e9],
                    curAssets=[3e9], sharesDil=[560e6, 530e6, 510e6, 500e6],
                    epsTrend={"+1y": {"cur": 5.4, "d30": 5.8, "d90": 6.0}}), pm)
    bad = {k: crit(weak, k)[4] for k in ("fcfM", "cov", "liq", "dil", "cash", "fcfPos", "fcfY", "rev90")}
    ok &= check("Schwache Cashflows, Zinslast, Verwässerung fallen durch", not any(bad.values()), bad)
    ok &= check("Strenger als vorher: schwächere Firma klar schlechter", weak["f"] <= r["f"] - 3, (weak["f"], r["f"]))

    # 5. Analystenbild
    ok &= check("Analystenbild 0-10 berechnet", r["a"] == 9.0 and r["an"]["up"] == 25.0, (r["a"], r["an"]))
    none = one(base(y={}), pm)
    ok &= check("Ohne Analysten kein Analystenbild, Gesamtscore trotzdem da", none["a"] is None and none["tot"] > 0,
                (none["a"], none["tot"]))

    # 6. Gewinn je Aktie
    s = one(base(trailingEps=0.12, forwardEps=5.5), pm)
    ok &= check("Split-Ausreißer erkannt", s["es"] == "f" and abs(s["pe"] - 18.2) < 0.1 and s["q"] != "A",
                (s["es"], s["pe"], s["q"]))
    s = one(base(trailingEps=9.0, forwardEps=9.5, dilutedEps=[2.0, 4.6, 4.2, 3.8],
                 netIncome=[1.0e9, 2.3e9, 2.1e9, 1.9e9]), pm)
    ok &= check("Gewinnsprung bleibt", s["es"] == "t" and abs(s["pe"] - 11.1) < 0.1, (s["es"], s["pe"]))
    s = one(base(trailingEps=-2.0, forwardEps=None), pm)
    ok &= check("Verlust: kein KGV, Bewertungskriterium verfehlt", s["pe"] is None and crit(s, "pe5")[4] is False,
                s["pe"])
    s = one(base(revGrowthEst=0.99), pm)
    ok &= check("Umsatzschätzung +99 % verworfen", crit(s, "revE")[2] is None and len(s["qn"]) == 1, s["qn"])
    ok &= check("Ziel-KGV höchstens heutiges KGV", r["kgve"] <= r["pe"], (r["kgve"], r["pe"]))

    # 7. Chartanalyse
    ok &= check("Aufwärtstrend erkannt", r["st"] == "Aufwärtstrend" and r["tr"] == [1, 1, 1], (r["st"], r["tr"]))
    ok &= check("Unterstützungszonen unter dem Kurs", len(r["zs"]) >= 2 and all(z[1] <= 100.0 for z in r["zs"]),
                r["zs"])
    ok &= check("RSI und 52-Wochen-Spanne", 50 < r["rsi"] <= 100 and r["l52"] < r["p"] <= r["h52"],
                (r["rsi"], r["l52"], r["h52"]))
    ok &= check("Anlauf-Signale gezählt", r["hot"] >= 6 and len(r["H"]) == 8, (r["hot"], r["H"]))
    down = one(base(), scoring.price_metrics(frame(100.0, drop=0.35)))
    ok &= check("Einbruch: kein Aufwärtstrend, Turnaround-Prüfungen aktiv",
                down["st"] in ("Trendbruch", "Abwärtstrend", "Korrektur") and down["U"] is not None
                and down["su"] in ("Qualität im Rücksetzer", "Turnaround-Kandidat"), (down["st"], down["su"]))
    ok &= check("Kursbild senkt den Gesamtscore, nicht die Qualität", down["tot"] < r["tot"] and down["ql"] >= r["ql"] - 8,
                (down["tot"], r["tot"], down["ql"], r["ql"]))
    ok &= check("Laufende Korrektur gemessen", down["sym"][1] is not None and down["sym"][1] > 30, down["sym"])
    ok &= check("Einordnungstext vorhanden", "Einordnung:" in down["txt"] and len(down["con"]) >= 1, down["txt"])
    ok &= check("Termin in 8 Tagen als Hinweis", any("Quartalszahlen" in c for c in r["con"]), r["con"])

    try:
        import json
        json.dumps([r, down, weak, none])
        ok &= check("Ergebnis lässt sich als JSON speichern", True)
    except TypeError as e:
        ok &= check("Ergebnis lässt sich als JSON speichern", False, e)
    print("Selbsttest", "bestanden" if ok else "FEHLGESCHLAGEN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
