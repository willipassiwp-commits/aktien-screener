"""Selbsttest der Bewertungslogik. Läuft vor jedem Datenlauf. Schlägt ein Test fehl, wird nichts veröffentlicht."""
import os
import sys
from datetime import datetime, timezone

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

    # 4. Normale Firma: 22 Prüfungen (v7: Gewinnqualität), neue Cashflow- und Bilanzprüfungen
    r = one(base(), pm)
    ok &= check("22 Prüfungen, alle auswertbar", r["fn"] == 22 and r["fa"] == 22, (r["fn"], r["fa"]))
    # Gewinnqualität: (2,5 − 2,8 Mrd.) / 20 Mrd. = −1,5 % → erfüllt; (2,5 − 0,5) / 20 = 10 % → nicht erfüllt
    ok &= check("v7 Gewinnqualität −1,5 %, erfüllt", crit(r, "accr")[2] == -1.5 and crit(r, "accr")[4], crit(r, "accr"))
    r_acc = one(base(ocf=[0.5e9, 2.5e9, 2.3e9, 2.0e9]), pm)
    ok &= check("v7 Gewinnqualität 10 %, nicht erfüllt", crit(r_acc, "accr")[2] == 10.0 and crit(r_acc, "accr")[4] is False,
                crit(r_acc, "accr"))
    ok &= check("v7 Risiko-Gründe vorhanden", any(k[0] == "accr" for k in r["rkx"]) and any(k[0] == "vola" for k in r["rkx"]),
                r["rkx"])
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

    # 8. Nachrichtenlage, Stand der Wende, Trendalter, Einstiegsmarken
    from datetime import date
    nv, nz, ns = scoring.news_view(
        [["Testfirma beats estimates and raises outlook", "Q", "2026-11-10", "u"],
         ["Testfirma faces lawsuit over recall", "Q", "2026-11-09", "u"],
         ["Musterbank upgrade for Testfirma", "Q", "2026-11-08", "u"],
         ["Markets mixed ahead of data", "Q", "2026-11-08", "u"],
         ["Testfirma posts record quarter", "Q", "2026-09-01", "u"]], "Testfirma AG", date(2026, 11, 13))
    ok &= check("Schlagzeilen: nur aktuelle mit Firmennamen gewertet", nz == dict(n=3, pos=2, neg=1) and ns == 6.2
                and [v[4] for v in nv] == [1, -1, 1, None], (nz, ns, [v[4] for v in nv]))
    ok &= check("Unter drei Schlagzeilen keine Note",
                scoring.news_view([["Testfirma beats", "Q", "2026-11-10", "u"]], "Testfirma AG", date(2026, 11, 13))[2] is None)
    ok &= check("Stand der Wende nur im Rücksetzer", r["ph"] is None and down["ph"] is not None
                and 1 <= down["ph"][0] <= 4 and len(down["ph"]) == 4, (r["ph"], down["ph"]))
    ok &= check("Trendalter im Aufwärtstrend", r["d200"] > 0 and r["tm"] in ("früh", "etabliert", "reif")
                and (down["d200"] > 0) == (down["tm"] is not None), (r["d200"], r["tm"], down["d200"], down["tm"]))
    ok &= check("Einstiegsmarken aus der Bewertung liegen unter dem Kurs",
                all(e[0] in ("V", "A") and (e[2] is None or e[2] < r["p"]) for e in r["ez"]), r["ez"])
    ok &= check("Nachrichtenlage fließt in den Gesamtscore ein", scoring.WEIGHTS.get("n") == 5
                and sum(scoring.WEIGHTS.values()) == 100, scoring.WEIGHTS)

    # 9. Regelwerk v5: Nachrichtenfilter, Reihenfolge der Einordnung, Chart-Deckel
    tk = scoring.name_keys("Taiwan Semiconductor Manufacturing Company Limited", "2330.TW")
    tv = scoring.news_view(
        [["Nvidia crisis sends AMD up 30%", "Q", "2026-11-10", "u"], ["TSMC books record month", "Q", "2026-11-10", "u"],
         ["Taiwan Semiconductor beats estimates", "Q", "2026-11-09", "u"], ["Taiwan stocks fall", "Q", "2026-11-09", "u"]],
        "Taiwan Semiconductor Manufacturing Company Limited", date(2026, 11, 13), "2330.TW")
    ok &= check("Schlagzeilen: Kurzname TSMC erkannt, fremde Meldungen aussortiert",
                tk == (["taiwan semiconductor"], ["TSMC"]) and len(tv[0]) == 2 and tv[1]["n"] == 2, (tk, tv[0], tv[1]))
    ok &= check("Schlagzeilen: Allerweltswort allein reicht nicht",
                scoring.name_keys("Bank of America Corporation", "BAC")[0] == ["bank of america"]
                and scoring.name_keys("L'Oréal S.A.", "OR.PA")[0] == ["l oreal"]
                and scoring.name_keys("Target Corporation", "TGT") == ([], ["TGT"])
                and scoring.name_keys("NVIDIA Corporation", "NVDA") == (["nvidia"], ["NVDA"]),
                (scoring.name_keys("Bank of America Corporation", "BAC"), scoring.name_keys("L'Oréal S.A.", "OR.PA")))
    ok &= check("Einordnung im Aufwärtstrend nur nahe am Hoch und über der 50-Tage-Linie",
                r["su"] in ("Anlauf", "Trendstark", "Überhitzt", "Neutral") and r["d52"] <= 10 and not r["tc"],
                (r["su"], r["d52"], r["tc"]))
    ok &= check("Rücksetzer-Einordnungen nur in Phase 1 bis 3",
                down["su"] not in ("Turnaround-Kandidat", "Qualität im Rücksetzer") or down["ph"][0] <= 3,
                (down["su"], down["ph"]))
    ok &= check("Qualitäts-Dip: sechs Prüfungen ausgewiesen", down["D"] is None or len(down["D"]) == 6, down["D"])

    # 11. Kurzfristige Lage und Tagesbewegung
    ok &= check("Lage: Einstufung mit Belegen", r["kl"] is not None and len(r["kl"]) == 5 and len(r["kl"][3]) >= 8
                and r["kl"][0] in "UGFDASEN", r["kl"] and r["kl"][:3])
    ok &= check("Lage: Aufwärtstrend ist weder 'fällt noch' noch 'läuft aus'", r["kl"][0] in ("S", "U", "D", "N", "E"), r["kl"][:2])
    ok &= check("Lage: nach Einbruch nicht 'läuft stabil' oder 'überdehnt'", down["kl"][0] not in ("S", "U"), down["kl"][:2])
    ok &= check("Tages- und Wochenbewegung vorhanden", r["p1d"] is not None and r["p1w"] is not None, (r["p1d"], r["p1w"]))

    # 12. Zusatzsignale und Unterzeile
    from datetime import date as _d
    isum = scoring.insider_summary([["2026-11-01", "A", "CEO", "B", 100, 2e6], ["2026-10-01", "B", "Director", "B", 10, 5e4],
                                    ["2026-09-01", "C", "CFO", "S", 50, 3e6], ["2025-01-01", "D", "CEO", "B", 9, 9e6],
                                    ["2026-11-02", "The Toronto-Dominion Bank", "Issuer", "B", 1, 5e8],
                                    ["2026-11-03", "Unilever Plc", "", "B", 1, 4e8]],
                                   _d(2026, 11, 15), company="Unilever PLC")
    ok &= check("Insider: nur sechs Monate, Käufe und Verkäufe getrennt",
                isum["nb"] == 2 and isum["nbp"] == 2 and isum["vb"] == 2.05e6 and isum["ns"] == 1 and isum["top"][0][1] == "A",
                isum)
    ok &= check("Unterzeile nur bei neutral", (r["nu"] is None) or r["su"] == "Neutral", (r["su"], r["nu"]))
    ok &= check("Rücksetzer-Tief liegt nicht über dem Kurs", down["ul"] is None or down["ul"] >= 0, down["ul"])

    # 10. Trefferquote: Eintritt, Sperrfrist, Horizont mit Währung, Überrendite, Einfrieren, Löschen, Größe
    import tracking as TK
    import tempfile
    import os as _os

    def tday(d, rows, fx_usd=0.9, mw=100.0):
        # rows: {Kürzel: (Einordnung, Gesamtscore, tri)}; 60 Füllaktien für den Durchschnitt
        p = {s: [TK.SU_CODES.index(su), tot, tri] + [None] * 9 for s, (su, tot, tri) in rows.items()}
        for k in range(60):
            p["F%02d" % k] = [0, 10, 100.0 * (1.1 if d > "2026-11-20" else 1.0)] + [None] * 9
        return {"d": d, "v": 5, "fx": {"USD": fx_usd, "EUR": 1.0}, "mw": mw, "p": p, "o": {}}
    curs = {"AAA": "USD", "BBB": "EUR"}
    curs.update({"F%02d" % k: "EUR" for k in range(60)})
    cs = []
    d1 = tday("2026-11-13", {"AAA": ("Anlauf", 80, 100.0), "BBB": ("Neutral", 50, 100.0)})
    TK.update_cases(cs, d1, {"AAA": {"cur": "USD", "px": 50.0}}, start=True)
    ok &= check("Trefferquote: Startbestand mit Merkmal start",
                any(c["s"] == "AAA" and c["k"] == "Anlauf" and c["st"] == 1 for c in cs)
                and not any(c["s"] == "BBB" and c["k"] == "Anlauf" for c in cs), [(c["s"], c["k"], c["st"]) for c in cs])
    ok &= check("Trefferquote: Eintrittskurs in Euro", next(c for c in cs if c["k"] == "Anlauf")["pe"] == 45.0)
    d2 = tday("2026-11-20", {"AAA": ("Neutral", 60, 110.0), "BBB": ("Anlauf", 81, 105.0)}, fx_usd=0.99, mw=102.0)
    TK.update_cases(cs, d2, {})
    a_case = next(c for c in cs if c["s"] == "AAA" and c["k"] == "Anlauf")
    ok &= check("Trefferquote: Austritt und neuer Eintritt erkannt", a_case["x"] == "2026-11-20"
                and any(c["s"] == "BBB" and c["k"] == "Anlauf" and c["st"] == 0 for c in cs))
    d3 = tday("2026-12-05", {"AAA": ("Anlauf", 82, 112.0), "BBB": ("Anlauf", 81, 106.0)}, fx_usd=0.99, mw=103.0)
    n_before = len(cs)
    TK.update_cases(cs, d3, {})
    ok &= check("Trefferquote: Rückkehr in 30 Tagen öffnet den alten Fall", len(cs) == n_before and a_case["x"] is None,
                (len(cs), n_before, a_case["x"]))
    TK.fill_results(cs, [d1, d2, d3], curs, "2026-12-05")
    r7 = a_case["r"].get("7")
    # AAA: tri 100 -> 110 (+10 %), Dollar 0,90 -> 0,99 Euro (+10 %): in Euro +21 %. MSCI: 100 -> 102 und +10 %
    ok &= check("Trefferquote: Horizont 7 Tage mit Währungsumrechnung", r7 is not None and r7[0] == 21.0
                and r7[1] == 12.2 and r7[2] is not None, r7)
    st = TK._stats([(r7[0], r7[1], r7[2], "AAA"), (5.0, 12.2, 9.0, "BBB")])
    ok &= check("Trefferquote: Überrendite und Anteile", st[0] == 2 and st[3] == round((21.0 - 12.2 + 5.0 - 12.2) / 2, 2)
                and st[4] == 50.0, st)
    # Einfrieren und Löschen: ein Fall mit allen Horizonten
    full = dict(a_case, i=99, d="2024-01-10", r={str(h): [1.0, 0.5, 0.2] for h in TK.HORIZONS}, st=0)
    sm = TK.build_summary([full], None, [], "2026-12-05", "2024-01-10")
    ok &= check("Trefferquote: fertiger Monat wird eingefroren", "2024-01" in sm["frozen"] and "2024-01" not in sm["months"])
    kept = TK.prune_cases([full, a_case], sm, "2026-12-05")
    ok &= check("Trefferquote: Fälle nach 24 Monaten gelöscht, wenn eingefroren", len(kept) == 1 and kept[0] is a_case)
    kept2 = TK.prune_cases([full], {"frozen": {}}, "2026-12-05")
    ok &= check("Trefferquote: ohne Einfrieren wird nichts gelöscht", len(kept2) == 1)
    sm2 = TK.build_summary([a_case], sm, [], "2026-12-06", "2024-01-10")
    ok &= check("Trefferquote: Eingefrorenes bleibt erhalten und zählt seit Start mit",
                sm2["frozen"]["2024-01"] == sm["frozen"]["2024-01"]
                and sm2["win"]["all"]["a"]["Anlauf"]["alle"]["365"][0] == 1, sm2["win"]["all"]["a"].get("Anlauf"))
    with tempfile.TemporaryDirectory() as td:
        TK.save_cases(td, [dict(a_case, i=k) for k in range(3000)])
        size = sum(_os.path.getsize(_os.path.join(td, f)) for f in _os.listdir(td))
    ok &= check("Trefferquote: 3000 Fälle bleiben unter 1,5 MB", size < 1.5e6, size)
    ok &= check("Trefferquote: Rangkorrelation", TK.spearman(list(range(40)), [v * 2 for v in range(40)]) == 1.0)

    # Trefferquote: Gesamtrendite aus dem Schlusskurs des Vortags (Korrektur 20.11.2026)
    import tracking as TRK
    tdays = [{"d": "2026-11-13", "p": {"AAA": [0, 50, 100.0] + [None] * 9}, "mw": 100.0},
             {"d": "2026-11-14", "p": {"AAA": [0, 50, 100.0] + [None] * 9}, "mw": 100.0},
             {"d": "2026-11-17", "p": {"AAA": [0, 50, 100.0] + [None] * 9}, "mw": 100.0}]
    tpx = {"2026-11-12": 10.0, "2026-11-13": 11.0, "2026-11-16": 12.1}
    tfn = lambda s_: max((v for k_, v in tpx.items() if k_ <= s_), default=None)
    TRK.repair_tri(tdays, {"AAA": tfn}, tfn)
    ok &= check("Trefferquote: tri aus Vortagskursen neu verkettet", [d_["p"]["AAA"][2] for d_ in tdays] == [100.0, 110.0, 121.0]
                and tdays[2]["mw"] == 121.0, [d_["p"]["AAA"][2] for d_ in tdays])
    nd_ = TRK.make_day("2026-11-18", 6, [], {}, tdays[-1], {"EUR": 1.0}, lambda s_: {"2026-11-16": 12.1, "2026-11-17": 13.31}.get(s_, 12.1))
    ok &= check("Trefferquote: neuer Tag nutzt Schlusskurs des Vortags", abs(nd_["mw"] - 133.1) < 0.01, nd_["mw"])
    ok &= check("Extern: kurze Londoner Kürzel mit Punkt", __import__("external").tv_ticker("RR.L")[0] == "LSE:RR.")

    # Erweiterte Kurzfristanalyse und Marktampel
    import shortplus as SP
    import market as MK
    xi = pd.bdate_range(end="2026-11-20", periods=2800)
    rng_ = np.random.default_rng(11)
    mret = rng_.normal(0.0004, 0.01, len(xi))
    mser = pd.Series(100 * np.exp(np.cumsum(mret)), index=xi)
    sret = 1.5 * mret + rng_.normal(0, 0.008, len(xi))
    sret[-30:] -= 0.012                                         # Absturz am Ende
    sc_ = 100 * np.exp(np.cumsum(sret))
    sdf = pd.DataFrame({"Open": sc_, "High": sc_ * 1.008, "Low": sc_ * 0.992, "Close": sc_,
                        "Volume": np.full(len(xi), 1e6)}, index=xi)
    kx, rsl = SP.extend(sdf, mser, None, [40, 40, 15, 60, 12], 80, "orange", "S&P 500")
    ok &= check("Erweiterung: Beta etwa 1,5", kx and kx.get("mk") and 1.2 < (kx["mk"]["b250"] or 0) < 1.8, kx and kx.get("mk"))
    ok &= check("Erweiterung: Wende-Treppe bei Absturz mit Fortschritt 0", kx.get("tl") and kx["tl"]["prog"] == 0, kx.get("tl"))
    ok &= check("Erweiterung: Erschöpfung erkannt", kx.get("ex") and kx["ex"]["n"] >= 3.5, kx.get("ex"))
    ok &= check("Erweiterung: Fazit Markt × Aktie", kx.get("mx") and "Gegenwind" in kx["mx"]["t"], kx.get("mx"))
    ok &= check("Erweiterung: RS-Linie mit 170 Werten", rsl is not None and len(rsl) == 170)
    an_ = kx.get("an")
    ok &= check("Erweiterung: frühere Fälle mit Text", an_ and an_.get("t"), an_)
    st_ = {}
    c1, _ = MK.apply_hysteresis(st_, "green", 0, "2026-11-18")
    c2, _ = MK.apply_hysteresis(st_, "orange", 0, "2026-11-19")
    c3, _ = MK.apply_hysteresis(st_, "orange", 0, "2026-11-19")
    c4, _ = MK.apply_hysteresis(st_, "orange", 0, "2026-11-20")
    ok &= check("Marktampel: Wechsel erst am zweiten Tag", (c1, c2, c3, c4) == ("green", "green", "green", "orange"), (c1, c2, c3, c4))
    c5, _ = MK.apply_hysteresis(st_, "orange", 2, "2026-11-23")
    c6, _ = MK.apply_hysteresis(st_, "orange", 2, "2026-11-24")
    ok &= check("Marktampel: nach Abwärtsdruck mit zwei Bodensignalen blau", c6 == "blue", (c5, c6))
    trg, _ = MK.triggers([{"t": "Stocks fall after hot jobs report", "d": "2026-11-20T15:00Z"}],
                         datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc))
    ok &= check("Marktupdate: Auslöser Arbeitsmarkt erkannt", trg and trg[0][0] == "Arbeitsmarktdaten", trg)
    evv = MK.events_view("2026-12-03", [])
    ok &= check("Termine: Arbeitsmarktbericht morgen", any(e[1] == "US-Arbeitsmarktbericht" and e[3] == 1 for e in evv), evv[:3])

    # Analystenkonsens im Vergleich zum Land
    items = [("India", "Asien-Pazifik", 1.5)] * 10 + [("Germany", "Europa", 2.2)] * 10 + [("Malta", "Europa", 2.0)] + \
            [("United States", "Nordamerika", 1.9)] * 20
    cr = scoring.consensus_refs(items)
    ok &= check("Konsens: Landesmedian Indien 1,5, Weltmedian 1,9", abs(cr["India"][0] - 1.5) < 1e-9 and abs(cr["India"][1] - 1.9) < 1e-9, cr["India"])
    ok &= check("Konsens: kleines Land nutzt Region", cr["Malta"][2] == "Europa", cr["Malta"])
    ra = scoring.score(base(), pm, 50e9, 1.0, TODAY, cons_ref=(1.5, 1.9, "India"))      # Konsens 1,9 bei Landesschnitt 1,5
    rb = scoring.score(base(), pm, 50e9, 1.0, TODAY, cons_ref=(2.3, 1.9, "Germany"))    # Konsens 1,9 bei Landesschnitt 2,3
    ca = next(c for c in ra["A"] if c[0] == "cons")
    cb = next(c for c in rb["A"] if c[0] == "cons")
    ok &= check("Konsens: 1,9 in Indien zählt weniger als 1,9 in Deutschland", ca[2] < cb[2] and abs(ca[1][1] - 2.3) < 1e-9, (ca, cb))
    rc = scoring.score(base(), pm, 50e9, 1.0, TODAY, cons_ref=(2.5, 1.9, "Switzerland"))  # 1,9 − 2,5 + 1,9 = 1,3
    rd = scoring.score(base(), pm, 50e9, 1.0, TODAY, cons_ref=(3.5, 1.9, "Switzerland"))  # wäre 0,3 → auf 1 begrenzt
    ok &= check("Konsens: bereinigter Wert unter 1 wird auf 1 begrenzt, nicht verworfen",
                rd["a"] is not None and rd["an"]["adj"] == 1.0 and rc["a"] is not None, (rd.get("an"), rd.get("a")))
    ok &= check("Konsens: Anzeige behält den echten Wert", ra["an"]["mean"] == 1.9 and ra["an"]["adj"] == 2.3, ra["an"])

    # Kurzfristige Chartanalyse
    kix = pd.bdate_range("2025-01-01", periods=300)
    def kframe(step):
        cc = 100 * np.exp(np.arange(300) * step + 0.004 * np.sin(np.arange(300)))
        return pd.DataFrame({"Open": cc * 0.999, "High": cc * 1.01, "Low": cc * 0.99, "Close": cc,
                             "Volume": np.where(np.arange(300) % 2 == 0, 2e6, 1e6)}, index=kix)
    ka_up, ser_up = scoring.short_analysis(kframe(0.003))
    ka_dn, _ = scoring.short_analysis(kframe(-0.003))
    ok &= check("Kurzfristanalyse: Anstieg ergibt Note über 6", ka_up["n"] > 6, ka_up["n"])
    ok &= check("Kurzfristanalyse: Rückgang ergibt Note unter 4", ka_dn["n"] < 4, ka_dn["n"])
    ok &= check("Kurzfristanalyse: Note zwischen 0 und 10, Signale gezählt",
                0 <= ka_dn["n"] <= 10 and ka_up["pos"] + ka_up["neg"] <= ka_up["cnt"])
    ok &= check("Kurzfristanalyse: Pivot-Marken geordnet",
                ka_up["lv"]["s2"] <= ka_up["lv"]["s1"] <= ka_up["lv"]["pp"] <= ka_up["lv"]["r1"] <= ka_up["lv"]["r2"], ka_up["lv"])
    ok &= check("Kurzfristanalyse: 170 Tageskurse für den Chart", len(ser_up["c"]) == 170 and len(ser_up["v"]) == 170)
    ok &= check("Kurzfristanalyse: zu wenig Daten ergibt nichts", scoring.short_analysis(kframe(0.001).iloc[:100]) is None)
    import external as EX
    ok &= check("Extern: TradingView-Kürzel", EX.tv_ticker("0700.HK") == ["HKEX:700"] and EX.tv_ticker("VOLV-B.ST") == ["OMXSTO:VOLV_B"]
                and EX.tv_ticker("BRK-B")[0] == "NASDAQ:BRK.B" and EX.tv_ticker("X.ZZ") == [])
    tcp = EX._tc_parse({"finance": {"result": {"instrumentInfo": {"technicalEvents": {"shortTermOutlook": {"direction": "Bullish", "score": 3}},
                        "keyTechnicals": {"support": 10, "resistance": 12}}, "events": []}}})
    ok &= check("Extern: Trading Central gelesen", tcp["s"][:2] == ["Bullish", 3] and tcp["sup"] == 10, tcp)
    ok &= check("Extern: leere Antwort ergibt nichts", EX._tc_parse({"finance": {"result": {}}}) is None)
    old_c = {"tc": {"AAA": {"s": ["Bullish", 1], "ts": 0}}, "tv": {}}
    ok &= check("Extern: veraltete Werte werden nicht angezeigt", EX.view("AAA", old_c, now=10 * 86400) is None)

    # v7: Kaufdruck (Volumen an Plus- gegen Minustagen) und Stärke-Rang mit zwei Punkten
    idx = pd.bdate_range("2026-01-01", periods=80)
    cc = pd.Series(100 + np.arange(80) % 2, index=idx, dtype=float)     # abwechselnd +1 / −1
    vv = pd.Series(np.where(np.arange(80) % 2 == 1, 3e6, 1e6), index=idx, dtype=float)  # Plustage 3× Volumen
    ok &= check("v7 Kaufdruck: Plustage mit 3-fachem Volumen ergibt etwa 3", 2.85 <= scoring.acc_ratio(cc, vv) <= 3.15,
                scoring.acc_ratio(cc, vv))
    ok &= check("v7 Kaufdruck: ohne Volumen nichts", scoring.acc_ratio(cc, None) is None)
    ok &= check("v7 Kaufdruck-Noten", [scoring.acc_grade(v) for v in (95, 60, 59, 20, 3, None)] == ["A", "B", "C", "D", "E", None])
    recs = []
    for i in range(60):
        rr = scoring.score(base(), scoring.price_metrics(frame(100, growth=0.02 + i * 0.004)), 50e9, 1.0, TODAY)
        rr["ud"] = 0.5 + i * 0.05
        recs.append(rr)
    recs = scoring.finalize(recs)
    top, mid, low = recs[-1], recs[35], recs[0]
    pts = lambda x, k: next(t for t in x["T"] if t[0] == k)[2]          # noqa: E731
    ok &= check("v7 Stärke-Rang: Spitze 2 Punkte, Mitte 1, Ende 0",
                (pts(top, "rs"), pts(mid, "rs"), pts(low, "rs")) == (2, 1, 0) and top["rsr"] == 99,
                (pts(top, "rs"), pts(mid, "rs"), pts(low, "rs"), top["rsr"]))
    ok &= check("v7 Kaufdruck: Spitze A mit Punkt, Ende E ohne", top["ac"] == "A" and pts(top, "acc") == 1
                and low["ac"] == "E" and pts(low, "acc") == 0, (top["ac"], low["ac"]))
    ok &= check("v7 Chart-Note höchstens 10", all(0 <= x["t"] <= 10 for x in recs))
    # Trefferquote nach Score-Stufen
    import tracking as TR
    d0 = {"d": "2026-01-05", "v": 7, "fx": {"USD": 1.0}, "p": {}}
    d1 = {"d": "2026-01-12", "v": 7, "fx": {"USD": 1.0}, "p": {}}
    for i in range(100):
        sc_ = 30 + i * 0.6                                  # Scores 30 … 89
        row0, row1 = [0, sc_, 100.0] + [None] * 9, [0, sc_, 100.0 + (sc_ - 60) / 10] + [None] * 9
        d0["p"]["S%d" % i], d1["p"]["S%d" % i] = row0, row1
    bk = TR.update_buckets([], [d0, d1], {"S%d" % i: "USD" for i in range(100)})
    st = {b[0]: b for b in bk[0][3]} if bk else {}
    ok &= check("Trefferquote: Score-Stufen getrennt, 80+ vor unter 50",
                bk and bk[0][1] == 7 and st["80+"][1] > 0 > st["unter 50"][1] and sum(b[4] for b in bk[0][3]) == 100, bk)

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
