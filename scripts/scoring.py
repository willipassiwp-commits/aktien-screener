"""Regelbasierte Bewertung. Jede Aktie durchläuft dieselbe Rechnung, es gibt keine Ermessenspunkte.

Aufbau des Ergebnisses je Aktie (kurze Schlüssel, weil die Website die Datei lädt):
  F  zehn Fundamental-Kriterien  [Schlüssel, Wert, Schwelle, erfüllt]
  M  fünf Burggraben-Kennzahlen  [Schlüssel, Wert, Punkte 0-5]
  T  sechs Technik-Bausteine     [Schlüssel, Wert, Punkte, Maximum]
  q  Datenqualität A/B/C, qn Hinweise dazu
"""
import math
from datetime import datetime

import numpy as np
import pandas as pd

VERSION = 2

# Schwellen je Lynch-Kategorie. Belegt aus öffentlichen Analysen: Average 6/6/6/10 und Verschuldung 3,0,
# Fast Grower EBIT-Marge 20. Die übrigen Werte sind Annahmen.
TH = {
    "Slow Grower": dict(revH=2, revE=2, ebitH=3, epsE=4, eq=25, roe=10, debt=3.0, margin=10, pe5=20, divG=5, cap=18),
    "Average Grower": dict(revH=6, revE=6, ebitH=6, epsE=10, eq=30, roe=15, debt=3.0, margin=15, pe5=25, cap=22),
    "Fast Grower": dict(revH=15, revE=15, ebitH=15, epsE=20, eq=30, roe=15, debt=1.5, margin=20, pe5=30, cap=28),
    "Zykliker": dict(revH=4, revE=4, ebitH=4, epsE=6, eq=30, roe=12, debt=2.5, margin=10, pe5=15, cap=15),
}
# Banken, Versicherer und andere Bilanz-Finanzwerte: Eigenkapitalquote, Nettoverschuldung/EBITDA und EBIT-Marge
# sagen dort nichts aus. Stattdessen drei branchenübliche Kennzahlen. Die Schwellen sind so gewählt, dass sie
# etwa so oft bestanden werden wie die drei ersetzten Kriterien bei den übrigen Unternehmen (rund 60 bis 78 %).
FIN_TH = {
    "bank": dict(eqA=6.0, roa=0.7, nm=28.0),
    "insurer": dict(eqA=6.0, roa=0.7, nm=9.0),
    "other": dict(eqA=8.0, roa=1.0, nm=20.0),
}
# Finanzwerte sind an der Börse dauerhaft niedriger bewertet. Ihre KGV-Schwelle und ihr KGV-Deckel liegen deshalb
# um diesen Faktor tiefer, sonst bestünden sie das Bewertungskriterium fast immer.
FIN_PE_FACTOR = 0.5
FIN_OTHER_INDUSTRIES = {"Capital Markets", "Credit Services", "Mortgage Finance", "Financial Conglomerates",
                        "Asset Management"}
CYC_SECTORS = {"Energy"}
CYC_INDUSTRIES = {"Auto Manufacturers", "Auto Parts", "Airlines", "Residential Construction", "Marine Shipping",
                  "Farm & Heavy Construction Machinery", "Building Materials", "Steel", "Aluminum", "Copper", "Gold",
                  "Silver", "Other Industrial Metals & Mining", "Other Precious Metals & Mining", "Coking Coal",
                  "Thermal Coal", "Chemicals", "Agricultural Inputs", "Paper & Paper Products",
                  "Lumber & Wood Production"}
G_CAP = {"Slow Grower": 0.08, "Average Grower": 0.15, "Fast Grower": 0.20, "Zykliker": 0.08}
SUBUNIT = {"GBp": 0.01, "GBX": 0.01, "ZAc": 0.01, "ILA": 0.01}
WEIGHTS = dict(f=30, m=20, t=20, r=30)


def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def clamp(x, a, b):
    return max(a, min(b, x))


def r1(x, d=1):
    return None if x is None else round(x, d)


def pct(v, d=1):
    return r1(v * 100, d) if v is not None else None


def de(v, d=1):
    """Zahl in deutscher Schreibweise für Hinweistexte."""
    return ("%.*f" % (d, v)).replace(".", ",")


def bounded(v, lo, hi):
    """Werte außerhalb des plausiblen Bereichs gelten als Datenfehler."""
    return v if v is not None and lo <= v <= hi else None


def first(lst):
    """Jüngster vorhandener Wert (Yahoo liefert das neueste Jahr manchmal leer)."""
    for x in lst or []:
        if x is not None:
            return x
    return None


def pair(a, b):
    """Jüngstes Jahr, für das beide Reihen einen Wert haben."""
    for x, y in zip(a or [], b or []):
        if x is not None and y is not None:
            return x, y
    return None, None


def hist_growth(lst):
    """Wachstum p. a. über die vorhandenen Jahresabschlüsse (neuester zuerst)."""
    idx = [i for i, x in enumerate(lst or []) if x is not None]
    if len(idx) < 2 or idx[-1] - idx[0] < 2:
        return None
    new, old, yrs = lst[idx[0]], lst[idx[-1]], idx[-1] - idx[0]
    if new <= 0:
        return -1.0
    if old <= 0:
        return None
    return (new / old) ** (1 / yrs) - 1


def declines(lst):
    v = [x for x in (lst or []) if x is not None]
    if len(v) < 3:
        return None
    return sum(1 for i in range(len(v) - 1) if v[i] < v[i + 1])


def perf(c, days):
    if len(c) <= days or c.iloc[-days - 1] <= 0:
        return None
    return float(c.iloc[-1] / c.iloc[-days - 1] - 1)


def price_metrics(df):
    """Kennzahlen aus der Kurshistorie. Erwartet Tagesdaten mit Schlusskursen, die um Aktiensplits, aber nicht um
    Dividenden bereinigt sind (wie im Chart). Gibt None zurück, wenn zu wenig Daten da sind."""
    if df is None or "Close" not in df:
        return None
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    c = c[c > 0]
    if len(c) < 250:
        return None
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)
        df = df.copy()
        df.index = df.index.tz_localize(None)
    last, end = float(c.iloc[-1]), c.index[-1]
    years = (end - c.index[0]).days / 365.25
    c5 = c[c.index >= end - pd.DateOffset(years=5)]
    y5 = (end - c5.index[0]).days / 365.25
    cagr5 = (last / float(c5.iloc[0])) ** (1 / y5) - 1 if y5 >= 1 else None
    sma200, sma50 = float(c.iloc[-200:].mean()), float(c.iloc[-50:].mean())
    sma200_prev = float(c.iloc[-260:-60].mean()) if len(c) >= 260 else None
    sma50_prev = float(c.iloc[-70:-20].mean()) if len(c) >= 70 else None
    high = float(c.max())
    mdd5 = float((1 - c5 / c5.cummax()).max())
    vol_above = None
    if "Volume" in df:
        c3 = c[c.index >= end - pd.DateOffset(years=3)]
        v3 = pd.to_numeric(df["Volume"], errors="coerce").reindex(c3.index).fillna(0)
        tot = float(v3.sum())
        if tot > 0:
            vol_above = float(v3[c3 > last].sum()) / tot
    w = c5.resample("W").last().dropna()
    r2, slope = None, None
    if len(w) >= 100:
        x = np.arange(len(w))
        y = np.log(w.values)
        slope, icpt = np.polyfit(x, y, 1)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(((y - (slope * x + icpt)) ** 2).sum()) / ss if ss > 0 else None
    rets = np.log(c.iloc[-253:]).diff().dropna()
    vola = float(rets.std() * math.sqrt(252)) if len(rets) > 50 else None
    div_ttm, div_growth = 0.0, None
    if "Dividends" in df:
        d = pd.to_numeric(df["Dividends"], errors="coerce").fillna(0)
        d = d[d > 0]
        div_ttm = float(d[d.index > end - pd.Timedelta(days=365)].sum())
        by_year = d.groupby(d.index.year).sum()
        y1, y0 = end.year - 1, end.year - 6
        if y1 in by_year.index and y0 in by_year.index and by_year[y0] > 0:
            div_growth = (float(by_year[y1]) / float(by_year[y0])) ** (1 / 5) - 1
        elif div_ttm == 0:
            div_growth = 0.0

    def trend(avg, prev):
        if prev is None:
            return 0
        if last > avg and avg > prev:
            return 1
        if last < avg and avg < prev:
            return -1
        return 0

    tr = [0 if cagr5 is None else 1 if cagr5 > 0.03 else -1 if cagr5 < -0.03 else 0,
          trend(sma200, sma200_prev), trend(sma50, sma50_prev)]
    return dict(last=last, date=str(end.date()), years=years, cagr5=cagr5, y5=y5, sma200=sma200, sma50=sma50,
                sma200_prev=sma200_prev, sma50_prev=sma50_prev, dist=1 - last / high, mdd5=mdd5, vol_above=vol_above,
                r2=r2, slope=slope, vola=vola, div_ttm=div_ttm, div_growth=div_growth, series=c, tr=tr,
                p1m=perf(c, 21), p3m=perf(c, 63), p6m=perf(c, 126), p1y=perf(c, 252),
                weekly=[float("%.4g" % v) for v in w.values])


def price_on(series, date):
    try:
        v = series.asof(pd.Timestamp(date))
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def fin_type(fund):
    """Bilanz-Finanzwert? Banken und Versicherer immer, andere Finanzdienstleister nur, wenn Yahoo für sie kein
    EBITDA ausweist (Broker, Kreditkartenbanken). Börsen, Zahlungsnetzwerke und Makler zählen als normale Firmen."""
    if fund.get("sector") != "Financial Services":
        return None
    ind = fund.get("industry") or ""
    if ind.startswith("Banks"):
        return "bank"
    if ind.startswith("Insurance -"):
        return "insurer"
    if ind in FIN_OTHER_INDUSTRIES and first(fund.get("ebitda")) is None:
        return "other"
    return None


def lynch(fund, rev_h):
    if fund.get("sector") in CYC_SECTORS or fund.get("industry") in CYC_INDUSTRIES:
        return "Zykliker"
    g = fund.get("revGrowthEst")
    if g is None:
        g = rev_h
    if g is None:
        return None
    return "Slow Grower" if g < 0.055 else "Average Grower" if g < 0.15 else "Fast Grower"


def ge(v, t):
    return None if v is None else bool(v >= t)


def le(v, t):
    return None if v is None else bool(v <= t)


def steps(v, cuts, higher=True):
    """Punkte 0-5 nach fünf Stufen."""
    if v is None:
        return None
    return sum(1 for c in cuts if (v >= c if higher else v <= c))


def total(rec, w=WEIGHTS):
    """Gesamtscore 0-100 mit der Standardgewichtung. Trendbruch und Abwärtstrend sind bei 61 gedeckelt."""
    ws = sum(w.values()) or 1
    t = (w["f"] * rec["f"] * 10 + w["m"] * rec["m"] * 4 + w["t"] * rec["t"] * 10 + w["r"] * rec["rs"]) / ws
    return min(t, 61.0) if rec["st"] in ("Abwärtstrend", "Trendbruch") else t


def score(fund, pm, mcap_usd=None, fx_fin=None, today=None):
    """Bewertet eine Aktie. fund: Fundamentaldaten aus dem Cache, pm: Ergebnis von price_metrics.

    fx_fin: Umrechnung von der Kurswährung (Haupteinheit) in die Bilanzwährung, None wenn unbekannt.
    Yahoo liefert trailingEps/forwardEps in der Kurswährung, die Abschlüsse in der Bilanzwährung.
    """
    if pm is None:
        return None
    notes = []
    y = fund.get("y") or {}
    rev, ebit, ni = fund.get("revenue") or [], fund.get("ebit") or [], fund.get("netIncome") or []
    equity = fund.get("equity") or []
    ft = fin_type(fund)
    fin = ft is not None
    rev_h = bounded(hist_growth(rev), -0.9, 5)
    # Unplausible Umsatzschätzung (Yahoo-Fehler, z. B. +100 % bei japanischen Werten) verwerfen
    rg = fund.get("revGrowthEst")
    if rg is not None and (rg > 0.5 and (rev_h is None or rev_h < 0.25) or rg < -0.5):
        fund = dict(fund, revGrowthEst=None)
        notes.append("Umsatzschätzung von Yahoo unplausibel (%s %%) und verworfen" % de(rg * 100, 0))
    cat = lynch(fund, rev_h)
    if cat is None:
        return None
    T = TH[cat]
    pe_factor = FIN_PE_FACTOR if fin else 1.0
    cap = T["cap"] * pe_factor
    unit = SUBUNIT.get(fund.get("currency"), 1.0)
    price = pm["last"]
    price_m = price * unit  # Kurs in der Haupteinheit (Pfund statt Pence)
    if fx_fin is None and fund.get("finCurrency") and fund.get("finCurrency") == fund.get("currency"):
        fx_fin = 1.0
    # Aktienzahl aus Börsenwert und Kurs zum Abrufzeitpunkt (deckt auch mehrere Aktiengattungen ab)
    shares = None
    if fund.get("mcap") and fund.get("priceAtFetch"):
        shares = fund["mcap"] / (fund["priceAtFetch"] * unit)

    # ---------- Fundamental: zehn Kriterien, je ein Punkt ----------
    ebit_h = bounded(hist_growth(ebit), -1, 5)
    rev_e, eps_e = fund.get("revGrowthEst"), bounded(fund.get("epsGrowthEst"), -1, 10)
    dec = declines(ni)
    eq = first(equity)
    eq_a, assets = pair(equity, fund.get("assets"))
    eq_ratio = bounded(eq_a / assets if eq_a is not None and assets else None, -2, 1)
    ni0, eq_n = pair(ni, equity)
    roe = ni0 / eq_n if eq_n and eq_n > 0 and ni0 is not None else None
    if roe is None and eq is None:
        roe = y.get("returnOnEquity")
    roe = bounded(roe, -3, 5)
    debt, cash, ebitda = first(fund.get("totalDebt")), first(fund.get("cash")), first(fund.get("ebitda"))
    if debt is None:
        debt = y.get("totalDebt")
    if cash is None:
        cash = y.get("totalCash")
    if ebitda is None and not fin:
        ebitda = y.get("ebitda")
    nd = None
    if debt is not None and ebitda is not None:
        net = debt - (cash or 0)
        nd = net / ebitda if ebitda > 0 else (0.0 if net <= 0 else 99.0)
    ebit0, rev0 = pair(ebit, rev)
    margin = ebit0 / rev0 if ebit0 is not None and rev0 and rev0 > 0 else y.get("operatingMargins")
    margin = bounded(margin, -2, 1)
    ni_a, assets_n = pair(ni, fund.get("assets"))
    roa = bounded(ni_a / assets_n if ni_a is not None and assets_n else None, -1, 1)
    ni_r, rev_n = pair(ni, rev)
    net_margin = bounded(ni_r / rev_n if ni_r is not None and rev_n and rev_n > 0 else None, -2, 1)

    # Gewinn je Aktie: Yahoos Wert der letzten zwölf Monate gilt, wenn er zu mindestens einer anderen Angabe passt
    # (letzter Jahresabschluss, Jahresgewinn durch Aktienzahl, Schätzung fürs nächste Jahr). Das fängt Aktiensplits
    # und Einheitenfehler ab, ohne echte Gewinnsprünge zu verwerfen.
    eps_t, eps_f = fund.get("trailingEps"), fund.get("forwardEps")
    ni_first, d_first = first(ni), first(fund.get("dilutedEps"))
    ref_eps = d_first / fx_fin if fx_fin and d_first and d_first > 0 else None
    ref_ni = ni_first / shares / fx_fin if fx_fin and shares and ni_first and ni_first > 0 else None
    refs = [r for r in (ref_eps, ref_ni, eps_f) if r and r > 0]
    eps, eps_src = None, None
    if eps_t is not None and eps_t > 0 and (not refs or any(r / 3 <= eps_t <= r * 3 for r in refs)):
        eps, eps_src = eps_t, "t"
    elif eps_f is not None and eps_f > 0:
        eps, eps_src = eps_f, "f"
        if eps_t is not None and eps_t <= 0:
            notes.append("Verlust in den letzten zwölf Monaten, KGV mit der Gewinnschätzung für das nächste Jahr "
                         "gerechnet")
        else:
            notes.append("Gewinn je Aktie der letzten zwölf Monate fehlt oder passt zu keiner anderen Angabe, KGV "
                         "mit der Schätzung für das nächste Jahr gerechnet")
    elif (eps_t is not None and eps_t <= 0) or (eps_t is None and d_first is not None and d_first <= 0):
        eps_src = "loss"
    elif ref_eps or ref_ni:
        eps, eps_src = ref_eps or ref_ni, "a"
        notes.append("Yahoo liefert keinen aktuellen Gewinn je Aktie, KGV mit dem letzten Jahresabschluss gerechnet")
    pe = bounded(price_m / eps if eps else None, 0.5, 1000)
    ype = y.get("trailingPE")
    if eps_src == "t" and pe and ype and ype > 0 and abs(pe / ype - 1) > 0.25:
        notes.append("KGV weicht von der Yahoo-Angabe ab (%s gegenüber %s)" % (de(pe), de(ype)))

    # Dauerhaft angesetztes Gewinnwachstum: Langfristschätzung der Analysten, sonst Schätzung für das laufende
    # und nächste Jahr. Höchstens Umsatzwachstum + 5 Punkte und gedeckelt je Kategorie, damit ein einzelnes
    # Erholungsjahr nicht fünf Jahre fortgeschrieben wird.
    ltg = fund.get("ltg")
    rev_ref = rev_e if rev_e is not None else rev_h
    if ltg is not None and 0 < ltg <= 0.6:
        g_raw, g_src = ltg, "L"
    elif eps_e is not None:
        g_raw, g_src = eps_e, "K"
    else:
        g_raw, g_src = rev_ref, "U"
    if g_raw is not None and rev_ref is not None:
        g_raw = min(g_raw, rev_ref + 0.05)
    g = clamp(g_raw, 0.0, G_CAP[cat]) if g_raw is not None else None
    pe5 = pe / (1 + g) ** 5 if pe is not None and g is not None else None

    if cat == "Slow Grower":
        c3 = ["divG", pct(pm["div_growth"]), T["divG"], ge(pm["div_growth"], T["divG"] / 100)]
    else:
        c3 = ["ebitH", pct(ebit_h), T["ebitH"], ge(ebit_h, T["ebitH"] / 100)]
    if fin:
        FT = FIN_TH[ft]
        c6 = ["eqA", pct(eq_ratio), FT["eqA"], ge(eq_ratio, FT["eqA"] / 100)]
        c8 = ["roa", pct(roa, 2), FT["roa"], ge(roa, FT["roa"] / 100)]
        c9 = ["nm", pct(net_margin), FT["nm"], ge(net_margin, FT["nm"] / 100)]
    else:
        c6 = ["eq", pct(eq_ratio), T["eq"], ge(eq_ratio, T["eq"] / 100)]
        c8 = ["debt", r1(nd), T["debt"], le(nd, T["debt"])]
        c9 = ["margin", pct(margin), T["margin"], ge(margin, T["margin"] / 100)]
    pe5_th = round(T["pe5"] * pe_factor, 1)
    F = [
        ["revH", pct(rev_h), T["revH"], ge(rev_h, T["revH"] / 100)],
        ["revE", pct(rev_e), T["revE"], ge(rev_e, T["revE"] / 100)],
        c3,
        ["epsE", pct(eps_e), T["epsE"], ge(eps_e, T["epsE"] / 100)],
        ["dec", dec, 1, le(dec, 1)],
        c6,
        ["roe", pct(roe), T["roe"], (False if eq is not None and eq <= 0 else ge(roe, T["roe"] / 100))],
        c8,
        c9,
        ["pe5", r1(pe5), pe5_th, (False if eps_src == "loss" else le(pe5, pe5_th))],
    ]
    avail = [x for x in F if x[3] is not None]
    if len(avail) < 6:
        return None
    fund10 = round(sum(1 for x in avail if x[3]) / len(avail) * 10, 1)

    # ---------- Burggraben als Kennzahlen-Ersatz für Porter: fünf Größen, je 0-5 Punkte ----------
    gross = bounded(fund.get("grossMargin"), -1, 1)
    margins = [e / r for e, r in zip(ebit, rev) if e is not None and r and r > 0]
    m_std = float(np.std(margins)) * 100 if len(margins) >= 3 else None
    roic = None
    if first(ebit) is not None and eq is not None and debt is not None and eq + debt > 0:
        roic = bounded(first(ebit) * 0.75 / (eq + debt), -2, 3)
    rev_dec = declines(rev)
    dec_pts = {0: 5, 1: 3, 2: 1}
    if fin:
        roes = [n / e for n, e in zip(ni, equity) if n is not None and e and e > 0]
        roe_min = min(roes) if len(roes) >= 3 else None
        M = [
            ["roe", pct(roe), steps(roe, [0.08, 0.11, 0.14, 0.17, 0.20])],
            ["roeMin", pct(roe_min), steps(roe_min, [0.06, 0.08, 0.11, 0.14, 0.17])],
            ["roa", pct(roa, 2), steps(roa, [0.005, 0.008, 0.012, 0.018, 0.025])],
            ["niDec", dec, None if dec is None else dec_pts.get(dec, 0)],
            ["revDec", rev_dec, None if rev_dec is None else dec_pts.get(rev_dec, 0)],
        ]
    else:
        M = [
            ["gross", pct(gross), steps(gross, [0.2, 0.3, 0.4, 0.5, 0.6])],
            ["margin", pct(margin), steps(margin, [0.10, 0.15, 0.20, 0.25, 0.30])],
            ["mStd", r1(m_std), steps(m_std, [12, 8, 5, 3, 1.5], higher=False)],
            ["roic", pct(roic), steps(roic, [0.06, 0.10, 0.15, 0.20, 0.25])],
            ["revDec", rev_dec, None if rev_dec is None else dec_pts.get(rev_dec, 0)],
        ]
    mv = [x[2] for x in M if x[2] is not None]
    if len(mv) < 3:
        return None
    moat25 = round(sum(mv) / (len(mv) * 5) * 25, 1)

    # ---------- Technik: zehn Punkte aus der Kurshistorie ----------
    above = price > pm["sma200"]
    c5g = pm["cagr5"]
    p_long = 0 if c5g is None else 2 if c5g >= 0.05 else 1 if c5g > 0 else 0
    p_stab = 1 if pm["y5"] >= 4.9 and pm["mdd5"] < 0.30 else 0
    p_hist = 2 if pm["years"] >= 19.5 else 1 if pm["years"] >= 10 else 0
    va = pm["vol_above"]
    if va is not None:
        p_vol = 2 if va <= 0.15 else 1 if va <= 0.35 else 0
    else:
        p_vol = 2 if pm["dist"] <= 0.10 else 1 if pm["dist"] <= 0.25 else 0
    q1 = 1 if pm["sma200_prev"] and pm["sma50"] > pm["sma200"] > pm["sma200_prev"] else 0
    q2 = 1 if pm["r2"] is not None and pm["r2"] >= 0.7 and pm["slope"] > 0 else 0
    Tk = [
        ["long", pct(c5g), p_long, 2], ["sma200", pct(price / pm["sma200"] - 1), 1 if above else 0, 1],
        ["stab", pct(pm["mdd5"]), p_stab, 1], ["hist", r1(pm["years"], 0), p_hist, 2],
        ["vol", pct(va), p_vol, 2], ["qual", r1(pm["r2"], 2), q1 + q2, 2],
    ]
    tech = sum(x[2] for x in Tk)

    # ---------- Renditeerwartung: Exit-KGV über fünf Jahre ----------
    hist_pe = None
    if fx_fin:
        pes, fixed = [], False
        # Passen Gewinn je Aktie und Jahresgewinn durch Aktienzahl im jüngsten Jahr zusammen, lassen sich ältere
        # Jahre gegenprüfen. Bei mehreren Aktiengattungen passen sie nie zusammen, dann gilt der Gewinn je Aktie.
        simple = bool(ref_eps and ref_ni and abs(ref_eps / ref_ni - 1) <= 0.35)
        for d, e, n in zip(fund.get("incDates") or [], fund.get("dilutedEps") or [], ni):
            p = price_on(pm["series"], d)
            if not p:
                continue
            p_fin = p * unit * fx_fin
            pe_eps = p_fin / e if e and e > 0 else None
            pe_ni = p_fin * shares / n if simple and n and n > 0 else None
            # Der Gewinn je Aktie alter Abschlüsse ist nach Aktiensplits oft nicht angepasst. Weicht er stark vom
            # Wert aus Jahresgewinn und heutiger Aktienzahl ab, gilt der zweite.
            if pe_eps and pe_ni and abs(pe_eps / pe_ni - 1) > 0.35:
                pe_eps, fixed = None, True
            v = pe_eps or pe_ni
            if v and 1 < v < 300:
                pes.append(v)
        if len(pes) >= 2:
            hist_pe = float(np.median(pes))
            if fixed:
                notes.append("Historisches KGV über Jahresgewinn und Aktienzahl gerechnet, weil der Gewinn je Aktie "
                             "alter Abschlüsse nicht zum Kurs passt (meist ein Aktiensplit)")
    dy = pm["div_ttm"] / price
    if dy > 0.15:
        notes.append("Dividendenrendite von %s %% unplausibel und auf 0 gesetzt" % de(dy * 100, 0))
        dy = 0.0
    # Ziel-KGV: historischer Median, gedeckelt je Kategorie und nie über dem heutigen KGV.
    # Die Rendite soll aus Gewinnwachstum und Dividende kommen, nicht aus einer höheren Bewertung.
    kgve = min(x for x in (hist_pe, cap, pe) if x is not None)
    rend = lo = hi = None
    if pe is not None and g is not None:
        rend = ((1 + g) ** 5 * kgve / pe) ** 0.2 - 1 + dy
        rend = min(rend, 0.25)  # Deckel gegen Ausreißer
        lo = (1 + 0.6 * g) ** 5 * 0.85 * kgve / pe - 1
        hi = (1 + g) ** 5 * min(hist_pe or pe, cap * 1.2) / pe - 1
    rs = 0.0 if rend is None else clamp(rend / 0.12, 0, 1) * 100

    # ---------- Risikoklasse 1 (defensiv) bis 5 (offensiv) ----------
    risk = 1
    v = pm["vola"]
    if v is not None and v >= 0.25:
        risk += 1
    if v is not None and v >= 0.40:
        risk += 1
    if pm["mdd5"] >= 0.40:
        risk += 1
    if (not fin and nd is not None and nd > 3) or (fin and c6[3] is False):
        risk += 1
    if dec is not None and dec >= 2:
        risk += 1
    risk = clamp(risk, 1, 5)

    # ---------- Trendstatus ----------
    dist = pm["dist"]
    if above and dist <= 0.10:
        status = "Aufwärtstrend"
    elif above:
        status = "Erholung"
    elif dist <= 0.20:
        status = "Korrektur"
    elif price > pm["sma50"] and pm["sma50_prev"] and pm["sma50"] > pm["sma50_prev"]:
        status = "Bodenbildung"
    elif c5g is not None and c5g > 0:
        status = "Trendbruch"
    else:
        status = "Abwärtstrend"

    # ---------- Datenqualität ----------
    ym, yr = y.get("operatingMargins"), y.get("returnOnEquity")
    if not fin and margin is not None and ym is not None and abs(margin - ym) > 0.12:
        notes.append("EBIT-Marge im letzten Jahresabschluss %s %%, laut Yahoo in den letzten zwölf Monaten %s %%"
                     % (de(margin * 100), de(ym * 100)))
    if roe is not None and yr is not None and abs(roe) < 1 and abs(yr) < 1 and abs(roe - yr) > 0.15:
        notes.append("Eigenkapitalrendite im letzten Jahresabschluss %s %%, laut Yahoo zuletzt %s %%"
                     % (de(roe * 100), de(yr * 100)))
    today = today or datetime.utcnow()
    inc_dates = fund.get("incDates") or []
    if inc_dates:
        age = (today - datetime.strptime(inc_dates[0], "%Y-%m-%d")).days
        if age > 550:
            notes.append("Letzter Jahresabschluss vom %s, also über 18 Monate alt" % inc_dates[0])
    if fx_fin is None:
        notes.append("Wechselkurs zwischen Kurs- und Bilanzwährung fehlt, historisches KGV nicht berechnet")
    if fund.get("sector") == "Real Estate":
        notes.append("Immobilienwert: Das KGV ist dort wenig aussagekräftig, weil Abschreibungen den Gewinn drücken")
    hard = [n for n in notes if not n.startswith(("Historisches KGV", "Immobilienwert"))]
    quality = "A" if len(avail) == 10 and not hard else "B" if len(avail) >= 8 and len(hard) <= 2 else "C"

    growth = g_raw or 0
    strat = "Dividende" if dy >= 0.025 else "Wachstum" if cat == "Fast Grower" or growth >= 0.10 else "Cashflow"
    rec = dict(
        cat=cat, strat=strat, fin=ft or "", p=float("%.5g" % price), d=pm["date"],
        mc=r1(mcap_usd / 1e9) if mcap_usd else None,
        F=F, f=fund10, fa=len(avail), M=M, m=moat25, T=Tk, t=tech,
        pe=r1(pe), es=eps_src, hpe=r1(hist_pe), kgve=r1(kgve), g=pct(g), gs=g_src, dy=r1(dy * 100, 2),
        r=pct(rend), rs=round(rs), lo=r1(lo * 100, 0) if lo is not None else None,
        hi=r1(hi * 100, 0) if hi is not None else None, dist=pct(dist), risk=risk, st=status,
        tr=pm["tr"], pf=[pct(pm["p1m"]), pct(pm["p3m"]), pct(pm["p6m"]), pct(pm["p1y"])],
        vo=pct(pm["vola"], 0), q=quality, qn=notes,
    )
    rec["tot"] = round(total(rec), 1)
    return rec
