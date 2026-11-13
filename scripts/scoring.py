"""Regelbasierte Bewertung. Jede Aktie durchläuft dieselbe Rechnung, es gibt keine Ermessenspunkte.

Aufbau nach dem Muster professioneller Analyse-Werkzeuge:
  - Checklisten mit vielen Einzelprüfungen je Themenblock (wie die 30 Prüfungen bei Simply Wall St oder die neun
    Tests des Piotroski-Scores),
  - getrennte Faktoren für Qualität, Bewertung, Kursbild und Analystenbild (wie die Faktornoten bei Seeking Alpha
    oder der Smart Score bei TipRanks),
  - Einordnung relativ zum Sektor und zum gesamten Universum.

Ergebnis je Aktie (kurze Schlüssel, weil die Website die Datei lädt):
  F  Fundamental-Prüfungen   [Block, Schlüssel, Wert, Schwelle, erfüllt]
  M  Burggraben-Kennzahlen   [Schlüssel, Wert, Punkte 0-5]
  T  Chart-Prüfungen         [Schlüssel, Wert, Punkte, Maximum]
  A  Analystenbild           [Schlüssel, Wert, Punkte, Maximum]
  H  Anlauf-Signale          [Schlüssel, Wert, erfüllt]
  U  Turnaround-Prüfungen    [Schlüssel, Wert, erfüllt]
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

VERSION = 3

# Schwellen je Lynch-Kategorie. Belegt aus öffentlichen Analysen: Average 6/6/6/10 und Verschuldung 3,0,
# Fast Grower EBIT-Marge 20. Die übrigen Werte sind Annahmen.
TH = {
    "Slow Grower": dict(revH=2, revE=2, ebitH=3, epsE=4, niH=3, eq=25, roe=10, roic=8, fcfM=5, debt=3.0, margin=10,
                        pe5=20, divG=5, cap=18),
    "Average Grower": dict(revH=6, revE=6, ebitH=6, epsE=10, niH=7, eq=30, roe=15, roic=10, fcfM=8, debt=3.0,
                           margin=15, pe5=25, cap=22),
    "Fast Grower": dict(revH=15, revE=15, ebitH=15, epsE=20, niH=15, eq=30, roe=15, roic=12, fcfM=10, debt=1.5,
                        margin=20, pe5=30, cap=28),
    "Zykliker": dict(revH=4, revE=4, ebitH=4, epsE=6, niH=4, eq=30, roe=12, roic=8, fcfM=5, debt=2.5, margin=10,
                     pe5=15, cap=15),
}
# Für alle Kategorien gleich
TH_ALL = dict(cov=6.0, liq=1.0, dil=2.0, cash=0.9, fcfY=3.0, rev90=-1.0, mTrend=-1.0, payout=75.0, roeMin=8.0)
# Banken, Versicherer und andere Bilanz-Finanzwerte: branchenübliche Kennzahlen statt Eigenkapitalquote,
# Verschuldung und EBIT-Marge. Schwellen so gewählt, dass sie etwa so oft bestanden werden wie die ersetzten.
FIN_TH = {
    "bank": dict(eqA=6.0, roa=0.7, nm=28.0),
    "insurer": dict(eqA=6.0, roa=0.7, nm=9.0),
    "other": dict(eqA=8.0, roa=1.0, nm=20.0),
}
FIN_PE_FACTOR = 0.5  # Finanzwerte sind dauerhaft niedriger bewertet: halbe KGV-Schwelle, halber KGV-Deckel
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
WEIGHTS = dict(f=30, m=15, t=20, r=20, a=15)
HOT_MIN = 7        # von 8 Anlauf-Signalen
T_MAX = 12         # Rohpunkte der Chart-Prüfungen, auf 10 umgerechnet


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


def sig(v, n=4):
    return None if v is None else float("%.*g" % (n, v))


def de(v, d=1):
    """Zahl in deutscher Schreibweise für Texte."""
    return ("%.*f" % (d, v)).replace(".", ",")


def sde(v, d=0):
    return ("+" if v > 0 else "−" if v < 0 else "±") + de(abs(v), d)


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


def ge(v, t):
    return None if v is None else bool(v >= t)


def le(v, t):
    return None if v is None else bool(v <= t)


def steps(v, cuts, higher=True):
    """Punkte 0-5 nach fünf Stufen."""
    if v is None:
        return None
    return sum(1 for c in cuts if (v >= c if higher else v <= c))


# ----------------------------------------------------------------------------------------------------------------
# Chartanalyse
# ----------------------------------------------------------------------------------------------------------------

def perf(c, days):
    if len(c) <= days or c.iloc[-days - 1] <= 0:
        return None
    return float(c.iloc[-1] / c.iloc[-days - 1] - 1)


def rsi(c, n=14):
    d = c.diff().dropna()
    if len(d) < n * 3:
        return None
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    return 100.0 if dn == 0 else float(100 - 100 / (1 + up / dn))


def volume_profile(c, v, bins=40):
    """Volumen-Cluster: Kursbereiche, in denen überdurchschnittlich viel gehandelt wurde. [von, bis, Anteil]"""
    lo, hi, tot = float(c.min()), float(c.max()), float(v.sum())
    if hi <= lo or tot <= 0:
        return []
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(c.values, edges) - 1, 0, bins - 1)
    hist = np.bincount(idx, weights=v.values, minlength=bins) / tot
    sm = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    nodes = []
    for i in range(bins):
        left = sm[i - 1] if i > 0 else -1
        right = sm[i + 1] if i < bins - 1 else -1
        if sm[i] >= left and sm[i] > right and sm[i] >= 1.3 / bins:
            nodes.append((float(edges[i]), float(edges[i + 1]), float(hist[max(i - 1, 0):i + 2].sum())))
    return nodes


def merge_zones(cands):
    """Überlappende Bereiche zusammenfassen. cands: [von, bis, Bezeichnung]"""
    out = []
    for lo, hi, lab in sorted(cands, key=lambda z: z[0]):
        if out and lo <= out[-1][1] * 1.005:
            out[-1][1] = max(out[-1][1], hi)
            if lab not in out[-1][2]:
                out[-1][2].append(lab)
        else:
            out.append([lo, hi, [lab]])
    return out


def corrections(w):
    """Frühere Korrekturen ab 10 % (Hoch bis Tief) und die laufende. w: Wochenschlusskurse."""
    peak = trough = w[0]
    past = []
    for v in w[1:]:
        if v >= peak:
            dd = 1 - trough / peak
            if dd >= 0.10:
                past.append(dd)
            peak = trough = v
        else:
            trough = min(trough, v)
    return past, 1 - w[-1] / peak, 1 - trough / peak


def price_metrics(df):
    """Kennzahlen und Chartanalyse aus der Kurshistorie. Erwartet Tagesdaten mit Schlusskursen, die um Aktiensplits,
    aber nicht um Dividenden bereinigt sind (wie im Chart). Gibt None zurück, wenn zu wenig Daten da sind."""
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
    c52 = c.iloc[-252:]
    hi52, lo52 = float(c52.max()), float(c52.min())
    mdd5 = float((1 - c5 / c5.cummax()).max())

    vol = None
    if "Volume" in df:
        vol = pd.to_numeric(df["Volume"], errors="coerce").reindex(c.index).fillna(0)
    vol_above, nodes, vol_ratio = None, [], None
    if vol is not None:
        c3 = c[c.index >= end - pd.DateOffset(years=3)]
        v3 = vol.reindex(c3.index)
        tot = float(v3.sum())
        if tot > 0:
            vol_above = float(v3[c3 > last].sum()) / tot
            nodes = volume_profile(c3, v3)
        v100 = float(vol.iloc[-100:].mean())
        if v100 > 0:
            vol_ratio = float(vol.iloc[-20:].mean()) / v100

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

    # Kreuzung von 50- und 200-Tage-Linie in den letzten 60 Handelstagen: [Richtung, vor wie vielen Tagen]
    cross = None
    if len(c) >= 270:
        diff = (c.rolling(50).mean() - c.rolling(200).mean()).iloc[-61:].dropna()
        sign = np.sign(diff.values)
        for i in range(len(sign) - 1, 0, -1):
            if sign[i] != sign[i - 1] and sign[i] != 0:
                cross = [int(sign[i]), len(sign) - 1 - i]
                break

    # Tage seit dem letzten 3-Monats-Hoch
    c63 = c.iloc[-63:]
    since_high = int(len(c63) - 1 - int(np.argmax(c63.values)))

    # Marktsymmetrie: laufende Korrektur gegen frühere Korrekturen der letzten zehn Jahre
    w10 = c[c.index >= end - pd.DateOffset(years=10)].resample("W").last().dropna().values
    past, cur_dd, cur_depth = corrections(list(w10)) if len(w10) > 60 else ([], 0.0, 0.0)
    sym = [pct(cur_dd), pct(cur_depth), pct(float(np.median(past))) if past else None,
           pct(max(past)) if past else None, len(past)]

    # Unterstützungen unter dem Kurs und Widerstände darüber
    low6m = float(c.iloc[-126:].min())
    sup, res = [], []
    for lo, hi, share in nodes:
        if hi <= last:
            sup.append([lo, hi, "Volumen-Cluster"])
        elif lo >= last:
            res.append([lo, hi, "Volumen-Cluster"])
        else:
            sup.append([lo, last, "Volumen-Cluster am aktuellen Kurs"])
    for level, lab in ((sma200, "200-Tage-Linie"), (sma50, "50-Tage-Linie")):
        (sup if level <= last else res).append([level * 0.985, level * 1.015, lab])
    if low6m < last * 0.98:
        sup.append([low6m * 0.985, low6m * 1.015, "Tief der letzten 6 Monate"])
    if lo52 < low6m * 0.97:
        sup.append([lo52 * 0.985, lo52 * 1.015, "52-Wochen-Tief"])
    if hi52 > last * 1.02:
        res.append([hi52 * 0.985, hi52 * 1.015, "52-Wochen-Hoch"])
    if high > hi52 * 1.03:
        res.append([high * 0.985, high * 1.015, "Allzeithoch"])
    zs = [z for z in merge_zones(sup) if z[1] >= last * 0.55]
    zs = sorted(zs, key=lambda z: -z[1])[:3]
    zr = sorted(merge_zones(res), key=lambda z: z[0])[:2]
    zs = [[sig(z[0]), sig(min(z[1], last)), z[2], pct(z[1] / last - 1) if z[1] < last else 0.0] for z in zs]
    zr = [[sig(z[0]), sig(z[1]), z[2], pct(z[0] / last - 1)] for z in zr]

    return dict(last=last, date=str(end.date()), years=years, cagr5=cagr5, y5=y5, sma200=sma200, sma50=sma50,
                sma200_prev=sma200_prev, sma50_prev=sma50_prev, dist=1 - last / high, mdd5=mdd5, vol_above=vol_above,
                r2=r2, slope=slope, vola=vola, div_ttm=div_ttm, div_growth=div_growth, series=c, tr=tr,
                p1m=perf(c, 21), p3m=perf(c, 63), p6m=perf(c, 126), p1y=perf(c, 252),
                hi52=hi52, lo52=lo52, rsi=rsi(c), vol_ratio=vol_ratio, cross=cross, since_high=since_high,
                sym=sym, zs=zs, zr=zr, weekly=[float("%.4g" % v) for v in w.values])


def price_on(series, date):
    try:
        v = series.asof(pd.Timestamp(date))
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------
# Einordnung
# ----------------------------------------------------------------------------------------------------------------

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


def _parts(rec, w, with_chart):
    parts = [(w["f"], rec["f"] * 10), (w["m"], rec["m"] * 4), (w["r"], rec["rs"])]
    if with_chart:
        parts.append((w["t"], rec["t"] * 10))
    if rec.get("a") is not None:
        parts.append((w["a"], rec["a"] * 10))
    ws = sum(p[0] for p in parts) or 1
    return sum(p[0] * p[1] for p in parts) / ws


def total(rec, w=WEIGHTS):
    """Gesamtscore 0-100. Fehlt das Analystenbild, verteilt sich sein Gewicht auf die übrigen Bausteine."""
    return _parts(rec, w, True)


def quality(rec, w=WEIGHTS):
    """Qualitätsscore 0-100: alles außer dem Kursbild."""
    return _parts(rec, w, False)


REC_WORD = [(1.5, "klar Kaufen"), (2.5, "Kaufen"), (3.5, "Halten"), (4.5, "Verkaufen"), (9, "klar Verkaufen")]


def rec_word(mean):
    return next(wd for lim, wd in REC_WORD if mean <= lim)


# ----------------------------------------------------------------------------------------------------------------
# Bewertung einer Aktie
# ----------------------------------------------------------------------------------------------------------------

def score(fund, pm, mcap_usd=None, fx_fin=None, today=None):
    """Bewertet eine Aktie. fund: Fundamentaldaten aus dem Cache, pm: Ergebnis von price_metrics.

    fx_fin: Umrechnung von der Kurswährung (Haupteinheit) in die Bilanzwährung, None wenn unbekannt.
    Yahoo liefert trailingEps/forwardEps in der Kurswährung, die Abschlüsse in der Bilanzwährung.
    Danach muss finalize() über alle Aktien laufen.
    """
    if pm is None:
        return None
    today = today or datetime.utcnow()
    notes = []
    y = fund.get("y") or {}
    rev, ebit, ni = fund.get("revenue") or [], fund.get("ebit") or [], fund.get("netIncome") or []
    equity = fund.get("equity") or []
    ft = fin_type(fund)
    fin = ft is not None
    rev_h = bounded(hist_growth(rev), -0.9, 5)
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
    shares = None
    if fund.get("mcap") and fund.get("priceAtFetch"):
        shares = fund["mcap"] / (fund["priceAtFetch"] * unit)
    mcap_fin = shares * price_m * fx_fin if shares and fx_fin else None  # Börsenwert in Bilanzwährung

    # ---------- Kennzahlen ----------
    ebit_h = bounded(hist_growth(ebit), -1, 5)
    ni_h = bounded(hist_growth(ni), -1, 5)
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
    roic = None
    if first(ebit) is not None and eq is not None and debt is not None and eq + debt > 0:
        roic = bounded(first(ebit) * 0.75 / (eq + debt), -2, 3)
    # Margenentwicklung: jüngste Marge gegen die älteste vorliegende (meist vor drei Jahren), in Prozentpunkten
    base_series = ni if fin else ebit
    ms = [e / r for e, r in zip(base_series, rev) if e is not None and r and r > 0]
    m_trend = (ms[0] - ms[-1]) * 100 if len(ms) >= 3 else None
    # Cashflow
    ocf, fcf = fund.get("ocf") or [], fund.get("fcf") or []
    fcf0, rev_f = pair(fcf, rev)
    fcf_m = bounded(fcf0 / rev_f if fcf0 is not None and rev_f and rev_f > 0 else None, -2, 1)
    ocf0, ni_c = pair(ocf, ni)
    cash_conv = bounded(ocf0 / ni_c if ocf0 is not None and ni_c and ni_c > 0 else None, -20, 50)
    fcf_vals = [v for v in fcf if v is not None]
    fcf_pos = (sum(1 for v in fcf_vals if v > 0), len(fcf_vals)) if len(fcf_vals) >= 3 else None
    fcf_y = bounded(first(fcf) / mcap_fin if first(fcf) is not None and mcap_fin else None, -1, 1)
    # Zinsdeckung, Liquidität, Verwässerung
    interest = first(fund.get("interest"))
    cov = None
    if first(ebit) is not None and interest is not None:
        cov = clamp(first(ebit) / abs(interest) if abs(interest) > 0 else 99.0, -99.0, 99.0)
    elif first(ebit) is not None and debt is not None and debt == 0:
        cov = 99.0
    ca, cl = pair(fund.get("curAssets"), fund.get("curLiab"))
    liq = bounded(ca / cl if ca is not None and cl else None, 0, 50)
    dil = bounded(hist_growth(fund.get("sharesDil")), -0.5, 1)
    payout = bounded(y.get("payoutRatio"), 0, 5)
    roes = [n / e for n, e in zip(ni, equity) if n is not None and e and e > 0]
    roe_min = min(roes) if len(roes) >= 3 else None
    # Analystenschätzungen: Gewinn je Aktie fürs nächste Jahr heute gegen vor 90 Tagen
    trend_all = fund.get("epsTrend") or {}
    et = trend_all.get("+1y") or trend_all.get("0y") or {}
    rev90 = None
    if et.get("cur") and et.get("d90") and et["d90"] > 0 and et["cur"] > 0:
        rev90 = bounded(et["cur"] / et["d90"] - 1, -0.9, 3)

    # ---------- Gewinn je Aktie und KGV ----------
    # Yahoos Wert der letzten zwölf Monate gilt, wenn er zu mindestens einer anderen Angabe passt (letzter
    # Jahresabschluss, Jahresgewinn durch Aktienzahl, Schätzung fürs nächste Jahr). Das fängt Aktiensplits und
    # Einheitenfehler ab, ohne echte Gewinnsprünge zu verwerfen.
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
    # und nächste Jahr. Höchstens Umsatzwachstum + 5 Punkte und gedeckelt je Kategorie.
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
    pe5_th = round(T["pe5"] * pe_factor, 1)

    # ---------- Fundamental: Prüfungen in vier Blöcken ----------
    A = TH_ALL
    W = [
        ["revH", pct(rev_h), T["revH"], ge(rev_h, T["revH"] / 100)],
        ["revE", pct(rev_e), T["revE"], ge(rev_e, T["revE"] / 100)],
        (["divG", pct(pm["div_growth"]), T["divG"], ge(pm["div_growth"], T["divG"] / 100)] if cat == "Slow Grower"
         else ["ebitH", pct(ebit_h), T["ebitH"], ge(ebit_h, T["ebitH"] / 100)]),
        ["niH", pct(ni_h), T["niH"], ge(ni_h, T["niH"] / 100)],
        ["epsE", pct(eps_e), T["epsE"], ge(eps_e, T["epsE"] / 100)],
    ]
    roe_ok = False if eq is not None and eq <= 0 else ge(roe, T["roe"] / 100)
    pe_row = ["pe5", r1(pe5), pe5_th, (False if eps_src == "loss" else le(pe5, pe5_th))]
    rev_row = ["rev90", pct(rev90), A["rev90"], ge(rev90, A["rev90"] / 100)]
    dil_row = ["dil", pct(dil), A["dil"], le(dil, A["dil"] / 100)]
    if fin:
        FT = FIN_TH[ft]
        P = [
            ["nm", pct(net_margin), FT["nm"], ge(net_margin, FT["nm"] / 100)],
            ["roe", pct(roe), T["roe"], roe_ok],
            ["roa", pct(roa, 2), FT["roa"], ge(roa, FT["roa"] / 100)],
            ["roeMin", pct(roe_min), A["roeMin"], ge(roe_min, A["roeMin"] / 100)],
            ["nmTrend", r1(m_trend), A["mTrend"], ge(m_trend, A["mTrend"])],
        ]
        B = [
            ["eqA", pct(eq_ratio), FT["eqA"], ge(eq_ratio, FT["eqA"] / 100)],
            ["payout", pct(payout, 0), A["payout"], le(payout, A["payout"] / 100)],
            dil_row,
        ]
        Q = [["dec", dec, 1, le(dec, 1)], pe_row, rev_row]
    else:
        P = [
            ["margin", pct(margin), T["margin"], ge(margin, T["margin"] / 100)],
            ["roe", pct(roe), T["roe"], roe_ok],
            ["roic", pct(roic), T["roic"], ge(roic, T["roic"] / 100)],
            ["fcfM", pct(fcf_m), T["fcfM"], ge(fcf_m, T["fcfM"] / 100)],
            ["mTrend", r1(m_trend), A["mTrend"], ge(m_trend, A["mTrend"])],
        ]
        B = [
            ["eq", pct(eq_ratio), T["eq"], ge(eq_ratio, T["eq"] / 100)],
            ["debt", r1(nd), T["debt"], le(nd, T["debt"])],
            ["cov", r1(cov), A["cov"], ge(cov, A["cov"])],
            ["liq", r1(liq, 2), A["liq"], ge(liq, A["liq"])],
            dil_row,
        ]
        Q = [
            ["dec", dec, 1, le(dec, 1)],
            ["cash", pct(cash_conv, 0), A["cash"] * 100, ge(cash_conv, A["cash"])],
            ["fcfPos", None if fcf_pos is None else fcf_pos[0], None if fcf_pos is None else fcf_pos[1],
             None if fcf_pos is None else fcf_pos[0] == fcf_pos[1]],
            pe_row,
            ["fcfY", pct(fcf_y), A["fcfY"], ge(fcf_y, A["fcfY"] / 100)],
            rev_row,
        ]
    F = [["W"] + c for c in W] + [["P"] + c for c in P] + [["B"] + c for c in B] + [["Q"] + c for c in Q]
    avail = [x for x in F if x[4] is not None]
    if len(avail) < (8 if fin else 10):
        return None
    f_ok = sum(1 for x in avail if x[4])
    fund10 = round(f_ok / len(avail) * 10, 1)
    fb = {}
    for blk in "WPBQ":
        rows = [x for x in avail if x[0] == blk]
        fb[blk] = [sum(1 for x in rows if x[4]), len(rows)]

    # ---------- Burggraben: fünf Kennzahlen, je 0-5 Punkte ----------
    gross = bounded(fund.get("grossMargin"), -1, 1)
    margins = [e / r for e, r in zip(ebit, rev) if e is not None and r and r > 0]
    m_std = float(np.std(margins)) * 100 if len(margins) >= 3 else None
    rev_dec = declines(rev)
    dec_pts = {0: 5, 1: 3, 2: 1}
    if fin:
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

    # ---------- Chart-Prüfungen (zwölf Rohpunkte, auf zehn umgerechnet; relative Stärke folgt in finalize) ----------
    above = price > pm["sma200"]
    c5g = pm["cagr5"]
    va = pm["vol_above"]
    dist52 = 1 - price / pm["hi52"]
    if va is not None:
        p_vol = 2 if va <= 0.15 else 1 if va <= 0.35 else 0
    else:
        p_vol = 2 if pm["dist"] <= 0.10 else 1 if pm["dist"] <= 0.25 else 0
    up200 = bool(pm["sma200_prev"] and pm["sma200"] > pm["sma200_prev"])
    Tk = [
        ["long", pct(c5g), 0 if c5g is None else 2 if c5g >= 0.05 else 1 if c5g > 0 else 0, 2],
        ["sma200", pct(price / pm["sma200"] - 1), 1 if above else 0, 1],
        ["up200", 1 if up200 else 0, 1 if up200 else 0, 1],
        ["cross", pct(pm["sma50"] / pm["sma200"] - 1), 1 if pm["sma50"] > pm["sma200"] else 0, 1],
        ["stab", pct(pm["mdd5"]), 1 if pm["y5"] >= 4.9 and pm["mdd5"] < 0.30 else 0, 1],
        ["hist", r1(pm["years"], 0), 1 if pm["years"] >= 10 else 0, 1],
        ["vol", pct(va), p_vol, 2],
        ["qual", r1(pm["r2"], 2), 1 if pm["r2"] is not None and pm["r2"] >= 0.7 and pm["slope"] > 0 else 0, 1],
        ["hi52", pct(dist52), 1 if dist52 <= 0.15 else 0, 1],
        ["rs", None, 0, 1],
    ]

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
    kgve = min(x for x in (hist_pe, cap, pe) if x is not None)
    rend = lo = hi = None
    if pe is not None and g is not None:
        rend = ((1 + g) ** 5 * kgve / pe) ** 0.2 - 1 + dy
        rend = min(rend, 0.25)
        lo = (1 + 0.6 * g) ** 5 * 0.85 * kgve / pe - 1
        hi = (1 + g) ** 5 * min(hist_pe or pe, cap * 1.2) / pe - 1
    rs = 0.0 if rend is None else clamp(rend / 0.12, 0, 1) * 100

    # ---------- Analystenbild (0-10) ----------
    n_an = y.get("numberOfAnalystOpinions")
    rmean = bounded(y.get("recommendationMean"), 1, 5)
    tgt = y.get("targetMeanPrice")
    upside = bounded(tgt / price - 1 if tgt and tgt > 0 else None, -0.6, 1.5)
    if tgt and upside is None:
        notes.append("Kursziel der Analysten passt nicht zum Kurs und wurde verworfen")
    cutoff = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    ud = [u for u in (fund.get("updown") or []) if u and u[0] >= cutoff]
    ups = sum(1 for u in ud if u[4] == "up")
    downs = sum(1 for u in ud if u[4] == "down")
    recs = fund.get("rec") or {}
    An, a10 = None, None
    if n_an and n_an >= 3 and rmean is not None:
        An = [
            ["cons", r1(rmean, 2), 3 if rmean <= 1.8 else 2 if rmean <= 2.2 else 1 if rmean <= 2.6 else 0, 3],
            ["tgt", pct(upside), 0 if upside is None else 3 if upside >= 0.20 else 2 if upside >= 0.10
             else 1 if upside >= 0 else 0, 3],
            ["rev", pct(rev90), 0 if rev90 is None else 2 if rev90 >= 0.02 else 1 if rev90 >= -0.01 else 0, 2],
            ["ud", [ups, downs], 1 if ups > downs else 0, 1],
            ["cov", int(n_an), 1 if n_an >= 10 else 0, 1],
        ]
        a10 = float(sum(x[2] for x in An))

    # ---------- Risikoklasse 1 (defensiv) bis 5 (offensiv) ----------
    risk = 1
    v = pm["vola"]
    if v is not None and v >= 0.25:
        risk += 1
    if v is not None and v >= 0.40:
        risk += 1
    if pm["mdd5"] >= 0.40:
        risk += 1
    if (not fin and nd is not None and nd > 3) or (fin and B[0][3] is False):
        risk += 1
    if dec is not None and dec >= 2:
        risk += 1
    risk = clamp(risk, 1, 5)

    # ---------- Trendstatus ----------
    dist = pm["dist"]
    if above and dist52 <= 0.10:
        status = "Aufwärtstrend"
    elif above:
        status = "Erholung"
    elif dist52 <= 0.20:
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
    inc_dates = fund.get("incDates") or []
    if inc_dates and (today - datetime.strptime(inc_dates[0], "%Y-%m-%d")).days > 550:
        notes.append("Letzter Jahresabschluss vom %s, also über 18 Monate alt" % inc_dates[0])
    if fx_fin is None:
        notes.append("Wechselkurs zwischen Kurs- und Bilanzwährung fehlt, historisches KGV nicht berechnet")
    if fund.get("sector") == "Real Estate":
        notes.append("Immobilienwert: Das KGV ist dort wenig aussagekräftig, weil Abschreibungen den Gewinn drücken")
    hard = [n for n in notes if not n.startswith(("Historisches KGV", "Immobilienwert"))]
    share = len(avail) / len(F)
    quality_grade = "A" if share >= 0.9 and not hard else "B" if share >= 0.75 and len(hard) <= 2 else "C"

    # ---------- Termine ----------
    earn, earn_days = fund.get("earn"), None
    if earn:
        try:
            earn_days = (datetime.strptime(earn, "%Y-%m-%d") - today).days
        except ValueError:
            earn = None
    if earn_days is not None and not -1 <= earn_days <= 120:
        earn, earn_days = None, None

    growth = g_raw or 0
    strat = "Dividende" if dy >= 0.025 else "Wachstum" if cat == "Fast Grower" or growth >= 0.10 else "Cashflow"
    er = fund.get("epsRev") or {}
    rec = dict(
        cat=cat, strat=strat, fin=ft or "", p=float("%.5g" % price), d=pm["date"],
        mc=r1(mcap_usd / 1e9) if mcap_usd else None,
        F=F, f=fund10, fa=len(avail), fo=f_ok, fn=len(F), fb=fb, M=M, m=moat25, T=Tk, t=0.0, A=An, a=a10,
        pe=r1(pe), es=eps_src, hpe=r1(hist_pe), kgve=r1(kgve), g=pct(g), gs=g_src, dy=r1(dy * 100, 2),
        r=pct(rend), rs=round(rs), lo=r1(lo * 100, 0) if lo is not None else None,
        hi=r1(hi * 100, 0) if hi is not None else None, dist=pct(dist), d52=pct(dist52), risk=risk, st=status,
        tr=pm["tr"], pf=[pct(pm["p1m"]), pct(pm["p3m"]), pct(pm["p6m"]), pct(pm["p1y"])],
        vo=pct(pm["vola"], 0), rsi=r1(pm["rsi"], 0), s50=sig(pm["sma50"], 5), s200=sig(pm["sma200"], 5),
        h52=sig(pm["hi52"], 5), l52=sig(pm["lo52"], 5), vr=r1(pm["vol_ratio"], 2), cx=pm["cross"],
        sym=pm["sym"], zs=pm["zs"], zr=pm["zr"],
        an=dict(n=int(n_an) if n_an else None, mean=r1(rmean, 2), tgt=sig(tgt, 5), up=pct(upside),
                hi=sig(y.get("targetHighPrice"), 5), lo=sig(y.get("targetLowPrice"), 5),
                rec=recs.get("0m"), rec3=recs.get("-3m"), rev90=pct(rev90),
                rev30=[er.get("up30"), er.get("down30")], ud=ud[:6]),
        ev=dict(earn=earn, ed=earn_days, exdiv=fund.get("exDiv")),
        news=(fund.get("news") or [])[:6], q=quality_grade, qn=notes, po=pct(payout, 0), nd=r1(nd),
    )
    rec["_x"] = dict(above=above, up200=up200, up50=bool(pm["sma50_prev"] and pm["sma50"] > pm["sma50_prev"]),
                     over50=price > pm["sma50"], since_high=pm["since_high"], ext=price / pm["sma200"] - 1,
                     upside=upside, rmean=rmean, rev90=rev90, ups=ups, downs=downs, margin=margin, roic=roic,
                     roe=roe, fcf_y=fcf_y, payout=payout, cov=cov, nd=nd, rev_e=rev_e,
                     net_cash=(debt is not None and cash is not None and cash > debt), fin=fin, dec=dec)
    return rec


# ----------------------------------------------------------------------------------------------------------------
# Zweiter Durchgang: Vergleich im Universum, Signale, Einordnung, Texte
# ----------------------------------------------------------------------------------------------------------------

def pct_rank(values, v):
    if v is None or not values:
        return None
    return round(100.0 * sum(1 for x in values if x <= v) / len(values))


def finalize(recs):
    """Ergänzt alles, was den Vergleich mit den anderen Aktien braucht, und schreibt die Texte."""
    p6 = [r["pf"][2] for r in recs if r["pf"][2] is not None]
    p12 = [r["pf"][3] for r in recs if r["pf"][3] is not None]
    med12 = float(np.median(p12)) if p12 else None
    for r in recs:
        x = r.pop("_x")
        # Relative Stärke
        r["rs6"], r["rs12"] = pct_rank(p6, r["pf"][2]), pct_rank(p12, r["pf"][3])
        rs_row = next(t for t in r["T"] if t[0] == "rs")
        rs_row[1] = r["rs12"]
        rs_row[2] = 1 if r["pf"][3] is not None and med12 is not None and r["pf"][3] > med12 else 0
        r["t"] = round(sum(t[2] for t in r["T"]) / T_MAX * 10, 1)
        r["tot"] = round(total(r), 1)
        r["ql"] = round(quality(r), 1)

        # Anlauf-Signale: läuft die Aktie gerade an?
        rsi_v, vr = r["rsi"], r["vr"]
        H = [
            ["over50", None, bool(x["over50"] and x["up50"])],
            ["stack", None, bool(r["T"][3][2] == 1 and x["up200"])],
            ["newHigh", x["since_high"], x["since_high"] <= 10],
            ["near52", r["d52"], r["d52"] is not None and r["d52"] <= 8],
            ["rs", r["rs6"], r["rs6"] is not None and r["rs6"] >= 75],
            ["volume", vr, vr is not None and vr >= 1.15],
            ["rsi", rsi_v, rsi_v is not None and 55 <= rsi_v <= 75],
            ["notExt", pct(x["ext"]), x["ext"] <= 0.25],
        ]
        H = [[h[0], h[1], bool(h[2])] for h in H]
        r["H"], r["hot"] = H, sum(1 for h in H if h[2])

        # Turnaround-Prüfungen: nur sinnvoll, wenn die Aktie deutlich unter ihrem Hoch steht
        sym = r["sym"]
        U = None
        if r["d52"] is not None and r["d52"] >= 15:
            U = [
                ["fund", r["f"], r["f"] >= 7],
                ["moat", r["m"], r["m"] >= 14],
                ["stab", None, bool(r["st"] == "Bodenbildung" or (x["over50"] and (r["pf"][0] or 0) > 0))],
                ["cons", r1(x["rmean"], 2), x["rmean"] is not None and x["rmean"] <= 2.3],
                ["tgt", pct(x["upside"]), x["upside"] is not None and x["upside"] >= 0.15],
                ["rev", pct(x["rev90"]), x["rev90"] is not None and x["rev90"] >= -0.02],
                ["sym", sym[1], sym[3] is not None and sym[1] is not None and sym[1] <= sym[3]],
            ]
        if U:
            U = [[u[0], u[1], bool(u[2])] for u in U]
        r["U"], r["tu"] = U, (sum(1 for u in U if u[2]) if U else None)

        # Einordnung
        ql, st = r["ql"], r["st"]
        weak_chart = st in ("Korrektur", "Trendbruch", "Abwärtstrend", "Bodenbildung")
        if (rsi_v is not None and rsi_v >= 78) or x["ext"] >= 0.40:
            setup = "Überhitzt"
        elif r["hot"] >= HOT_MIN:
            setup = "Anlauf"
        elif U and r["tu"] >= 5 and r["d52"] >= 25:
            setup = "Turnaround-Kandidat"
        elif ql >= 70 and weak_chart and r["d52"] >= 15:
            setup = "Qualität im Rücksetzer"
        elif ql >= 65 and st == "Aufwärtstrend":
            setup = "Trendstark"
        elif ql < 45 and st in ("Trendbruch", "Abwärtstrend"):
            setup = "Schwach"
        else:
            setup = "Neutral"
        r["su"] = setup
        r["pro"], r["con"], r["txt"] = texts(r, x)

    # Rang im Sektor nach Qualität
    by_sec = {}
    for r in recs:
        by_sec.setdefault(r.get("sec") or "", []).append(r)
    for rows in by_sec.values():
        rows.sort(key=lambda r: -r["ql"])
        for i, r in enumerate(rows):
            r["sq"] = [i + 1, len(rows)]
    return recs


def texts(r, x):
    """Stärken, Punkte zum Aufpassen und eine Einordnung in Sätzen. Jede Aussage nennt die Zahl, auf der sie beruht."""
    pro, con = [], []
    f, m = r["f"], r["m"]
    if x["rev_e"] is not None and x["rev_e"] >= 0.10:
        pro.append("Analysten erwarten %s %% Umsatzwachstum" % de(x["rev_e"] * 100, 0))
    if not x["fin"] and x["roic"] is not None and x["roic"] >= 0.15:
        pro.append("Hohe Kapitalrendite von %s %%" % de(x["roic"] * 100, 0))
    if x["fin"] and x["roe"] is not None and x["roe"] >= 0.14:
        pro.append("Eigenkapitalrendite von %s %%" % de(x["roe"] * 100, 0))
    if not x["fin"] and x["margin"] is not None and x["margin"] >= 0.25:
        pro.append("EBIT-Marge von %s %%" % de(x["margin"] * 100, 0))
    if not x["fin"] and x["net_cash"]:
        pro.append("Mehr Barmittel als Schulden")
    if x["fcf_y"] is not None and x["fcf_y"] >= 0.05:
        pro.append("Free-Cashflow-Rendite von %s %%" % de(x["fcf_y"] * 100))
    if x["dec"] == 0:
        pro.append("Kein Gewinnrückgang in den vorliegenden Jahresabschlüssen")
    if x["rev90"] is not None and x["rev90"] >= 0.03:
        pro.append("Gewinnschätzung in 90 Tagen um %s %% angehoben" % de(x["rev90"] * 100, 0))
    if x["upside"] is not None and x["upside"] >= 0.20 and r["an"]["n"]:
        pro.append("Mittleres Kursziel von %d Analysten liegt %s %% über dem Kurs"
                   % (r["an"]["n"], de(x["upside"] * 100, 0)))
    if r["st"] == "Aufwärtstrend":
        pro.append("Intakter Aufwärtstrend, %s %% unter dem 52-Wochen-Hoch" % de(r["d52"], 0))
    if r["rs12"] is not None and r["rs12"] >= 80:
        pro.append("Kurs in zwölf Monaten stärker als %d %% der Aktien im Universum" % r["rs12"])
    if r["r"] is not None and r["r"] >= 12:
        pro.append("Renditeerwartung von %s %% p. a. nach dem Rechenmodell" % de(r["r"]))

    ev = r["ev"]
    if ev["ed"] is not None and 0 <= ev["ed"] <= 14:
        con.append("Quartalszahlen in %d Tagen (%s): rund um den Termin schwankt der Kurs oft stark"
                   % (ev["ed"], ev["earn"]))
    if x["rev90"] is not None and x["rev90"] <= -0.05:
        con.append("Gewinnschätzung in 90 Tagen um %s %% gesenkt" % de(abs(x["rev90"]) * 100, 0))
    if x["downs"] > x["ups"] and x["downs"] >= 2:
        con.append("%d Abstufungen durch Analysten in 90 Tagen, %d Hochstufungen" % (x["downs"], x["ups"]))
    if r["st"] in ("Trendbruch", "Abwärtstrend"):
        con.append("%s: Kurs %s %% unter dem 52-Wochen-Hoch und unter der 200-Tage-Linie"
                   % (r["st"], de(r["d52"], 0)))
    if r["pf"][0] is not None and r["pf"][0] <= -15:
        con.append("Kurs in einem Monat um %s %% gefallen" % de(abs(r["pf"][0]), 0))
    if r["rsi"] is not None and r["rsi"] >= 75:
        con.append("RSI bei %d: kurzfristig überkauft" % r["rsi"])
    if x["ext"] >= 0.35:
        con.append("Kurs %s %% über der 200-Tage-Linie, also weit vorausgelaufen" % de(x["ext"] * 100, 0))
    if r["es"] == "loss":
        con.append("Verlust in den letzten zwölf Monaten")
    if r["pe"] is not None and r["hpe"] is not None and r["pe"] > r["hpe"] * 1.3:
        con.append("KGV %s liegt deutlich über dem eigenen Schnitt von %s" % (de(r["pe"]), de(r["hpe"])))
    if not x["fin"] and x["nd"] is not None and x["nd"] > 3:
        con.append("Nettoverschuldung beim %s-Fachen des EBITDA" % de(min(x["nd"], 99)))
    if not x["fin"] and x["cov"] is not None and x["cov"] < 3:
        con.append("Zinsdeckung nur %s (EBIT durch Zinsaufwand)" % de(x["cov"]))
    if x["payout"] is not None and x["payout"] > 0.9 and r["dy"] and r["dy"] > 1:
        con.append("Ausschüttungsquote %s %%: Dividende kaum durch den Gewinn gedeckt" % de(x["payout"] * 100, 0))
    if x["dec"] is not None and x["dec"] >= 2:
        con.append("%d Gewinnrückgänge in den vorliegenden Jahresabschlüssen" % x["dec"])
    if x["upside"] is not None and x["upside"] < 0 and r["an"]["n"]:
        con.append("Mittleres Kursziel der Analysten liegt %s %% unter dem Kurs" % de(abs(x["upside"]) * 100, 0))
    if r["q"] == "C":
        con.append("Datenqualität C: Kennzahlen lückenhaft oder auffällig")

    fw = "sehr stark" if f >= 8.5 else "stark" if f >= 7 else "solide" if f >= 5.5 else "schwach"
    mw = "stark" if m >= 18 else "solide" if m >= 14 else "schwach"
    s = ["Fundamental %s: %d von %d Prüfungen bestanden (%s/10). Burggraben %s (%s/25)."
         % (fw, r["fo"], r["fa"], de(f), mw, de(m))]
    rel = "über" if r["T"][1][2] else "unter"
    s.append("Der Kurs steht %s %% unter dem 52-Wochen-Hoch und %s der 200-Tage-Linie (%s)."
             % (de(r["d52"], 0), rel, r["st"]))
    an = r["an"]
    if an["mean"] is not None and an["n"]:
        t = "Analysten: Konsens „%s“ (Note %s von 5, %d Analysten)" % (rec_word(an["mean"]), de(an["mean"]), an["n"])
        if an["up"] is not None:
            t += ", mittleres Kursziel %s %% zum Kurs" % sde(an["up"])
        s.append(t + ".")
    su = r["su"]
    tail = {
        "Anlauf": "Einordnung: Anlauf. %d von 8 Anlauf-Signalen sind erfüllt." % r["hot"],
        "Trendstark": "Einordnung: trendstark. Kennzahlen und Kursbild zeigen in dieselbe Richtung.",
        "Qualität im Rücksetzer": "Einordnung: Qualität im Rücksetzer. Die Kennzahlen sind stark, das Kursbild "
                                  "ist es derzeit nicht.",
        "Turnaround-Kandidat": "Einordnung: Turnaround-Kandidat. %d von 7 Turnaround-Prüfungen sind erfüllt."
                               % (r["tu"] or 0),
        "Überhitzt": "Einordnung: überhitzt. Der Kurs ist kurzfristig weit vorausgelaufen.",
        "Schwach": "Einordnung: schwach. Weder Kennzahlen noch Kursbild überzeugen.",
        "Neutral": "Einordnung: neutral. Kein klares Bild aus Kennzahlen und Kurs.",
    }[su]
    sym = r["sym"]
    if su in ("Qualität im Rücksetzer", "Turnaround-Kandidat") and sym[2] is not None:
        tail += (" Der laufende Rückgang beträgt bis zu %s %%, frühere Korrekturen lagen im Mittel bei %s %% "
                 "und höchstens bei %s %%." % (de(sym[1], 0), de(sym[2], 0), de(sym[3], 0)))
    s.append(tail)
    return pro[:6], con[:7], " ".join(s)
