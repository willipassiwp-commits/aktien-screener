"""Regelbasierte Bewertung. Jede Aktie durchläuft dieselbe Rechnung, es gibt keine Ermessenspunkte."""
import math

import numpy as np
import pandas as pd

# Schwellen je Lynch-Kategorie. Belegt aus öffentlichen Analysen: Average 6/6/6/10 und Verschuldung 3,0, Fast EBIT-Marge 20.
TH = {
    "Slow Grower": dict(revH=2, revE=2, ebitH=3, epsE=4, eq=25, roe=10, debt=3.0, margin=10, pe5=20, divG=5, cap=18),
    "Average Grower": dict(revH=6, revE=6, ebitH=6, epsE=10, eq=30, roe=15, debt=3.0, margin=15, pe5=25, cap=22),
    "Fast Grower": dict(revH=15, revE=15, ebitH=15, epsE=20, eq=30, roe=15, debt=1.5, margin=20, pe5=30, cap=28),
    "Zykliker": dict(revH=4, revE=4, ebitH=4, epsE=6, eq=30, roe=12, debt=2.5, margin=10, pe5=15, cap=15),
}
CYC_SECTORS = {"Energy", "Basic Materials"}
CYC_INDUSTRIES = {"Auto Manufacturers", "Auto Parts", "Airlines", "Residential Construction", "Marine Shipping",
                  "Farm & Heavy Construction Machinery", "Building Materials", "Steel"}


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


def first(lst):
    return lst[0] if lst else None


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


def price_metrics(df):
    """Kennzahlen aus der Kurshistorie (Tagesdaten, bereinigt). Gibt None zurück, wenn zu wenig Daten da sind."""
    if df is None or "Close" not in df:
        return None
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(c) < 250:
        return None
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)
        df = df.copy()
        df.index = df.index.tz_localize(None)
    last, end = float(c.iloc[-1]), c.index[-1]
    if last <= 0:
        return None
    years = (end - c.index[0]).days / 365.25
    c5 = c[c.index >= end - pd.DateOffset(years=5)]
    y5 = (end - c5.index[0]).days / 365.25
    cagr5 = (last / float(c5.iloc[0])) ** (1 / y5) - 1 if y5 >= 1 and c5.iloc[0] > 0 else None
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
    if len(w) >= 100 and (w > 0).all():
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
    return dict(last=last, date=str(end.date()), years=years, cagr5=cagr5, y5=y5, sma200=sma200, sma50=sma50,
                sma200_prev=sma200_prev, sma50_prev=sma50_prev, dist=1 - last / high, mdd5=mdd5, vol_above=vol_above,
                r2=r2, slope=slope, vola=vola, div_ttm=div_ttm, div_growth=div_growth, series=c)


def price_on(series, date):
    try:
        v = series.asof(pd.Timestamp(date))
        return None if pd.isna(v) else float(v)
    except Exception:
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


def score(fund, pm, mcap_usd=None):
    """Bewertet eine Aktie. fund: Fundamentaldaten aus dem Cache, pm: Ergebnis von price_metrics."""
    rev, ebit, ni = fund.get("revenue") or [], fund.get("ebit") or [], fund.get("netIncome") or []
    fin = fund.get("sector") == "Financial Services"
    rev_h = hist_growth(rev)
    cat = lynch(fund, rev_h)
    if cat is None or pm is None:
        return None
    T = TH[cat]
    price = pm["last"]

    # --- Fundamental: zehn Kriterien, je ein Punkt ---
    ebit_h = hist_growth(ebit)
    rev_e, eps_e = fund.get("revGrowthEst"), fund.get("epsGrowthEst")
    dec = declines(ni)
    eq, assets, ni0 = first(fund.get("equity")), first(fund.get("assets")), first(ni)
    eq_ratio = eq / assets if eq is not None and assets else None
    roe = ni0 / eq if eq and eq > 0 and ni0 is not None else None
    debt, cash, ebitda = first(fund.get("totalDebt")), first(fund.get("cash")), first(fund.get("ebitda"))
    nd = None
    if debt is not None and ebitda is not None:
        net = debt - (cash or 0)
        nd = net / ebitda if ebitda > 0 else (0.0 if net <= 0 else 99.0)
    rev0, ebit0 = first(rev), first(ebit)
    margin = ebit0 / rev0 if ebit0 is not None and rev0 and rev0 > 0 else None
    eps_t, eps_f = fund.get("trailingEps"), fund.get("forwardEps")
    eps = eps_t if eps_t and eps_t > 0 else (eps_f if eps_f and eps_f > 0 else None)
    pe = price / eps if eps else None
    g_raw = eps_e if eps_e is not None else (rev_e if rev_e is not None else rev_h)
    g = clamp(g_raw, 0.0, 0.20) if g_raw is not None else None
    pe5 = pe / (1 + g) ** 5 if pe is not None and g is not None else None

    def pct(v):
        return r1(v * 100) if v is not None else None

    if cat == "Slow Grower":
        c3 = [pct(pm["div_growth"]), T["divG"], ge(pm["div_growth"], T["divG"] / 100)]
    else:
        c3 = [pct(ebit_h), T["ebitH"], ge(ebit_h, T["ebitH"] / 100)]
    F = [
        [pct(rev_h), T["revH"], ge(rev_h, T["revH"] / 100)],
        [pct(rev_e), T["revE"], ge(rev_e, T["revE"] / 100)],
        c3,
        [pct(eps_e), T["epsE"], ge(eps_e, T["epsE"] / 100)],
        [dec, 1, le(dec, 1)],
        [pct(eq_ratio), T["eq"], None if fin else ge(eq_ratio, T["eq"] / 100)],
        [pct(roe), T["roe"], (False if eq is not None and eq <= 0 else ge(roe, T["roe"] / 100))],
        [r1(nd), T["debt"], None if fin else le(nd, T["debt"])],
        [pct(margin), T["margin"], None if fin else ge(margin, T["margin"] / 100)],
        [r1(pe5), T["pe5"], (False if pe is None and (eps_t is not None or eps_f is not None) else le(pe5, T["pe5"]))],
    ]
    avail = [x for x in F if x[2] is not None]
    if len(avail) < 6:
        return None
    fund10 = round(sum(1 for x in avail if x[2]) / len(avail) * 10, 1)

    # --- Burggraben als Kennzahlen-Ersatz für Porter: fünf Größen, je 0-5 Punkte ---
    gross = fund.get("grossMargin")
    margins = [e / r for e, r in zip(ebit, rev) if e is not None and r and r > 0]
    m_std = float(np.std(margins)) * 100 if len(margins) >= 3 else None
    roic = None
    if ebit0 is not None and eq is not None and debt is not None and eq + debt > 0:
        roic = ebit0 * 0.75 / (eq + debt)
    rev_dec = declines(rev)
    M = [
        [pct(gross), None if fin else steps(gross, [0.2, 0.3, 0.4, 0.5, 0.6])],
        [pct(margin), None if fin else steps(margin, [0.10, 0.15, 0.20, 0.25, 0.30])],
        [r1(m_std), None if fin else steps(m_std, [12, 8, 5, 3, 1.5], higher=False)],
        [pct(roe if fin else roic), steps(roe, [0.06, 0.09, 0.12, 0.15, 0.18]) if fin
         else steps(roic, [0.06, 0.10, 0.15, 0.20, 0.25])],
        [rev_dec, None if rev_dec is None else {0: 5, 1: 3, 2: 1}.get(rev_dec, 0)],
    ]
    mv = [x[1] for x in M if x[1] is not None]
    if len(mv) < 2:
        return None
    moat25 = round(sum(mv) / (len(mv) * 5) * 25, 1)

    # --- Technik: zehn Punkte aus der Kurshistorie ---
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
        [pct(c5g), p_long], [1 if above else 0, 1 if above else 0], [pct(pm["mdd5"]), p_stab],
        [r1(pm["years"], 0), p_hist], [pct(va) if va is not None else None, p_vol], [r1(pm["r2"], 2), q1 + q2],
    ]
    tech = sum(x[1] for x in Tk)

    # --- Renditeerwartung: Exit-KGV über fünf Jahre ---
    hist_pe = None
    if fund.get("finCurrency") and fund.get("finCurrency") == fund.get("currency"):
        pes = []
        for d, e in zip(fund.get("incDates") or [], fund.get("dilutedEps") or []):
            p = price_on(pm["series"], d)
            if e and e > 0 and p:
                pes.append(p / e)
        if len(pes) >= 2:
            hist_pe = float(np.median(pes))
    dy = pm["div_ttm"] / price
    kgve = min(hist_pe or T["cap"], T["cap"])
    rend = lo = hi = None
    if pe is not None and g is not None:
        rend = ((1 + g) ** 5 * kgve / pe) ** 0.2 - 1 + dy
        rend = min(rend, 0.25)  # Deckel gegen Ausreißer durch fehlerhafte Gewinnangaben
        lo = (1 + 0.6 * g) ** 5 * 0.85 * kgve / pe - 1
        hi = (1 + g) ** 5 * min(hist_pe or T["cap"], T["cap"] * 1.2) / pe - 1
    rs = 0.0 if rend is None else clamp(rend / 0.12, 0, 1) * 100

    # --- Risikoklasse 1 (defensiv) bis 5 (offensiv) ---
    risk = 1
    v = pm["vola"]
    if v is not None and v >= 0.25:
        risk += 1
    if v is not None and v >= 0.40:
        risk += 1
    if pm["mdd5"] >= 0.40:
        risk += 1
    if nd is not None and nd > 3 and not fin:
        risk += 1
    if dec is not None and dec >= 2:
        risk += 1
    risk = clamp(risk, 1, 5)

    # --- Trendstatus ---
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

    strat = "Dividende" if dy >= 0.025 else "Wachstum" if cat == "Fast Grower" or (g_raw or 0) >= 0.10 else "Cashflow"
    return dict(
        cat=cat, strat=strat, fin=fin, p=round(price, 2), d=pm["date"], mc=r1(mcap_usd / 1e9) if mcap_usd else None,
        F=F, f=fund10, fa=len(avail), M=M, m=moat25, T=Tk, t=tech,
        pe=r1(pe), hpe=r1(hist_pe), kgve=r1(kgve), g=pct(g), dy=r1(dy * 100, 2),
        r=pct(rend), rs=round(rs), lo=r1(lo * 100, 0) if lo is not None else None,
        hi=r1(hi * 100, 0) if hi is not None else None, dist=pct(dist), risk=risk, st=status,
    )
