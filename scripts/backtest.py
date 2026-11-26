"""Rückrechnung (Prognose-Paket 3, 26.11.2026). Nur Anzeige, fließt nicht in Score oder Einordnung ein.

Läuft einmal im Monat in .github/workflows/backtest.yml (nicht im täglichen Lauf). Lädt Tageskurse ab 2009 für das
Universum, berechnet je Aktie und Woche (Freitag) Merkmale, die es an diesem Tag gab (keine Vorwegnahme), und misst,
was in den 5/20/60 Handelstagen danach passierte – als Mehrrendite gegenüber der mittleren Aktie (Median) derselben
Woche. Ergebnis docs/bt.json (klein):
  - set:  feste Lagen („Setups“) mit Fallzahl, Anteil besser als Ø, Median-Mehrrendite, getrennt nach Lernzeitraum
          (bis 2018) und Prüfzeitraum (ab 2019) und nach Marktphase; Kennzeichen ok = in beiden Zeiträumen gleich
          gerichtet und deutlich (bewährt).
  - dec:  Merkmale in Zehnteln (je Woche über alle Aktien gerankt): Anteil besser als Ø und Median-Mehrrendite je Zehntel.
  - base: was ein beliebiger Tag brachte (Median-Rendite und Anteil im Plus) für 5/20/60 Tage.
Die Merkmal-Funktion features() nutzt auch prognose.py im täglichen Lauf, damit Rückrechnung und Anzeige gleich rechnen.
"""
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "bt.json")
CACHE = os.path.join(ROOT, "_btcache", "prices.pkl")
START = "2009-01-01"
SPLIT = 2019              # Lernzeitraum bis 2018, Prüfzeitraum ab 2019
HZ = (5, 20, 60, 120)     # 120 seit 26.11. (Prognosemodell)

# Setups: Name, Erklärung in einfachen Worten
SETUPS = {
    "hoch52": ("Nahe am 52-Wochen-Hoch", "Kurs höchstens 2 % unter dem Jahreshoch"),
    "ausbruch": ("Ausbruch mit Volumen", "Schluss über dem Hoch der letzten 60 Tage bei deutlich erhöhtem Volumen"),
    "ruecksetzer": ("Rücksetzer im Aufwärtstrend", "langfristiger Aufwärtstrend, Kurs an der 50-Tage-Linie, kurzfristig abgekühlt"),
    "kaufdruck": ("Aufwärtstrend mit Kaufdruck", "Aufwärtstrend und deutlich mehr Volumen an Plus- als an Minustagen"),
    "ueberhitzt": ("Überhitzt", "sehr schneller Anstieg: RSI über 75 oder mehr als 40 % über der 200-Tage-Linie"),
    "ueberverkauft": ("Stark überverkauft", "RSI unter 30 und mehr als 10 % unter der 50-Tage-Linie"),
    "kapitulation": ("Ausverkauf mit hohem Volumen", "über 15 % Verlust in einem Monat bei stark erhöhtem Volumen"),
    "wende": ("Erholung nach tiefem Fall", "über 25 % unter dem Jahreshoch, aber wieder über der 50-Tage-Linie"),
    "abwaerts": ("Fallender Trend", "unter fallender 200-Tage-Linie und unter der 50-Tage-Linie"),
    "boden": ("Fallender Trend, aber überverkauft", "fallender Trend und RSI unter 35"),
}
DECF = {"mom12": "Kursentwicklung 12 Monate (ohne letzten Monat)", "mom1": "Kursentwicklung letzter Monat",
        "d52h": "Abstand zum 52-Wochen-Hoch", "d200": "Abstand zur 200-Tage-Linie", "rsi": "RSI 14 Tage",
        "vol60": "Schwankung 60 Tage", "ud": "Kaufdruck (Volumen Plus- zu Minustagen)", "volr": "Volumen gegenüber Schnitt"}


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _prep(df):
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce")
    ok = c.notna() & (c > 0)
    df, c = df[ok], c[ok]
    h = pd.to_numeric(df["High"], errors="coerce").fillna(c) if "High" in df else c
    l = pd.to_numeric(df["Low"], errors="coerce").fillna(c) if "Low" in df else c
    v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) if "Volume" in df else pd.Series(0.0, index=c.index)
    return c, np.maximum(h, c), np.minimum(l, c), v


def features(df):
    """Merkmale je Handelstag, nur aus Daten bis zu diesem Tag. Rückgabe DataFrame oder None."""
    p = _prep(df)
    if p is None or len(p[0]) < 260:
        return None
    c, h, l, v = p
    F = pd.DataFrame(index=c.index)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    F["mom12"] = c.shift(21) / c.shift(252) - 1
    F["mom1"] = c / c.shift(21) - 1
    F["d52h"] = c / h.rolling(252).max() - 1
    F["d50"] = c / s50 - 1
    F["d200"] = c / s200 - 1
    F["sl200"] = s200 / s200.shift(20) - 1
    F["rsi"] = _rsi(c, 14)
    lr = np.log(c).diff()
    F["vol60"] = lr.rolling(60).std() * np.sqrt(252)
    vm100 = v.rolling(100).mean()
    F["volr"] = (v.rolling(5).mean() / vm100).where(vm100 > 0)
    dc = c.diff()
    up, dn = v.where(dc > 0, 0).rolling(65).sum(), v.where(dc < 0, 0).rolling(65).sum()
    F["ud"] = (up / dn).where(dn > 0)
    F["trend"] = ((c > s200) & (s50 > s200)).astype(float)
    F["brk"] = (c >= h.rolling(60).max().shift(1)).astype(float)
    # seit 26.11.2026 (Prognosemodell, Ausbau): weitere bekannte Merkmale, nur zusätzliche Spalten
    F["mom3"] = c / c.shift(63) - 1
    F["mom6"] = c.shift(21) / c.shift(126) - 1
    F["rev1w"] = c / c.shift(5) - 1
    F["d52l"] = c / l.rolling(252).min() - 1
    F["maxr"] = lr.rolling(21).max()
    F.loc[s200.isna(), "trend"] = np.nan
    return F


def setups_of(r):
    """Welche Setups treffen auf eine Merkmalszeile (dict oder Series) zu? Fehlende Werte = trifft nicht zu."""
    g = lambda k: r.get(k) if hasattr(r, "get") else r[k]  # noqa: E731
    def v(k):
        x = g(k)
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
    out = []
    tr, d52h, d50, d200, rsi, mom1, mom12 = v("trend"), v("d52h"), v("d50"), v("d200"), v("rsi"), v("mom1"), v("mom12")
    volr, ud, sl200, brk = v("volr"), v("ud"), v("sl200"), v("brk")
    if d52h is not None and d52h >= -0.02:
        out.append("hoch52")
    if brk == 1 and volr is not None and volr >= 1.5:
        out.append("ausbruch")
    if tr == 1 and d50 is not None and -0.05 <= d50 <= 0.01 and rsi is not None and rsi < 45 and mom12 is not None and mom12 > 0.1:
        out.append("ruecksetzer")
    if tr == 1 and ud is not None and ud >= 1.4:
        out.append("kaufdruck")
    if (rsi is not None and rsi > 75) or (d200 is not None and d200 > 0.4):
        out.append("ueberhitzt")
    if rsi is not None and rsi < 30 and d50 is not None and d50 < -0.1:
        out.append("ueberverkauft")
    if mom1 is not None and mom1 < -0.15 and volr is not None and volr >= 1.8:
        out.append("kapitulation")
    if d52h is not None and d52h < -0.25 and d50 is not None and d50 > 0:
        out.append("wende")
    falling = d200 is not None and d200 < 0 and sl200 is not None and sl200 < 0 and d50 is not None and d50 < 0
    if falling:
        out.append("abwaerts")
    if falling and rsi is not None and rsi < 35:
        out.append("boden")
    return out


def market_phase(sp, vx=None):
    """Marktphase je Tag aus S&P 500 (und VIX): 'auf', 'korrektur', 'stress'."""
    c = pd.to_numeric(sp["Close"], errors="coerce").dropna()
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    dd = c / c.rolling(252, min_periods=60).max() - 1
    v = None
    if vx is not None:
        v = pd.to_numeric(vx["Close"], errors="coerce")
        if getattr(v.index, "tz", None) is not None:
            v.index = v.index.tz_localize(None)
        v = v.reindex(c.index).ffill()
    ph = pd.Series("korrektur", index=c.index)
    ph[(c > s200) & (c > s50)] = "auf"
    stress = (c < s200) & ((dd <= -0.10) | ((v >= 25) if v is not None else False))
    ph[stress] = "stress"
    return ph


def _fwd(c, l):
    o = {}
    for n in HZ:
        o["f%d" % n] = c.shift(-n) / c - 1
    o["mae"] = l[::-1].rolling(60, min_periods=60).min()[::-1].shift(-1) / c - 1
    return pd.DataFrame(o)


def panel(prices, phase=None, meta=None):
    """Wöchentliche Fälle über alle Aktien: Merkmale, Setups, Folgerenditen, Mehrrendite gegen Ø derselben Woche."""
    rows = []
    for s, df in prices.items():
        try:
            F = features(df)
            if F is None:
                continue
            c, h, l, v = _prep(df)
            X = pd.concat([F, _fwd(c, l)], axis=1)
            W = X.resample("W-FRI").last()
            W = W[W["mom12"].notna() & W["d200"].notna()]
            W["s"] = s
            mt = (meta or {}).get(s) or {}
            W["rg"] = mt.get("region") or "alle"
            W["sec"] = mt.get("sector") or "?"
            rows.append(W)
        except Exception as e:  # noqa: BLE001
            print("Rückrechnung: Merkmale fehlgeschlagen", s, type(e).__name__, flush=True)
    if not rows:
        return None
    P = pd.concat(rows)
    P.index.name = "w"
    P = P.reset_index()
    # Mehrrendite gegen die mittlere Aktie derselben Woche UND derselben Region (seit 26.11.: Region, damit nicht
    # Japan-gegen-USA-Bewegungen als Rauschen im Ziel landen)
    for n in HZ:
        k = "f%d" % n
        P["x%d" % n] = P[k] - P.groupby(["w", "rg"])[k].transform("median")
    P["secm"] = P.groupby(["w", "sec"])["mom6"].transform("median")   # Branchen-Momentum (6 Monate)
    if phase is not None:
        ph = phase.resample("W-FRI").last()
        P["ph"] = P["w"].map(ph).fillna("korrektur")
    else:
        P["ph"] = "korrektur"
    P["yr"] = P["w"].dt.year
    return P


def _stat(x):
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return None
    return [int(len(x)), round(float((x > 0).mean()) * 100, 1), round(float(np.median(x)) * 100, 2)]


def analyse(P):
    out = {"v": 2, "dedup": 28, "ziel": "region", "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "split": SPLIT,
           "von": int(P["yr"].min()), "bis": int(P["yr"].max()), "n": int(len(P)), "nam": SETUPS, "decn": DECF}
    out["base"] = {str(n): [round(float(np.nanmedian(P["f%d" % n])) * 100, 2), round(float((P["f%d" % n] > 0).mean()) * 100, 1)]
                   for n in HZ}
    # Setups
    flags = {k: [] for k in SETUPS}
    for r in P[["trend", "d52h", "d50", "d200", "rsi", "mom1", "mom12", "volr", "ud", "sl200", "brk"]].itertuples(index=False):
        hit = set(setups_of(r._asdict()))
        for k in SETUPS:
            flags[k].append(k in hit)
    # entdoppeln (26.11.): je Aktie und Setup höchstens ein Fall in 4 Wochen, sonst zählt dieselbe Lage mehrfach
    sv, wv = P["s"].values, P["w"].values.astype("datetime64[D]").astype(np.int64)
    for k in SETUPS:
        m, last = flags[k], {}
        for i in range(len(m)):
            if m[i]:
                if sv[i] in last and wv[i] - last[sv[i]] < 28:
                    m[i] = False
                else:
                    last[sv[i]] = wv[i]
    tr = P["yr"] < SPLIT
    st = {}
    for k in SETUPS:
        m = np.array(flags[k])
        e = {}
        for n in HZ:
            x = P["x%d" % n].values
            a, b = _stat(x[m & tr.values]), _stat(x[m & ~tr.values])
            e[str(n)] = {"all": _stat(x[m]), "tr": a, "te": b,
                         "ph": {ph: _stat(x[m & (P["ph"].values == ph)]) for ph in ("auf", "korrektur", "stress")}}
        mae = P["mae"].values[m]
        mae = mae[np.isfinite(mae)]
        e["mae"] = round(float(np.median(mae)) * 100, 1) if len(mae) >= 30 else None
        a, b = e["60"]["tr"], e["60"]["te"]
        ok = 0
        if a and b and a[0] >= 300 and b[0] >= 300:
            if a[1] >= 53 and b[1] >= 53 and a[2] > 0 and b[2] > 0:
                ok = 1
            elif a[1] <= 47 and b[1] <= 47 and a[2] < 0 and b[2] < 0:
                ok = -1
        e["ok"] = ok
        st[k] = e
    out["set"] = st
    # Zehntel je Merkmal (je Woche gerankt)
    dec = {}
    for f in DECF:
        q = P.groupby("w")[f].rank(pct=True)
        d = np.ceil(q * 10).clip(1, 10)
        res = {}
        for n in (20, 60):
            x = P["x%d" % n]
            res[str(n)] = [[_stat(x[(d == i) & tr].values), _stat(x[(d == i) & ~tr].values)] for i in range(1, 11)]
        dec[f] = res
    out["dec"] = dec
    return out


def download(symbols):
    import yfinance as yf
    prices = {}
    for i in range(0, len(symbols), 50):
        b = symbols[i:i + 50]
        px = None
        for att in range(3):
            try:
                px = yf.download(b, start=START, auto_adjust=False, group_by="ticker", progress=False, threads=True)
                if px is not None and not px.empty:
                    break
            except Exception as e:  # noqa: BLE001
                print("Kursabruf-Fehler", type(e).__name__, flush=True)
            time.sleep(20 * (att + 1))
        for s in b:
            try:
                df = px[s].dropna(how="all") if px is not None else None
                if df is not None and len(df) > 300:
                    prices[s] = df[["Close", "High", "Low", "Volume"]].astype("float32")
            except Exception:
                pass
        print("Kurse:", min(i + 50, len(symbols)), "/", len(symbols), flush=True)
        time.sleep(1)
    return prices


def main():
    U = json.load(open(os.path.join(ROOT, "data", "universe.json")))["stocks"]
    uni = [x["symbol"] for x in U]
    meta = {x["symbol"]: {"region": x.get("region"), "sector": x.get("sector")} for x in U}
    prices = None
    if os.path.exists(CACHE):                      # nur wiederverwenden, wenn höchstens 5 Tage alt (erneuter Versuch)
        try:
            ck = pickle.load(open(CACHE, "rb"))
            if time.time() - ck.get("ts", 0) < 5 * 86400:
                prices = ck["p"]
                print("Kurse aus dem Zwischenspeicher:", len(prices), flush=True)
        except Exception:
            prices = None
    if not prices:
        prices = download(uni + ["^GSPC", "^VIX"])
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pickle.dump({"ts": time.time(), "p": prices}, open(CACHE, "wb"))
    sp, vx = prices.pop("^GSPC", None), prices.pop("^VIX", None)
    phase = market_phase(sp, vx) if sp is not None else None
    prices = {s: d for s, d in prices.items() if s in set(uni)}
    t = time.time()
    P = panel(prices, phase, meta)
    if P is None or len(P) < 10000:
        print("Rückrechnung: zu wenige Fälle, nichts geschrieben", flush=True)
        return 1
    print("Fälle:", len(P), "Aktien:", P["s"].nunique(), "Sekunden:", round(time.time() - t), flush=True)
    res = analyse(P)
    try:                                           # Prognosemodell (Paket 5); bei Fehler bleibt die Rückrechnung gültig
        rep, params = model_fit(P)
        res["mdl"] = rep
        if params:
            with open(os.path.join(ROOT, "data", "prog_model.json"), "w") as f:
                json.dump(params, f, separators=(",", ":"))
    except Exception as e:  # noqa: BLE001
        print("Prognosemodell übersprungen:", type(e).__name__, e, flush=True)
    try:                                           # Prüfung der Simulation (26.11.)
        sc = sim_check(prices)
        if sc:
            res["sim"] = sc
    except Exception as e:  # noqa: BLE001
        print("Prüfung der Simulation übersprungen:", type(e).__name__, e, flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, separators=(",", ":"), ensure_ascii=False)
    for k, e in res["set"].items():
        print("%-14s 60T alle=%s Lern=%s Prüf=%s ok=%s" % (k, e["60"]["all"], e["60"]["tr"], e["60"]["te"], e["ok"]), flush=True)
    print("Geschrieben:", OUT, os.path.getsize(OUT), "Bytes", flush=True)
    return 0



# ------------------------------------------------------------------------------------------------ Prognosemodell (Paket 5)
# Ausbau 26.11.2026: Ziel = schlägt die mittlere Aktie derselben Region; Merkmale je Woche und Region gerankt; mehr
# Merkmale (3/6 Monate, letzte Woche, Abstand zum Tief, größter Tagesgewinn, Branchen-Momentum); Zeiträume 20/60/120
# Tage; zwei vorher festgelegte Modelle (A: logistische Regression mit Rang und Rang², B: Gradient Boosting mit
# Marktphase, erkennt Zusammenhänge zwischen Merkmalen). Welches Modell gilt, entscheiden NUR die Prüfjahre bis
# AUSWAHL_BIS; die Abnahme-Regeln sind unverändert (alle Prüfjahre) plus: auch in den Jahren danach Ø AUC ≥ 0,525 und
# mindestens so gut wie Momentum (−0,003). Kalibrierung über die Rangfolge innerhalb der Woche (gleich bei Prüfung und heute), gelernt
# nur an Jahren bis AUSWAHL_BIS, Kalibrierungsfehler gemessen an den Jahren danach.
MF = ["mom12", "mom1", "d52h", "d50", "d200", "sl200", "rsi", "vol60", "volr", "ud"]   # alter Satz (altes Modell)
MB = ["trend", "brk"]
MF2 = MF + ["mom3", "mom6", "rev1w", "d52l", "maxr", "secm"]
MHZ = (20, 60, 120)
AUSWAHL_BIS = 2020
PHN = {"auf": 0.0, "korrektur": 1.0, "stress": 2.0}
MNAME = {"mom12": "Kursentwicklung 12 Monate", "mom1": "Kursentwicklung letzter Monat", "d52h": "Abstand zum Jahreshoch",
         "d50": "Abstand zur 50-Tage-Linie", "d200": "Abstand zur 200-Tage-Linie", "sl200": "Richtung der 200-Tage-Linie",
         "rsi": "RSI 14 Tage", "vol60": "Schwankung", "volr": "Volumen zuletzt", "ud": "Kaufdruck",
         "trend": "Aufwärtstrend", "brk": "Hoch der letzten 60 Tage", "mom3": "Kursentwicklung 3 Monate",
         "mom6": "Kursentwicklung 6 Monate", "rev1w": "Kursentwicklung letzte Woche", "d52l": "Abstand zum Jahrestief",
         "maxr": "größter Tagesgewinn im Monat", "secm": "Stärke der eigenen Branche", "phn": "Marktphase"}


def _gkeys(D):
    return ["w", "rg"] if "rg" in D else ["w"]


def add_ranks(P, mf=None):
    for f in (mf or MF2):
        if f not in P:
            P[f] = np.nan
        P["r_" + f] = P.groupby(_gkeys(P))[f].rank(pct=True)
    return P


def design(D, mf=None, mb=None, kind="lin"):
    """Merkmalsmatrix und Namen je Spalte. lin: Rang−0,5 und (Rang−0,5)²; gbm: Rang−0,5 und Marktphase."""
    mf, mb = mf or MF, mb or MB
    cols, names = [], []
    for f in mf:
        r = D["r_" + f].fillna(0.5).values
        cols.append(r - 0.5); names.append(f)
        if kind == "lin":
            cols.append((r - 0.5) ** 2 - 1 / 12); names.append(f)
    for f in mb:
        cols.append((D[f] if f in D else pd.Series(0.5, index=D.index)).fillna(0.5).values - 0.5); names.append(f)
    if kind == "gbm":
        cols.append((D["phn"] if "phn" in D else pd.Series(1.0, index=D.index)).fillna(1.0).values); names.append("phn")
    return np.column_stack(cols).astype(float), names


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=1500,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def _fit(kind, X, y):
    if kind == "gbm":
        return _gbm().fit(X, y)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=400).fit(X, y)


def _score(m, X):
    return m.decision_function(X)


def _pct(score, keys):
    """Rangfolge 0–1 innerhalb der Gruppe (Woche, Region)."""
    return pd.Series(score).groupby([pd.Series(k) for k in keys]).rank(pct=True).values if keys else pd.Series(score).rank(pct=True).values


def _export_gbm(m, X):
    """Bäume als Listen (ohne scikit-learn anwendbar). Prüft die Nachrechnung; bei Abweichung None."""
    try:
        trees = []
        for it in m._predictors:
            nd = it[0].nodes
            trees.append([nd["feature_idx"].astype(int).tolist(), [float(v) for v in nd["num_threshold"]],
                          nd["left"].astype(int).tolist(), nd["right"].astype(int).tolist(),
                          nd["is_leaf"].astype(int).tolist(), [float(v) for v in nd["value"]]])
        base = float(np.ravel(m._baseline_prediction)[0])
        k = min(2000, len(X))
        ok = np.allclose(_gbm_score({"trees": trees, "b0": base}, X[:k]), m.decision_function(X[:k]), atol=1e-6)
        return {"trees": trees, "b0": base} if ok else None
    except Exception as e:  # noqa: BLE001
        print("Gradient Boosting nicht exportierbar:", type(e).__name__, e, flush=True)
        return None


def _gbm_score(M, X):
    out = np.full(len(X), M["b0"], dtype=float)
    for f, t, l, r, leaf, v in M["trees"]:
        f, t, l, r, leaf, v = np.array(f), np.array(t), np.array(l), np.array(r), np.array(leaf, bool), np.array(v)
        idx = np.zeros(len(X), dtype=int)
        for _ in range(64):
            act = ~leaf[idx]
            if not act.any():
                break
            ii = np.where(act)[0]
            go_l = X[ii, f[idx[ii]]] <= t[idx[ii]]
            idx[ii] = np.where(go_l, l[idx[ii]], r[idx[ii]])
        out += v[idx]
    return out


def _evaluate(D, X, y, x, kind, years):
    """Walk-forward: je Prüfjahr nur mit Daten lernen, die ≥ 90 Tage vor Jahresbeginn enden."""
    w, yr = D["w"].values, D["yr"].values
    keys = [D[k].values for k in _gkeys(D)]
    rows, out = [], []
    for Y in years:
        if Y < years[0] + 5:
            continue
        te = yr == Y
        tr = w < np.datetime64("%d-01-01" % Y) - np.timedelta64(90, "D")
        if te.sum() < 2000 or tr.sum() < 20000:
            continue
        m = _fit(kind, X[tr], y[tr])
        sc = _score(m, X[te])
        u = _pct(sc, [k[te] for k in keys])
        a = _auc(y[te], sc)
        b = _auc(y[te], D["r_mom12"].fillna(0.5).values[te])
        q = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False).values
        top, bot = x[te][q == 9], x[te][q == 0]
        rows.append([int(Y), round(a, 3) if a else None, round(b, 3) if b else None,
                     round(float((top > 0).mean()) * 100, 1), round(float(np.median(top)) * 100, 2),
                     round(float((bot > 0).mean()) * 100, 1), round(float(np.median(bot)) * 100, 2), int(te.sum())])
        out.append((Y, u, y[te], x[te]))
    return rows, out


def _summ(rows):
    a = [r[1] for r in rows if r[1] is not None]
    b = [r[2] for r in rows if r[2] is not None]
    if not a:
        return None
    return {"auc": float(np.mean(a)), "mom": float(np.mean(b)) if b else 0.5, "share": float(np.mean([v > 0.5 for v in a])),
            "tb": float(np.mean([r[3] > r[5] for r in rows]))}


def model_fit(P):
    """Rückgabe (Bericht für bt.json, Modellparameter oder None)."""
    from sklearn.isotonic import IsotonicRegression
    P = P.copy()
    for f in MF2 + MB:
        if f not in P:
            P[f] = np.nan
    if "phn" not in P:
        P["phn"] = P["ph"].map(PHN).fillna(1.0) if "ph" in P else 1.0
    P = add_ranks(P, MF2)
    rep = {"hz": {}, "v": 2, "sel": AUSWAHL_BIS, "ziel": "region" if "rg" in P else "alle"}
    params = {"v": 2, "mf": MF2, "mb": MB, "nam": MNAME, "hz": {}}
    years = sorted(P["yr"].unique())
    for hz in MHZ:
        k = "x%d" % hz
        if k not in P:
            continue
        D = P[np.isfinite(P[k])]
        if len(D) < 30000:
            continue
        y, x = (D[k].values > 0).astype(int), D[k].values
        res = {}
        for kind in ("lin", "gbm"):
            try:
                X, _ = design(D, MF2, MB, kind)
                rows, oos = _evaluate(D, X, y, x, kind, years)
                if len(rows) >= 4:
                    res[kind] = (rows, oos)
            except Exception as e:  # noqa: BLE001
                print("Modell", kind, hz, "fehlgeschlagen:", type(e).__name__, e, flush=True)
        if not res:
            continue
        val = {kd: _summ([r for r in v[0] if r[0] <= AUSWAHL_BIS]) for kd, v in res.items()}
        kind = max(res, key=lambda kd: (val[kd] or {"auc": 0})["auc"])
        rows, oos = res[kind]
        S, H = _summ(rows), _summ([r for r in rows if r[0] > AUSWAHL_BIS])
        # Kalibrierung: an Jahren bis AUSWAHL_BIS lernen, an späteren messen
        ue = np.concatenate([o[1] for o in oos if o[0] <= AUSWAHL_BIS] or [np.array([0.5])])
        ye = np.concatenate([o[2] for o in oos if o[0] <= AUSWAHL_BIS] or [np.array([0])])
        iso0 = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(ue, ye)
        cal = []
        if H:
            uh = np.concatenate([o[1] for o in oos if o[0] > AUSWAHL_BIS])
            yh = np.concatenate([o[2] for o in oos if o[0] > AUSWAHL_BIS])
            ph_ = iso0.predict(uh)
            bins = np.clip((ph_ * 20).astype(int), 0, 19)
            for b_ in np.unique(bins):
                kk = bins == b_
                if kk.sum() >= 500:
                    cal.append([round(float(ph_[kk].mean()) * 100, 1), round(float(yh[kk].mean()) * 100, 1), int(kk.sum())])
        ece = sum(abs(c[0] - c[1]) * c[2] for c in cal) / max(1, sum(c[2] for c in cal)) if cal else 99.0
        ok = bool(S["auc"] >= 0.53 and S["share"] >= 0.7 and S["tb"] >= 0.7 and ece < 3 and S["auc"] >= S["mom"] - 0.003
                  and H is not None and H["auc"] >= 0.525 and H["auc"] >= H["mom"] - 0.003)
        # Endgültige Kalibrierung und Mehrrendite je Stufe über alle Prüfjahre
        ua, ya, xa = (np.concatenate([o[i] for o in oos]) for i in (1, 2, 3))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(ua, ya)
        qb = np.clip((iso.predict(ua) * 20).astype(int), 0, 19)
        ex = {}
        for b_ in np.unique(qb):
            kk = qb == b_
            if kk.sum() >= 300:
                ex[str(int(b_))] = [round(float(np.median(xa[kk])) * 100, 2), round(float(np.percentile(xa[kk], 25)) * 100, 2),
                                    round(float(np.percentile(xa[kk], 75)) * 100, 2), int(kk.sum())]
        rep["hz"][str(hz)] = {"yrs": rows, "auc": round(S["auc"], 3), "auc_mom": round(S["mom"], 3), "auc_share": round(S["share"] * 100),
                              "topbot": round(S["tb"] * 100), "ece": round(ece, 2), "cal": cal, "ok": ok, "ex": ex, "kind": kind,
                              "val": {kd: round(v["auc"], 3) for kd, v in val.items() if v},
                              "hold": [round(H["auc"], 3), round(H["mom"], 3)] if H else None}
        X, names = design(D, MF2, MB, kind)
        m = _fit(kind, X, y)
        if kind == "gbm":
            g = _export_gbm(m, X)
            if g is None:
                rep["hz"][str(hz)]["ok"] = ok = False
                continue
            par = {"kind": "gbm", **g}
        else:
            par = {"kind": "lin", "b0": float(m.intercept_[0]), "b": [float(v) for v in m.coef_[0]]}
        par.update({"names": names, "cal": "rang", "iso": [[float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_]],
                    "ok": ok, "ex": ex})
        params["hz"][str(hz)] = par
        print("Modell %d Tage (%s, Auswahl %s): AUC %.3f (Momentum %.3f), Jahre >0,5: %d %%, Spitze>Boden: %d %%, "
              "ab %d: AUC %s, Kalibrierung %.2f Pp., ok=%s" % (hz, kind, {kd: round(v["auc"], 3) for kd, v in val.items() if v}, S["auc"], S["mom"], S["share"] * 100,
                                                               S["tb"] * 100, AUSWAHL_BIS + 1, H and round(H["auc"], 3), ece, ok), flush=True)
    rep["ok"] = any(v["ok"] for v in rep["hz"].values())
    return rep, (params if params["hz"] else None)


def model_apply(params, rows, phase=None):
    """rows: {Kürzel: Merkmalszeile (dict, optional rg/sec)} für heute. Rangfolge über alle heutigen Aktien (je Region),
    Wahrscheinlichkeit (kalibriert) und die Beiträge der Merkmale. Rückgabe {Kürzel: {hz: [p, Gründe…]}}."""
    if not params or not rows:
        return {}
    mf, mb = params.get("mf") or MF, params.get("mb") or MB
    D = pd.DataFrame.from_dict(rows, orient="index")
    for f in set(mf + MF2 + mb + ["mom6"]):
        if f not in D:
            D[f] = np.nan
        D[f] = pd.to_numeric(D[f], errors="coerce")
    if "sec" in D and D["secm"].isna().all():
        D["secm"] = D.groupby("sec")["mom6"].transform("median")
    D["w"] = 0
    if "rg" in D:
        D["rg"] = D["rg"].fillna("alle")
    D["phn"] = PHN.get(phase or "", 1.0)
    D = add_ranks(D, list(dict.fromkeys(mf + MF2)))
    out = {s: {} for s in D.index}
    keys = [D[k].values for k in _gkeys(D)]
    for hz, m in params["hz"].items():
        kind = m.get("kind", "lin")
        X, names = design(D, mf, mb, kind)
        if kind == "gbm":
            sc = _gbm_score(m, X)
            f_sc = lambda Z: _gbm_score(m, Z)  # noqa: E731
        else:
            b = np.array(m["b"])
            sc = m["b0"] + X @ b
            f_sc = lambda Z, b=b, b0=m["b0"]: b0 + Z @ b  # noqa: E731
        if m.get("cal") == "rang":
            p = np.interp(_pct(sc, keys), m["iso"][0], m["iso"][1])
        else:                                       # altes Modell (v1)
            p = np.interp(1 / (1 + np.exp(-sc)), m["iso"][0], m["iso"][1])
        # Beitrag je Merkmal: Wert dieses Merkmals auf neutral setzen und Unterschied messen
        con = {}
        for f in dict.fromkeys(names):
            if f == "phn":
                continue
            Z = X.copy()
            for j, nm in enumerate(names):
                if nm == f:
                    Z[:, j] = 0.0 if not (kind == "lin" and j > 0 and names[j - 1] == f) else -1 / 12
            con[f] = sc - f_sc(Z)
        for i, s in enumerate(D.index):
            top = sorted(((f, v[i]) for f, v in con.items()), key=lambda t: -abs(t[1]))[:6]
            out[s][hz] = [round(float(p[i]) * 100, 1),
                          [[f, round(float(v), 3), round(float(D.loc[s, "r_" + f]) * 100) if ("r_" + f) in D and np.isfinite(D.loc[s, "r_" + f]) else None]
                           for f, v in top]]
    return out


# ------------------------------------------------------------------------------------------------ Prüfung der Simulation
def sim_check(prices, n_cases=3000, n_paths=2000, seed=7):
    """Stimmt die Simulation (prognose.simulate) mit dem überein, was früher wirklich passiert ist?
    Für zufällige frühere Tage ab 2015: Ziel = Hoch der letzten 60 Tage, Stopp = Tief der letzten 60 Tage (mindestens
    3 % Abstand, höchstens 30 %), Richtung = übliche mittlere Aktie. Simuliert wird nur mit Kursen bis zu diesem Tag;
    verglichen wird mit dem echten Verlauf der Schlusskurse in den 60 Tagen danach."""
    import prognose as PG
    rng = np.random.default_rng(seed)
    syms = [s for s, d in prices.items() if len(d) > 1100]
    if not syms:
        return None
    allf = []
    for s in syms[:200]:
        c = pd.to_numeric(prices[s]["Close"], errors="coerce").dropna().values
        allf.append(np.log(c[60:] / c[:-60]))
    base = float(np.median(np.concatenate(allf))) / 60 if allf else 0.0
    rec = []
    tries = 0
    while len(rec) < n_cases and tries < n_cases * 4:
        tries += 1
        s = syms[rng.integers(len(syms))]
        df = prices[s]
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        c = pd.to_numeric(df["Close"], errors="coerce").values.astype(float)
        ok_i = np.where(idx >= pd.Timestamp("2015-01-01"))[0]
        ok_i = ok_i[(ok_i >= 800) & (ok_i < len(c) - 61)]
        if not len(ok_i):
            continue
        i = int(ok_i[rng.integers(len(ok_i))])
        hist = c[:i + 1]
        if not np.isfinite(hist[-260:]).all():
            continue
        p0 = hist[-1]
        up = float(np.clip(np.nanmax(hist[-60:]), p0 * 1.03, p0 * 1.30))
        dn = float(np.clip(np.nanmin(hist[-60:]), p0 * 0.70, p0 * 0.97))
        mc = PG.simulate(hist, up, dn, base, seed=int(rng.integers(1 << 30)), n=n_paths)
        if not mc or "pu" not in mc:
            continue
        fut = c[i + 1:i + 61]
        if not np.isfinite(fut).all():
            continue
        hu = np.where(fut >= up)[0]
        hd = np.where(fut <= dn)[0]
        tu, td = (hu[0] if len(hu) else 99), (hd[0] if len(hd) else 99)
        act = "u" if tu < td else "d" if td < tu else "n"
        rec.append((mc["pu"] / 100, mc["pd"] / 100, mc["p60"] / 100, act, float(fut[-1] > p0)))
    if len(rec) < 500:
        return None
    A = np.array([[r[0], r[1], r[2], r[3] == "u", r[3] == "d", r[4]] for r in rec], dtype=float)

    def cal(pred, act):
        b = np.clip((pred * 10).astype(int), 0, 9)
        rows = []
        for k in range(10):
            m = b == k
            if m.sum() >= 40:
                rows.append([round(float(pred[m].mean()) * 100, 1), round(float(act[m].mean()) * 100, 1), int(m.sum())])
        e = sum(abs(r[0] - r[1]) * r[2] for r in rows) / max(1, sum(r[2] for r in rows))
        return rows, round(e, 1)
    cu, eu = cal(A[:, 0], A[:, 3])
    cd, ed = cal(A[:, 1], A[:, 4])
    c6, e6 = cal(A[:, 2], A[:, 5])
    pn = 1 - A[:, 0] - A[:, 1]
    an = 1 - A[:, 3] - A[:, 4]
    br = float(np.mean((A[:, 0] - A[:, 3]) ** 2 + (A[:, 1] - A[:, 4]) ** 2 + (pn - an) ** 2))
    mu, md_ = A[:, 3].mean(), A[:, 4].mean()
    br0 = float(np.mean((mu - A[:, 3]) ** 2 + (md_ - A[:, 4]) ** 2 + ((1 - mu - md_) - an) ** 2))
    res = {"n": len(rec), "u": cu, "eu": eu, "d": cd, "ed": ed, "p60": c6, "e60": e6,
           "pu": [round(float(A[:, 0].mean()) * 100, 1), round(float(mu) * 100, 1)],
           "pd": [round(float(A[:, 1].mean()) * 100, 1), round(float(md_) * 100, 1)],
           "br": round(br, 4), "br0": round(br0, 4), "ok": bool(eu < 5 and ed < 5 and e6 < 5)}
    print("Prüfung Simulation:", res["n"], "Fälle, Ziel zuerst vorhergesagt/eingetreten", res["pu"], "Stopp", res["pd"],
          "Abweichung", eu, ed, e6, "Brier", res["br"], "ohne Simulation", res["br0"], "ok", res["ok"], flush=True)
    return res


if __name__ == "__main__":
    sys.exit(main())
