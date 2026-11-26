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
HZ = (5, 20, 60)

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


def panel(prices, phase=None):
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
            rows.append(W)
        except Exception as e:  # noqa: BLE001
            print("Rückrechnung: Merkmale fehlgeschlagen", s, type(e).__name__, flush=True)
    if not rows:
        return None
    P = pd.concat(rows)
    P.index.name = "w"
    P = P.reset_index()
    for n in HZ:
        k = "f%d" % n
        P["x%d" % n] = P[k] - P.groupby("w")[k].transform("median")
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
    out = {"v": 1, "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "split": SPLIT,
           "von": int(P["yr"].min()), "bis": int(P["yr"].max()), "n": int(len(P)), "nam": SETUPS, "decn": DECF}
    out["base"] = {str(n): [round(float(np.nanmedian(P["f%d" % n])) * 100, 2), round(float((P["f%d" % n] > 0).mean()) * 100, 1)]
                   for n in HZ}
    # Setups
    flags = {k: [] for k in SETUPS}
    for r in P[["trend", "d52h", "d50", "d200", "rsi", "mom1", "mom12", "volr", "ud", "sl200", "brk"]].itertuples(index=False):
        hit = set(setups_of(r._asdict()))
        for k in SETUPS:
            flags[k].append(k in hit)
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
    uni = [x["symbol"] for x in json.load(open(os.path.join(ROOT, "data", "universe.json")))["stocks"]]
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
    P = panel(prices, phase)
    if P is None or len(P) < 10000:
        print("Rückrechnung: zu wenige Fälle, nichts geschrieben", flush=True)
        return 1
    print("Fälle:", len(P), "Aktien:", P["s"].nunique(), "Sekunden:", round(time.time() - t), flush=True)
    res = analyse(P)
    with open(OUT, "w") as f:
        json.dump(res, f, separators=(",", ":"), ensure_ascii=False)
    for k, e in res["set"].items():
        print("%-14s 60T alle=%s Lern=%s Prüf=%s ok=%s" % (k, e["60"]["all"], e["60"]["tr"], e["60"]["te"], e["ok"]), flush=True)
    print("Geschrieben:", OUT, os.path.getsize(OUT), "Bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
