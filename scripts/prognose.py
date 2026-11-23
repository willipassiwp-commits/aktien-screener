"""Prognose-Bausteine (Paket 2, 24.11.2026). Nur Anzeige, fließt nicht in Score oder Einordnung ein.

- Bewährte Unterstützungen und Widerstände: Wie oft hat der Kurs eine Zone in den letzten 5 Jahren getestet,
  wie oft hat sie gehalten, wie stark war die Bewegung danach?
- Erwartungs-Korridor: Verteilung der Kursveränderung nach 20 und 60 Handelstagen in früheren ähnlichen Lagen
  dieser Aktie (dieselben Lagen wie in shortplus.analogs), dazu der Vergleich mit allen Tagen derselben Zeitspanne,
  damit sichtbar wird, ob die Lage wirklich besser oder schlechter war als ein zufälliger Tag.
- Schwankungsband: übliche Schwankung der letzten 60 Tage, hochgerechnet auf 20 und 60 Tage (Ersatz, wenn es zu
  wenige ähnliche Lagen gibt).

Alle Renditen als Anteile (0,05 = +5 %), auf 3 signifikante Stellen gerundet. Jede Funktion gibt bei Problemen None zurück.
"""
import math

import numpy as np
import pandas as pd

HZ = (20, 60)
QS = (10, 25, 50, 75, 90)


def _sig(v, d=3):
    if v is None or not np.isfinite(v):
        return None
    return float("%.*g" % (d, v))


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
    return c, np.maximum(h, c), np.minimum(l, c)


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------------------------------------------ Zonen
def zone_record(c, h, l, lo, hi, kind, years=5, gap=15, look=20):
    """Tests einer Zone in den letzten `years` Jahren.
    Unterstützung: Tief erreicht die Zone (±1 %), nachdem der Schluss in den 10 Tagen davor mindestens 3 % über der
    Oberkante lag. Gehalten: kein Schluss mehr als 3 % unter der Unterkante in den 10 Tagen danach UND danach ein
    Anstieg von mindestens 5 % vom Tief innerhalb von 20 Tagen. Widerstand spiegelbildlich.
    Tests der letzten 20 Handelstage zählen nicht (Ausgang noch offen). Rückgabe [Tests, gehalten, Median-Bewegung,
    letzter Test JJJJ-MM] oder None."""
    n = len(c)
    if n < 300 or not lo or not hi or hi < lo:
        return None
    C, H, L = c.values, h.values, l.values
    start = max(10, n - int(years * 252))
    tests, held, moves, last = 0, 0, [], None
    prev = -10 ** 9
    for i in range(start, n - look):
        if i - prev < gap:
            continue
        if kind == "s":
            touch = L[i] <= hi * 1.01 and L[i] >= lo * 0.9 and C[i - 10:i].max() > hi * 1.03
        else:
            touch = H[i] >= lo * 0.99 and H[i] <= hi * 1.1 and C[i - 10:i].min() < lo * 0.97
        if not touch:
            continue
        prev = i
        tests += 1
        last = c.index[i]
        if kind == "s":
            ext = L[i:i + 6].min()
            mv = H[i + 1:i + 1 + look].max() / ext - 1
            ok = C[i:i + 11].min() >= lo * 0.97 and mv >= 0.05
        else:
            ext = H[i:i + 6].max()
            mv = L[i + 1:i + 1 + look].min() / ext - 1
            ok = C[i:i + 11].max() <= hi * 1.03 and mv <= -0.05
        if ok:
            held += 1
        moves.append(mv)
    if not tests:
        return [0, 0, None, None]
    return [tests, held, _sig(float(np.median(moves))), last.strftime("%Y-%m")]


# ------------------------------------------------------------------------------------------------ Korridor
def analog_events(c, h, l, years=10):
    """Dieselben ähnlichen Lagen wie shortplus.analogs (RSI 14, Abstand zur 20-Tage-Linie in Tagesspannen, Trend)."""
    n = len(c)
    if n < 600:
        return None, None
    try:                                     # dieselben Rechenwege wie die Tabelle „frühere Fälle“
        import shortplus
        r14, atr = shortplus._rsi(c, 14), shortplus._atr(h, l, c)
    except Exception:
        r14, atr = _rsi(c, 14), _atr(h, l, c)
    e20 = c.ewm(span=20, adjust=False).mean()
    dist = (c - e20) / atr.replace(0, np.nan)
    s200 = c.rolling(200).mean()
    up = c > s200
    cr, cd, cu = float(r14.iloc[-1]), float(dist.iloc[-1]), bool(up.iloc[-1])
    start = max(220, n - int(years * 252))
    if cr < 30 or cd < -2:
        cond = (r14 <= max(cr + 6, 25)) & (dist <= min(cd * 0.75, -1.5))
    elif cr > 70 or cd > 2:
        cond = (r14 >= min(cr - 6, 70)) & (dist >= max(cd * 0.75, 1.5))
    else:
        cond = ((r14 - cr).abs() <= 5) & ((dist - cd).abs() <= 0.5) & (up == cu)
    idx = [i for i in np.where(cond.values)[0] if start <= i <= n - 61]
    ev = []
    for i in idx:
        if not ev or i - ev[-1] >= 10:
            ev.append(i)
    return ev, start


def corridor(c, ev, start, min_n=8):
    """Quantile der Rendite nach 20/60 Tagen in den ähnlichen Lagen und dieselben Kennzahlen über alle Tage."""
    n = len(c)
    C = c.values
    out = {}
    for hz in HZ:
        allr = C[start + hz:n] / C[start:n - hz] - 1 if n - hz > start else np.array([])
        allr = allr[np.isfinite(allr)]
        b = None
        if len(allr) >= 100:
            b = [_sig(float(np.median(allr))), round(float((allr > 0).mean()) * 100)]
        e = None
        if ev and len(ev) >= min_n:
            r = np.array([C[i + hz] / C[i] - 1 for i in ev if i + hz < n])
            if len(r) >= min_n:
                e = {"q": [_sig(float(np.percentile(r, q))) for q in QS], "hit": round(float((r > 0).mean()) * 100),
                     "n": int(len(r))}
        out[str(hz)] = {"b": b, "e": e}
    return out


def vol_band(c, days=60):
    """Übliche Schwankung: Standardabweichung der Tagesrenditen der letzten `days` Tage, hochgerechnet (Wurzel-Zeit).
    Rückgabe je Zeitraum [10-%-, 90-%-Grenze] als Rendite (Normalverteilung der Logrenditen, ohne Drift)."""
    if len(c) < days + 2:
        return None
    lr = np.log(c.values[-days - 1:]).astype(float)
    sd = float(np.std(np.diff(lr), ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return {str(hz): [_sig(math.exp(-1.2816 * sd * math.sqrt(hz)) - 1), _sig(math.exp(1.2816 * sd * math.sqrt(hz)) - 1)]
            for hz in HZ}


def compute(df, zs=None, zr=None):
    """Alles für eine Aktie. zs/zr wie in data.json: [[Unterkante, Oberkante, [Bezeichnungen], Abstand %], …]."""
    try:
        p = _prep(df)
        if p is None:
            return None
        c, h, l = p
        if len(c) < 300:
            return None
        out = {}
        z = {}
        for key, zones, kind in (("s", zs, "s"), ("r", zr, "r")):
            if zones:
                z[key] = [zone_record(c, h, l, float(q[0]), float(q[1]), kind) for q in zones]
        if z:
            out["z"] = z
        ev, start = analog_events(c, h, l)
        if start is not None:
            k = corridor(c, ev, start)
            if any(v["b"] or v["e"] for v in k.values()):
                out["k"] = k
        vb = vol_band(c)
        if vb:
            out["v"] = vb
        return out or None
    except Exception:
        return None
