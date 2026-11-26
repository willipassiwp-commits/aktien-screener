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
import json
import math
import os

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


def compute(df, zs=None, zr=None, sym=""):
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
        try:
            mc = scenario(df, zs, zr, sym)
            if mc:
                out["mc"] = mc
        except Exception:
            pass
        return out or None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ Simulation (26.11.2026)
# 10.000 mögliche Kursverläufe je Aktie für 60 Handelstage. Bausteine sind die echten Tagesbewegungen der Aktie aus den
# letzten 3 Jahren, jeweils durch die damalige Schwankung geteilt und mit der heutigen Schwankung multipliziert
# (gefilterte historische Simulation), gezogen in Blöcken zu 5 Tagen, damit Serien von Plus- und Minustagen erhalten
# bleiben. Richtung: neutral (mittlere Aktie aller Tage aus der Rückrechnung), plus der Vorteil bewährter Setups, wenn
# die Rückrechnung für die heutige Lage einen gefunden hat (ok ≠ 0, in Lern- und Prüfzeitraum gleich gerichtet).
NSIM, HSIM, BLOCK = 10000, 60, 5
MC_DAYS = (5, 10, 20, 40, 60)
_CTX = {"bt": None, "phase": None, "mdl": None, "rows": {}, "sim": {}}


def set_context(MD=None, prog_path=None):
    """Einmal je Lauf: Rückrechnungsergebnis (docs/bt.json) laden und heutige Marktphase aus den Marktdaten bestimmen."""
    try:
        if prog_path:
            p = os.path.join(os.path.dirname(prog_path), "bt.json")
            if os.path.exists(p):
                _CTX["bt"] = json.load(open(p))
    except Exception:
        _CTX["bt"] = None
    try:                                           # Prognosemodell (Paket 5), gelernt im monatlichen Ablauf
        if prog_path:
            p = os.path.join(os.path.dirname(os.path.dirname(prog_path)), "data", "prog_model.json")
            if os.path.exists(p):
                _CTX["mdl"] = json.load(open(p))
    except Exception:
        _CTX["mdl"] = None
    try:
        if MD and "^GSPC" in MD:
            import backtest
            ph = backtest.market_phase(MD["^GSPC"], MD.get("^VIX"))
            _CTX["phase"] = str(ph.iloc[-1])
    except Exception:
        _CTX["phase"] = None


def targets(p, zs, zr):
    """Ziel (nächster Widerstand, Unterkante) und Stopp (1 % unter der Unterkante der nächsten Unterstützung),
    wie im Ausblick der Seite (abLv)."""
    sup = [z for z in (zs or []) if z[0] < p]
    res = [z for z in (zr or []) if z[1] > p]
    if res and res[0][0] <= p * 1.015 and len(res) > 1:
        res = res[1:]
    up = float(res[0][0]) if res else None
    dn = float(sup[0][0]) * 0.99 if sup else None
    return up, dn


def setup_drift(F):
    """Setups der heutigen Lage und daraus die tägliche Mehrrendite (Prüfzeitraum, 60 Tage) bewährter Setups."""
    bt = _CTX["bt"]
    if F is None or len(F) == 0:
        return [], 0.0, None
    try:
        import backtest
        row = F.iloc[-1].to_dict()
        st = backtest.setups_of(row)
    except Exception:
        return [], 0.0, None
    if not bt or "set" not in bt:
        return st, 0.0, None
    ph, ds, used = _CTX["phase"], [], []
    for k in st:
        e = bt["set"].get(k)
        if not e or not e.get("ok"):
            continue
        q = (e["60"].get("ph") or {}).get(ph) if ph else None
        te = e["60"].get("te")
        v = q[2] if q and q[0] >= 200 and np.sign(q[2]) == np.sign(te[2]) else te[2]
        ds.append(v / 100.0)
        used.append(k)
    if not ds:
        return st, 0.0, None
    d = float(np.mean(ds)) / HSIM
    return st, float(np.clip(d, -0.0015, 0.0015)), used


def simulate(c, up=None, dn=None, drift=0.0, seed=0, n=NSIM, h=HSIM):
    """Rückgabe dict: q = Quantile 5/25/50/75/95 der Kursveränderung an den Tagen MC_DAYS; pu/pd/pn = Anteil der
    Verläufe, die zuerst das Ziel, zuerst den Stopp bzw. keins von beiden in h Tagen erreichen; tu/td = Median-Tage bis
    dahin; p20/p60 = Anteil über dem heutigen Kurs."""
    c = np.asarray(c, dtype=float)
    c = c[np.isfinite(c) & (c > 0)][-757:]
    if len(c) < 300:
        return None
    lr = np.diff(np.log(c))
    lam = 0.94
    var = np.empty(len(lr))
    var[0] = np.var(lr[:20]) if len(lr) > 20 else lr[0] ** 2
    for i in range(1, len(lr)):
        var[i] = lam * var[i - 1] + (1 - lam) * lr[i - 1] ** 2
    sig = np.sqrt(np.maximum(var, 1e-10))
    z = lr / sig
    z = (z - z.mean()) / (z.std() or 1.0)
    # heutige Schwankung: halb kurzfristig (EWMA, reagiert auf die letzten Tage), halb 60 Tage, begrenzt auf das
    # 0,6- bis 1,6-Fache der Jahresschwankung (einzelne Extremtage sollen den Fächer nicht aufblähen)
    v_ew = lam * var[-1] + (1 - lam) * lr[-1] ** 2
    v_60 = float(np.var(lr[-60:]))
    s_now = float(np.sqrt(0.5 * v_ew + 0.5 * v_60))
    s_year = float(np.std(lr[-252:]))
    s_now = float(np.clip(s_now, 0.6 * s_year, 1.6 * s_year))
    rng = np.random.default_rng(seed)
    nb = -(-h // BLOCK)
    starts = rng.integers(0, len(z) - BLOCK, size=(n, nb))
    Z = z[starts[:, :, None] + np.arange(BLOCK)[None, None, :]].reshape(n, nb * BLOCK)[:, :h]
    # Medianpfad ohne Schwankungsabschlag: drift ist die gewünschte mittlere Tagesrendite (log)
    R = np.cumsum(drift + s_now * Z, axis=1)
    out = {"q": [], "d": list(MC_DAYS), "s": round(float(s_now * np.sqrt(252) * 100), 1), "dr": round(drift * HSIM * 100, 2)}
    for d in MC_DAYS:
        r = np.exp(R[:, d - 1]) - 1
        out["q"].append([_sig(float(np.percentile(r, q))) for q in (5, 25, 50, 75, 95)])
    out["p20"] = round(float((R[:, 19] > 0).mean()) * 100, 1)
    out["p60"] = round(float((R[:, h - 1] > 0).mean()) * 100, 1)
    p0 = c[-1]
    if up and dn and dn < p0 < up:
        lu, ld = np.log(up / p0), np.log(dn / p0)
        hu, hd = R >= lu, R <= ld
        tu = np.where(hu.any(1), hu.argmax(1), h + 1)
        td = np.where(hd.any(1), hd.argmax(1), h + 1)
        fu, fd = tu < td, td < tu
        out["pu"] = round(float(fu.mean()) * 100, 1)
        out["pd"] = round(float(fd.mean()) * 100, 1)
        out["pn"] = round(100 - out["pu"] - out["pd"], 1)
        out["tu"] = int(np.median(tu[fu]) + 1) if fu.any() else None
        out["td"] = int(np.median(td[fd]) + 1) if fd.any() else None
        out["up"], out["dn"] = _sig(up, 5), _sig(dn, 5)
    return out


def scenario(df, zs=None, zr=None, sym=""):
    """Simulation plus Setups für eine Aktie (Teil von compute)."""
    p = _prep(df)
    if p is None:
        return None
    c = p[0]
    try:
        import backtest
        F = backtest.features(df)
    except Exception:
        F = None
    st, dr, used = setup_drift(F)
    try:                                           # für das Prognosemodell nach der Schleife (finalize)
        if F is not None and len(F):
            _CTX["rows"][sym] = {k: (None if not np.isfinite(v) else float(v)) for k, v in F.iloc[-1].items()}
    except Exception:
        pass
    bt = _CTX["bt"]
    base = 0.0
    if bt and bt.get("base"):
        base = float(np.log1p(bt["base"]["60"][0] / 100.0)) / HSIM     # mittlere Aktie an beliebigen Tagen
    up, dn = targets(float(c.iloc[-1]), zs, zr)
    import zlib
    seed = zlib.crc32((sym + str(c.index[-1].date())).encode())
    m = simulate(c.values, up, dn, base + dr, seed=seed)
    _CTX["sim"][sym] = (c.values, up, dn, seed, base)
    if not m:
        return None
    m["dr"], m["db"] = round(dr * HSIM * 100, 2), round(base * HSIM * 100, 2)   # Vorteil der Setups / üblicher Anstieg
    m["st"] = st
    m["su"] = used or []
    m["ph"] = _CTX["phase"]
    m["n"] = NSIM
    return m



def finalize(prog):
    """Nach der Schleife im täglichen Lauf: Prognosemodell auf alle heutigen Aktien anwenden (Ränge über alle Aktien)
    und, wenn das 60-Tage-Modell die Prüfung bestanden hat, die Simulation mit der Richtung des Modells neu rechnen.
    Schreibt prog[s]["pm"] = {"20": [p, Gründe], "60": [...], "ok": {...}, "ex": [Mitte, 25 %, 75 %] }."""
    M = _CTX.get("mdl")
    if not M or not _CTX["rows"]:
        return 0
    import backtest
    res = backtest.model_apply(M, _CTX["rows"])
    n = 0
    for s, r in res.items():
        if s not in prog:
            continue
        pm = {hz: v for hz, v in r.items()}
        pm["ok"] = {hz: bool(M["hz"][hz].get("ok")) for hz in M["hz"]}
        m60 = M["hz"].get("60")
        if m60 and "60" in r:
            b_ = str(int(min(19, max(0, r["60"][0] / 5))))
            ex = (m60.get("ex") or {}).get(b_)
            if ex:
                pm["ex"] = ex[:3]
            if m60.get("ok") and ex and s in _CTX["sim"]:
                c, up, dn, seed, base = _CTX["sim"][s]
                d = float(np.clip(np.log1p(ex[0] / 100.0) / HSIM, -0.0015, 0.0015))
                mc = simulate(c, up, dn, base + d, seed=seed)
                if mc:
                    old = prog[s].get("mc") or {}
                    mc.update({"dr": round(d * HSIM * 100, 2), "db": round(base * HSIM * 100, 2), "st": old.get("st", []),
                               "su": ["modell"], "ph": old.get("ph"), "n": NSIM})
                    prog[s]["mc"] = mc
        prog[s]["pm"] = pm
        n += 1
    return n
