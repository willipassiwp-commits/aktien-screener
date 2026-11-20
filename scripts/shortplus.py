"""Erweiterte Kurzfristanalyse je Aktie (nur Anzeige, fließt nicht in den Gesamtscore ein):

- Markt: Beta, relative Stärke gegen Markt und Branche, grobe Zerlegung des Rückgangs, Fazit „Markt × Aktie“
- Erschöpfung: wie stark der Verkaufsdruck schon ausgereizt ist
- Divergenzen: Kurs macht neues Tief, RSI/MACD/OBV nicht (und umgekehrt am Hoch)
- Wende-Treppe: sechs Stufen mit konkreten Kursmarken und „Was müsste passieren“
- Rücksetzer im Aufwärtstrend (das am besten belegte Kurzfristmuster)
- Frühere Fälle: was in ähnlicher Lage bei dieser Aktie danach passierte, inklusive weiterem Rückgang bis zum Tief

Beschreibt den Kursverlauf bis zum letzten Handelstag und frühere Häufigkeiten. Nichts davon ist eine Vorhersage.
"""
import math

import numpy as np
import pandas as pd

from scoring import de, sig


def _sp(v, d=1):
    if v is None or not np.isfinite(v):
        return "–"
    return ("+" if v >= 0 else "−") + de(abs(v) * 100, d) + " %"


def _pr(v):
    if v is None:
        return "–"
    return de(v, 2 if v < 100 else 0)


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)


def _atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swings(s, k=3, kind="low"):
    """Indexpositionen von Swing-Tiefs/-Hochs: Wert ist Minimum/Maximum von k Tagen links und rechts."""
    a = s.values
    out = []
    for i in range(k, len(a) - k):
        w = a[i - k:i + k + 1]
        if (kind == "low" and a[i] == w.min()) or (kind == "high" and a[i] == w.max()):
            if not out or i - out[-1] > k:
                out.append(i)
    return out


def prep(df):
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce")
    ok = c.notna() & (c > 0)
    df, c = df[ok], c[ok]
    h = pd.to_numeric(df["High"], errors="coerce").fillna(c) if "High" in df else c
    l = pd.to_numeric(df["Low"], errors="coerce").fillna(c) if "Low" in df else c
    o = pd.to_numeric(df["Open"], errors="coerce").fillna(c) if "Open" in df else c
    v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) if "Volume" in df else None
    return c, np.maximum(h, c), np.minimum(l, c), o, v


# ------------------------------------------------------------------------------------------------ Markt
def market_link(c, mkt, sec=None, mlight=None, mname="S&P 500", sname=None):
    """Beta, relative Stärke und grobe Zerlegung gegen den Markt (mkt) und die Branche (sec)."""
    if mkt is None or len(mkt) < 120:
        return None
    j = pd.concat([c, mkt], axis=1, keys=["a", "m"]).dropna()
    if len(j) < 120:
        return None
    r = j.pct_change().dropna()
    out = {"mn": mname}

    def beta(n):
        x = r.iloc[-n:]
        vm = float(x["m"].var())
        return float(x["a"].cov(x["m"]) / vm) if vm > 0 and len(x) >= 40 else None
    b60, b250 = beta(60), beta(250)
    corr = float(r["a"].iloc[-250:].corr(r["m"].iloc[-250:])) if len(r) >= 120 else None
    out.update(b60=sig(b60, 3), b250=sig(b250, 3), corr=sig(corr, 2))

    def ret(s, n):
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n else None
    rs = {}
    for n in (5, 20, 60):
        a, m = ret(j["a"], n), ret(j["m"], n)
        rs[str(n)] = [sig(a, 3), sig(m, 3), sig(None if a is None or m is None else a - m, 3)]
    out["rs"] = rs
    if sec is not None and len(sec) > 70:
        js = pd.concat([c, sec, mkt], axis=1, keys=["a", "s", "m"]).dropna()
        if len(js) > 70:
            out["sec"] = {"n": sname, "r20": sig(ret(js["s"], 20), 3), "r60": sig(ret(js["s"], 60), 3),
                          "x20": sig(ret(js["a"], 20) - ret(js["s"], 20), 3)}
            # grobe Zerlegung der letzten 20 Tage: Markt (Beta × Markt), Branche (Branche minus Markt), Rest = Aktie selbst
            bb = b250 if b250 is not None else b60
            a20, m20, s20 = ret(js["a"], 20), ret(js["m"], 20), ret(js["s"], 20)
            if bb is not None and None not in (a20, m20, s20):
                mp = bb * m20
                sp_ = s20 - m20
                out["dec"] = [sig(a20, 3), sig(mp, 3), sig(sp_, 3), sig(a20 - mp - sp_, 3)]
    # RS-Linie (Aktie / Markt, auf 100 normiert) für die letzten 170 Tage
    rl = (j["a"] / j["m"]).iloc[-170:]
    out["rsl"] = [sig(float(x / rl.iloc[0] * 100), 4) for x in rl.values]
    rl250 = (j["a"] / j["m"]).iloc[-250:]
    out["rsHigh"] = bool(rl250.iloc[-1] >= rl250.max() * 0.995)
    # Text
    t = []
    bb = b250 if b250 is not None else b60
    if bb is not None and bb < 0.3:
        t.append("Die Aktie bewegt sich weitgehend unabhängig vom %s (Beta %s)." % (mname, de(bb, 2)))
    elif bb is not None:
        t.append("Die Aktie bewegt sich etwa %s so stark wie der %s (Beta %s)%s." % (
            "doppelt" if 1.8 <= bb < 2.3 else "halb" if 0.4 <= bb < 0.6 else de(bb, 1) + "-mal", mname, de(bb, 2),
            ": fällt der Markt um weitere 5 %%, wären für diese Aktie rund %s %% üblich" % de(5 * bb, 0) if bb >= 0.3 else ""))
    x20 = rs["20"][2]
    if x20 is not None:
        t.append("In 20 Handelstagen lief sie %s %s als der Markt (%s gegen %s)." % (
            de(abs(x20) * 100, 1) + " Prozentpunkte", "besser" if x20 >= 0 else "schlechter", _sp(rs["20"][0]), _sp(rs["20"][1])))
    if out.get("sec") and out["sec"].get("x20") is not None:
        xs = out["sec"]["x20"]
        t.append("Gegenüber ihrer Branche (%s) lief sie %s." % (
            sname, "besser" if xs > 0.01 else "schlechter" if xs < -0.01 else "etwa gleich"))
    if out.get("dec"):
        a20, mp, sp_, own = out["dec"]
        t.append("Von %s in 20 Tagen gehen rechnerisch etwa %s auf den Markt, %s auf die Branche und %s auf die Aktie selbst." % (
            _sp(a20, 0), _sp(mp, 0), _sp(sp_, 0), _sp(own, 0)))
    if out["rsHigh"]:
        t.append("Die relative Stärke gegenüber dem Markt steht auf einem Jahreshoch: solche Aktien führen Erholungen oft an.")
    out["t"] = " ".join(t)
    return out


# ------------------------------------------------------------------------------------------------ Erschöpfung
def exhaustion(c, h, l, o, v, sym=None):
    r14, r2 = _rsi(c, 14), _rsi(c, 2)
    atr = _atr(h, l, c)
    e20 = c.ewm(span=20, adjust=False).mean()
    a = float(atr.iloc[-1]) or 1e-9
    dist = float((c.iloc[-1] - e20.iloc[-1]) / a)
    rows, pts = [], 0
    x = float(r14.iloc[-1])
    p = 2 if x < 20 else 1 if x < 30 else 0
    pts += p
    rows.append(["RSI 14", de(x, 0), p, "extrem überverkauft" if x < 20 else "überverkauft" if x < 30 else "nicht überverkauft"])
    x2 = float(r2.iloc[-1])
    p = 1 if x2 < 5 else 0
    pts += p
    rows.append(["RSI 2 (sehr kurzfristig)", de(x2, 0), p, "sehr kurzfristig ausgereizt" if x2 < 5 else "nicht ausgereizt"])
    p = 2 if dist <= -3 else 1 if dist <= -2 else 0
    pts += p
    rows.append(["Abstand zur 20-Tage-Linie", "%s Tagesspannen" % de(dist, 1), p,
                 "selten weit darunter (über 3 Tagesspannen)" if dist <= -3 else "deutlich darunter" if dist <= -2 else "kein extremer Abstand"])
    if sym and sym[1] is not None and sym[2]:
        ratio = sym[1] / sym[2] if sym[2] else None
        p = 2 if ratio and ratio >= 2 else 1 if ratio and ratio >= 1.3 else 0
        pts += p
        rows.append(["Korrektur gegenüber früheren", "%s %% (üblich %s %%, tiefste %s %%)" % (
            de(sym[1], 0), de(sym[2], 0), de(sym[3], 0) if sym[3] is not None else "–"), p,
            "deutlich tiefer als üblich" if p == 2 else "tiefer als üblich" if p == 1 else "im üblichen Rahmen"])
    cap = None
    if v is not None and len(v) > 25:
        v20 = v.rolling(20).mean()
        for i in range(len(c) - 5, len(c)):
            rng = float(h.iloc[i] - l.iloc[i])
            if v20.iloc[i - 1] > 0 and v.iloc[i] >= 2 * v20.iloc[i - 1] and rng >= 1.5 * float(atr.iloc[i - 1]) and \
                    rng > 0 and (c.iloc[i] - l.iloc[i]) / rng >= 0.6 and l.iloc[i] < c.iloc[i - 1]:
                cap = str(c.index[i].date())
        p = 2 if cap else 0
        pts += p
        rows.append(["Kapitulationstag (5 Tage)", cap or "nein", p,
                     "sehr hohes Volumen, großer Tagesbereich, Kurs vom Tief zurückgekauft" if cap else "kein Kapitulationstag"])
    dn = 0
    for i in range(len(c) - 1, 0, -1):
        if c.iloc[i] < c.iloc[i - 1]:
            dn += 1
        else:
            break
    gaps = int(sum(1 for i in range(len(c) - 10, len(c)) if o.iloc[i] < c.iloc[i - 1] * 0.98))
    p = 1 if dn >= 5 or gaps >= 2 else 0
    pts += p
    rows.append(["Minustage in Folge / Abwärtslücken (10 Tage)", "%d / %d" % (dn, gaps), p,
                 "anhaltender Verkaufsdruck" if p else "unauffällig"])
    mx = 2 + 1 + 2 + (2 if sym and sym[1] is not None and sym[2] else 0) + (2 if v is not None else 0) + 1
    note = round(10 * pts / mx, 1)
    lb = ("Kapitulationszeichen" if cap and note >= 5 else "extrem überverkauft" if note >= 6 else "überverkauft" if note >= 3.5
          else "leicht überverkauft" if note >= 1.5 else "kein Ausverkauf")
    if note >= 6:
        t = "Der Verkaufsdruck ist sehr weit ausgereizt. Das heißt nicht, dass das Tief erreicht ist, macht eine Gegenbewegung aber wahrscheinlicher als sonst."
    elif note >= 3.5:
        t = "Die Aktie ist überverkauft. Gegenbewegungen sind häufiger, ein Boden ist damit aber noch nicht bestätigt."
    elif note >= 1.5:
        t = "Leicht überverkauft, aber noch kein Ausverkauf."
    else:
        t = "Kein Zeichen von Ausverkauf oder Erschöpfung."
    return {"n": note, "lb": lb, "rows": rows, "t": t, "dist": sig(dist, 3), "cap": cap}


# ------------------------------------------------------------------------------------------------ Divergenzen
def divergences(c, h, l, v, look=70):
    r14 = _rsi(c, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum() if v is not None else None
    out = []
    n = len(c)
    base = max(0, n - look)
    lw = l.iloc[base:]
    lows = [base + i for i in swings(lw, 3, "low")]
    # das aktuelle Tief zählt auch, wenn es in den letzten 3 Tagen liegt
    tail = int(np.argmin(l.iloc[-4:].values)) + n - 4
    if tail not in lows and l.iloc[tail] <= l.iloc[base:].min() * 1.001:
        lows.append(tail)
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if l.iloc[i2] < l.iloc[i1] and i2 - i1 >= 5:
            for name, ind in (("RSI", r14), ("MACD-Histogramm", hist), ("OBV", obv)):
                if ind is not None and ind.iloc[i2] > ind.iloc[i1]:
                    out.append(["pos", name, str(c.index[i1].date()), str(c.index[i2].date()),
                                "Kurs mit neuem Tief, %s aber mit höherem Tief: der Verkaufsdruck lässt nach" % name])
    hw = h.iloc[base:]
    highs = [base + i for i in swings(hw, 3, "high")]
    tailh = int(np.argmax(h.iloc[-4:].values)) + n - 4
    if tailh not in highs and h.iloc[tailh] >= h.iloc[base:].max() * 0.999:
        highs.append(tailh)
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if h.iloc[i2] > h.iloc[i1] and i2 - i1 >= 5 and i2 >= n - 15:
            for name, ind in (("RSI", r14), ("MACD-Histogramm", hist)):
                if ind.iloc[i2] < ind.iloc[i1]:
                    out.append(["neg", name, str(c.index[i1].date()), str(c.index[i2].date()),
                                "Kurs mit neuem Hoch, %s aber mit tieferem Hoch: der Schwung lässt nach" % name])
    # nur frische Divergenzen (zweites Tief/Hoch in den letzten 15 Tagen)
    cut = str(c.index[max(0, n - 15)].date())
    return [d for d in out if d[3] >= cut]


# ------------------------------------------------------------------------------------------------ Wende-Treppe
def ladder(c, h, l, mkt=None):
    """Sechs Stufen der Wende mit Kursmarken. Nur sinnvoll, wenn die Aktie unter der 50-Tage-Linie oder deutlich
    unter ihrem Hoch steht."""
    n = len(c)
    last = float(c.iloc[-1])
    s50 = float(c.rolling(50).mean().iloc[-1])
    e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    hi60 = float(h.iloc[-60:].max())
    if last > s50 and last >= hi60 * 0.92:
        return None
    w = l.iloc[-60:]
    li = int(np.argmin(w.values)) + n - 60                 # tiefster Punkt der letzten 60 Tage
    low = float(l.iloc[li])
    since = n - 1 - li
    steps = []
    steps.append(["3 Handelstage ohne neues Tief", since >= 3, None,
                  "erfüllt: letztes Tief vor %d Tagen (%s)" % (since, _pr(low)) if since >= 3 else
                  "noch nicht: letztes Tief %s (%s)" % ("heute" if since == 0 else "vor %d Tag%s" % (since, "" if since == 1 else "en"), _pr(low))])
    h5 = float(h.iloc[-6:-1].max())
    steps.append(["Schluss über dem Hoch der letzten 5 Tage", last > h5, h5, "bei einem Schlusskurs über %s" % _pr(h5)])
    steps.append(["Schluss über der 20-Tage-Linie", last > e20, e20, "über %s" % _pr(e20)])
    # höheres Tief: Swing-Tief nach dem Tief, das höher liegt, und danach Anstieg
    sw = [i for i in swings(l.iloc[li:], 2, "low")] if n - li > 6 else []
    hl = [li + i for i in sw if li + i > li + 1 and l.iloc[li + i] > low * 1.005]
    ok4 = bool(hl) and last > float(l.iloc[hl[-1]])
    steps.append(["Höheres Tief gebildet", ok4, float(l.iloc[hl[-1]]) if hl else low,
                  "erfüllt: höheres Tief bei %s" % _pr(float(l.iloc[hl[-1]])) if ok4 else
                  "ein Rücksetzer, der über %s endet, und danach wieder Anstieg" % _pr(low)])
    # letztes Zwischenhoch vor dem Tief
    pre = h.iloc[max(0, li - 60):li]
    shs = swings(pre, 3, "high") if len(pre) > 8 else []
    zh = float(pre.iloc[shs[-1]]) if shs else (float(pre.max()) if len(pre) else None)
    steps.append(["Schluss über dem letzten Zwischenhoch", zh is not None and last > zh, zh,
                  "über %s" % _pr(zh) if zh else "–"])
    steps.append(["Schluss über der 50-Tage-Linie", last > s50, s50, "über %s" % _pr(s50)])
    done = 0
    for s in steps:
        if s[1]:
            done += 1
        else:
            break
    met = sum(1 for s in steps if s[1])
    nxt = next((s for s in steps if not s[1]), None)
    out = {"steps": [[s[0], bool(s[1]), sig(s[2]) if s[2] else None, s[3]] for s in steps], "prog": done, "met": met,
           "low": sig(low), "lowd": str(c.index[li].date()), "inv": sig(low)}
    if nxt:
        out["t"] = "Wende-Fortschritt %d von 6. Nächste Stufe: %s (%s)." % (done, nxt[0], nxt[3])
    else:
        out["t"] = "Alle sechs Stufen erfüllt: die Wende ist charttechnisch bestätigt."
    out["t"] += " Fällt der Kurs unter das Tief bei %s, beginnt die Treppe von vorn." % _pr(low)
    if mkt is not None and len(mkt) > 30:
        me20 = float(mkt.ewm(span=20, adjust=False).mean().iloc[-1])
        out["mk"] = [bool(float(mkt.iloc[-1]) > me20), sig(me20)]
    return out


# ------------------------------------------------------------------------------------------------ Rücksetzer im Aufwärtstrend
def pullback(c, h, l, v, quality=None):
    n = len(c)
    if n < 220:
        return None
    s200 = c.rolling(200).mean()
    last = float(c.iloc[-1])
    if not (last > s200.iloc[-1] and s200.iloc[-1] > s200.iloc[-21]):
        return None
    e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    s50 = float(c.rolling(50).mean().iloc[-1])
    atr = float(_atr(h, l, c).iloc[-1])
    r2 = float(_rsi(c, 2).iloc[-1])
    dn = 0
    for i in range(n - 1, 0, -1):
        if c.iloc[i] < c.iloc[i - 1]:
            dn += 1
        else:
            break
    near = [nm for nm, lv in (("20-Tage-Linie", e20), ("50-Tage-Linie", s50)) if abs(last - lv) <= atr]
    vol_ok = None
    if v is not None and n > 30:
        seg = c.diff().iloc[-10:]
        dv = v.reindex(seg.index)[seg < 0]
        uv = v.reindex(seg.index)[seg > 0]
        if len(dv) >= 2 and len(uv) >= 1:
            vol_ok = bool(dv.mean() < v.iloc[-60:].mean())
    hi20 = float(h.iloc[-20:].max())
    depth = (hi20 - last) / atr if atr else None
    conds = [["Kurs über steigender 200-Tage-Linie", True],
             ["Qualität ab 70", quality is not None and quality >= 70],
             ["Kurzfristig ausgereizt (RSI 2 unter 10 oder 3 Minustage in Folge)", r2 < 10 or dn >= 3],
             ["Kurs an 20- oder 50-Tage-Linie (Abstand unter 1 Tagesspanne)", bool(near)],
             ["Volumen im Rücksetzer unterdurchschnittlich", bool(vol_ok)]]
    pb = {"c": conds, "near": near, "dp": sig(depth, 3), "vo": vol_ok}
    pb_text(pb)
    return pb


def pb_text(pb):
    """Zählt erfüllte Bedingungen und schreibt den Text (auch nachträglich, wenn die Qualität feststeht)."""
    met = sum(1 for x in pb["c"] if x[1])
    near = pb.get("near") or []
    t = ("Rücksetzer im intakten Aufwärtstrend%s; Tiefe %s Tagesspannen vom 20-Tage-Hoch. " % (
        (" an der " + " und ".join(near)) if near else "", de(pb.get("dp") or 0, 1)))
    t += ("Solche Rücksetzer gelten als das am besten belegte Kurzfristmuster, alle Bedingungen sind erfüllt."
          if met == 5 else "%d von 5 Bedingungen erfüllt." % met)
    if pb.get("vo") is False:
        t += " Das Volumen an Minustagen ist erhöht: das kann auf Verkäufe größerer Anleger hindeuten."
    pb["met"], pb["t"] = met, t
    return pb


# ------------------------------------------------------------------------------------------------ Frühere Fälle
def analogs(c, h, l, mkt=None, years=10):
    """Wann stand die Aktie früher ähnlich da (RSI 14 und Abstand zur 20-Tage-Linie in Tagesspannen, Trendrichtung)?
    Was passierte danach: Rendite nach 5/20/60 Tagen und weiterer Rückgang bis zum Tief innerhalb von 20 Tagen."""
    n = len(c)
    if n < 600:
        return None
    r14 = _rsi(c, 14)
    atr = _atr(h, l, c)
    e20 = c.ewm(span=20, adjust=False).mean()
    dist = (c - e20) / atr.replace(0, np.nan)
    s200 = c.rolling(200).mean()
    up = c > s200
    cr, cd, cu = float(r14.iloc[-1]), float(dist.iloc[-1]), bool(up.iloc[-1])
    start = max(220, n - int(years * 252))
    if cr < 30 or cd < -2:
        cond = (r14 <= max(cr + 6, 25)) & (dist <= min(cd * 0.75, -1.5))
        kind = "überverkauft"
    elif cr > 70 or cd > 2:
        cond = (r14 >= min(cr - 6, 70)) & (dist >= max(cd * 0.75, 1.5))
        kind = "überkauft"
    else:
        cond = ((r14 - cr).abs() <= 5) & ((dist - cd).abs() <= 0.5) & (up == cu)
        kind = "ähnlich"
    idx = [i for i in np.where(cond.values)[0] if start <= i <= n - 61]
    ev = []
    for i in idx:
        if not ev or i - ev[-1] >= 10:
            ev.append(i)
    mk_weak = None
    if mkt is not None and len(mkt) > 300:
        m = mkt.reindex(c.index).ffill()
        mk_weak = (m < m.rolling(50).mean()).values

    def stats(evs):
        if len(evs) < 5:
            return None
        res = {}
        for hz in (5, 20, 60):
            r = np.array([c.iloc[i + hz] / c.iloc[i] - 1 for i in evs])
            res[str(hz)] = [sig(float(np.median(r)), 3), round(float((r > 0).mean()) * 100)]
        mae = np.array([min(0.0, float(l.iloc[i + 1:i + 21].min() / c.iloc[i] - 1)) for i in evs])
        res["mae"] = [sig(float(np.median(mae)), 3), sig(float(mae.min()), 3)]
        res["n"] = len(evs)
        return res
    out = {"kind": kind, "all": stats(ev)}
    if mk_weak is not None:
        out["weak"] = stats([i for i in ev if mk_weak[i]])
    if not out["all"]:
        out["t"] = "Zu wenige vergleichbare Lagen in den letzten %d Jahren (mindestens 5 nötig)." % years
        return out
    a = out["all"]
    lab = {"überverkauft": "so stark überverkauft wie heute", "überkauft": "so stark überkauft wie heute",
           "ähnlich": "in einer ähnlichen Lage wie heute"}[kind]
    t = ("In den letzten %d Jahren war die Aktie %d-mal %s. 20 Handelstage später stand der Kurs im Mittel (Median) %s, "
         "in %d %% der Fälle im Plus. Zwischendurch fiel der Kurs innerhalb dieser 20 Tage im Mittel noch um %s %%, "
         "im schlechtesten Fall um %s %%." % (years, a["n"], lab, _sp(a["20"][0]), a["20"][1],
                                               de(abs(a["mae"][0] or 0) * 100, 1), de(abs(a["mae"][1] or 0) * 100, 1)))
    if out.get("weak"):
        w = out["weak"]
        t += " Nur Fälle, in denen auch der Markt schwach war (%d): nach 20 Tagen %s, %d %% im Plus, zwischendurch im Mittel noch %s %% tiefer." % (
            w["n"], _sp(w["20"][0]), w["20"][1], de(abs(w["mae"][0] or 0) * 100, 1))
    t += " Vergangenheit, kleine Fallzahlen, keine Vorhersage."
    out["t"] = t
    return out


# ------------------------------------------------------------------------------------------------ Fazit Markt × Aktie
def matrix_text(mcode, stock_state, beta=None, rel_strong=False):
    """stock_state: 'boden' (Wende-Fortschritt ≥ 2), 'faellt', 'trend' (Aufwärtstrend/Rücksetzer)."""
    g = mcode in ("green",)
    y = mcode in ("yellow", "gray", "orange", "blue")
    r = mcode == "red"
    T = {
        ("g", "boden"): "Gute Ausgangslage: die Aktie dreht, der Markt hilft.",
        ("g", "faellt"): "Die Schwäche liegt an der Aktie selbst, der Markt ist nicht der Grund.",
        ("g", "trend"): "Rücksetzer im starken Umfeld: häufig eine Gelegenheit.",
        ("y", "boden"): "Die Aktie zeigt erste Stärke, der Markt kann sie aber noch einmal mitziehen.",
        ("y", "faellt"): "Gegenwind von beiden Seiten. Ein Boden der Aktie vor einem Boden des Marktes ist unwahrscheinlich" +
                         (", vor allem bei dieser hohen Marktabhängigkeit." if beta and beta >= 1.3 else "."),
        ("y", "trend"): "Die Aktie hält sich gegen den schwächeren Markt: relative Stärke.",
        ("r", "boden"): "Frühes Zeichen in einer Stressphase. Tiefs sind erst im Nachhinein sicher.",
        ("r", "faellt"): "Ausverkauf: abwarten, bis der Markt Bodensignale zeigt (siehe Reiter Markt).",
        ("r", "trend"): "Selten: die Aktie trotzt dem Ausverkauf.",
    }
    k = "g" if g else "r" if r else "y"
    t = T.get((k, stock_state), "")
    if rel_strong and stock_state != "trend":
        t += " Pluspunkt: die Aktie hält sich relativ zum Markt gut."
    return t


def extend(df, mkt=None, sec=None, sym=None, quality=None, mcode=None, mname="S&P 500", sname=None, live=False):
    """Alle Erweiterungen für eine Aktie. Gibt (dict, RS-Linie) zurück; einzelne Teile fehlen, wenn Daten fehlen."""
    c, h, l, o, v = prep(df)
    if len(c) < 130:
        return None, None
    if live and len(c) > 130:                       # laufender Tag: Volumen für Kapitulationserkennung unvollständig
        v = v.copy() if v is not None else None
        if v is not None:
            v.iloc[-1] = np.nan
            v = v.ffill()
    out = {}
    ml = None
    try:
        ml = market_link(c, mkt, sec, mcode, mname, sname)
    except Exception:
        ml = None
    rsl = None
    if ml:
        rsl = ml.pop("rsl", None)
        out["mk"] = ml
    for k, fn in (("ex", lambda: exhaustion(c, h, l, o, v, sym)), ("dv", lambda: divergences(c, h, l, v)),
                  ("tl", lambda: ladder(c, h, l, mkt)), ("pb", lambda: pullback(c, h, l, v, quality)),
                  ("an", lambda: analogs(c, h, l, mkt))):
        try:
            r = fn()
            if r:
                out[k] = r
        except Exception:
            pass
    # Fazit Markt × Aktie
    tl, pb = out.get("tl"), out.get("pb")
    state = "trend" if not tl else ("boden" if tl["prog"] >= 2 else "faellt")
    if mcode:
        bb = (ml or {}).get("b250") or (ml or {}).get("b60")
        rel = bool(ml and ml["rs"]["20"][2] is not None and ml["rs"]["20"][2] > 0.03)
        out["mx"] = {"s": state, "t": matrix_text(mcode, state, bb, rel)}
    return out, rsl
