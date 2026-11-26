"""Paket 7: Prognosen im Echtbetrieb messen (nur Anzeige, eigenes Skript, jeder Fehler wird abgefangen).

Einmal im Monat (erster täglicher Lauf eines Monats) werden die Vorhersagen aller Aktien festgehalten:
Prognosemodell (20/60/120 Tage, „schlägt die mittlere Aktie derselben Region“), Rückschlag-Risiko (60 Tage, mehr als
15 % zwischendurch) und Simulation (zuerst Widerstand / zuerst Unterstützung, über dem heutigen Kurs nach 60 Tagen).
Sind 20/60/120 Handelstage vergangen, wird mit den echten Kursen gemessen, ob es so kam. Gemessen wird auch, wenn das
Modell die Rückrechnung nicht bestanden hat: Das ist die Prüfung, die sich nicht schönrechnen lässt.

Dateien: data/prog_track.json (Momentaufnahmen und Einzelergebnisse, nur Zahlen, höchstens 3 Jahre),
docs/progtrack.json (kleine Auswertung für die Seite).
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HZ = (20, 60, 120)
DD_LIM = 0.15
KEEP_DAYS = 3 * 365
_S = {"root": None, "st": None, "obs": {}, "dirty": False}


def _empty():
    return {"v": 1, "snap": [], "out": {"20": [], "60": [], "120": [], "dd60": [], "sim": []}}


def load(root):
    _S["root"], _S["obs"], _S["dirty"] = root, {}, False
    p = os.path.join(root, "data", "prog_track.json")
    try:
        _S["st"] = json.load(open(p)) if os.path.exists(p) else _empty()
    except Exception:
        _S["st"] = None                            # beschädigt: nichts überschreiben
    return _S["st"]


def observe(sym, df):
    """Im täglichen Lauf je Aktie: Kurse ab dem Tag einer offenen Momentaufnahme merken."""
    st = _S["st"]
    if not st or not st.get("snap") or df is None or not len(df):
        return
    need = [sn for sn in st["snap"] if sym in sn.get("x", {}) and len(sn.get("done", [])) < len(HZ)]
    if not need:
        return
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce").values.astype(float)
    lo = pd.to_numeric(df["Low"], errors="coerce").values.astype(float) if "Low" in df else c
    for sn in need:
        i0 = int(np.searchsorted(idx.values, np.datetime64(sn["d"]), side="right")) - 1
        if i0 < 0 or not np.isfinite(c[i0]):
            continue
        _S["obs"][(sn["d"], sym)] = (c[i0:i0 + max(HZ) + 1], lo[i0 + 1:i0 + 61])


def auc(y, p):
    """Trennschärfe ohne scikit-learn (Rangsummen, Gleichstände gemittelt)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _measure(st, today):
    """Fällige Zeiträume messen. Fällig = mindestens 80 % der beobachteten Aktien haben genug Handelstage danach."""
    for sn in st["snap"]:
        done = set(sn.get("done", []))
        obs = {s: _S["obs"][(sn["d"], s)] for s in sn["x"] if (sn["d"], s) in _S["obs"]}
        if len(obs) < max(20, 0.5 * len(sn["x"])):
            continue
        for h in HZ:
            if str(h) in done:
                continue
            ok_ = [s for s, (c, lo) in obs.items() if len(c) > h and np.isfinite(c[h])]
            if len(ok_) < 0.8 * len(obs):
                continue
            r = {s: float(obs[s][0][h] / obs[s][0][0] - 1) for s in ok_}
            rg = {s: sn["x"][s][1] or "alle" for s in ok_}
            med = pd.Series(r).groupby(pd.Series(rg)).median().to_dict()
            hk = {20: 2, 60: 3, 120: 4}[h]
            for s in ok_:
                v = sn["x"][s]
                if v[hk] is not None:
                    st["out"][str(h)].append([sn["d"], s, rg[s], v[hk], round(r[s], 4), round(r[s] - med[rg[s]], 4)])
            if h == 60:
                for s in ok_:
                    v = sn["x"][s]
                    c, lo = obs[s]
                    lw = lo[:60][np.isfinite(lo[:60])]
                    mae = float(lw.min() / c[0] - 1) if len(lw) else None
                    if v[5] is not None and mae is not None:
                        st["out"]["dd60"].append([sn["d"], s, v[5], round(mae, 4)])
                    if v[6] is not None and v[8] and v[9]:
                        fut = c[1:61]
                        hu, hd = np.where(fut >= v[8])[0], np.where(fut <= v[9])[0]
                        tu, td = (hu[0] if len(hu) else 99), (hd[0] if len(hd) else 99)
                        hit = "u" if tu < td else "d" if td < tu else "n"
                        st["out"]["sim"].append([sn["d"], s, v[6], v[7], v[10], hit, int(c[60] > c[0])])
            done.add(str(h))
            sn["done"] = sorted(done)
            _S["dirty"] = True
            print("Messung der Prognosen: Momentaufnahme", sn["d"], "nach", h, "Tagen gemessen,", len(ok_), "Aktien", flush=True)
    # fertige Momentaufnahmen entfernen, alte Ergebnisse begrenzen
    n0 = len(st["snap"])
    st["snap"] = [sn for sn in st["snap"] if len(sn.get("done", [])) < len(HZ)]
    lim = (pd.Timestamp(today) - pd.Timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    for k in st["out"]:
        m0 = len(st["out"][k])
        st["out"][k] = [r for r in st["out"][k] if r[0] >= lim]
        _S["dirty"] |= len(st["out"][k]) != m0
    _S["dirty"] |= len(st["snap"]) != n0


def _snapshot(st, prog, ctx, today):
    if any(sn["d"][:7] == today[:7] for sn in st["snap"]) or st.get("last", "")[:7] == today[:7]:
        return
    rows = (ctx or {}).get("rows") or {}
    sim = (ctx or {}).get("sim") or {}
    x = {}
    for s, v in prog.items():
        pm, mc = v.get("pm") or {}, v.get("mc") or {}
        g = lambda k: (pm.get(k) or [None])[0]  # noqa: E731
        sm = sim.get(s)
        c0 = float(sm[0][-1]) if sm is not None and len(sm[0]) else None
        up, dn = (float(sm[1]) if sm and sm[1] else None), (float(sm[2]) if sm and sm[2] else None)
        rec = [c0, (rows.get(s) or {}).get("rg"), g("20"), g("60"), g("120"), g("dd60"),
               mc.get("pu"), mc.get("pd"), up, dn, mc.get("p60")]
        if any(r is not None for r in rec[2:8]):
            x[s] = [round(r, 4) if isinstance(r, float) else r for r in rec]
    if len(x) < 50:
        return
    M = (ctx or {}).get("mdl") or {}
    st["snap"].append({"d": today, "x": x, "done": [],
                       "ok": {h: bool(m.get("ok")) for h, m in (M.get("hz") or {}).items()},
                       "sim_use": bool((((ctx or {}).get("bt") or {}).get("sim") or {}).get("use"))})
    st["last"] = today
    st.setdefault("since", today)
    _S["dirty"] = True
    print("Messung der Prognosen: neue Momentaufnahme", today, len(x), "Aktien", flush=True)


def summary(st, today):
    out, res = st["out"], {"v": 1, "ts": today, "since": st.get("since"), "open": [], "hz": {}}
    for sn in st["snap"]:
        for h in HZ:
            if str(h) not in sn.get("done", []):
                eta = (pd.Timestamp(sn["d"]) + pd.offsets.BDay(h + 1)).strftime("%Y-%m-%d")
                res["open"].append([sn["d"], h, eta])
    for h in HZ:
        R = out[str(h)]
        if not R:
            continue
        p = np.array([r[3] for r in R], float)
        xx = np.array([r[5] for r in R], float)
        y = (xx > 0).astype(int)
        q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).values if len(R) >= 50 else None
        res["hz"][str(h)] = {"n": len(R), "m": len(set(r[0] for r in R)), "auc": None if auc(y, p) is None else round(auc(y, p), 3),
                             "pm": round(float(p.mean()), 1), "ist": round(float(y.mean()) * 100, 1),
                             "top": None if q is None else round(float(y[q == 9].mean()) * 100, 1),
                             "bot": None if q is None else round(float(y[q == 0].mean()) * 100, 1)}
    if out["dd60"]:
        R = out["dd60"]
        p = np.array([r[2] for r in R], float)
        y = (np.array([r[3] for r in R], float) < -DD_LIM).astype(int)
        cal = []
        for lo_, hi_ in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 101)):
            m = (p >= lo_) & (p < hi_)
            if m.sum() >= 20:
                cal.append([lo_, hi_, int(m.sum()), round(float(p[m].mean()), 1), round(float(y[m].mean()) * 100, 1)])
        res["dd60"] = {"n": len(R), "auc": None if auc(y, p) is None else round(auc(y, p), 3), "pm": round(float(p.mean()), 1),
                       "ist": round(float(y.mean()) * 100, 1), "cal": cal}
    if out["sim"]:
        R = out["sim"]
        res["sim"] = {"n": len(R),
                      "pu": [round(float(np.mean([r[2] for r in R])), 1), round(float(np.mean([r[5] == "u" for r in R])) * 100, 1)],
                      "pd": [round(float(np.mean([r[3] for r in R])), 1), round(float(np.mean([r[5] == "d" for r in R])) * 100, 1)],
                      "q60": [round(float(np.mean([r[4] for r in R if r[4] is not None] or [np.nan])), 1),
                              round(float(np.mean([r[6] for r in R])) * 100, 1)]}
    return res


def run(prog, ctx=None, today=None):
    st, root = _S["st"], _S["root"]
    if st is None or not root:
        return None
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _measure(st, today)
    _snapshot(st, prog or {}, ctx, today)
    sp = os.path.join(root, "docs", "progtrack.json")
    if _S["dirty"] or not os.path.exists(sp):
        with open(os.path.join(root, "data", "prog_track.json"), "w") as f:
            json.dump(st, f, separators=(",", ":"), ensure_ascii=False)
        with open(sp, "w") as f:
            json.dump(summary(st, today), f, separators=(",", ":"), ensure_ascii=False)
        _S["dirty"] = False
    return st
