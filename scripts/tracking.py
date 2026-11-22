"""Trefferquote: Tagesstände, Fälle und Zusammenfassungen.

Prinzip
- Jeder Kalendertag bekommt höchstens einen Tagesstand (data/track/JJJJ-MM.json). Darin je Aktie die Einordnung,
  die Scores und ein eigener Gesamtrendite-Index „tri“ in Börsenwährung. tri wird von Tag zu Tag mit dem
  dividendenbereinigten Schlusskurs (Adjusted Close) aus derselben, aktuellen Kursreihe fortgeschrieben. So bleiben
  Dividenden enthalten, ohne dass sich alte Werte durch spätere Bereinigungen verschieben.
- Jeder Tagesstand enthält die Wechselkurse (Euro je Währungseinheit). Renditen in Euro ergeben sich aus
  tri-Verhältnis mal Wechselkursverhältnis.
- Ein Fall entsteht, wenn eine Aktie neu in eine verfolgte Kategorie kommt. Seine Ergebnisse (Rendite nach 7, 30,
  91, 182 und 365 Tagen, dazu MSCI World und Durchschnitt aller Aktien) werden gefüllt, sobald der Horizont erreicht
  ist, unabhängig davon, ob die Aktie die Kategorie schon verlassen hat.
- Zusammenfassungen je Eintrittsmonat werden eingefroren, sobald alle Fälle des Monats ihren 12-Monats-Wert haben.
  Einzelfälle werden 24 Monate nach Eintritt gelöscht, aber nur, wenn ihr Monat eingefroren ist.
"""
import json
import math
import os
from datetime import datetime, timedelta

SCHEMA = 2
HORIZONS = [7, 30, 91, 182, 365]
TRACKED = ["Anlauf", "Trendstark", "Qualitäts-Dip", "Turnaround-Kandidat", "Qualität im Rücksetzer", "Überhitzt",
           "Schwach"]
TOP_CATS = [["Rangliste Top 3", 3], ["Rangliste Top 20", 20]]
SU_CODES = ["Neutral", "Anlauf", "Trendstark", "Qualitäts-Dip", "Turnaround-Kandidat", "Qualität im Rücksetzer",
            "Überhitzt", "Schwach"]
COOLDOWN_DAYS = 30
CASE_KEEP_MONTHS = 24
TRACK_KEEP_MONTHS = 14          # Tagesstände: reicht für 365 Tage plus Puffer
LATE_DAYS = 30                  # so lange nach dem Horizont wird auf einen Kurs gewartet, dann „nicht messbar“
DIMS = ["kl", "ph", "ta", "cd", "sg", "t3", "rg", "sc"]   # Unterkategorien, je einzeln ausgewertet

# Indizes in einem Tagesstand je Aktie
I_SU, I_TOT, I_TRI, I_F, I_M, I_T, I_R, I_A, I_N, I_PH, I_D52, I_SG = range(12)


def d2s(d):
    return d.strftime("%Y-%m-%d")


def s2d(s):
    return datetime.strptime(s, "%Y-%m-%d")


def sig5(v):
    return float("%.6g" % v) if v is not None else None


# ---------------------------------------------------------------------------------------------- Dateien
def load_days(track_dir, legacy_path=None):
    """Alle Tagesstände, ältester zuerst. Übernimmt einmalig die alte data/track.json (Format [Kurs, Einordnung,
    Gesamtscore]); dort fehlen tri und Wechselkurse, sie werden beim nächsten Lauf rückwirkend auf 100 gesetzt."""
    days = []
    if os.path.isdir(track_dir):
        for fn in sorted(os.listdir(track_dir)):
            if fn.endswith(".json") and fn[:4].isdigit():
                days += json.load(open(os.path.join(track_dir, fn))).get("days", [])
    if not days and legacy_path and os.path.exists(legacy_path):
        for d in json.load(open(legacy_path)).get("days", []):
            p = {}
            for s, v in d.get("p", {}).items():
                su = SU_CODES.index(v[1]) if v[1] in SU_CODES else 0
                p[s] = [su, v[2], None, None, None, None, None, None, None, None, None, None]
            days.append({"d": d["d"], "v": d.get("v"), "fx": {}, "mw": None, "p": p, "px": {s: v[0] for s, v in d["p"].items()}})
    days.sort(key=lambda d: d["d"])
    return days


def save_days(track_dir, days, today):
    """Schreibt nur die Monatsdatei, die sich geändert hat, und entfernt Monate außerhalb der Aufbewahrung."""
    os.makedirs(track_dir, exist_ok=True)
    keep_from = (s2d(today).replace(day=1) - timedelta(days=31 * (TRACK_KEEP_MONTHS - 1))).strftime("%Y-%m")
    months = sorted({d["d"][:7] for d in days})
    for m in months:
        if m < keep_from:
            continue
        rows = [d for d in days if d["d"][:7] == m]
        path = os.path.join(track_dir, m + ".json")
        txt = json.dumps({"schema": SCHEMA, "days": rows}, ensure_ascii=False, separators=(",", ":"))
        if not os.path.exists(path) or open(path).read() != txt:
            with open(path, "w") as f:
                f.write(txt)
    for fn in os.listdir(track_dir):
        if fn.endswith(".json") and fn[:4].isdigit() and fn[:7] < keep_from:
            os.remove(os.path.join(track_dir, fn))
    return [d for d in days if d["d"][:7] >= keep_from]


# ---------------------------------------------------------------------------------------------- Tagesstand
def make_day(today, version, stocks, adj, prev, fx, mw_adj, extra_adj=None):
    """Neuer Tagesstand.
    stocks: Datensätze aus scoring (mit s, su, tot, f, m, t, rs, a, ns, ph, d52, hot, tu, D)
    adj: {Kürzel: Funktion(Datum) -> Adjusted Close am Datum oder davor, None wenn unbekannt}
    prev: letzter Tagesstand oder None; fx: {Währung: Euro je Einheit}; mw_adj: wie adj für den MSCI-World-ETF
    extra_adj: Kurse für Aktien offener Fälle, die nicht mehr im Universum sind."""
    def roll(sym, pv, fn):
        """tri fortschreiben. Regel: gemessen wird immer der Schlusskurs (Adjusted Close) des Vortags, weil der
        tägliche Messlauf früh am Morgen läuft und der heutige Kurs dann noch fehlt oder vorläufig ist."""
        if fn is None:
            return None
        a_now = fn(day_before(today))
        if a_now is None or a_now <= 0:
            return None
        if prev is None or pv is None:
            return 100.0
        a_prev = fn(day_before(prev["d"]))
        if a_prev is None or a_prev <= 0:
            return None
        return sig5(pv * a_now / a_prev)

    def prev_tri(sym, store="p"):
        if prev is None:
            return None
        row = (prev.get(store) or {}).get(sym)
        if row is None:
            return None
        v = row[I_TRI] if store == "p" else row
        return 100.0 if v is None else v          # alter Stand ohne tri: rückwirkend 100

    p = {}
    for x in stocks:
        if x.get("tot") is None:
            continue
        su = SU_CODES.index(x["su"]) if x.get("su") in SU_CODES else 0
        tri = roll(x["s"], prev_tri(x["s"]) if prev and x["s"] in prev.get("p", {}) else None, adj.get(x["s"]))
        ph = x["ph"][0] if x.get("ph") else None
        sg = signals(x)
        p[x["s"]] = [su, round(x["tot"]), tri, _r(x.get("f")), _r(x.get("m")), _r(x.get("t")), _r(x.get("rs")),
                     _r(x.get("a")), _r(x.get("ns")), ph, _r(x.get("d52")), sg]
    o = {}
    for s, fn in (extra_adj or {}).items():
        pv = None
        if prev is not None:
            pv = (prev.get("o") or {}).get(s)
            if pv is None and s in prev.get("p", {}):
                pv = prev_tri(s)
        t = roll(s, pv, fn)
        if t is not None:
            o[s] = t
    mw_prev = None if prev is None else (prev.get("mw") if prev.get("mw") is not None else 100.0)
    hz = {x["s"]: [_r(v) for v in x["hz"]] for x in stocks if x.get("hz") and x["s"] in p}
    day = {"d": today, "v": version, "tx": 1, "fx": {k: sig5(v) for k, v in fx.items()},
           "mw": roll("_mw", mw_prev, mw_adj), "p": p, "o": o}
    if hz:
        day["hz"] = hz                              # Score je Zeitraum [3 Monate, 2 Jahre+]
    return day


def day_before(s):
    return d2s(s2d(s) - timedelta(days=1))


def repair_tri(days, adj, mw_adj, extra_adj=None):
    """Einmalige Korrektur (20.11.2026): Bis dahin verglich der Messlauf den letzten verfügbaren Kurs mit dem Kurs am
    Datum des Vortags; früh am Morgen ist das derselbe Schlusskurs, tri blieb deshalb bei 100. Hier wird tri für alle
    gespeicherten Tage nach der neuen Regel (Schlusskurs des Vortags) neu verkettet. Gibt die Zahl der Kürzel zurück."""
    fns = dict(extra_adj or {})
    fns.update(adj or {})
    last, mw_last, fixed = {}, None, set()
    for d in days:
        ds = day_before(d["d"])
        for store in ("p", "o"):
            rows = d.get(store) or {}
            for sym in list(rows):
                fn = fns.get(sym)
                if fn is None:
                    continue
                a = fn(ds)
                if a is None or a <= 0:
                    continue
                tri = sig5(last[sym][0] * a / last[sym][1]) if sym in last else 100.0
                last[sym] = (tri, a)
                fixed.add(sym)
                if store == "p":
                    rows[sym][I_TRI] = tri
                else:
                    rows[sym] = tri
        if mw_adj is not None:
            a = mw_adj(ds)
            if a:
                mw = 100.0 if mw_last is None else sig5(mw_last[0] * a / mw_last[1])
                d["mw"], mw_last = mw, (mw, a)
        d["tx"] = 1
    return len(fixed)


def _r(v):
    return None if v is None else round(float(v), 1)


def signals(x):
    su = x.get("su")
    if su == "Turnaround-Kandidat" and x.get("tu") is not None:
        return x["tu"]
    if su == "Qualitäts-Dip" and x.get("D"):
        return sum(1 for c in x["D"] if c[2])
    return x.get("hot")


# ---------------------------------------------------------------------------------------------- Renditen
def eur_ratio(day_a, day_b, sym, cur):
    """Gesamtrendite in Euro zwischen zwei Tagesständen als Faktor, oder None."""
    ta = _tri(day_a, sym)
    tb = _tri(day_b, sym)
    if ta is None or tb is None or ta <= 0:
        return None
    fa, fb = (day_a.get("fx") or {}).get(cur), (day_b.get("fx") or {}).get(cur)
    if cur == "EUR":
        fa = fb = 1.0
    if not fa or not fb:
        return None
    return tb / ta * fb / fa


def _tri(day, sym):
    row = day["p"].get(sym)
    if row is not None and row[I_TRI] is not None:
        return row[I_TRI]
    return (day.get("o") or {}).get(sym)


def mw_ratio(day_a, day_b):
    a, b = day_a.get("mw"), day_b.get("mw")
    fa, fb = (day_a.get("fx") or {}).get("USD"), (day_b.get("fx") or {}).get("USD")
    if not a or not b or not fa or not fb:
        return None
    return b / a * fb / fa


def avg_ratio(day_a, day_b, curs):
    """Gleichgewichteter Durchschnitt aller Aktien des Universums, die an beiden Tagen einen Wert haben."""
    vals = []
    for s in day_a["p"]:
        if s in day_b["p"]:
            r = eur_ratio(day_a, day_b, s, curs.get(s))
            if r is not None and 0.1 < r < 6:
                vals.append(r)
    return sum(vals) / len(vals) if len(vals) >= 50 else None


def day_at_or_after(days, idx_by_date, target):
    """Index des ersten Tagesstands am Zieltag oder danach."""
    for k in range(0, 15):
        d = d2s(s2d(target) + timedelta(days=k))
        if d in idx_by_date:
            return idx_by_date[d]
    return None


# ---------------------------------------------------------------------------------------------- Fälle
def memberships(day, top_n_list=TOP_CATS):
    """Kategorien je Aktie an einem Tagesstand, inkl. Rangliste Top N nach Gesamtscore."""
    m = {}
    for s, row in day["p"].items():
        su = SU_CODES[row[I_SU]] if row[I_SU] is not None and row[I_SU] < len(SU_CODES) else "Neutral"
        if su in TRACKED:
            m.setdefault(s, set()).add(su)
    ranked = sorted((s for s, r in day["p"].items() if r[I_TOT] is not None), key=lambda s: -day["p"][s][I_TOT])
    for name, n in top_n_list:
        for s in ranked[:n]:
            m.setdefault(s, set()).add(name)
    return m, {s: i + 1 for i, s in enumerate(ranked)}


def top3_in_cat(day, cat, sym):
    if cat.startswith("Rangliste"):
        return None
    code = SU_CODES.index(cat)
    members = sorted((s for s, r in day["p"].items() if r[I_SU] == code), key=lambda s: -day["p"][s][I_TOT])
    return 1 if sym in members[:3] else 0


def update_cases(cases, day, info, start=False):
    """Eintritte und Austritte für einen Tagesstand. cases: Liste von Fällen (wird verändert).
    info: {Kürzel: {cur, rg, sc, px(Kurs Börsenwährung), ta, cd}} aus dem aktuellen Lauf.
    start=True: Startbestand (alle aktuellen Mitglieder werden als Fall mit st=1 angelegt)."""
    today = day["d"]
    mem, rank = memberships(day)
    open_by = {(c["s"], c["k"]): c for c in cases if c["x"] is None}
    closed_by = {}
    for c in cases:
        if c["x"] is not None:
            key = (c["s"], c["k"])
            if key not in closed_by or c["x"] > closed_by[key]["x"]:
                closed_by[key] = c
    next_id = max([c["i"] for c in cases], default=0) + 1
    new = 0
    for s, cats in mem.items():
        for k in cats:
            if (s, k) in open_by:
                continue
            old = closed_by.get((s, k))
            if old is not None and (s2d(today) - s2d(old["x"])).days <= COOLDOWN_DAYS:
                old["x"] = None                       # Rückkehr innerhalb der Sperrfrist: alter Fall läuft weiter
                continue
            row, inf = day["p"][s], info.get(s, {})
            fx = (day.get("fx") or {}).get(inf.get("cur"))
            if inf.get("cur") == "EUR":
                fx = 1.0
            px = inf.get("px")
            cases.append({
                "i": next_id, "s": s, "d": today, "k": k, "v": day.get("v"), "st": 1 if start else 0,
                "t": row[I_TOT], "sc": row[I_F:I_N + 1], "rk": rank.get(s),
                "pe": sig5(px * fx) if px and fx else None, "x": None, "cur": inf.get("cur"),
                "m": {"ph": row[I_PH] if k in ("Turnaround-Kandidat", "Qualität im Rücksetzer", "Qualitäts-Dip") else None,
                      "ta": inf.get("ta"), "cd": inf.get("cd"), "sg": row[I_SG], "t3": top3_in_cat(day, k, s),
                      "rg": inf.get("rg"), "sc": inf.get("sc"), "kl": inf.get("kl")},
                "r": {}})
            next_id += 1
            new += 1
    for (s, k), c in open_by.items():
        if k not in mem.get(s, set()):
            c["x"] = today
    return new


def fill_results(cases, days, curs, today):
    """Füllt fällige Horizonte. Ergebnis je Horizont: [Rendite %, MSCI World %, Durchschnitt aller Aktien %] in Euro,
    oder "na", wenn LATE_DAYS nach dem Horizont noch immer kein Kurs vorliegt."""
    idx = {d["d"]: i for i, d in enumerate(days)}
    cache = {}
    filled = 0
    for c in cases:
        if c["d"] not in idx:
            continue
        a = days[idx[c["d"]]]
        for h in HORIZONS:
            key = str(h)
            if key in c["r"]:
                continue
            target = d2s(s2d(c["d"]) + timedelta(days=h))
            if target > today:
                break
            j = day_at_or_after(days, idx, target)
            res = None
            if j is not None:
                b = days[j]
                r = eur_ratio(a, b, c["s"], c.get("cur") or curs.get(c["s"]))
                if r is not None:
                    ck = (a["d"], b["d"])
                    if ck not in cache:
                        cache[ck] = (mw_ratio(a, b), avg_ratio(a, b, curs))
                    mw, av = cache[ck]
                    res = [round((r - 1) * 100, 1), None if mw is None else round((mw - 1) * 100, 1),
                           None if av is None else round((av - 1) * 100, 1)]
            if res is None and (s2d(today) - s2d(target)).days > LATE_DAYS:
                res = "na"
            if res is not None:
                c["r"][key] = res
                filled += 1
    return filled


# ---------------------------------------------------------------------------------------------- Zusammenfassung
def _stats(rows):
    """rows: Liste (Rendite, MSCI, Durchschnitt, Kürzel). Werte: [n, Ø, Median, Ø Überrendite MSCI,
    Anteil > MSCI, Anteil > Durchschnitt, bester, schlechtester (je [Kürzel, Rendite])]"""
    n = len(rows)
    if not n:
        return None
    rets = sorted(r[0] for r in rows)
    med = rets[n // 2] if n % 2 else (rets[n // 2 - 1] + rets[n // 2]) / 2
    mw = [r for r in rows if r[1] is not None]
    av = [r for r in rows if r[2] is not None]
    best = max(rows, key=lambda r: r[0])
    worst = min(rows, key=lambda r: r[0])
    return [n, round(sum(rets) / n, 2), round(med, 2),
            round(sum(r[0] - r[1] for r in mw) / len(mw), 2) if mw else None,
            round(100 * sum(1 for r in mw if r[0] > r[1]) / len(mw), 1) if mw else None,
            round(100 * sum(1 for r in av if r[0] > r[2]) / len(av), 1) if av else None,
            [best[3], best[0]], [worst[3], worst[0]],
            round(sum(r[1] for r in mw) / len(mw), 2) if mw else None,
            round(sum(r[2] for r in av) / len(av), 2) if av else None]


def _short(rows):
    """Kurzform für Unterkategorien: [n, Ø, Median, Anteil > MSCI, Anteil > Durchschnitt]."""
    st = _stats(rows)
    return None if st is None else [st[0], round(st[1], 1), round(st[2], 1), st[4], st[5]]


def group_stats(cases, dims=True):
    """{Kategorie: {"alle": {Horizont: Werte}, Dimension: {Wert: {Horizont: Kurzwerte}}}}."""
    acc = {}
    for c in cases:
        for h in HORIZONS:
            r = c["r"].get(str(h))
            if not isinstance(r, list):
                continue
            row = (r[0], r[1], r[2], c["s"])
            g = acc.setdefault(c["k"], {})
            g.setdefault("alle", {}).setdefault(str(h), []).append(row)
            if dims:
                for dm in DIMS:
                    v = c["m"].get(dm)
                    if v is not None:
                        g.setdefault(dm, {}).setdefault(str(v), {}).setdefault(str(h), []).append(row)
    out = {}
    for k, g in acc.items():
        out[k] = {"alle": {h: _stats(rows) for h, rows in g["alle"].items()}}
        for dm in DIMS:
            if dm in g:
                out[k][dm] = {v: {h: _short(rows) for h, rows in hs.items()} for v, hs in g[dm].items()}
    return out


def merge_stats(parts):
    """Fügt eingefrorene Monatswerte zusammen (für Monate, deren Einzelfälle schon gelöscht sind). Der Median ist
    dann der nach n gewichtete Median der Monatsmediane und als Näherung markiert."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    n = sum(p[0] for p in parts)

    def wavg(i):
        xs = [(p[i], p[0]) for p in parts if p[i] is not None]
        w = sum(b for _, b in xs)
        return round(sum(a * b for a, b in xs) / w, 2) if w else None
    meds = sorted((p[2], p[0]) for p in parts)
    acc, med = 0, meds[-1][0]
    for v, w in meds:
        acc += w
        if acc >= n / 2:
            med = v
            break
    best = max((p[6] for p in parts), key=lambda b: b[1])
    worst = min((p[7] for p in parts), key=lambda b: b[1])
    return [n, wavg(1), med, wavg(3), wavg(4), wavg(5), best, worst, wavg(8), wavg(9), "≈"]


def spearman(xs, ys):
    n = len(xs)
    if n < 30:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def update_corr(corr, days, curs):
    """Wöchentliche Rangkorrelation zwischen Gesamtscore und späterer Rendite (30 und 91 Tage)."""
    done = {(c[0], c[1]) for c in corr}
    idx = {d["d"]: i for i, d in enumerate(days)}
    seen = set()
    for i, d in enumerate(days):
        wk = s2d(d["d"]).isocalendar()[:2]
        if wk in seen:
            continue
        seen.add(wk)
        for h in (30, 91):
            if (d["d"], h) in done:
                continue
            j = day_at_or_after(days, idx, d2s(s2d(d["d"]) + timedelta(days=h)))
            if j is None:
                continue
            xs, ys = [], []
            for s, row in d["p"].items():
                if row[I_TOT] is None or s not in days[j]["p"]:
                    continue
                r = eur_ratio(d, days[j], s, curs.get(s))
                if r is not None and 0.1 < r < 6:
                    xs.append(row[I_TOT])
                    ys.append(r)
            rho = spearman(xs, ys)
            if rho is not None:
                corr.append([d["d"], h, rho, len(xs)])
    corr.sort()
    return corr


HZ_CORR = [(0, "3m", 91), (1, "2j", 365)]        # Index in day["hz"], Schlüssel, Horizont in Tagen


def update_corr_hz(corr, days, curs):
    """Wie update_corr, aber für die Scores je Zeitraum: 3-Monats-Score gegen Rendite nach 91 Tagen,
    2-Jahres-Score gegen Rendite nach 365 Tagen. Einträge [Datum, Schlüssel, Horizont, rho, n]."""
    done = {(c[0], c[1]) for c in corr}
    idx = {d["d"]: i for i, d in enumerate(days)}
    seen = set()
    for d in days:
        if not d.get("hz"):
            continue
        wk = s2d(d["d"]).isocalendar()[:2]
        if wk in seen:
            continue
        seen.add(wk)
        for i, key, h in HZ_CORR:
            if (d["d"], key) in done:
                continue
            j = day_at_or_after(days, idx, d2s(s2d(d["d"]) + timedelta(days=h)))
            if j is None:
                continue
            xs, ys = [], []
            for s, v in d["hz"].items():
                if v[i] is None or s not in days[j]["p"]:
                    continue
                r = eur_ratio(d, days[j], s, curs.get(s))
                if r is not None and 0.1 < r < 6:
                    xs.append(v[i])
                    ys.append(r)
            rho = spearman(xs, ys)
            if rho is not None:
                corr.append([d["d"], key, h, rho, len(xs)])
    corr.sort()
    return corr


BUCKETS = [["80+", 80], ["65–80", 65], ["50–65", 50], ["unter 50", -1]]   # Score-Stufen (Gesamtscore)
BUCKET_H = (7, 30, 91)


def update_buckets(bk, days, curs):
    """Trefferquote nach Score-Stufen: wöchentlich, je Stufe die Rendite in Euro nach 7, 30 und 91 Tagen im Vergleich
    zum Durchschnitt aller Aktien. Einträge [Datum, Horizont, Regelwerk, [[Stufe, Ø Mehrrendite %, Median %, Anteil
    besser als Ø %, n], ...]]. Klein: ein Eintrag je Woche und Horizont."""
    done = {(b[0], b[1]) for b in bk}
    idx = {d["d"]: i for i, d in enumerate(days)}
    seen = set()
    for d in days:
        wk = s2d(d["d"]).isocalendar()[:2]
        if wk in seen:
            continue
        seen.add(wk)
        for h in BUCKET_H:
            if (d["d"], h) in done:
                continue
            j = day_at_or_after(days, idx, d2s(s2d(d["d"]) + timedelta(days=h)))
            if j is None:
                continue
            rows = []
            for s, row in d["p"].items():
                if row[I_TOT] is None or s not in days[j]["p"]:
                    continue
                r = eur_ratio(d, days[j], s, curs.get(s))
                if r is not None and 0.1 < r < 6:
                    rows.append((row[I_TOT], r))
            if len(rows) < 50:
                continue
            avg = sum(r for _, r in rows) / len(rows)
            out = []
            for name, lim in BUCKETS:
                hi = min((b[1] for b in BUCKETS if b[1] > lim), default=None)
                ex = sorted((r - avg) * 100 for t, r in rows if t >= lim and (hi is None or t < hi))
                if not ex:
                    out.append([name, None, None, None, 0])
                    continue
                med = ex[len(ex) // 2] if len(ex) % 2 else (ex[len(ex) // 2 - 1] + ex[len(ex) // 2]) / 2
                out.append([name, round(sum(ex) / len(ex), 2), round(med, 2),
                            round(100 * sum(1 for e in ex if e > 0) / len(ex)), len(ex)])
            bk.append([d["d"], h, d.get("v"), out])
    bk.sort(key=lambda b: (b[0], b[1]))
    return bk


MD_NAMES = [t[0] for t in TOP_CATS] + TRACKED      # Musterdepots je Kategorie


def update_models(md, days, curs):
    """Musterdepots: je Kategorie ein Index (Start 100), der täglich gleichgewichtet alle Aktien hält, die in der
    Kategorie stehen. Aufnahme mit einem Tag Verzögerung: Die Rendite von Tagesstand j-1 nach j gilt für die Mitglieder
    am Tagesstand j-2, damit nichts eingerechnet wird, was man beim Kauf noch nicht wissen konnte. Dazu MSCI World
    („_mw“) und der gleichgewichtete Durchschnitt aller Aktien („_av“), alles in Euro mit Dividenden.
    md: {"d": [Datum], "s": {Name: [Index]}, "nl": {Name: Anzahl im letzten Schritt}}. Wird nur fortgeschrieben."""
    names = MD_NAMES + ["_mw", "_av"]
    md = json.loads(json.dumps(md)) if md and md.get("d") else None
    if not days:
        return md
    if md is None:
        md = {"d": [days[0]["d"]], "s": {k: [100.0] for k in names}, "nl": {}}
    for k in names:                                  # neue Kategorie: ab dem aktuellen Stand mit 100 beginnen
        if k not in md["s"]:
            md["s"][k] = [None] * (len(md["d"]) - 1) + [100.0]
    idx = {d["d"]: i for i, d in enumerate(days)}
    i = idx.get(md["d"][-1])
    if i is None:
        return md
    for j in range(i + 1, len(days)):
        a, b = days[j - 1], days[j]
        mem = memberships(days[j - 2])[0] if j >= 2 else {}
        by_cat = {}
        for s, cats in mem.items():
            for k in cats:
                by_cat.setdefault(k, []).append(s)
        md["d"].append(b["d"])
        for k in MD_NAMES:
            vals = []
            for s in by_cat.get(k, []):
                r = eur_ratio(a, b, s, curs.get(s))
                if r is not None and 0.1 < r < 6:
                    vals.append(r)
            f = sum(vals) / len(vals) if vals else 1.0
            prev = md["s"][k][-1]
            md["s"][k].append(None if prev is None else round(prev * f, 3))
            md["nl"][k] = len(vals)
        for k, f in (("_mw", mw_ratio(a, b)), ("_av", avg_ratio(a, b, curs))):
            prev = md["s"][k][-1]
            md["s"][k].append(None if prev is None else round(prev * (f or 1.0), 3))
    return md


def open_cases(cases, days, curs, sectors):
    """Laufende Fälle (Aktie noch in der Kategorie): Ergebnis seit Eintritt bis zum letzten Tagesstand in Euro,
    MSCI World und Durchschnitt aller Aktien im selben Zeitraum, Durchschnitt der Aktien derselben Branche
    (mindestens 3) und größter Rückgang seit Eintritt (Tiefpunkt gegenüber dem Einstieg, höchstens 0).
    Einträge [Kürzel, Kategorie, Eintritt, Rendite %, MSCI %, Ø alle %, Ø Branche %, größter Rückgang %, Startbestand]."""
    if not days:
        return []
    idx = {d["d"]: i for i, d in enumerate(days)}
    last = days[-1]
    bench, out = {}, []
    for c in cases:
        if c["x"] is not None or c["d"] not in idx:
            continue
        i = idx[c["d"]]
        a = days[i]
        cur = c.get("cur") or curs.get(c["s"])
        r = eur_ratio(a, last, c["s"], cur)
        if r is None:
            continue
        if c["d"] not in bench:
            sec = {}
            for s in a["p"]:
                if s in last["p"] and sectors.get(s):
                    q = eur_ratio(a, last, s, curs.get(s))
                    if q is not None and 0.1 < q < 6:
                        sec.setdefault(sectors[s], []).append(q)
            mw, av = mw_ratio(a, last), avg_ratio(a, last, curs)
            bench[c["d"]] = (None if mw is None else round((mw - 1) * 100, 1),
                             None if av is None else round((av - 1) * 100, 1),
                             {k: round((sum(v) / len(v) - 1) * 100, 1) for k, v in sec.items() if len(v) >= 3})
        mw, av, sec = bench[c["d"]]
        low = 1.0
        for d in days[i + 1:]:
            q = eur_ratio(a, d, c["s"], cur)
            if q is not None and q < low:
                low = q
        sc = (c.get("m") or {}).get("sc") or sectors.get(c["s"])
        out.append([c["s"], c["k"], c["d"], round((r - 1) * 100, 1), mw, av, sec.get(sc), round((low - 1) * 100, 1),
                    c.get("st", 0)])
    out.sort(key=lambda o: (o[2], o[0]))
    return out


def build_summary(cases, old_summary, corr, today, since):
    """Monate × Kategorie × Dimension × Horizont, Zeitfenster „letzte 12 Monate“ und „seit Start“
    (je mit und ohne Startbestand), eingefrorene Monate bleiben unverändert."""
    frozen = dict((old_summary or {}).get("frozen", {}))
    by_month = {}
    for c in cases:
        by_month.setdefault(c["d"][:7], []).append(c)
    live = {}
    for m, cs in sorted(by_month.items()):
        if m in frozen:
            continue
        done = all(str(HORIZONS[-1]) in c["r"] for c in cs)
        e = [c for c in cs if not c["st"]]
        entry = {"n": len(cs), "a": group_stats(cs, dims=False)}
        if len(e) != len(cs):
            entry["e"] = group_stats(e, dims=False)
        if done:
            entry["fz"] = today
            frozen[m] = entry
        else:
            live[m] = entry
    win = {}
    l12 = d2s(s2d(today) - timedelta(days=365))
    have = {c["d"][:7] for c in cases}
    for wname, pick in (("l12", lambda c: c["d"] >= l12), ("all", lambda c: True)):
        sel = [c for c in cases if pick(c)]
        w = {"a": group_stats(sel)}
        if any(c["st"] for c in sel):
            w["e"] = group_stats([c for c in sel if not c["st"]])
        if wname == "all":                        # eingefrorene Monate ohne Einzelfälle dazurechnen
            gone = [frozen[m] for m in frozen if m not in have]
            if gone:
                for var in list(w):
                    w[var] = _merge_into(w[var], [g.get(var) or g["a"] for g in gone])
        win[wname] = w
    years = sorted({c["d"][:4] for c in cases} | set((old_summary or {}).get("years", [])))
    return {"schema": SCHEMA, "since": since, "updated": today, "horizons": HORIZONS,
            "cats": [t[0] for t in TOP_CATS] + TRACKED, "dims": DIMS, "years": years,
            "win": win, "months": live, "frozen": frozen, "corr": corr,
            "n_open": sum(1 for c in cases if c["x"] is None), "n_cases": len(cases)}


def _merge_into(base, extra):
    """Rechnet eingefrorene Monatswerte (nur Ebene „alle“) in ein Zeitfenster ein."""
    out = json.loads(json.dumps(base))
    for g in extra:
        for k, cat in (g or {}).items():
            for h, st in cat.get("alle", {}).items():
                cur = out.setdefault(k, {}).setdefault("alle", {}).get(h)
                out[k]["alle"][h] = merge_stats([cur, st])
    return out


def prune_cases(cases, summary, today):
    """Löscht Einzelfälle älter als CASE_KEEP_MONTHS, deren Monat eingefroren ist."""
    limit = d2s(s2d(today) - timedelta(days=int(CASE_KEEP_MONTHS * 30.44)))
    frozen = set((summary or {}).get("frozen", {}))
    return [c for c in cases if not (c["d"] < limit and c["d"][:7] in frozen)]


def save_cases(cases_dir, cases):
    os.makedirs(cases_dir, exist_ok=True)
    by_year = {}
    for c in cases:
        by_year.setdefault(c["d"][:4], []).append(c)
    for y, cs in by_year.items():
        path = os.path.join(cases_dir, "cases-%s.json" % y)
        txt = json.dumps({"schema": SCHEMA, "cases": cs}, ensure_ascii=False, separators=(",", ":"))
        if not os.path.exists(path) or open(path).read() != txt:
            with open(path, "w") as f:
                f.write(txt)
    for fn in os.listdir(cases_dir):
        if fn.startswith("cases-") and fn[6:10] not in by_year:
            os.remove(os.path.join(cases_dir, fn))


def load_cases(cases_dir):
    out = []
    if os.path.isdir(cases_dir):
        for fn in sorted(os.listdir(cases_dir)):
            if fn.startswith("cases-") and fn.endswith(".json"):
                out += json.load(open(os.path.join(cases_dir, fn))).get("cases", [])
    return out


def run(today, version, stocks, adj, fx, mw_adj, info, paths, extra_adj=None):
    """Ein Tracking-Schritt. Höchstens einmal je Kalendertag: Gibt es für heute schon einen Tagesstand, passiert
    nichts. Rückgabe: Kurzinfo für data.json oder None."""
    days = load_days(paths["track"], paths.get("legacy"))
    need_fix = any(not d.get("tx") for d in days)
    if days and days[-1]["d"] == today and not need_fix:
        return None
    cases = load_cases(paths["cases"])
    try:
        old_summary = json.load(open(paths["summary"]))
    except Exception:
        old_summary = None
    curs = dict(json.load(open(paths["meta"]))) if os.path.exists(paths["meta"]) else {}
    curs.update({s: v["cur"] for s, v in info.items() if v.get("cur")})
    if need_fix and days:
        n = repair_tri(days, adj, mw_adj, extra_adj)
        first = days[0]["d"]
        for c in cases:
            if c["d"] >= first:
                c["r"] = {}
        if old_summary:
            old_summary["corr"] = []
        print("Trefferquote: Gesamtrendite für", n, "Kürzel rückwirkend korrigiert", flush=True)
        if days[-1]["d"] == today:               # heute schon gemessen: nur neu auswerten
            fill_results(cases, days, curs, today)
            corr = update_corr([], days, curs)
            summary = build_summary(cases, old_summary, corr, today, (old_summary or {}).get("since") or days[0]["d"])
            summary["chg"] = (old_summary or {}).get("chg")
            days = save_days(paths["track"], days, today)
            save_cases(paths["cases"], cases)
            with open(paths["summary"], "w") as f:
                json.dump(summary, f, ensure_ascii=False, separators=(",", ":"))
            return {"schema": SCHEMA, "since": summary["since"], "cases": len(cases), "open": summary["n_open"]}
    prev = days[-1] if days else None
    first_run = not cases and not old_summary
    if first_run and prev is not None:
        # Startbestand: Mitglieder am ersten Messtag; der erste Messtag bekommt rückwirkend tri 100
        start_day = days[0]
        for row in start_day["p"].values():
            if row[I_TRI] is None:
                row[I_TRI] = 100.0
        if start_day.get("mw") is None:
            start_day["mw"] = 100.0
        if not start_day.get("fx"):
            start_day["fx"] = {k: sig5(v) for k, v in fx.items()}
        sinfo = {}
        for s, inf in info.items():
            sinfo[s] = dict(inf)
            if "px" in start_day and s in start_day["px"]:
                sinfo[s]["px"] = start_day["px"][s]
        update_cases(cases, start_day, sinfo, start=True)
        prev = days[-1]
    open_syms = {c["s"] for c in cases if c["x"] is None or len(c["r"]) < len(HORIZONS)}
    extra = {s: fn for s, fn in (extra_adj or {}).items() if s in open_syms}
    day = make_day(today, version, stocks, adj, prev, fx, mw_adj, extra)
    days.append(day)
    if first_run and prev is None:
        update_cases(cases, day, info, start=True)
    else:
        update_cases(cases, day, info)
    fill_results(cases, days, curs, today)
    corr = update_corr(list((old_summary or {}).get("corr", [])), days, curs)
    since = days[0]["d"]
    summary = build_summary(cases, old_summary, corr, today, (old_summary or {}).get("since") or since)
    summary["corrh"] = update_corr_hz(list((old_summary or {}).get("corrh", [])), days, curs)
    try:                                          # Score-Stufen; Fehler dürfen die Trefferquote nicht aufhalten
        summary["bk"] = update_buckets(list((old_summary or {}).get("bk", [])), days, curs)
    except Exception as e:  # noqa: BLE001
        print("Score-Stufen nicht berechnet:", type(e).__name__, flush=True)
        summary["bk"] = (old_summary or {}).get("bk", [])
    try:                                          # Musterdepots; Fehler dürfen die Trefferquote nicht aufhalten
        summary["md"] = update_models((old_summary or {}).get("md"), days, curs)
    except Exception as e:  # noqa: BLE001
        print("Musterdepots nicht berechnet:", type(e).__name__, flush=True)
        summary["md"] = (old_summary or {}).get("md")
    try:                                          # laufende Fälle mit Branchenvergleich und größtem Rückgang
        summary["op"] = open_cases(cases, days, curs, {s: v.get("sc") for s, v in info.items()})
    except Exception as e:  # noqa: BLE001
        print("Laufende Fälle nicht berechnet:", type(e).__name__, flush=True)
        summary["op"] = (old_summary or {}).get("op", [])
    # Wechsel seit dem letzten Messtag (für die Liste „Neu seit dem letzten Messtag“)
    summary["chg"] = {"d": today, "prev": prev["d"] if prev is not None else None,
                      "in": [[c["s"], c["k"]] for c in cases if c["d"] == today and not c["st"]],
                      "out": [[c["s"], c["k"]] for c in cases if c["x"] == today]}
    cases = prune_cases(cases, summary, today)
    days = save_days(paths["track"], days, today)
    save_cases(paths["cases"], cases)
    os.makedirs(os.path.dirname(paths["summary"]), exist_ok=True)
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, ensure_ascii=False, separators=(",", ":"))
    with open(paths["meta"], "w") as f:
        json.dump(sorted(curs.items()), f, separators=(",", ":"))
    return {"schema": SCHEMA, "since": summary["since"], "cases": len(cases), "open": summary["n_open"]}
