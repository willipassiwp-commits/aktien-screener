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
import re
import unicodedata
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

VERSION = 6

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
WEIGHTS = dict(f=30, m=15, t=20, r=15, a=15, n=5)
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
    """Zahl in deutscher Schreibweise für Texte: kaufmännisch gerundet (wie auf der Website), Tausenderpunkt,
    echtes Minuszeichen."""
    if v is None:
        return "–"
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    neg = q < 0
    txt = "{:,.{}f}".format(abs(q), d)                     # 24,392.00
    ip, _, fp = txt.partition(".")
    ip = ip.replace(",", ".")
    out = ip + ("," + fp if fp else "")
    return ("−" if neg and float(q) != 0 else "") + out


def sde(v, d=0):
    return ("+" if v > 0 else "−" if v < 0 else "±") + de(abs(v), d)


def bounded(v, lo, hi):
    """Werte außerhalb des plausiblen Bereichs gelten als Datenfehler."""
    return v if v is not None and lo <= v <= hi else None


def clamp(v, lo, hi):
    """Wert auf den Bereich lo–hi begrenzen (anders als bounded, das Ausreißer verwirft)."""
    return None if v is None else max(lo, min(hi, v))


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


def short_term(c, vol, weekly):
    """Kennzahlen für die kurzfristige Lage. Alles aus Schlusskursen und Volumen der letzten Monate."""
    if len(c) < 130:
        return None
    last = float(c.iloc[-1])
    sma20 = float(c.iloc[-20:].mean())
    sma20_prev = float(c.iloc[-25:-5].mean())
    low60_age = int(59 - int(np.argmin(c.iloc[-60:].values)))          # Handelstage seit dem 60-Tage-Tief
    higher_low = bool(float(c.iloc[-10:].min()) > float(c.iloc[-30:-10].min()))
    hi3m = float(c.iloc[-63:].max())
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    rets = c.pct_change().iloc[-60:].dropna()
    sd = float(rets.std()) if len(rets) > 20 else None
    sma50 = float(c.iloc[-50:].mean())
    stretch = (last / sma50 - 1) / (sd * np.sqrt(20)) if sd else None   # Abstand zur 50-Tage-Linie in üblichen Schwankungen
    udv = None
    if vol is not None:
        d = c.diff().iloc[-20:]
        v = vol.reindex(d.index)
        up, dn = float(v[d > 0].sum()), float(v[d < 0].sum())
        udv = up / dn if dn > 0 else None
    return dict(sma20=sma20, up20=sma20 > sma20_prev, cross20=bool(float(c.iloc[-6]) < float(c.iloc[-26:-6].mean())
                                                                   and last > sma20),
                low60_age=low60_age, higher_low=higher_low, dd3m=1 - last / hi3m if hi3m > 0 else None,
                macd_h=float(hist.iloc[-1]), macd_h5=float(hist.iloc[-6]), rsi5=rsi(c.iloc[:-5]),
                rsi_w=rsi(weekly) if weekly is not None and len(weekly) >= 45 else None,
                stretch=float(stretch) if stretch is not None else None, udv=udv)


def is_insider(i, company=""):
    """Zählt Geschäfte von Insidern: Vorstände, Aufsichtsräte, Großaktionäre (auch deren Gesellschaften, etwa ein
    Familienbüro). Nicht gezählt werden Aktienrückkäufe des Unternehmens selbst: Position „Issuer“ oder der Name des
    Unternehmens als Käufer."""
    pos = (i[2] or "").strip().lower()
    if pos == "issuer":
        return False
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", (company or "").lower()).split() if len(w) >= 3 and w != "the"]
    name = " " + re.sub(r"[^a-z0-9 ]", " ", (i[1] or "").lower()) + " "
    return not (words and " %s " % words[0] in name)


def insider_summary(items, today, days=182, company=""):
    """Insiderkäufe und -verkäufe der letzten sechs Monate. items: [Datum, Name, Position, "B"/"S", Stück, Wert]."""
    cut = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = [i for i in items if i and i[0] >= cut and is_insider(i, company)]
    buys = [i for i in rows if i[3] == "B"]
    sells = [i for i in rows if i[3] == "S"]
    vb = sum(i[5] or 0 for i in buys)
    vs = sum(i[5] or 0 for i in sells)
    top = sorted(buys, key=lambda i: -(i[5] or 0))[:5]
    return dict(nb=len(buys), vb=float("%.4g" % vb) if vb else 0, nbp=len({i[1] for i in buys}),
                ns=len(sells), vs=float("%.4g" % vs) if vs else 0, nsp=len({i[1] for i in sells}),
                top=top, last=rows[:8], n12=len(items))


def revenue_view(fund):
    """Umsatz der letzten Geschäftsjahre mit Wachstum zum Vorjahr und die Analystenschätzung für das laufende und das
    nächste Jahr. Nur Anzeige. Werte in der Berichtswährung."""
    revs, dates = fund.get("revenue") or [], fund.get("incDates") or []
    h = []
    for i, (d, v) in enumerate(zip(dates, revs)):
        if v is None or v <= 0:
            continue
        prev = revs[i + 1] if i + 1 < len(revs) else None
        g = (v / prev - 1) * 100 if prev and prev > 0 else None
        h.append([str(d)[:4], float("%.4g" % v), None if g is None or abs(g) > 500 else round(g, 1)])
    e = fund.get("revEstYrs")
    e = [None if v is None or abs(v) > 3 else round(v * 100, 1) for v in e] if e else None
    ea = fund.get("revGrowthEst")
    ea = None if ea is None or abs(ea) > 3 else round(ea * 100, 1)
    if not h and not e and ea is None:
        return None
    return {"h": h[:4], "e": e, "ea": ea, "cur": fund.get("finCurrency") or fund.get("currency")}


def money(v, cur):
    cur = {"GBp": "GBP", "ZAc": "ZAR", "ILA": "ILS"}.get(cur, cur or "")
    if v >= 1e9:
        return "%s Mrd. %s" % (de(v / 1e9, 1), cur)
    if v >= 1e6:
        return "%s Mio. %s" % (de(v / 1e6, 1), cur)
    return "%s %s" % (de(v, 0), cur)


def lage(price, pm, rsi_d, zs, sym):
    """Kurzfristige Lage: eine Einstufung mit Belegen. Beschreibt den Kursverlauf bis heute, keine Vorhersage.
    Rückgabe: [Code, Bezeichnung, Satz, Belege [[Text, Wert, positiv?], ...], Überdehnt-Zähler].
    Codes: U überdehnt, G Gegenbewegung, F fällt noch, D dreht nach oben, A Rücksetzer läuft aus, S läuft stabil,
    E Erholung im Abwärtstrend (kurzfristig über den Linien, langfristig noch nicht), N uneinheitlich."""
    st = pm.get("st")
    if not st:
        return None
    s20, s50 = st["sma20"], pm["sma50"]
    over20, over50 = price > s20, price > s50
    rw, sx = st["rsi_w"], st["stretch"]
    hot = [rsi_d is not None and rsi_d >= 70, rw is not None and rw >= 70, sx is not None and sx >= 2]
    new_low = st["low60_age"] <= 2
    p1w = pm.get("p1w")
    rsi_up = rsi_d is not None and st["rsi5"] is not None and rsi_d > st["rsi5"] + 3
    macd_turn = st["macd_h"] > 0 >= st["macd_h5"]
    sup = zs[0] if zs else None
    near_sup = sup is not None and sup[3] >= -4
    B = [["Kurs zur 20-Tage-Linie", pct(price / s20 - 1), over20], ["Kurs zur 50-Tage-Linie", pct(price / s50 - 1), over50],
         ["Handelstage seit dem 60-Tage-Tief", st["low60_age"], not new_low],
         ["Tief der letzten 10 Tage höher als davor", None, st["higher_low"]],
         ["RSI 14 Tage (vor 5 Tagen)", "%s (%s)" % (de(rsi_d, 0), de(st["rsi5"], 0)), rsi_up],
         ["RSI 14 Wochen", de(rw, 0), None if rw is None else rw < 70],
         ["MACD-Histogramm", "positiv" if st["macd_h"] > 0 else "negativ", st["macd_h"] > 0],
         ["Abstand zur 50-Tage-Linie in üblichen Schwankungen", de(sx, 1), None if sx is None else sx < 2]]
    if st["udv"] is not None:
        B.append(["Volumen an Plus- zu Minustagen (20 Tage)", de(st["udv"], 2), st["udv"] >= 1])
    dd = (st["dd3m"] or 0) * 100
    med = sym[2] if sym else None
    if sum(hot) >= 2:
        code, lab = "U", "Überdehnt"
        txt = ("Der Kurs ist kurzfristig weit gelaufen: %d von 3 Kriterien (RSI 14 Tage ab 70, RSI 14 Wochen ab 70, "
               "mindestens zwei übliche Schwankungen über der 50-Tage-Linie). Nach solchen Phasen folgen oft "
               "Verschnaufpausen; die 50-Tage-Linie liegt bei %s (%s %%)." % (sum(hot), de(s50, 2), sde((s50 / price - 1) * 100)))
    elif not over20 and p1w is not None and p1w >= 0.04 and not new_low:
        code, lab = "G", "Gegenbewegung"
        txt = ("Der Kurs hat sich in einer Woche um %s %% erholt, liegt aber noch unter der 20-Tage-Linie (%s)%s. "
               "Ob daraus eine Wende wird, zeigt sich erst über der 20-Tage-Linie."
               % (de(p1w * 100, 0), de(s20, 2), "" if over50 else " und der 50-Tage-Linie (%s)" % de(s50, 2)))
    elif not over20 and not over50 and (new_low or not st["up20"]):
        code, lab = "F", "Fällt noch"
        txt = ("Der Kurs liegt unter der 20- und der 50-Tage-Linie%s. Ein Wendezeichen gibt es noch nicht."
               % (" und hat in den letzten Tagen ein neues 60-Tage-Tief markiert" if new_low else ", die 20-Tage-Linie fällt"))
    elif over20 and (st["cross20"] or macd_turn) and (not over50 or st["higher_low"]):
        code, lab = "D", "Dreht nach oben"
        txt = ("Der Kurs ist über die 20-Tage-Linie (%s) zurückgekehrt%s%s. Bestätigt wäre die Wende mit einem Kurs über "
               "der 50-Tage-Linie (%s)." % (de(s20, 2), ", das letzte Tief liegt höher als das vorige" if st["higher_low"] else "",
                                            ", das MACD-Histogramm ist ins Plus gedreht" if macd_turn else "", de(s50, 2))
               if not over50 else
               "Der Kurs ist über die 20-Tage-Linie zurückgekehrt und liegt auch über der 50-Tage-Linie; das letzte Tief "
               "liegt höher als das vorige.")
    elif not new_low and not over20 and near_sup and (rsi_up or (st["udv"] is not None and st["udv"] >= 1)):
        code, lab = "A", "Rücksetzer läuft aus"
        txt = ("Seit %d Handelstagen kein neues Tief, der Kurs steht nahe der Unterstützung %s – %s (%s), und der "
               "Verkaufsdruck lässt nach. Ein Wendezeichen wäre ein Kurs über der 20-Tage-Linie (%s)."
               % (st["low60_age"], de(sup[0], 2), de(sup[1], 2), ", ".join(sup[2]), de(s20, 2)))
    elif over20 and over50 and price > pm["sma200"] and s50 >= pm["sma200"]:
        code, lab = "S", "Läuft stabil"
        txt = "Der Kurs liegt über der 20-, der 50- und der 200-Tage-Linie, ohne überdehnt zu sein."
    elif over20 and over50:
        code, lab = "E", "Erholung im Abwärtstrend"
        txt = ("Kurzfristig erholt: Der Kurs liegt über der 20- und der 50-Tage-Linie. Langfristig ist das Bild noch "
               "schwach: %s." % ("der Kurs liegt unter der 200-Tage-Linie (%s)" % de(pm["sma200"], 2)
                                 if price <= pm["sma200"] else "die 50-Tage-Linie liegt noch unter der 200-Tage-Linie"))
    else:
        code, lab = "N", "Uneinheitlich"
        txt = ("Die Signale zeigen in verschiedene Richtungen: Kurs %s der 20-Tage-Linie und %s der 50-Tage-Linie."
               % ("über" if over20 else "unter", "über" if over50 else "unter"))
    if dd >= 5 and code != "U":
        txt += " Der Kurs steht %s %% unter dem Hoch der letzten drei Monate" % de(dd, 0)
        if med is not None:
            txt += "; frühere Korrekturen dieser Aktie lagen im Mittel bei %s %%" % de(med, 0)
        if sup is not None and sup[3] < 0:
            txt += ", die nächste Unterstützung liegt %s %% tiefer" % de(-sup[3], 0)
        txt += "."
    return [code, lab, txt, [[b[0], b[1], None if b[2] is None else bool(b[2])] for b in B], int(sum(hot))]


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

    # Kurzfristige Lage: Kennzahlen der letzten Wochen
    st_ = short_term(c, vol, w)

    # Abstand zum 52-Wochen-Tief und Alter des laufenden Trends
    since_low = int(len(c52) - 1 - int(np.argmin(c52.values)))
    # Tief des laufenden Rücksetzers: tiefster Schlusskurs seit dem 52-Wochen-Hoch
    ih = int(np.argmax(c52.values))
    seg = c52.values[ih:]
    pb_low = float(seg.min())
    pb_age = int(len(seg) - 1 - int(np.argmin(seg)))
    s200 = c.rolling(200).mean().iloc[-520:].dropna()
    side = np.sign((c.reindex(s200.index) - s200).values)
    days200 = 0
    for v in side[::-1]:
        if v != side[-1] or v == 0:
            break
        days200 += 1
    days200 = int(days200 * side[-1]) if len(side) else 0
    capped200 = bool(len(side) and abs(days200) >= len(side))

    # Tage seit dem letzten 3-Monats-Hoch
    c63 = c.iloc[-63:]
    since_high = int(len(c63) - 1 - int(np.argmax(c63.values)))

    # Marktsymmetrie: laufende Korrektur gegen frühere Korrekturen der letzten zehn Jahre
    w10 = c[c.index >= end - pd.DateOffset(years=10)].resample("W").last().dropna().values
    past, cur_dd, cur_depth = corrections(list(w10)) if len(w10) > 60 else ([], 0.0, 0.0)
    # Wochenschlusskurse glätten Spitzen weg: Die laufende Korrektur ist nie kleiner als der Abstand zum
    # 52-Wochen-Hoch aus Tagesschlusskursen.
    if hi52 > 0:
        cur_dd = max(cur_dd, 1 - last / hi52)
        cur_depth = max(cur_depth, cur_dd)
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
                since_low=since_low, days200=days200, capped200=capped200, st=st_, pb_low=pb_low, pb_age=pb_age,
                p1d=perf(c, 1), p1w=perf(c, 5),
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
    if rec.get("ns") is not None:
        parts.append((w.get("n", 0), rec["ns"] * 10))
    ws = sum(p[0] for p in parts) or 1
    return sum(p[0] * p[1] for p in parts) / ws


# Score je Zeitraum: dieselben Bausteine, anders gewichtet. 1 Jahr = Gesamtscore (WEIGHTS).
# k = Kurzfrist-Baustein aus Kurzfrist-Note, relativer Stärke und Wende-Fortschritt.
HZ_WEIGHTS = {"3m": dict(t=25, k=30, a=15, n=10, f=10, m=5, r=5),
              "2j": dict(f=35, m=25, r=30, a=5, t=5)}


def short_component(rec):
    """Kurzfrist-Baustein 0–100 aus der Kurzfrist-Note (50 %), der relativen Stärke der letzten 20 Tage gegen den
    Vergleichsindex (25 %, ±10 Prozentpunkte = 0 bzw. 100) und dem Wende-Fortschritt (25 %; ohne Wende-Treppe, also
    ohne Rückgang, 70). Liest die ausführlichen Felder (ka, kx) oder die Kurzfassung (ka, ks)."""
    ka = (rec.get("ka") or {}).get("n")
    if ka is None:
        return None
    kx, ks = rec.get("kx") or {}, rec.get("ks") or {}
    rs = ((kx.get("mk") or {}).get("rs") or {}).get("20") if kx else ks.get("rs")
    tp = (kx.get("tl") or {}).get("prog") if kx else ks.get("tp")
    if kx and not kx.get("tl"):
        tp = None
    rs_s = clamp(50 + rs[2] * 500, 0, 100) if rs and rs[2] is not None else 50
    tp_s = tp / 6 * 100 if tp is not None else 70
    return 0.5 * ka * 10 + 0.25 * rs_s + 0.25 * tp_s


def horizon_scores(rec):
    """[3 Monate, 2 Jahre und länger], je 0–100; fehlende Bausteine verteilen ihr Gewicht."""
    val = {"f": rec["f"] * 10, "m": rec["m"] * 4, "r": rec.get("rs"), "t": rec["t"] * 10,
           "a": None if rec.get("a") is None else rec["a"] * 10, "n": None if rec.get("ns") is None else rec["ns"] * 10,
           "k": short_component(rec)}
    out = []
    for key in ("3m", "2j"):
        parts = [(w, val[k]) for k, w in HZ_WEIGHTS[key].items() if val.get(k) is not None]
        ws = sum(p[0] for p in parts)
        out.append(round(sum(p[0] * p[1] for p in parts) / ws, 1) if ws else None)
    return out


def total(rec, w=WEIGHTS):
    """Gesamtscore 0-100. Fehlt das Analystenbild, verteilt sich sein Gewicht auf die übrigen Bausteine."""
    return _parts(rec, w, True)


def quality(rec, w=WEIGHTS):
    """Qualitätsscore 0-100: alles außer dem Kursbild."""
    return _parts(rec, w, False)


NEG_WORDS = ["downgrade", "cuts ", "cut to", "slashes", "lawsuit", "sues", "sued", "probe", "investigation",
             "recall", "fraud", "warning", "warns", "misses", "missed", "plunge", "tumble", "sinks", "slump",
             "layoffs", "antitrust", "short seller", "resigns", "steps down", "fined", " ban", "halts", "bankrupt",
             "scandal", "subpoena", "lowers", "selloff", "sell-off", "falls", "drops", "slides", "weak", "concern",
             "disappoint", "delay", "loss ", "losses", "crash", "underperform", "bearish"]
POS_WORDS = ["upgrade", "beats", "beat ", "raises", "raised", "record", "buyback", "surges", "soars", "jumps",
             "rallies", "rally", "wins", "approval", "approved", "tops", "outperform", "all-time high", "boosts",
             "hikes", "strong", "gains", "climbs", "rises", "bullish", "breakthrough", "expands"]
NAME_SKIP = {"the", "inc", "corp", "corporation", "company", "co", "ltd", "plc", "group", "holdings", "holding",
             "ag", "se", "sa", "nv", "and"}


NAME_GENERIC = {"bank", "china", "taiwan", "american", "united", "general", "national", "first", "international",
                "royal", "new", "japan", "tokyo", "hong", "deutsche", "british", "canadian", "pacific", "global",
                "advanced", "applied", "western", "southern", "northern", "union", "standard", "industrial",
                "home", "target", "progressive", "public", "analog", "digital", "realty", "crown", "dollar", "capital",
                "enterprise", "energy", "constellation", "carrier", "delta", "sea", "trade", "sun", "mitsubishi",
                "sumitomo", "mitsui", "tata", "ping", "banco", "nippon", "east", "west", "central", "siemens",
                "republic", "waste", "best", "block", "service", "samsung", "hyundai", "williams", "simon"}
LEGAL = {"inc", "corp", "corporation", "company", "co", "ltd", "limited", "plc", "ag", "se", "sa", "nv", "holdings",
         "holding", "group", "the", "kgaa", "spa", "asa", "ab", "oyj", "class", "incorporated", "aktiengesellschaft", "publ"}
NAME_ALIAS = {"alphabet": ["google"], "meta": ["facebook", "instagram"], "walt": ["disney"], "eli": ["lilly"]}
NAME_RISKY = {"target", "block", "public", "best", "service", "trade", "energy", "capital", "home", "general"}
NAME_SHORT = {"taiwan semiconductor": ["TSMC"], "international business": ["IBM"], "advanced micro": ["AMD"],
              "industrial and": ["ICBC"], "royal bank": ["RBC"], "canadian imperial": ["CIBC"],
              "united overseas": ["UOB"], "british american": ["BAT"], "lvmh moet": ["LVMH"],
              "hong kong": ["HKEX"], "united parcel": ["UPS"], "at": ["AT&T"], "bp": ["BP"], "s": ["S&P Global"]}
TICKER_WORDS = {"NOW", "ALL", "KEY", "CAT", "LOW", "ARE", "BIG", "COST", "REAL", "WELL", "FAST", "TECH", "PLAY", "SNOW",
                "TEAM", "OPEN", "LIFE", "GOLD", "HAS", "FOR", "CAN", "ANY", "ONE", "TWO", "DAY", "MAN", "AIR", "CAR"}


def _plain(t):
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", t).split()) + " "


def name_keys(name, symbol=None):
    """Woran eine Schlagzeile als Nachricht zu diesem Unternehmen erkannt wird: am Namen (bei Allerweltswörtern
    wie „Bank“ oder „Taiwan“ und bei Zwei-Buchstaben-Namen an den ersten Wörtern zusammen), an einem gängigen
    Kurznamen (TSMC, IBM) oder am Börsenkürzel. Rückgabe: (Namensteile klein, Kürzel in Großbuchstaben)."""
    toks = _plain(name).split()
    while toks and (toks[-1] in LEGAL or len(toks[-1]) == 1) and len(toks) > 1:
        toks.pop()
    while len(toks) > 1 and toks[0] in LEGAL:
        toks = toks[1:]
    low, caps = [], []
    if toks:
        if toks[0] in NAME_GENERIC or len(toks[0]) < 3:
            k = 1
            while k < len(toks) - 1 and toks[k] in ("of", "and", "de", "the"):
                k += 1
            if k < len(toks) and max(len(t_) for t_ in toks[:k + 1]) >= 3:
                low.append(" ".join(toks[:k + 1]))
            elif k < len(toks):
                low += [t_ for t_ in toks[k + 1:k + 2] if len(t_) >= 5 and t_ not in NAME_GENERIC]  # U.S. Bancorp
            elif toks[0] not in NAME_RISKY and (len(toks[0]) >= 3 or any(ch.isdigit() for ch in toks[0])):
                low.append(toks[0])
        else:
            low.append(toks[0])
        low += NAME_ALIAS.get(toks[0], [])
        caps += NAME_SHORT.get(" ".join(toks[:2]), []) + NAME_SHORT.get(toks[0], [])
    root = (symbol or "").split(".")[0]
    if root.isalpha() and len(root) >= 3 and root not in TICKER_WORDS and root not in caps:
        caps.append(root)
    return low, caps


def news_view(items, name, today, symbol=None):
    """Schlagzeilen mit Stichwort-Einordnung. Behalten werden nur Schlagzeilen, die das Unternehmen im Titel nennen
    (Name, Kürzel oder Börsenkürzel); Branchen- und Konkurrenzmeldungen fallen weg. Gewertet werden die der letzten
    30 Tage. Eine Stichwortsuche erkennt den Ton einer Überschrift, nicht den Inhalt des Artikels.
    Rückgabe: Liste [Titel, Quelle, Datum, Link, Wertung], Zusammenfassung, Score 0-10 oder None."""
    keys_low, keys_cap = name_keys(name, symbol)
    cutoff = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    out, pos, neg, n = [], 0, 0, 0
    for it in items:
        if len(it) < 4:
            continue
        title, low = it[0], " " + it[0].lower() + " "
        pl = _plain(title)
        if not (any(" " + k + " " in pl for k in keys_low)
                or any(re.search(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", title) for k in keys_cap)):
            continue
        flag = None
        if it[2] >= cutoff:
            p_ = any(w in low for w in POS_WORDS)
            n_ = any(w in low for w in NEG_WORDS)
            flag = 1 if p_ and not n_ else -1 if n_ and not p_ else 0
            n += 1
            pos += flag == 1
            neg += flag == -1
        out.append([title, it[1], it[2], it[3], flag])
    score10 = round(clamp(5 + 5 * (pos - neg) / max(n, 4), 0, 10), 1) if n >= 3 else None
    return out, dict(n=n, pos=int(pos), neg=int(neg)), score10


REC_WORD = [(1.5, "klar Kaufen"), (2.5, "Kaufen"), (3.5, "Halten"), (4.5, "Verkaufen"), (9, "klar Verkaufen")]


def rec_word(mean):
    return next(wd for lim, wd in REC_WORD if mean <= lim)


# ----------------------------------------------------------------------------------------------------------------
# Kurzfristige Chartanalyse (nur Anzeige, fließt nicht in den Gesamtscore ein)
# ----------------------------------------------------------------------------------------------------------------

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi_series(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def _f(v, d=1, unit=""):
    return "–" if v is None else de(v, d) + unit


def _sp(v, d=1):
    """Vorzeichen-Prozent als Text, v als Anteil (0,05 = +5 %)."""
    if v is None:
        return "–"
    return ("+" if v >= 0 else "−") + de(abs(v) * 100, d) + " %"


def short_analysis(df, live=False):
    """Ausführliche kurzfristige Chartanalyse aus Tageskursen (Eröffnung, Hoch, Tief, Schluss, Volumen).
    Rückgabe: dict mit Kurzfrist-Note 0–10, Einstufung, Zusammenfassung, Signalgruppen, kurzfristigen Marken und den
    Tagesreihen für den Chart. None, wenn zu wenig Daten da sind. Beschreibt den Kursverlauf bis heute, sagt nichts
    Sicheres über die Zukunft.
    live=True: der letzte Tag ist ein noch laufender Handelstag. Volumen- und Tagesspannen-Signale nutzen dann nur
    abgeschlossene Tage, die Seite zeigt den Hinweis „vorläufig“."""
    if df is None or "Close" not in df:
        return None
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce")
    ok = c.notna() & (c > 0)
    df, c = df[ok], c[ok]
    if len(c) < 130:
        return None
    h = pd.to_numeric(df["High"], errors="coerce").fillna(c) if "High" in df else c
    l = pd.to_numeric(df["Low"], errors="coerce").fillna(c) if "Low" in df else c
    o = pd.to_numeric(df["Open"], errors="coerce").fillna(c) if "Open" in df else c
    h, l = np.maximum(h, c), np.minimum(l, c)
    v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) if "Volume" in df else None
    if v is not None and float(v.iloc[-60:].sum()) <= 0:
        v = None
    last = float(c.iloc[-1])
    rows = {"Trend": [], "Momentum": [], "Volatilität": [], "Volumen": [], "Kursmuster": []}
    flags = []

    def add(g, label, value, sig, note):
        rows[g].append([label, value, sig, note])

    # ---------------- Trend
    e10, e20, s50 = _ema(c, 10), _ema(c, 20), c.rolling(50).mean()
    e20_now, e20_prev = float(e20.iloc[-1]), float(e20.iloc[-6])
    if last > e20_now and e20_now > e20_prev:
        s, t = 1, "Kurs über der steigenden EMA 20"
    elif last < e20_now and e20_now < e20_prev:
        s, t = -1, "Kurs unter der fallenden EMA 20"
    else:
        s, t = 0, "Kurs und EMA 20 uneinheitlich"
    add("Trend", "Kurs zur EMA 20", _sp(last / e20_now - 1), s, t)
    diff = (e10 - e20).iloc[-30:]
    sign = np.sign(diff.values)
    xd = None
    for i in range(len(sign) - 1, 0, -1):
        if sign[i] != sign[i - 1] and sign[i] != 0:
            xd = len(sign) - 1 - i
            break
    s = 1 if sign[-1] > 0 else -1
    t = ("EMA 10 über EMA 20" if s > 0 else "EMA 10 unter EMA 20") + (
        ", gekreuzt vor %d Handelstag%s" % (xd, "" if xd == 1 else "en") if xd is not None and xd <= 10 else "")
    add("Trend", "EMA 10 gegen EMA 20", "Abstand " + _sp(float(e10.iloc[-1] / e20.iloc[-1] - 1)), s, t)
    s50n = float(s50.iloc[-1])
    add("Trend", "Kurs zur 50-Tage-Linie", _sp(last / s50n - 1), 1 if last > s50n else -1,
        "über der 50-Tage-Linie" if last > s50n else "unter der 50-Tage-Linie")
    y = np.log(c.iloc[-20:].values)
    x = np.arange(20)
    slope, icpt = np.polyfit(x, y, 1)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - (slope * x + icpt)) ** 2).sum()) / ss if ss > 0 else 0.0
    m20 = math.exp(slope * 19) - 1
    s = 1 if m20 > 0.02 and r2 >= 0.5 else -1 if m20 < -0.02 and r2 >= 0.5 else 0
    add("Trend", "Steigung 20 Tage", "%s (Gleichmäßigkeit %s)" % (_sp(m20), de(r2, 2)), s,
        "gleichmäßiger Anstieg" if s > 0 else "gleichmäßiger Rückgang" if s < 0 else "kein gleichmäßiger Verlauf")
    # ADX
    up_m, dn_m = h.diff(), -l.diff()
    pdm = up_m.where((up_m > dn_m) & (up_m > 0), 0.0)
    ndm = dn_m.where((dn_m > up_m) & (dn_m > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    adx = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    pd_, nd_ = float(pdi.iloc[-1] or 0), float(ndi.iloc[-1] or 0)
    if adx >= 25:
        s = 1 if pd_ > nd_ else -1
        t = "kräftiger Trend nach " + ("oben" if s > 0 else "unten")
    elif adx < 20:
        s, t = 0, "kein ausgeprägter Trend, eher Seitwärtsphase"
    else:
        s, t = 0, "Trend schwach ausgeprägt (nach " + ("oben)" if pd_ > nd_ else "unten)")
    add("Trend", "ADX (Trendstärke)", "%s · +DI %s / −DI %s" % (de(adx, 0), de(pd_, 0), de(nd_, 0)), s, t)

    # ---------------- Momentum
    rs = _rsi_series(c)
    r_now, r_5 = float(rs.iloc[-1]), float(rs.iloc[-6])
    if r_now >= 70:
        s, t = 0, "überkauft (über 70), Rücksetzer häufiger"
        flags.append("überkauft")
    elif r_now <= 30:
        s, t = 0, "überverkauft (unter 30), Gegenbewegungen häufiger"
        flags.append("überverkauft")
    elif r_now >= 50:
        s, t = (1, "über 50 und steigend") if r_now >= r_5 else (0, "über 50, aber fallend")
    else:
        s, t = (-1, "unter 50 und fallend") if r_now < r_5 else (0, "unter 50, aber steigend")
    add("Momentum", "RSI 14", "%s (vor 5 Tagen %s)" % (de(r_now, 0), de(r_5, 0)), s, t)
    macd = _ema(c, 12) - _ema(c, 26)
    sigl = _ema(macd, 9)
    hist = macd - sigl
    hs = np.sign(hist.iloc[-30:].values)
    md = None
    for i in range(len(hs) - 1, 0, -1):
        if hs[i] != hs[i - 1] and hs[i] != 0:
            md = len(hs) - 1 - i
            break
    s = 1 if hs[-1] > 0 else -1
    t = ("MACD über der Signallinie" if s > 0 else "MACD unter der Signallinie") + (
        ", Kreuzung vor %d Handelstag%s" % (md, "" if md == 1 else "en") if md is not None and md <= 10 else "")
    add("Momentum", "MACD (12, 26, 9)", "%s / Signal %s" % (de(float(macd.iloc[-1]), 2), de(float(sigl.iloc[-1]), 2)), s, t)
    h3 = hist.iloc[-4:].values
    s = 1 if h3[-1] > h3[-2] > h3[-3] else -1 if h3[-1] < h3[-2] < h3[-3] else 0
    add("Momentum", "MACD-Histogramm", "3 Tage " + ("steigend" if s > 0 else "fallend" if s < 0 else "gemischt"), s,
        "Schwung nimmt zu" if s > 0 else "Schwung lässt nach" if s < 0 else "Schwung ohne klare Richtung")
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k_raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = k_raw.rolling(3).mean()
    dd = k.rolling(3).mean()
    kn, dn_ = float(k.iloc[-1]), float(dd.iloc[-1])
    if kn >= 80:
        s, t = 0, "stark gestiegen (überkauft)"
    elif kn <= 20:
        s, t = (1, "stark gefallen, dreht aber nach oben") if kn > dn_ else (0, "stark gefallen (überverkauft)")
    else:
        s, t = (1, "kurzfristiger Schwung zeigt nach oben") if kn > dn_ else (-1, "kurzfristiger Schwung zeigt nach unten")
    add("Momentum", "Stochastik (14, 3)", "%%K %s / %%D %s" % (de(kn, 0), de(dn_, 0)), s, t)
    roc = float(c.iloc[-1] / c.iloc[-11] - 1)
    add("Momentum", "Veränderung 10 Tage", _sp(roc), 1 if roc > 0.03 else -1 if roc < -0.03 else 0,
        "deutlich gestiegen" if roc > 0.03 else "deutlich gefallen" if roc < -0.03 else "kaum verändert")

    # ---------------- Volatilität
    m20s, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    up_b, lo_b = m20s + 2 * sd20, m20s - 2 * sd20
    pb = float((c.iloc[-1] - lo_b.iloc[-1]) / (up_b.iloc[-1] - lo_b.iloc[-1])) if float(up_b.iloc[-1] - lo_b.iloc[-1]) > 0 else 0.5
    if pb > 1:
        s, t = 0, "über dem oberen Band, kurzfristig überdehnt"
        flags.append("über dem oberen Bollinger-Band")
    elif pb < 0:
        s, t = 0, "unter dem unteren Band, kurzfristig stark gefallen"
    elif pb >= 0.5:
        s, t = 1, "obere Hälfte der Bänder"
    else:
        s, t = -1, "untere Hälfte der Bänder"
    add("Volatilität", "Bollinger-Bänder (20, 2)", "Position %s %%" % de(pb * 100, 0), s, t)
    bw = ((up_b - lo_b) / m20s).iloc[-120:].dropna()
    bwr = float((bw < bw.iloc[-1]).mean()) if len(bw) > 30 else None
    if bwr is not None:
        add("Volatilität", "Bandbreite", ("weiter als %s %% der letzten 6 Monate" % de(bwr * 100, 0)) if bwr >= 0.5 else
            ("enger als %s %% der letzten 6 Monate" % de((1 - bwr) * 100, 0)), None,
            "Bänder sehr eng: oft folgt eine größere Bewegung, Richtung offen" if bwr <= 0.15 else
            "Bänder weit: hohe Schwankung" if bwr >= 0.85 else "normale Bandbreite")
    atr_n = float(atr.iloc[-1])
    atrp = atr_n / last
    add("Volatilität", "ATR 14 (übliche Tagesspanne)", "%s · %s %% vom Kurs" % (de(atr_n, 2 if atr_n < 100 else 0),
                                                                           de(atrp * 100, 1)), None,
        "sehr ruhig" if atrp < 0.012 else "ruhig" if atrp < 0.02 else "lebhaft" if atrp < 0.035 else "sehr schwankungsreich")

    # ---------------- Volumen
    if v is not None:
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        ob = obv.iloc[-20:].values
        ob_s = np.polyfit(np.arange(20), ob, 1)[0] * 20
        vavg = float(v.iloc[-60:].mean()) or 1
        rel = ob_s / (vavg * 20)
        s = 1 if rel > 0.1 else -1 if rel < -0.1 else 0
        add("Volumen", "On-Balance-Volumen 20 Tage", "steigend" if s > 0 else "fallend" if s < 0 else "seitwärts", s,
            "es wird eher gekauft als verkauft" if s > 0 else "es wird eher verkauft als gekauft" if s < 0 else "Kauf- und Verkaufsdruck ausgeglichen")
        d20 = c.diff().iloc[-20:]
        vv = v.reindex(d20.index)
        upv, dnv = float(vv[d20 > 0].sum()), float(vv[d20 < 0].sum())
        udr = upv / dnv if dnv > 0 else None
        s = 0 if udr is None else 1 if udr > 1.2 else -1 if udr < 0.8 else 0
        add("Volumen", "Volumen an Plus- gegen Minustagen", "–" if udr is None else de(udr, 2) + " : 1", s,
            "mehr Volumen an Plustagen" if s > 0 else "mehr Volumen an Minustagen" if s < 0 else "ausgeglichen")
        k0 = -2 if live else -1                    # laufender Tag: Volumen noch unvollständig
        v20 = float(v.iloc[k0 - 20:k0].mean())
        vr = float(v.iloc[k0]) / v20 if v20 > 0 else None
        d1 = float(c.iloc[k0] - c.iloc[k0 - 1])
        if vr is not None and vr >= 1.5:
            s = 1 if d1 > 0 else -1 if d1 < 0 else 0
            t = "hohes Volumen am " + ("Plustag" if d1 > 0 else "Minustag" if d1 < 0 else "Handelstag")
        else:
            s, t = 0, "normales Volumen" if vr is None or vr >= 0.6 else "geringes Volumen"
        add("Volumen", "Volumen letzter abgeschlossener Handelstag" if live else "Volumen letzter Handelstag",
            "–" if vr is None else de(vr, 1) + "-fach des 20-Tage-Schnitts", s, t)

    # ---------------- Kursmuster
    h20p, l20p = float(c.iloc[-21:-1].max()), float(c.iloc[-21:-1].min())
    s = 1 if last > h20p else -1 if last < l20p else 0
    add("Kursmuster", "20-Tage-Spanne", "Hoch %s · Tief %s" % (_f(h20p, 2), _f(l20p, 2)), s,
        "Schlusskurs über dem 20-Tage-Hoch (Ausbruch)" if s > 0 else "Schlusskurs unter dem 20-Tage-Tief" if s < 0
        else "innerhalb der Spanne, %s %% vom Tief zum Hoch" % de((last - l20p) / (h20p - l20p) * 100 if h20p > l20p else 50, 0))
    hh_ = float(h.iloc[-10:].max()) > float(h.iloc[-20:-10].max())
    hl_ = float(l.iloc[-10:].min()) > float(l.iloc[-20:-10].min())
    s = 1 if hh_ and hl_ else -1 if (not hh_ and not hl_) else 0
    add("Kursmuster", "Hochs und Tiefs (10 gegen 10 Tage)",
        ("höheres" if hh_ else "tieferes") + " Hoch, " + ("höheres" if hl_ else "tieferes") + " Tief", s,
        "Aufwärtsstruktur" if s > 0 else "Abwärtsstruktur" if s < 0 else "gemischte Struktur")
    k1 = len(c) - 1 if live else len(c)          # laufender Tag: nur abgeschlossene Tage
    rng = (h - l).iloc[k1 - 3:k1]
    clv = ((2 * c - h - l) / (h - l).replace(0, np.nan)).iloc[k1 - 3:k1].fillna(0)
    cm = float(clv.mean()) if float(rng.sum()) > 0 else 0.0
    s = 1 if cm > 0.3 else -1 if cm < -0.3 else 0
    add("Kursmuster", "Schlusskurs in der Tagesspanne (3 Tage)", "Ø %s %% der Spanne" % de((cm + 1) * 50, 0), s,
        "Schluss meist nahe am Tageshoch" if s > 0 else "Schluss meist nahe am Tagestief" if s < 0 else "Schluss meist in der Mitte")
    gap = float(o.iloc[-1] / c.iloc[-2] - 1)
    streak, sgn = 0, np.sign(float(c.iloc[-1] - c.iloc[-2]))
    for i in range(len(c) - 1, 0, -1):
        if np.sign(float(c.iloc[i] - c.iloc[i - 1])) == sgn and sgn != 0:
            streak += 1
        else:
            break
    info = []
    if abs(gap) >= 0.02:
        info.append("Kurslücke %s zur Eröffnung" % _sp(gap))
    if streak >= 4:
        info.append("%d %s in Folge" % (streak, "Plustage" if sgn > 0 else "Minustage"))
    add("Kursmuster", "Letzte Tage", ("unverändert" if sgn == 0 else "%s %s in Folge" % (
        streak, ("Plustag" if sgn > 0 else "Minustag") + ("" if streak == 1 else "e"))),
        None, "; ".join(info) if info else "keine Kurslücke, keine auffällige Serie")

    # ---------------- Note und Text
    scored = [r for g in rows.values() for r in g if r[2] is not None]
    pos = sum(1 for r in scored if r[2] > 0)
    neg = sum(1 for r in scored if r[2] < 0)
    note = round(5 + 5 * (pos - neg) / len(scored), 1) if scored else 5.0
    lb = ("kurzfristig positiv" if note >= 7 else "kurzfristig eher positiv" if note >= 5.8 else
          "kurzfristig neutral" if note > 4.2 else "kurzfristig eher negativ" if note > 3 else "kurzfristig negativ")
    tr_s = sum(r[2] for r in rows["Trend"] if r[2] is not None)
    mo_s = sum(r[2] for r in rows["Momentum"] if r[2] is not None)
    part = []
    part.append("Der kurzfristige Trend zeigt " + ("nach oben" if tr_s >= 2 else "nach unten" if tr_s <= -2 else "keine klare Richtung"))
    if (tr_s <= -2 and mo_s >= 2):
        part.append("das Momentum dreht aber schon nach oben")
    elif (tr_s >= 2 and mo_s <= -1):
        part.append("das Momentum lässt aber nach")
    else:
        part.append("das Momentum ist " + ("positiv" if mo_s >= 2 else "negativ" if mo_s <= -2 else "gemischt"))
    txt = "%s von %s Signalen positiv, %s negativ. %s, %s." % (pos, len(scored), neg, part[0], part[1])
    if flags:
        txt += " Hinweis: " + ", ".join(dict.fromkeys(flags)) + "."

    # ---------------- Marken
    hlc = pd.DataFrame({"h": h, "l": l, "c": c})
    if atrp < 0.03:                                 # ruhige Aktie: Pivot der Vorwoche
        pv = "W"
        wk = hlc.resample("W-FRI").agg({"h": "max", "l": "min", "c": "last"}).dropna()
        if len(wk) >= 2 and wk.index[-1] >= c.index[-1]:
            wk = wk.iloc[:-1] if c.index[-1].weekday() < 4 else wk      # laufende Woche nicht verwenden
    else:                                           # schwankungsreiche Aktie: Pivot des Vormonats
        pv = "M"
        wk = hlc.resample("MS").agg({"h": "max", "l": "min", "c": "last"}).dropna()
        if len(wk) >= 2:
            wk = wk.iloc[:-1]                       # laufender Monat nicht verwenden
    pw = wk.iloc[-1]
    pp = (pw.h + pw.l + pw.c) / 3
    lv = {"pp": sig(pp), "r1": sig(2 * pp - pw.l), "s1": sig(2 * pp - pw.h), "r2": sig(pp + (pw.h - pw.l)),
          "s2": sig(pp - (pw.h - pw.l)), "h20": sig(float(h.iloc[-20:].max())), "l20": sig(float(l.iloc[-20:].min())),
          "atr": sig(atr_n), "e20": sig(e20_now), "s50": sig(s50n), "pv": pv}
    # ---------------- Auswertung in einfachen Worten
    def pr(val):
        return _f(val, 2 if val < 100 else 0)
    above20, above50 = last > e20_now, last > s50n
    if above20 and above50:
        tg = "Der Kurs liegt über seiner 20-Tage-Linie und über der 50-Tage-Linie: kurzfristig zeigt der Trend nach oben."
        tk = "up"
    elif above50:
        tg = ("Der größere Aufwärtstrend steht noch (Kurs über der 50-Tage-Linie), in den letzten Tagen ist der Kurs aber "
              "unter seine 20-Tage-Linie gerutscht: eine Verschnaufpause im Aufwärtstrend.")
        tk = "pause"
    elif above20:
        tg = ("Der Kurs erholt sich gerade (über der 20-Tage-Linie), liegt aber noch unter der 50-Tage-Linie: "
              "der größere Abwärtstrend ist noch nicht gedreht.")
        tk = "rebound"
    else:
        tg = "Der Kurs liegt unter seiner 20-Tage-Linie und unter der 50-Tage-Linie: kurzfristig zeigt der Trend nach unten."
        tk = "down"
    tg += (" Die Bewegung ist kräftig (ADX über 25)." if adx >= 25 else
           " Insgesamt bewegt sich die Aktie eher seitwärts, ohne klare Richtung (ADX unter 20)." if adx < 20 else "")
    macd_up = hs[-1] > 0
    hdir = 1 if h3[-1] > h3[-2] > h3[-3] else -1 if h3[-1] < h3[-2] < h3[-3] else 0
    if macd_up and hdir >= 0:
        mg = "Der Schwung nimmt zu: Käufer haben derzeit die Oberhand (MACD über seiner Signallinie)."
        mk = 1
    elif not macd_up and hdir <= 0:
        mg = "Der Schwung lässt nach: in den letzten Tagen überwiegen die Verkäufer (MACD unter seiner Signallinie)."
        mk = -1
    else:
        mg = "Der Schwung ist gemischt: " + ("die Käufer verlieren gerade etwas Kraft." if macd_up else "die Verkäufer verlieren gerade etwas Kraft.")
        mk = 0
    if round(r_now) >= 70:
        mg += " Der RSI über 70 heißt überkauft: die Aktie ist in kurzer Zeit stark gestiegen, danach folgt oft eine Pause oder ein Rücksetzer."
    elif round(r_now) <= 30:
        mg += " Der RSI unter 30 heißt überverkauft: die Aktie ist in kurzer Zeit stark gefallen, danach folgt oft eine Gegenbewegung nach oben."
    else:
        mg += " Der RSI von %s liegt im normalen Bereich (30 bis 70), %s." % (
            de(r_now, 0), "eher auf der starken Seite" if r_now >= 50 else "eher auf der schwachen Seite")
    vg = ("Die Aktie schwankt an einem normalen Tag um etwa ±%s %%. " % de(atrp * 100, 1)) + (
        "Sie liegt gerade über ihrer üblichen Schwankungsbreite (oberes Bollinger-Band) und ist damit kurzfristig überdehnt." if pb > 1 else
        "Sie liegt gerade unter ihrer üblichen Schwankungsbreite (unteres Bollinger-Band): ungewöhnlich stark gefallen." if pb < 0 else
        "Sie liegt im oberen Teil ihrer üblichen Schwankungsbreite." if pb >= 0.5 else "Sie liegt im unteren Teil ihrer üblichen Schwankungsbreite.")
    if bwr is not None and bwr <= 0.15:
        vg += " Die Schwankungen sind zuletzt ungewöhnlich klein geworden; danach kommt oft eine größere Bewegung, in welche Richtung ist offen."
    og = None
    if v is not None:
        ok_ = rel > 0.1 or (udr or 1) > 1.2
        bad_ = rel < -0.1 or (udr or 1) < 0.8
        og = ("An Tagen mit steigenden Kursen wird mehr gehandelt als an Tagen mit fallenden: das spricht dafür, dass eher gekauft wird." if ok_ and not bad_ else
              "An Tagen mit fallenden Kursen wird mehr gehandelt als an Tagen mit steigenden: das spricht dafür, dass eher verkauft wird." if bad_ and not ok_ else
              "Das Handelsvolumen gibt kein klares Bild, Käufer und Verkäufer halten sich in etwa die Waage.")
        if vr is not None and vr >= 1.5:
            og += " Am letzten Handelstag war das Volumen deutlich höher als üblich, die Bewegung an diesem Tag hat also Gewicht."
    kg = ("Die Aktie hat gerade ihr Hoch der letzten 20 Tage überschritten (Ausbruch nach oben)." if last > h20p else
          "Die Aktie ist gerade unter ihr Tief der letzten 20 Tage gefallen." if last < l20p else
          "Die Aktie bewegt sich innerhalb ihrer Spanne der letzten 20 Tage.")
    kg += (" Die Hochs und Tiefs werden höher, ein Zeichen für einen Aufwärtstrend." if hh_ and hl_ else
           " Die Hochs und Tiefs werden tiefer, ein Zeichen für einen Abwärtstrend." if not hh_ and not hl_ else "")
    if abs(gap) >= 0.02:
        kg += " Zur letzten Eröffnung gab es einen Kurssprung von %s, meist nach Nachrichten." % _sp(gap)
    # Fazit
    if tk == "up" and mk >= 0:
        fz = "Kurzfristig läuft es gut: Trend und Schwung zeigen nach oben."
    elif tk == "up":
        fz = "Der Trend zeigt noch nach oben, der Schwung lässt aber nach. Das kann eine kurze Pause sein oder der Beginn eines Rücksetzers."
    elif tk == "pause":
        fz = ("Der größere Aufwärtstrend ist intakt, kurzfristig macht die Aktie aber eine Pause" +
              (" und hat an Schwung verloren." if mk < 0 else ", der Schwung dreht bereits wieder nach oben." if mk > 0 else "."))
    elif tk == "rebound":
        fz = ("Die Aktie erholt sich nach einem Rückgang" + (" mit zunehmendem Schwung" if mk > 0 else "") +
              ", der größere Trend ist aber noch nicht gedreht. Ob daraus eine echte Wende wird, zeigt sich erst über der 50-Tage-Linie bei " + pr(s50n) + ".")
    else:
        fz = ("Kurzfristig ist das Bild schwach: Trend und Schwung zeigen nach unten." if mk <= 0 else
              "Der Trend zeigt nach unten, der Schwung dreht aber schon nach oben: ein mögliches erstes Zeichen für eine Gegenbewegung.")
    if "überkauft" in flags:
        fz += " Achtung: Die Aktie ist überkauft, nach solchen Phasen kommt es häufiger zu Rücksetzern."
    elif "überverkauft" in flags:
        fz += " Die Aktie ist überverkauft, nach solchen Phasen kommt es häufiger zu Gegenbewegungen."
    gap_min = 0.5 * atr_n                           # Marken direkt am Kurs sind keine Information
    gap_max = 4 * atr_n                             # weiter entfernte Marken sind nicht „in der Nähe“
    sup = max([q for q in (lv["s1"], lv["l20"], lv["s2"]) if q is not None and last - gap_max <= q < last - gap_min] or [None],
              key=lambda q: q or 0)
    res = min([q for q in (lv["r1"], lv["h20"], lv["r2"]) if q is not None and last + gap_min < q <= last + gap_max] or [None],
              key=lambda q: q or 1e18)
    if sup or res:
        fz += " Wichtige Marken in der Nähe:" + (" Unterstützung bei %s (%s)" % (pr(sup), _sp(sup / last - 1)) if sup else "") + \
              ("," if sup and res else "") + (" Widerstand bei %s (%s)" % (pr(res), _sp(res / last - 1)) if res else "") + "."
        if sup and res:
            fz += " Fällt der Kurs unter %s, trübt sich das Bild weiter ein; steigt er über %s, wäre das ein Zeichen von Stärke." % (pr(sup), pr(res))
    dt = {"fz": fz, "g": {"Trend": tg, "Momentum": mg, "Volatilität": vg, "Kursmuster": kg}}
    if og:
        dt["g"]["Volumen"] = og

    # ---------------- Reihen für den Tageschart (170 Tage, die Seite rechnet Linien und Indikatoren selbst)
    cc = c.iloc[-170:]
    months, prev = [], None
    for i, d in enumerate(cc.index):
        if d.month != prev:
            months.append([i, int(d.month)])
            prev = d.month
    ser = {"c": [sig(float(q), 5) for q in cc.values],
           "v": [int(q) for q in v.reindex(cc.index).fillna(0).values] if v is not None else None,
           "m": months, "d": str(cc.index[-1].date()),
           # Abstand der Handelstage in Kalendertagen (eine Ziffer je Tag, 9 = 9 oder mehr), für Datumsangaben im Chart
           "dg": "".join(str(min(9, (cc.index[i] - cc.index[i - 1]).days)) for i in range(1, len(cc)))}
    return {"n": note, "lb": lb, "txt": txt, "dt": dt, "live": bool(live), "pos": pos, "neg": neg, "cnt": len(scored),
            "g": [[k, r] for k, r in rows.items() if r], "lv": lv, "fl": list(dict.fromkeys(flags))}, ser


# ----------------------------------------------------------------------------------------------------------------
# Bewertung einer Aktie
# ----------------------------------------------------------------------------------------------------------------

def consensus_refs(items):
    """items: Liste (Land, Region, Konsens). Ergebnis: dict Land -> (Vergleichswert, Weltmedian, Bezeichnung).
    Länder mit mindestens 8 Werten nutzen ihren eigenen Median, sonst den Median ihrer Region."""
    vals = [r for _, _, r in items if r is not None]
    if len(vals) < 20:
        return {}
    g = float(np.median(vals))
    byc, byr = {}, {}
    for c, reg, r in items:
        if r is None:
            continue
        byc.setdefault(c, []).append(r)
        byr.setdefault(reg, []).append(r)
    out = {}
    for c, reg, _ in items:
        if c in out:
            continue
        if len(byc.get(c, [])) >= 8:
            out[c] = (float(np.median(byc[c])), g, c)
        elif len(byr.get(reg, [])) >= 8:
            out[c] = (float(np.median(byr[reg])), g, reg)
        else:
            out[c] = (g, g, "weltweit")
    return out


def score(fund, pm, mcap_usd=None, fx_fin=None, today=None, symbol=None, cons_ref=None):
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
    # Gegenprobe: Wachstum aus den Gewinnschätzungen für das laufende und das nächste Jahr. Gilt der niedrigere Wert.
    et = fund.get("epsTrend") or {}
    e0, e1 = (et.get("0y") or {}).get("cur"), (et.get("+1y") or {}).get("cur")
    if eps_e is not None and e0 and e1 and e0 > 0 and e1 > 0:
        eps_e = min(eps_e, e1 / e0 - 1)
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
    # Weicht die Marge aus dem Jahresabschluss stark von Yahoos Angabe für die letzten zwölf Monate ab, ist oft eine
    # der Zeilen falsch zugeordnet (z. B. Bruttogewinn statt EBIT). Dann zählt vorsichtshalber der niedrigere Wert.
    margin_raw, ym0 = margin, bounded(y.get("operatingMargins"), -2, 1)
    if margin is not None and ym0 is not None and abs(margin - ym0) > 0.12:
        margin = min(margin, ym0)
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
    # Negatives Eigenkapital (oft durch Aktienrückkäufe) gilt wie in der Vorlage als nicht erfüllt.
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
    if 0.08 < dy <= 0.15:
        notes.append("Dividendenrendite von %s %% aus den Zahlungen der letzten zwölf Monate; kann eine "
                     "Sonderdividende enthalten" % de(dy * 100, 1))
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
    if rmean is None:
        # Yahoo liefert den Konsens manchmal nicht, die Verteilung der Empfehlungen aber schon: dann daraus berechnen
        # (Stark kaufen = 1 bis Stark verkaufen = 5, wie Yahoos eigener Wert).
        rc = recs.get("0m")
        if n_an and rc and len(rc) == 5 and sum(rc) >= 3:
            rmean = bounded(sum((i + 1) * v for i, v in enumerate(rc)) / sum(rc), 1, 5)
    # Analysten urteilen je nach Land unterschiedlich freundlich (Indien, Korea, Taiwan im Schnitt um 1,5, Deutschland
    # um 2,1, Australien um 3). Gewertet wird deshalb der Konsens im Vergleich zum Durchschnitt des eigenen Landes,
    # umgerechnet auf den weltweiten Maßstab: bereinigt = Konsens − Landesmedian + Weltmedian.
    rraw, radj = rmean, rmean
    if rmean is not None and cons_ref and cons_ref[0] is not None and cons_ref[1] is not None:
        radj = clamp(rmean - cons_ref[0] + cons_ref[1], 1, 5)
    rmean = radj
    An, a10 = None, None
    # Nach einem starken Kursrutsch hinken die Kursziele oft hinterher. Dann zählt ein großes Kurspotenzial höchstens
    # wie ein knapp positives (1 von 3 Punkten), damit veraltete Ziele den Score nicht heben.
    drop = max(-(pm.get("p3m") or 0), dist52 or 0)          # Rückgang in drei Monaten oder vom 52-Wochen-Hoch
    stale_tgt = bool(upside is not None and upside >= 0.6 and ((pm.get("p3m") or 0) <= -0.25 or (dist52 or 0) >= 0.35))
    if stale_tgt:
        notes.append("Kursziel %s %% über dem Kurs nach einem Kursrückgang von %s %%; die Ziele sind vermutlich noch "
                     "nicht angepasst und zählen nur eingeschränkt" % (de(upside * 100, 0), de(drop * 100, 0)))
    if n_an and n_an >= 3 and rmean is not None:
        An = [
            ["cons", [r1(rraw, 2), r1(radj, 2), r1(cons_ref[0], 2) if cons_ref else None, cons_ref[2] if cons_ref else None],
             3 if rmean <= 1.8 else 2 if rmean <= 2.2 else 1 if rmean <= 2.6 else 0, 3],
            ["tgt", pct(upside), 0 if upside is None else 1 if stale_tgt else 3 if upside >= 0.20 else 2
             if upside >= 0.10 else 1 if upside >= 0 else 0, 3],
            ["rev", pct(rev90), 0 if rev90 is None else 2 if rev90 >= 0.02 else 1 if rev90 >= -0.01 else 0, 2],
            ["ud", [ups, downs], 1 if ups > downs else 0, 1],
            ["cov", int(n_an), 1 if n_an >= 10 else 0, 1],
        ]
        a10 = float(sum(x[2] for x in An))

    # ---------- Nachrichtenlage (0-10) ----------
    news, nz, ns10 = news_view(fund.get("news") or [], fund.get("name") or "", today, symbol)

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
        status = "Aufwärtstrend" if price > pm["sma50"] else "Aufwärtstrend (Konsolidierung)"
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
    if eq is not None and eq <= 0:
        notes.append("Eigenkapital negativ (oft durch Aktienrückkäufe): Eigenkapitalrendite hat keinen Wert und gilt "
                     "wie in der Vorlage als nicht erfüllt")
    if not fin and margin_raw is not None and ym is not None and abs(margin_raw - ym) > 0.12:
        notes.append("EBIT-Marge im letzten Jahresabschluss %s %%, laut Yahoo in den letzten zwölf Monaten %s %%; "
                     "gewertet wird der niedrigere Wert" % (de(margin_raw * 100), de(ym * 100)))
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
        an=dict(n=int(n_an) if n_an else None, mean=r1(rraw, 2), adj=r1(radj, 2), tgt=sig(tgt, 5), up=pct(upside), stale=stale_tgt,
                hi=sig(y.get("targetHighPrice"), 5), lo=sig(y.get("targetLowPrice"), 5),
                rec=recs.get("0m"), rec3=recs.get("-3m"), rev90=pct(rev90),
                rev30=[er.get("up30"), er.get("down30")], ud=ud[:6]),
        ev=dict(earn=earn, ed=earn_days, exdiv=fund.get("exDiv")),
        news=news[:8], nz=nz, ns=ns10, q=quality_grade, qn=notes, po=pct(payout, 0), nd=r1(nd),
        p1d=pct(pm.get("p1d")), p1w=pct(pm.get("p1w")),
        rsiw=r1((pm.get("st") or {}).get("rsi_w"), 0),
    )
    rec["kl"] = lage(price, pm, pm["rsi"], pm["zs"], pm["sym"])
    # Zusatzsignale (nur Anzeige und Texte, kein Einfluss auf den Score)
    rec["ins"] = (insider_summary(fund["ins"], today, company=fund.get("name") or "")
                  if "ins" in fund and fund["ins"] is not None else None)
    rec["eq"] = fund.get("surp") or None
    rec["rv"] = revenue_view(fund)
    if rec["rv"] and rev_e is None:          # vom Regelwerk als unplausibel verworfen: auch nicht anzeigen
        rec["rv"]["ea"], rec["rv"]["e"] = None, None
    sf, sr = bounded(y.get("shortPercentOfFloat"), 0, 1), bounded(y.get("shortRatio"), 0, 100)
    rec["sh"] = [pct(sf), r1(sr, 1)] if sf is not None or sr is not None else None
    rec["_x"] = dict(above=above, up200=up200, up50=bool(pm["sma50_prev"] and pm["sma50"] > pm["sma50_prev"]),
                     over50=price > pm["sma50"], since_high=pm["since_high"], ext=price / pm["sma200"] - 1,
                     upside=upside, rmean=rmean, rev90=rev90, ups=ups, downs=downs, margin=margin, roic=roic,
                     roe=roe, fcf_y=fcf_y, payout=payout, cov=cov, nd=nd, rev_e=rev_e,
                     net_cash=(debt is not None and cash is not None and cash > debt), fin=fin, dec=dec,
                     ocf_ok=bool((ni_first or 0) > 0) if fin else bool((first(ocf) or 0) > 0),
                     pe=pe, hist_pe=hist_pe, g=g, dy=dy, cap=cap, since_low=pm["since_low"], pb_low=pm["pb_low"], pb_age=pm["pb_age"],
                     days200=pm["days200"], capped200=pm["capped200"], lo52=pm["lo52"])
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
        r["tc"] = bool(not x["over50"] and r["t"] > 8.0)   # Kurs unter der 50-Tage-Linie: Chart-Note höchstens 8,0
        if r["tc"]:
            r["t"] = 8.0
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

        # Wie weit ist die Wende? Vier Phasen, jede an Kursdaten festgemacht.
        # Gezählt wird ab dem Tief des laufenden Rücksetzers (tiefster Kurs seit dem 52-Wochen-Hoch), nicht ab dem
        # 52-Wochen-Tief: Bei einer Aktie, die ein Jahr gestiegen ist, liegt das 52-Wochen-Tief vor dem Anstieg.
        r["sl"] = x["pb_age"]
        r["ul"] = pct(max(0.0, r["p"] / x["pb_low"] - 1)) if x["pb_low"] else None
        ph = None
        if r["d52"] is not None and r["d52"] >= 15:
            new_low = x["pb_age"] <= 20
            if x["above"]:
                ph = [4, "Wende bestätigt", "Der Kurs liegt wieder über der 200-Tage-Linie (%s)." % de(r["s200"], 2),
                      "Für einen neuen Aufwärtstrend fehlt noch ein Kurs nahe am 52-Wochen-Hoch (%s, %s %%)."
                      % (de(r["h52"], 2), sde(r["d52"] / (1 - r["d52"] / 100)))]
            elif x["over50"] and x["up50"]:
                ph = [3, "Bodenbildung", "Der Kurs liegt über einer steigenden 50-Tage-Linie, aber noch unter der "
                      "200-Tage-Linie.", "Bestätigt wäre die Wende über der 200-Tage-Linie bei %s (%s %% zum Kurs)."
                      % (de(r["s200"], 2), sde((r["s200"] / r["p"] - 1) * 100))]
            elif not new_low and (x["over50"] or (r["ul"] or 0) >= 10):
                ph = [2, "Stabilisierung", "Seit %d Handelstagen kein neues Tief im laufenden Rücksetzer, der Kurs "
                      "liegt %s %% über diesem Tief." % (x["pb_age"], de(r["ul"] or 0, 0)),
                      "Nächster Schritt wäre ein Kurs über einer steigenden 50-Tage-Linie (derzeit %s)."
                      % de(r["s50"], 2)]
            else:
                if new_low:
                    ph = [1, "Abwärtsbewegung läuft", "Das Tief des Rücksetzers ist erst %d Handelstage alt."
                          % x["pb_age"], "Ein erstes Zeichen wäre: 20 Handelstage ohne neues Tief und ein Kurs über "
                          "der 50-Tage-Linie (%s)." % de(r["s50"], 2)]
                else:
                    ph = [1, "Kein neues Tief, Trend noch abwärts", "Seit %d Handelstagen kein neues Tief, der Kurs "
                          "liegt aber unter einer fallenden 50-Tage-Linie." % x["pb_age"],
                          "Ein erstes Zeichen wäre ein Kurs über der 50-Tage-Linie (%s)." % de(r["s50"], 2)]
        r["ph"] = ph

        # Alter des Trends: Wie lange liegt der Kurs schon über der 200-Tage-Linie?
        d200 = x["days200"]
        r["d200"] = d200
        if d200 > 0:
            mature = d200 > 250 or x["ext"] >= 0.25
            r["tm"] = "früh" if d200 <= 60 else "reif" if mature else "etabliert"
        else:
            r["tm"] = None

        # Einstiegsmarken aus Bewertung und Analysten (die Chart-Zonen stehen in zs)
        ez, price = [], r["p"]
        pe, hpe, g, dy = x["pe"], x["hist_pe"], x["g"], x["dy"]
        if pe and hpe:
            if hpe < pe:
                ez.append(["V", "KGV wie im eigenen Schnitt der letzten Abschlüsse (%s)" % de(hpe), sig(price * hpe / pe),
                           pct(hpe / pe - 1)])
            else:
                ez.append(["V", "KGV liegt bereits unter dem eigenen Schnitt (%s gegenüber %s)" % (de(pe), de(hpe)),
                           None, None])
        if pe and g is not None and 1.10 - dy > 0:
            k = min(hpe or x["cap"], x["cap"])
            pe10 = (1 + g) ** 5 * k / (1.10 - dy) ** 5
            if pe10 < pe:
                ez.append(["V", "Renditeerwartung des Rechenmodells erreicht 10 % p. a.", sig(price * pe10 / pe),
                           pct(pe10 / pe - 1)])
            else:
                ez.append(["V", "Renditeerwartung des Rechenmodells liegt bereits bei mindestens 10 % p. a.",
                           None, None])
        an = r["an"]
        if an.get("n") and an.get("lo") and an["lo"] < price:
            ez.append(["A", "Niedrigstes Kursziel der %d Analysten" % an["n"], an["lo"], pct(an["lo"] / price - 1)])
        elif an.get("n") and an.get("lo"):
            ez.append(["A", "Kurs liegt unter dem niedrigsten Kursziel der %d Analysten (%s)"
                       % (an["n"], de(an["lo"], 2)), None, None])
        r["ez"] = ez

        # Einordnung: feste Reihenfolge, die erste zutreffende Regel gilt. Verglichen wird mit den Werten, wie die
        # Website sie zeigt (ganze Zahlen, kaufmännisch gerundet), damit eine angezeigte 65 auch als 65 zählt.
        d52, ph, an_ = r["d52"], r["ph"], r["an"]
        hr = lambda v: int(math.floor(v + 0.5))                # noqa: E731
        ql_nn = _parts(dict(r, ns=None), WEIGHTS, False)      # Qualität ohne Nachrichtenlage
        qr, dr = hr(r["ql"]), hr(d52)
        floor_ok = x["above"] or x["pb_age"] >= 10
        D = [["ql", round(ql_nn, 1), hr(ql_nn) >= 85], ["moat", r["m"], r["m"] >= 20],
             ["dd", d52, d52 is not None and 12 <= dr <= 35],
             ["rev", an_.get("rev90"), an_.get("rev90") is not None and an_["rev90"] >= 0],
             ["cash", None, x["ocf_ok"]], ["floor", x["pb_age"], floor_ok]]
        D = [[c[0], c[1], bool(c[2])] for c in D]
        dip = all(c[2] for c in D)
        r["D"] = D if (d52 is not None and 12 <= dr <= 35 and ql_nn >= 70) else None
        early = ph is not None and ph[0] <= 3
        if (rsi_v is not None and rsi_v >= 78) or x["ext"] >= 0.40:
            setup = "Überhitzt"
        elif r["hot"] >= HOT_MIN and dr <= 10:
            setup = "Anlauf"
        elif qr >= 65 and x["above"] and x["over50"] and dr <= 10:
            setup = "Trendstark"
        elif dip:
            setup = "Qualitäts-Dip"
        elif U and r["tu"] >= 5 and 15 <= dr <= 50 and early:
            setup = "Turnaround-Kandidat"
        elif qr >= 70 and 15 <= dr <= 40 and early:
            setup = "Qualität im Rücksetzer"
        elif qr < 45 and not x["above"]:
            setup = "Schwach"
        else:
            setup = "Neutral"
        # Im Qualitäts-Dip zählen negative Schlagzeilen nicht gegen die Aktie: Sie sind dort der Anlass des Rücksetzers.
        r["nsx"] = None
        if setup == "Qualitäts-Dip" and r.get("ns") is not None and r["ns"] < 5:
            r["nsx"], r["ns"] = r["ns"], None
            r["tot"], r["ql"] = round(total(r), 1), round(quality(r), 1)
        r["su"] = setup
        # Unterzeile für „neutral“: zeigt, warum eine starke Aktie in keine Kategorie fällt (nur Anzeige, keine Regel)
        nu = None
        if setup == "Neutral" and qr >= 65:
            if x["above"] and dr <= 10 and not x["over50"]:
                nu = "starke Qualität, kurze Konsolidierung"
            elif x["above"] and dr <= 10:
                nu = "starke Qualität, wenig Anlauf-Signale"
            elif x["above"] and dr < 15:
                nu = "starke Qualität, leichter Rücksetzer"
            elif x["above"]:
                nu = "starke Qualität, Wende bestätigt"
            else:
                nu = "starke Qualität, unter der 200-Tage-Linie"
        r["nu"] = nu
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


def neutral_text(r, x):
    """Begründung für „neutral“: welche Bedingung für eine andere Einordnung fehlt."""
    q, d52 = int(math.floor(r["ql"] + 0.5)), r["d52"] or 0
    if q >= 65 and x["above"]:
        miss = []
        if not x["over50"]:
            miss.append("der Kurs liegt unter der 50-Tage-Linie")
        if d52 > 10.5:
            miss.append("der Kurs steht %s %% unter dem 52-Wochen-Hoch (erlaubt sind höchstens 10 %%)" % de(d52, 0))
        if miss:
            return ("Einordnung: neutral. Die Kennzahlen sind stark (Qualität %d), für „trendstark“ fehlt aber: %s."
                    % (q, " und ".join(miss)))
    if q >= 65 and not x["above"] and d52 < 14.5:
        return ("Einordnung: neutral. Die Kennzahlen sind stark (Qualität %d), der Kurs liegt aber unter der "
                "200-Tage-Linie; für einen Rücksetzer oder Turnaround ist er nicht weit genug gefallen." % q)
    if q >= 65:
        return "Einordnung: neutral. Die Kennzahlen sind stark (Qualität %d), keine der Sonderlagen trifft zu." % q
    return "Einordnung: neutral. Keine der Sonderlagen trifft zu, die Kennzahlen sind mittelmäßig (Qualität %d)." % q


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
        pro.append("Mittleres Kursziel von %d Analysten liegt %s %% über dem Kurs%s"
                   % (r["an"]["n"], de(x["upside"] * 100, 0), " (nach dem Kursrutsch vermutlich noch nicht angepasst)"
                      if r["an"].get("stale") else ""))
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
    nz0 = r.get("nz") or {}
    if nz0.get("neg", 0) >= 3 and nz0["neg"] > nz0.get("pos", 0):
        con.append("%d von %d Schlagzeilen der letzten 30 Tage mit negativem Ton" % (nz0["neg"], nz0["n"]))
    if r.get("tm") == "reif" and r["st"].startswith("Aufwärtstrend") and 0.25 <= x["ext"] < 0.35:
        con.append("Kurs %s %% über der 200-Tage-Linie: Rücksetzer sind nach so einem Anstieg üblich"
                   % de(x["ext"] * 100, 0))
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
        "Qualitäts-Dip": "Einordnung: Qualitäts-Dip. Sehr starke Kennzahlen, Rücksetzer von %s %%, die Gewinnschätzungen "
                         "wurden in 90 Tagen nicht gesenkt und %s." % (de(r["d52"], 0), "der Kurs liegt über der "
                         "200-Tage-Linie" if x["above"] else "seit %d Handelstagen gab es kein neues Tief im Rücksetzer"
                         % x["pb_age"]),
        "Qualität im Rücksetzer": "Einordnung: Qualität im Rücksetzer. Die Kennzahlen sind stark, das Kursbild "
                                  "ist es derzeit nicht.",
        "Turnaround-Kandidat": "Einordnung: Turnaround-Kandidat. %d von 7 Turnaround-Prüfungen sind erfüllt."
                               % (r["tu"] or 0),
        "Überhitzt": "Einordnung: überhitzt. Der Kurs ist kurzfristig weit vorausgelaufen.",
        "Schwach": "Einordnung: schwach. Weder Kennzahlen noch Kursbild überzeugen.",
        "Neutral": neutral_text(r, x),
    }[su]
    sym = r["sym"]
    if su in ("Qualität im Rücksetzer", "Turnaround-Kandidat", "Qualitäts-Dip") and sym[2] is not None:
        tail += (" Der laufende Rückgang beträgt bis zu %s %%, frühere Korrekturen lagen im Mittel bei %s %% "
                 "und höchstens bei %s %%." % (de(sym[1], 0), de(sym[2], 0), de(sym[3], 0)))
    s.append(tail)
    if su == "Qualitäts-Dip" and r.get("nsx") is not None:
        s.append("Die Nachrichtenlage (%s von 10) geht hier nicht in den Score ein." % de(r["nsx"]))
    if r.get("ph") and su in ("Qualität im Rücksetzer", "Turnaround-Kandidat", "Qualitäts-Dip"):
        ph = r["ph"]
        s.append("Stand der Wende: Phase %d von 4 (%s). %s %s" % (ph[0], ph[1], ph[2], ph[3]))
    if r.get("tm") and r["st"].startswith("Aufwärtstrend"):
        mon = "über zwei Jahren" if x["capped200"] else "%d Handelstagen" % r["d200"]
        if r["tm"] == "reif":
            s.append("Der Trend ist reif: Der Kurs liegt seit %s über der 200-Tage-Linie und %s %% darüber. "
                     "Ein Rücksetzer bis zur 50-Tage-Linie (%s) wären %s %%."
                     % (mon, de(x["ext"] * 100, 0), de(r["s50"], 2), sde((r["s50"] / r["p"] - 1) * 100)))
        elif r["tm"] == "früh":
            s.append("Der Trend ist jung: Der Kurs liegt erst seit %s über der 200-Tage-Linie." % mon)
    nz = r.get("nz") or {}
    if r.get("ns") is not None:
        s.append("Nachrichten: %d Schlagzeilen zum Unternehmen in 30 Tagen, davon %d mit positivem und %d mit "
                 "negativem Ton (Stichwortsuche)." % (nz["n"], nz["pos"], nz["neg"]))
    # Zusatzsignale: Insider, Gewinnüberraschungen, Leerverkäufe
    ins, cur = r.get("ins"), r.get("cur")
    if ins and ins["nb"] and (ins["vb"] >= 1e6 or ins["nbp"] >= 2):
        t0 = ins["top"][0]
        pro.insert(0, "Insiderkäufe in sechs Monaten: %d %s von %d %s, zusammen %s; größter: %s (%s), %s am %s"
                   % (ins["nb"], "Kauf" if ins["nb"] == 1 else "Käufe", ins["nbp"], "Insider" if ins["nbp"] == 1 else "Insidern",
                      money(ins["vb"], cur), t0[1], t0[2] or "Insider", money(t0[5] or 0, cur),
                      ".".join(reversed(t0[0].split("-")))))
    elif ins and not ins["nb"] and ins["nsp"] >= 3 and ins["vs"] >= 2e7:
        con.append("Insider haben in sechs Monaten für %s verkauft (%d Insider), ohne einen Kauf; Verkäufe sind "
                   "oft planmäßig und weniger aussagekräftig als Käufe" % (money(ins["vs"], cur), ins["nsp"]))
    eq = [e for e in (r.get("eq") or []) if e[3] is not None]
    if len(eq) >= 4:
        beats = sum(1 for e in eq[:4] if e[3] > 0)
        if beats >= 3:
            pro.append("Gewinn je Aktie in %d der letzten 4 Quartale über der Schätzung" % beats)
        elif beats <= 1:
            con.append("Gewinn je Aktie in %d der letzten 4 Quartale unter der Schätzung" % (4 - beats))
    sh = r.get("sh")
    if sh and sh[0] is not None and sh[0] >= 10:
        con.append("Hohe Leerverkaufsquote: %s %% der frei handelbaren Aktien sind leerverkauft" % de(sh[0], 1))
    return pro[:6], con[:7], " ".join(s)
