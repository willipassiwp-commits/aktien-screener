"""Marktlage: Marktampel, Angst-/Gier-Index, Bodensignale, Marktupdate mit möglichem Auslöser, Terminkalender.

Nur Anzeige, fließt nicht in den Gesamtscore ein. Alle Daten über yfinance (Indizes, VIX, Anleihen-ETFs, Sektor-ETFs)
und aus den US-Aktien des eigenen Universums (Marktbreite). Fehler dürfen den Lauf nie aufhalten: dann bleibt der
letzte Stand aus data/market.json stehen (höchstens drei Tage).
"""
import json
import math
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

US_IDX = [["^GSPC", "S&P 500"], ["^NDX", "Nasdaq 100"], ["^DJI", "Dow Jones"], ["^RUT", "Russell 2000 (Nebenwerte)"]]
EU_IDX = [["^STOXX50E", "Euro Stoxx 50"], ["^GDAXI", "DAX"], ["^STOXX", "Stoxx Europe 600"]]
SECTORS = [["XLK", "Technologie"], ["SMH", "Halbleiter"], ["IGV", "Software"], ["XLC", "Kommunikation"],
           ["XLY", "Zyklischer Konsum"], ["XLF", "Finanzen"], ["XLV", "Gesundheit"], ["XLI", "Industrie"],
           ["XLE", "Energie"], ["XLB", "Grundstoffe"], ["XLP", "Basiskonsum"], ["XLU", "Versorger"],
           ["XLRE", "Immobilien"]]
OTHER = [["^VIX", "VIX (Angstbarometer)"], ["^VIX3M", "VIX 3 Monate"], ["^VIX9D", "VIX 9 Tage"],
         ["HYG", "Hochzinsanleihen"], ["IEF", "US-Staatsanleihen 7–10 J."], ["TLT", "US-Staatsanleihen 20+ J."],
         ["GLD", "Gold"], ["^TNX", "Rendite 10-jährige US-Anleihe"], ["DX-Y.NYB", "Dollar-Index"],
         ["EURUSD=X", "Euro in Dollar"], ["SPY", "S&P-500-ETF"], ["RSP", "S&P 500 gleichgewichtet"],
         ["CL=F", "Öl (WTI)"]]
HOME = ["^GSPTSE", "^FTSE", "^N225", "^HSI", "^TWII", "^KS11", "^NSEI", "^AXJO", "000001.SS", "^STI"]
ALL = [s for s, _ in US_IDX + EU_IDX + SECTORS + OTHER] + HOME
# Börsenendung -> Vergleichsindex (Name, Symbol)
HOME_IDX = {"": ("S&P 500", "^GSPC"), "TO": ("TSX", "^GSPTSE"), "L": ("FTSE 100", "^FTSE"), "T": ("Nikkei 225", "^N225"),
            "HK": ("Hang Seng", "^HSI"), "TW": ("Taiwan-Index", "^TWII"), "KS": ("KOSPI", "^KS11"),
            "NS": ("Nifty 50", "^NSEI"), "BO": ("Nifty 50", "^NSEI"), "AX": ("ASX 200", "^AXJO"),
            "SS": ("Shanghai Composite", "000001.SS"), "SZ": ("Shanghai Composite", "000001.SS"), "SI": ("Straits Times", "^STI")}


def home_index(symbol):
    suf = symbol.rsplit(".", 1)[1] if "." in symbol else ""
    return HOME_IDX.get(suf, ("Stoxx Europe 600", "^STOXX"))
NAMES = {s: n for s, n in US_IDX + EU_IDX + SECTORS + OTHER}
# Yahoo-Sektor -> Sektor-ETF (für den Branchenvergleich je Aktie)
SECTOR_ETF = {"Technology": "XLK", "Communication Services": "XLC", "Consumer Cyclical": "XLY",
              "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI", "Energy": "XLE",
              "Basic Materials": "XLB", "Consumer Defensive": "XLP", "Utilities": "XLU", "Real Estate": "XLRE"}

# Termine aus den offiziellen Kalendern (Fed, BLS, EZB), Stand 20.11.2026; einmal im Jahr ergänzen.
EVENTS = (
    [["2026-12-09", "Fed-Zinsentscheid (mit Projektionen)", "US"], ["2027-01-27", "Fed-Zinsentscheid", "US"],
     ["2027-03-17", "Fed-Zinsentscheid", "US"], ["2027-04-28", "Fed-Zinsentscheid", "US"],
     ["2027-06-09", "Fed-Zinsentscheid", "US"], ["2027-07-28", "Fed-Zinsentscheid", "US"],
     ["2027-09-15", "Fed-Zinsentscheid", "US"], ["2027-10-27", "Fed-Zinsentscheid", "US"],
     ["2027-11-17", "Fed-Zinsentscheid", "US"]]
    + [[d, "US-Arbeitsmarktbericht", "US"] for d in
       ["2026-11-06", "2026-12-04", "2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02", "2027-05-07",
        "2027-06-04", "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-08", "2027-11-05", "2027-12-03"]]
    + [[d, "US-Inflationsdaten (CPI)", "US"] for d in
       ["2026-11-10", "2026-12-10", "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13", "2027-05-12",
        "2027-06-10", "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14", "2027-11-10", "2027-12-10"]]
    + [[d, "EZB-Zinsentscheid", "EU"] for d in
       ["2026-12-17", "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-10", "2027-07-22", "2027-09-09",
        "2027-10-28", "2027-12-16"]]
)
EVENT_HINT = {"Fed": "rund um Zinsentscheide schwanken die Kurse oft stärker",
              "Arbeitsmarkt": "überraschende Arbeitsmarktzahlen bewegen Zinsen und Aktien",
              "Inflation": "überraschende Inflationsdaten bewegen Zinsen und Aktien",
              "EZB": "wichtig vor allem für europäische Aktien und den Euro",
              "Verfall": "großer Verfallstag an den Terminbörsen, oft hohe Umsätze",
              "Zahlen": "Zahlen der großen Technologiewerte bewegen oft den ganzen Markt"}

# Stichworte in Marktschlagzeilen -> möglicher Auslöser (Schlagzeilen sind englisch)
TRIGGERS = [
    ("Arbeitsmarktdaten", r"\b(jobs report|payrolls?|nonfarm|unemployment|jobless|labor market|hiring)\b"),
    ("Inflationsdaten", r"\b(cpi|inflation|consumer prices|pce|producer prices|ppi)\b"),
    ("Fed und Zinsen", r"\b(fed|federal reserve|powell|warsh|rate cut|rate hike|fomc|interest rates?)\b"),
    ("Anleiherenditen", r"\b(treasury yields?|bond yields?|10-year)\b"),
    ("Quartalszahlen", r"\b(earnings|results|guidance|quarterly)\b"),
    ("Künstliche Intelligenz / Chips", r"\b(ai|artificial intelligence|chip|chips|semiconductor|nvidia)\b"),
    ("Zölle und Handel", r"\b(tariffs?|trade war|trade deal|export controls?)\b"),
    ("Konjunktur", r"\b(gdp|recession|retail sales|consumer confidence|ism|pmi|economy)\b"),
    ("Öl und Energie", r"\b(oil|crude|opec)\b"),
    ("Geopolitik", r"\b(war|attack|missile|sanctions?|iran|russia|ukraine|china tensions|taiwan)\b"),
    ("Politik", r"\b(shutdown|election|congress|white house|trump|tax)\b"),
]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


def _sig(v, n=5):
    if v is None or not np.isfinite(v):
        return None
    return float("%.*g" % (n, v))


def _r(a, b):
    return None if a is None or b is None or not b else a / b - 1


def download(symbols, years=3):
    import yfinance as yf
    df = yf.download(symbols, period="%dy" % years, interval="1d", auto_adjust=False, group_by="ticker",
                     progress=False, threads=True)
    out = {}
    for s in symbols:
        try:
            sub = df[s] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub.dropna(subset=["Close"])
            if getattr(sub.index, "tz", None) is not None:
                sub.index = sub.index.tz_localize(None)
            if len(sub) > 30:
                out[s] = sub
        except Exception:
            pass
    return out


def idx_view(df):
    c = df["Close"].astype(float)
    last = float(c.iloc[-1])
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    ytd0 = c[c.index < pd.Timestamp(datetime(c.index[-1].year, 1, 1))]
    return {"p": _sig(last), "d1": _sig(_r(last, float(c.iloc[-2])), 3), "w1": _sig(_r(last, float(c.iloc[-6])), 3),
            "m1": _sig(_r(last, float(c.iloc[-22])), 3), "m3": _sig(_r(last, float(c.iloc[-64])), 3) if len(c) > 64 else None,
            "ytd": _sig(_r(last, float(ytd0.iloc[-1])), 3) if len(ytd0) else None,
            "y1": _sig(_r(last, float(c.iloc[-253])), 3) if len(c) > 253 else None,
            "o20": bool(last > e20.iloc[-1]), "o50": bool(last > s50.iloc[-1]),
            "o200": bool(last > s200.iloc[-1]) if s200.notna().iloc[-1] else None,
            "s50up": bool(s50.iloc[-1] > s50.iloc[-11]), "e20": _sig(float(e20.iloc[-1])),
            "s50": _sig(float(s50.iloc[-1])), "s200": _sig(float(s200.iloc[-1])) if s200.notna().iloc[-1] else None,
            "d": str(c.index[-1].date())}


def pct_rank(series, window=504):
    """Perzentil des letzten Werts innerhalb der letzten `window` Werte (0–100)."""
    s = series.dropna().iloc[-window:]
    if len(s) < 60:
        return None
    return float((s < s.iloc[-1]).mean() * 100)


def pct_rank_series(series, window=504):
    s = series.dropna()
    return s.rolling(window, min_periods=120).apply(lambda a: (a[:-1] < a[-1]).mean() * 100 if len(a) > 1 else np.nan,
                                                     raw=True)


def breadth(us):
    """Marktbreite aus den US-Aktien des Universums. us: dict Symbol -> Series Schlusskurse."""
    if not us or len(us) < 50:
        return None
    px = pd.DataFrame(us).sort_index()
    px = px[px.index >= px.index[-1] - pd.Timedelta(days=4 * 365)].ffill(limit=3)
    px = px.loc[:, px.notna().sum() > 300]
    if px.shape[1] < 50:
        return None
    n = px.notna().sum(axis=1)
    adv = (px.diff() > 0).sum(axis=1) / n
    dec = (px.diff() < 0).sum(axis=1) / n
    o50 = (px > px.rolling(50).mean()).sum(axis=1) / n
    o200 = (px > px.rolling(200).mean()).sum(axis=1) / n
    hi = (px >= px.rolling(252).max()).sum(axis=1) / n
    lo = (px <= px.rolling(252).min()).sum(axis=1) / n
    return pd.DataFrame({"adv": adv, "dec": dec, "o50": o50, "o200": o200, "hl": hi - lo, "n": n}).iloc[260:]


def fear_greed(D, br):
    """Eigener Angst-/Gier-Index nach dem Aufbau von CNN (ohne Put/Call). 0 = extreme Angst, 100 = extreme Gier."""
    comps, series = [], {}
    sp = D.get("^GSPC")
    if sp is not None:
        c = sp["Close"]
        series["mom"] = ("Marktschwung", c / c.rolling(125).mean() - 1, False,
                         "S&P 500 im Vergleich zu seinem 125-Tage-Schnitt")
    if br is not None:
        series["str"] = ("Kursstärke", br["hl"].rolling(5).mean(), False,
                         "Anteil der US-Aktien am 52-Wochen-Hoch minus Anteil am Tief")
        series["brd"] = ("Marktbreite", br["adv"].rolling(10).mean(), False, "Anteil steigender US-Aktien, 10-Tage-Schnitt")
    if "HYG" in D and "IEF" in D:
        a = D["HYG"]["Close"].pct_change(20) - D["IEF"]["Close"].pct_change(20)
        series["jnk"] = ("Nachfrage nach Hochzinsanleihen", a, False,
                         "20-Tage-Rendite Hochzinsanleihen minus Staatsanleihen")
    if "^VIX" in D:
        v = D["^VIX"]["Close"]
        series["vol"] = ("Volatilität", v / v.rolling(50).mean() - 1, True, "VIX im Vergleich zu seinem 50-Tage-Schnitt")
    if "SPY" in D and "TLT" in D:
        a = D["SPY"]["Close"].pct_change(20) - D["TLT"]["Close"].pct_change(20)
        series["safe"] = ("Sichere Häfen", a, False, "20-Tage-Rendite Aktien minus langlaufende Staatsanleihen")
    ranks = {}
    for k, (name, s, inv, desc) in series.items():
        rs = pct_rank_series(s.astype(float))
        if inv:
            rs = 100 - rs
        ranks[k] = rs
        v = rs.dropna()
        if len(v):
            comps.append({"k": k, "n": name, "v": round(float(v.iloc[-1])), "x": desc})
    if len(comps) < 4:
        return None
    tot = pd.DataFrame(ranks).dropna(how="all").mean(axis=1).dropna()
    val = float(tot.iloc[-1])

    def at(days):
        t = tot[tot.index <= tot.index[-1] - pd.Timedelta(days=days)]
        return round(float(t.iloc[-1])) if len(t) else None
    return {"v": round(val), "lb": fg_label(val), "w1": at(7), "m1": at(30), "y1": at(365), "c": comps,
            "h": [round(float(x)) for x in tot.iloc[-130:].values]}


def fg_label(v):
    return ("extreme Angst" if v < 25 else "Angst" if v < 45 else "neutral" if v <= 55 else "Gier" if v <= 75
            else "extreme Gier")


def fg_text(c):
    v, k = c["v"], c["k"]
    lo, hi = v < 35, v > 65
    T = {"mom": ("Der S&P 500 liegt deutlich unter seinem Halbjahresschnitt.", "Der S&P 500 liegt deutlich über seinem Halbjahresschnitt."),
         "str": ("Viele Aktien stehen auf Jahrestiefs, kaum welche auf Jahreshochs.", "Viele Aktien erreichen neue Jahreshochs."),
         "brd": ("An den meisten Tagen fallen mehr Aktien als steigen.", "An den meisten Tagen steigen mehr Aktien als fallen."),
         "jnk": ("Hochzinsanleihen werden verkauft: Anleger meiden Risiko.", "Hochzinsanleihen sind gefragt: Anleger gehen Risiko ein."),
         "vol": ("Der VIX ist stark gestiegen: Anleger sichern sich ab.", "Der VIX ist niedrig: wenig Angst im Markt."),
         "safe": ("Anleger flüchten in sichere Staatsanleihen.", "Aktien laufen deutlich besser als sichere Anleihen.")}
    t = T.get(k)
    if not t:
        return ""
    return t[0] if lo else t[1] if hi else "Im üblichen Bereich."


def cnn_fg(cache, now):
    """Originalwert von CNN (inoffizieller Abruf). Fällt er aus, bleibt der letzte Wert bis zu drei Tage stehen."""
    try:
        req = urllib.request.Request(CNN_URL, headers=dict(UA, Accept="application/json", Referer="https://edition.cnn.com/"))
        with urllib.request.urlopen(req, timeout=15) as r:
            j = json.loads(r.read().decode("utf-8"))
        fg = j.get("fear_and_greed") or {}
        if fg.get("score") is not None:
            cache["cnn"] = {"v": round(float(fg["score"])), "lb": fg.get("rating"), "w1": _num(fg.get("previous_1_week")),
                            "m1": _num(fg.get("previous_1_month")), "ts": int(now)}
    except Exception as e:  # noqa: BLE001
        print("CNN-Angst-/Gier-Index nicht geladen:", type(e).__name__, flush=True)
    c = cache.get("cnn")
    return c if c and now - c.get("ts", 0) <= 3 * 86400 else None


def _num(v):
    try:
        return round(float(v))
    except Exception:
        return None


def bottom_signals(D, br):
    """Bodensignale für den Gesamtmarkt; jedes mit erfüllt/nicht erfüllt und Erklärung."""
    out = []
    sp, vx, v3 = D.get("^GSPC"), D.get("^VIX"), D.get("^VIX3M")
    if vx is not None:
        v = vx["Close"]
        spike = bool((v.iloc[-15:] > 30).any())
        turn = spike and bool((v.iloc[-3:] < v.rolling(10).mean().iloc[-3:]).all())
        out.append(["VIX-Spitze dreht", turn, "VIX war in den letzten drei Wochen über 30 und liegt seit drei Tagen unter seinem 10-Tage-Schnitt"
                    if turn else ("VIX war über 30, ist aber noch nicht klar gefallen" if spike else "keine VIX-Spitze über 30 in den letzten drei Wochen")])
        if v3 is not None:
            j = pd.concat([v, v3["Close"]], axis=1, keys=["a", "b"]).dropna()
            inv = bool((j["a"].iloc[-15:] > j["b"].iloc[-15:]).any())
            norm = inv and bool(j["a"].iloc[-1] < j["b"].iloc[-1])
            out.append(["Angst-Laufzeitkurve normalisiert", norm, "VIX lag kürzlich über dem 3-Monats-VIX (akute Angst) und ist wieder darunter"
                        if norm else ("akute Angst: VIX liegt über dem 3-Monats-VIX" if inv and j["a"].iloc[-1] >= j["b"].iloc[-1]
                                      else "keine akute Angstphase in den letzten drei Wochen")])
    if br is not None and len(br) > 30:
        e = br["adv"].ewm(span=10, adjust=False).mean()
        w = e.iloc[-10:]
        thrust = bool(w.min() < 0.40 and w.iloc[-1] > 0.615 and w.idxmin() < w.index[-1])
        out.append(["Breitenschub (Zweig)", thrust, "Anteil steigender Aktien (10-Tage-Schnitt) stieg binnen 10 Tagen von unter 40 % auf über 61,5 %; sehr selten, historisch meist ein starkes Zeichen"
                    if thrust else "kein Breitenschub (Anteil steigender Aktien, 10-Tage-Schnitt: %d %%)" % round(e.iloc[-1] * 100)])
        cap = br["dec"].iloc[-10:]
        capd = cap[cap >= 0.9]
        out.append(["Kapitulationstag", bool(len(capd)), "an mindestens einem der letzten 10 Tage fielen 90 % oder mehr der Aktien (Ausverkauf)"
                    if len(capd) else "kein Tag mit 90 % fallenden Aktien in den letzten 10 Tagen"])
    if sp is not None:
        c, vol = sp["Close"], sp.get("Volume")
        low_i = int(np.argmin(c.iloc[-30:].values))
        days_from_low = 29 - low_i
        ftd = False
        if days_from_low >= 3:
            seg = c.iloc[-30:]
            vv = vol.iloc[-30:] if vol is not None else None
            for i in range(low_i + 4, 30):
                if seg.iloc[i] / seg.iloc[i - 1] - 1 >= 0.0125 and (vv is None or vv.iloc[i] > vv.iloc[i - 1]):
                    ftd = True
        out.append(["Follow-Through-Day", ftd, "ab Tag 4 nach dem Tief stieg der S&P 500 um mindestens 1,25 % bei höherem Umsatz"
                    if ftd else "noch kein kräftiger Bestätigungstag nach dem letzten Tief"])
        e20 = c.ewm(span=20, adjust=False).mean()
        above = bool(c.iloc[-1] > e20.iloc[-1])
        out.append(["S&P 500 über der 20-Tage-Linie", above, "über der 20-Tage-Linie (%s)" % fmt_de(e20.iloc[-1], 0) if above
                    else "noch unter der 20-Tage-Linie (bei %s)" % fmt_de(e20.iloc[-1], 0)])
    return out


def fmt_de(v, d=1):
    s = ("{:,.%df}" % d).format(float(v))
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def light(D, br, bsig):
    """Marktampel aus festen Regeln. Gibt (Code, Belege) zurück."""
    sp, nd, vx, v3 = D.get("^GSPC"), D.get("^NDX"), D.get("^VIX"), D.get("^VIX3M")
    if sp is None:
        return "gray", [], 0
    s, n = idx_view(sp), idx_view(nd) if nd is not None else None
    vix = float(vx["Close"].iloc[-1]) if vx is not None else None
    vix3 = float(v3["Close"].iloc[-1]) if v3 is not None else None
    b50 = float(br["o50"].iloc[-1]) if br is not None else None
    ev = []
    ev.append("S&P 500 " + ("über" if s["o50"] else "unter") + " der 50-Tage-Linie, " + ("über" if s["o200"] else "unter") +
              " der 200-Tage-Linie; die 50-Tage-Linie " + ("steigt" if s["s50up"] else "fällt"))
    if n:
        ev.append("Nasdaq 100 " + ("über" if n["o50"] else "unter") + " der 50-Tage-Linie, " + ("über" if n["o200"] else "unter") + " der 200-Tage-Linie")
    if b50 is not None:
        ev.append("%d %% der US-Aktien im Screener über ihrer 50-Tage-Linie" % round(b50 * 100))
    if vix is not None:
        ev.append("VIX %s%s" % (fmt_de(vix, 1), " und damit über dem 3-Monats-VIX (akute Angst)" if vix3 and vix > vix3 else ""))
    nbot = sum(1 for b in bsig if b[1])
    if (vix is not None and vix > 30) or (vix and vix3 and vix > vix3 * 1.02) or (b50 is not None and b50 < 0.20):
        code = "red"
    elif not s["o50"] and not s["s50up"] and (b50 is None or b50 < 0.40):
        code = "orange"
    elif s["o50"] and s["o200"] and s["s50up"] and (n is None or n["o50"]) and (b50 is None or b50 > 0.60) and (vix is None or vix < 20):
        code = "green"
    elif s["o200"] and not s["o50"]:
        code = "yellow"
    else:
        code = "gray"
    return code, ev, nbot


LIGHT = {"green": ("Rückenwind", "Der Markt steigt breit. Gute Aktien haben Rückenwind."),
         "yellow": ("Korrektur im Aufwärtstrend", "Rücksetzer im intakten Aufwärtstrend. Solche Phasen waren oft Gelegenheiten, können aber noch Wochen dauern."),
         "gray": ("Gemischt", "Der Markt zeigt kein klares Bild: einige Signale positiv, andere negativ."),
         "orange": ("Abwärtsdruck", "Der Markt fällt. Einzelne Aktien drehen selten gegen den Markt nach oben."),
         "red": ("Stress / Ausverkauf", "Akuter Stress. Die Schwankungen sind sehr hoch, Tiefs lassen sich erst im Nachhinein erkennen."),
         "blue": ("Boden in Bildung", "Erste Zeichen, dass der Verkaufsdruck nachlässt. Bestätigt ist die Wende noch nicht.")}


def apply_hysteresis(cache, code, nbot, today):
    """Ampel wechselt erst, wenn die neue Stufe an zwei verschiedenen Tagen festgestellt wurde. Nach Rot/Orange wird
    sie blau („Boden in Bildung“), sobald mindestens zwei Bodensignale erfüllt sind."""
    st = cache.get("light") or {}
    prev = st.get("code")
    hist = st.get("hist") or []                    # [[Datum, Code]] der letzten Wochen
    if prev in ("red", "orange", "blue") or any(h[1] in ("red", "orange") for h in hist[-15:]):
        if nbot >= 2 and code in ("red", "orange", "gray", "yellow"):
            code = "blue"
    if not prev:
        new = code
    elif code == prev:
        new = prev
    else:
        cand, cdates = st.get("cand"), st.get("cdates") or []
        if cand != code:
            cand, cdates = code, []
        if today not in cdates:
            cdates.append(today)
        new = code if len(cdates) >= 2 else prev
        st["cand"], st["cdates"] = (None, []) if new == code else (cand, cdates[-3:])
    if not hist or hist[-1][0] != today:
        hist.append([today, new])
    else:
        hist[-1][1] = new
    since = st.get("since") if new == prev else today
    st.update(code=new, hist=hist[-60:], since=since or today)
    if new == prev and st.get("cand") == new:
        st["cand"], st["cdates"] = None, []
    cache["light"] = st
    return new, st


def news(cache, now):
    """Marktschlagzeilen (englisch) zu S&P 500 und Nasdaq von Yahoo."""
    items = []
    try:
        import yfinance as yf
        for s in ("^GSPC", "SPY", "QQQ", "^IXIC"):
            try:
                for n in (yf.Ticker(s).news or [])[:20]:
                    c = n.get("content") or n
                    title = c.get("title")
                    if not title:
                        continue
                    ts = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")
                    if isinstance(ts, (int, float)):
                        t = datetime.fromtimestamp(ts, timezone.utc)
                    else:
                        try:
                            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        except Exception:
                            t = None
                    url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                           or c.get("link"))
                    pub = (c.get("provider") or {}).get("displayName") or c.get("publisher")
                    items.append({"t": title.strip(), "d": t.strftime("%Y-%m-%dT%H:%MZ") if t else None, "u": url, "p": pub})
            except Exception:
                continue
    except Exception as e:  # noqa: BLE001
        print("Marktnachrichten nicht geladen:", type(e).__name__, flush=True)
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["d"] or "", reverse=True):
        k = re.sub(r"\W+", "", it["t"].lower())[:60]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    if out:
        cache["news"] = {"items": out[:25], "ts": int(now)}
    c = cache.get("news")
    return c["items"] if c and now - c.get("ts", 0) <= 2 * 86400 else []


def triggers(items, now_dt, hours=36):
    cut = (now_dt - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%MZ")
    recent = [i for i in items if (i.get("d") or "") >= cut] or items[:10]
    cnt = []
    for name, rx in TRIGGERS:
        hits = [i for i in recent if re.search(rx, i["t"], re.I)]
        if hits:
            cnt.append([name, len(hits), [h["t"] for h in hits[:3]]])
    cnt.sort(key=lambda x: -x[1])
    return cnt[:3], recent[:8]


def events_view(today, mega_earn):
    """Termine der nächsten 21 Tage und der letzten 3 Tage."""
    t = datetime.strptime(today, "%Y-%m-%d").date()
    ev = list(EVENTS)
    # großer Verfallstag: dritter Freitag im März, Juni, September, Dezember
    for y in (t.year, t.year + 1):
        for m in (3, 6, 9, 12):
            d = datetime(y, m, 15).date()
            d += timedelta(days=(4 - d.weekday()) % 7)
            ev.append([str(d), "Großer Verfallstag", "US"])
    for name, d in mega_earn:
        ev.append([d, "Quartalszahlen " + name, "US"])
    out = []
    for d, name, reg in ev:
        try:
            dd = datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            continue
        diff = (dd - t).days
        if -3 <= diff <= 21:
            key = next((k for k in EVENT_HINT if k.lower() in name.lower()), None)
            out.append([d, name, reg, diff, EVENT_HINT.get(key, "")])
    return sorted(out)


def update_text(D, iv, br, trig, ev_recent, eur):
    """Marktupdate in einfachen Worten: was ist passiert, möglicher Auslöser, Europa."""
    sp, nd = iv.get("^GSPC"), iv.get("^NDX")
    if not sp:
        return None
    parts = []
    d1 = sp["d1"] or 0

    def p(v):
        return ("+" if v >= 0 else "−") + fmt_de(abs(v) * 100, 1) + " %"
    move = ("kräftig gestiegen" if d1 >= 0.015 else "gestiegen" if d1 >= 0.004 else "kräftig gefallen" if d1 <= -0.015
            else "gefallen" if d1 <= -0.004 else "kaum verändert")
    parts.append("Der S&P 500 ist am letzten Handelstag %s (%s)%s." % (
        move, p(d1), (", der Nasdaq 100 %s" % p(nd["d1"] or 0)) if nd else ""))
    if br is not None:
        a, dcl = float(br["adv"].iloc[-1]), float(br["dec"].iloc[-1])
        parts.append("Von den US-Aktien im Screener sind %d %% gestiegen und %d %% gefallen%s." % (
            round(a * 100), round(dcl * 100), ": ein breiter Ausverkauf" if dcl >= 0.85 else ": ein breiter Anstieg" if a >= 0.85 else ""))
    secs = [(NAMES[s], iv[s]["d1"]) for s, _ in SECTORS if iv.get(s) and iv[s].get("d1") is not None]
    if len(secs) >= 5:
        secs.sort(key=lambda x: x[1])
        parts.append("Am schwächsten: %s (%s), %s: %s (%s)." % (secs[0][0], p(secs[0][1]),
                     "am stärksten" if secs[-1][1] > 0 else "am besten gehalten", secs[-1][0], p(secs[-1][1])))
    vx = D.get("^VIX")
    if vx is not None:
        v, v1 = float(vx["Close"].iloc[-1]), float(vx["Close"].iloc[-2])
        parts.append("Das Angstbarometer VIX steht bei %s (%s Punkte zum Vortag)%s." % (
            fmt_de(v, 1), ("+" if v >= v1 else "−") + fmt_de(abs(v - v1), 1),
            ", ein Zeichen für hohe Nervosität" if v > 25 else ", ruhig" if v < 15 else ""))
    tn = D.get("^TNX")
    if tn is not None:
        y, y1 = float(tn["Close"].iloc[-1]), float(tn["Close"].iloc[-2])
        if abs(y - y1) >= 0.05:
            parts.append("Die Rendite zehnjähriger US-Anleihen %s auf %s %%: %s." % (
                "stieg" if y > y1 else "fiel", fmt_de(y, 2),
                "steigende Zinsen belasten vor allem teure Wachstumsaktien" if y > y1 else "sinkende Zinsen helfen vor allem Wachstumsaktien"))
    trig_txt = None
    ev_hit = [e for e in ev_recent if e[3] in (0, -1)]
    if ev_hit:
        trig_txt = "Am %s gab es: %s." % (", ".join(sorted({e[0] for e in ev_hit})), ", ".join(e[1] for e in ev_hit))
    if trig:
        t2 = "Die Marktschlagzeilen drehen sich vor allem um " + " und ".join(t[0] for t in trig[:2]) + "."
        trig_txt = (trig_txt + " " + t2) if trig_txt else t2
    eu = iv.get("^STOXX50E") or iv.get("^GDAXI")
    eu_txt = None
    if eu and eu.get("d1") is not None:
        eu_txt = "Europa: Euro Stoxx 50 %s, DAX %s am letzten Handelstag." % (
            p(iv["^STOXX50E"]["d1"]) if iv.get("^STOXX50E") else "–", p(iv["^GDAXI"]["d1"]) if iv.get("^GDAXI") else "–")
    if eur is not None:
        eu_txt = (eu_txt + " " if eu_txt else "") + (
            "Der Dollar hat in einem Monat %s %% gegenüber dem Euro %s: US-Aktien sind in Euro gerechnet entsprechend %s gelaufen." % (
                fmt_de(abs(eur) * 100, 1), "verloren" if eur < 0 else "gewonnen", "schlechter" if eur < 0 else "besser")
            if abs(eur) >= 0.01 else "")
    return {"txt": " ".join(parts), "trig": trig_txt, "eu": eu_txt}


def fetch():
    try:
        return download(ALL)
    except Exception as e:  # noqa: BLE001
        print("Marktdaten nicht geladen:", type(e).__name__, flush=True)
        return {}


def run(us_closes, path, mega_earn=None, D=None):
    """Hauptfunktion. us_closes: dict Symbol -> Series (Schlusskurse der US-Aktien des Universums)."""
    now = time.time()
    now_dt = datetime.now(timezone.utc)
    try:
        cache = json.load(open(path))
    except Exception:
        cache = {}
    if D is None:
        D = fetch()
    if len(D) < 10 or "^GSPC" not in D:
        m = cache.get("last")
        return (m if m and now - m.get("ts", 0) <= 3 * 86400 else None), D
    out = {"ts": int(now)}
    try:
        iv = {s: idx_view(D[s]) for s in D}
        out["us"] = [[s, NAMES[s], iv[s]] for s, _ in US_IDX if s in iv]
        out["eu"] = [[s, NAMES[s], iv[s]] for s, _ in EU_IDX if s in iv]
        out["sec"] = [[s, NAMES[s], iv[s]] for s, _ in SECTORS if s in iv]
        out["oth"] = [[s, NAMES[s], iv[s]] for s, _ in OTHER if s in iv and s not in ("SPY", "^VIX9D")]
        br = breadth(us_closes)
        if br is not None:
            out["br"] = {"o50": round(float(br["o50"].iloc[-1]) * 100), "o200": round(float(br["o200"].iloc[-1]) * 100),
                         "adv": round(float(br["adv"].iloc[-1]) * 100), "dec": round(float(br["dec"].iloc[-1]) * 100),
                         "hl": round(float(br["hl"].iloc[-1]) * 100, 1), "n": int(br["n"].iloc[-1]),
                         "o50h": [round(float(x) * 100) for x in br["o50"].iloc[-130:].values]}
        rsp, spy = D.get("RSP"), D.get("SPY")
        if rsp is not None and spy is not None:
            j = pd.concat([rsp["Close"], spy["Close"]], axis=1, keys=["a", "b"]).dropna()
            out["eqw"] = _sig(float(j["a"].iloc[-1] / j["a"].iloc[-22] - j["b"].iloc[-1] / j["b"].iloc[-22]), 3)
        bs = bottom_signals(D, br)
        out["bot"] = bs
        code, ev, nbot = light(D, br, bs)
        today = str(D["^GSPC"].index[-1].date())
        code, st = apply_hysteresis(cache, code, nbot, today)
        out["light"] = {"c": code, "lb": LIGHT[code][0], "t": LIGHT[code][1], "ev": ev, "since": st.get("since"),
                        "nbot": nbot, "hist": st.get("hist")[-20:]}
        fg = fear_greed(D, br)
        if fg:
            for c in fg["c"]:
                c["t"] = fg_text(c)
            out["fg"] = fg
        cn = cnn_fg(cache, now)
        if cn:
            out["cnn"] = cn
        eur = None
        if "EURUSD=X" in D:
            e = D["EURUSD=X"]["Close"]
            eur = float(1 / (e.iloc[-1] / e.iloc[-22]) - 1)        # Dollar gegen Euro in einem Monat
            out["eur"] = _sig(eur, 3)
        evs = events_view(today, mega_earn or [])
        out["ev"] = evs
        items = news(cache, now)
        trig, recent = triggers(items, now_dt)
        out["news"] = recent
        out["trig"] = trig
        out["upd"] = update_text(D, iv, br, trig, evs, eur)
        vx = D.get("^VIX")
        if vx is not None:
            v = float(vx["Close"].iloc[-1])
            out["vixnote"] = ("Historisch lag der S&P 500 nach VIX-Ständen zwischen 30 und 40 drei Monate später im Mittel "
                              "rund 6–7 % höher und in rund vier von fünf Fällen im Plus (1990–2026). Ab einem VIX von 40 ging es "
                              "aber, wie 2008, zum Teil noch deutlich weiter abwärts: Angst war langfristig oft eine Gelegenheit, "
                              "kurzfristig kann es trotzdem weiter fallen.") if v >= 28 else None
        out["d"] = today
    except Exception as e:  # noqa: BLE001
        print("Marktauswertung unvollständig:", type(e).__name__, e, flush=True)
    cache["last"] = out
    try:
        with open(path, "w") as f:
            json.dump(cache, f, separators=(",", ":"))
    except Exception:
        pass
    return out, D
