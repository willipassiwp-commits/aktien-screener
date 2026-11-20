"""Externe Chartanalysen, nur zur Anzeige (fließen nicht in die Bewertung ein).

- Trading Central über Yahoo Finance: Ausblick kurz-, mittel- und langfristig, Unterstützung und Widerstand,
  jüngste technische Ereignisse. Yahoo liefert das nur für US-Aktien.
- TradingView: technisches Rating aus gleitenden Durchschnitten und Oszillatoren (Tag, Woche, Monat), alle Regionen.

Beides wird im Cache data/ext.json mit Zeitstempel gehalten. Fällt eine Quelle aus, bleiben die letzten Werte bis zu
drei Tage stehen; danach wird nichts mehr angezeigt. Fehler dürfen den Lauf nie aufhalten.
"""
import json
import os
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json"}
TC_URL = "https://query2.finance.yahoo.com/ws/insights/v2/finance/insights?symbol="
TV_URL = "https://scanner.tradingview.com/global/scan"
TV_COLS = ["Recommend.All", "Recommend.MA", "Recommend.Other", "Recommend.All|1W", "Recommend.All|1M"]
TC_MAX_AGE = 3 * 3600          # Trading Central höchstens alle drei Stunden je Aktie neu laden
TC_PER_RUN = 140
SHOW_MAX_AGE = 3 * 86400

# Yahoo-Endung -> TradingView-Börse
TV_EX = {"DE": "XETR", "PA": "EURONEXT", "AS": "EURONEXT", "BR": "EURONEXT", "LS": "EURONEXT", "IR": "EURONEXT",
         "MI": "MIL", "MC": "BME", "L": "LSE", "SW": "SIX", "ST": "OMXSTO", "CO": "OMXCOP", "HE": "OMXHEX",
         "OL": "OSL", "VI": "VIE", "TO": "TSX", "AX": "ASX", "HK": "HKEX", "T": "TSE", "TW": "TWSE", "TWO": "TPEX",
         "KS": "KRX", "KQ": "KRX", "NS": "NSE", "BO": "BSE", "SI": "SGX", "NZ": "NZX", "SA": "BMFBOVESPA",
         "JK": "IDX", "SR": "TADAWUL", "MX": "BMV", "JO": "JSE", "WA": "GPW", "IS": "BIST", "TA": "TASE"}


def load(path):
    try:
        c = json.load(open(path))
        return {"tc": c.get("tc") or {}, "tv": c.get("tv") or {}}
    except Exception:
        return {"tc": {}, "tv": {}}


def save(path, cache):
    try:
        with open(path, "w") as f:
            json.dump(cache, f, separators=(",", ":"))
    except Exception as e:  # noqa: BLE001
        print("Externe Analysen nicht gespeichert:", type(e).__name__, flush=True)


def _get_json(url, data=None, timeout=20):
    req = urllib.request.Request(url, data=data, headers=dict(UA, **({"Content-Type": "application/json"} if data else {})))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _tc_get(sym):
    try:
        return _get_json(TC_URL + sym)
    except Exception:
        pass
    try:                                           # Ersatzweg mit Cookie und Crumb über yfinance
        from yfinance.data import YfData
        r = YfData().get(TC_URL + sym, timeout=20)
        return r.json()
    except Exception:
        return None


def _tc_parse(j):
    r = ((j or {}).get("finance") or {}).get("result") or {}
    ii = r.get("instrumentInfo") or {}
    te, kt = ii.get("technicalEvents") or {}, ii.get("keyTechnicals") or {}
    out = {}
    for k, key in (("s", "shortTermOutlook"), ("i", "intermediateTermOutlook"), ("l", "longTermOutlook")):
        o = te.get(key) or {}
        if o.get("direction"):
            out[k] = [o.get("direction"), o.get("score"), o.get("sectorDirection"), o.get("sectorScore"),
                      o.get("indexDirection"), o.get("indexScore")]
    if kt.get("support") or kt.get("resistance"):
        out["sup"], out["res"] = kt.get("support"), kt.get("resistance")
    ev = []
    for e in (r.get("events") or [])[:6]:
        if e.get("eventType"):
            ev.append([e.get("eventType"), time.strftime("%Y-%m-%d", time.gmtime(e.get("startDate") or 0)),
                       e.get("tradingHorizon"), e.get("tradeType")])
    if ev:
        out["ev"] = ev
    return out or None


def fetch_tc(symbols, cache, now=None):
    """Trading Central für US-Aktien (Symbol ohne Börsenendung), die ältesten zuerst, höchstens TC_PER_RUN je Lauf."""
    now = now or time.time()
    tc = cache["tc"]
    us = [s for s in symbols if "." not in s.replace("-", "")]
    todo = sorted((s for s in us if now - (tc.get(s) or {}).get("ts", 0) > TC_MAX_AGE),
                  key=lambda s: (tc.get(s) or {}).get("ts", 0))[:TC_PER_RUN]
    got, fails = 0, 0
    for s in todo:
        j = _tc_get(s)
        if j is None:
            fails += 1
            if fails >= 8 and got == 0:          # Quelle offenbar gesperrt: nicht weiter anfragen
                break
            continue
        p = _tc_parse(j)
        tc[s] = dict(p or {}, ts=int(now))
        got += 1
        time.sleep(0.25)
    return got, len(todo)


def tv_ticker(sym):
    """Mögliche TradingView-Kürzel für ein Yahoo-Symbol."""
    if "." in sym:
        base, suf = sym.rsplit(".", 1)
        ex = TV_EX.get(suf)
        if not ex:
            return []
        if suf == "HK":
            base = base.lstrip("0") or "0"
        elif suf in ("ST", "CO", "HE", "OL"):
            base = base.replace("-", "_")
        else:
            base = base.replace("-", ".")
        if suf == "L" and len(base) <= 2:           # TradingView schreibt kurze Londoner Kürzel mit Punkt (RR., BP.)
            return [ex + ":" + base + ".", ex + ":" + base]
        return [ex + ":" + base]
    b = sym.replace("-", ".")
    return ["NASDAQ:" + b, "NYSE:" + b, "AMEX:" + b]


def fetch_tv(symbols, cache, now=None):
    now = now or time.time()
    back = {}
    for s in symbols:
        for t in tv_ticker(s):
            back[t] = s
    tickers = list(back)
    got = 0
    for i in range(0, len(tickers), 300):
        chunk = tickers[i:i + 300]
        try:
            j = _get_json(TV_URL, json.dumps({"symbols": {"tickers": chunk, "query": {"types": []}},
                                              "columns": TV_COLS}).encode())
        except Exception as e:  # noqa: BLE001
            print("TradingView nicht erreichbar:", type(e).__name__, flush=True)
            break
        for row in j.get("data") or []:
            s = back.get(row.get("s"))
            d = row.get("d") or []
            if s and len(d) == len(TV_COLS) and d[0] is not None:
                cache["tv"][s] = {"r": [None if x is None else round(float(x), 3) for x in d], "ts": int(now),
                                  "t": row.get("s")}
                got += 1
        time.sleep(1)
    return got


def view(sym, cache, now=None):
    """Kompakte Anzeige-Daten für eine Aktie oder None."""
    now = now or time.time()
    out = {}
    tc = cache["tc"].get(sym)
    if tc and now - tc.get("ts", 0) <= SHOW_MAX_AGE and any(k in tc for k in ("s", "i", "l", "sup", "ev")):
        out["tc"] = {k: v for k, v in tc.items()}
    tv = cache["tv"].get(sym)
    if tv and now - tv.get("ts", 0) <= SHOW_MAX_AGE:
        out["tv"] = tv
    return out or None


def run(symbols, path):
    """Beide Quellen auffrischen; gibt den Cache zurück. Fängt alle Fehler ab."""
    cache = load(path)
    try:
        g, n = fetch_tc(symbols, cache)
        print("Trading Central:", g, "von", n, "geladen", flush=True)
    except Exception as e:  # noqa: BLE001
        print("Trading Central nicht geladen:", type(e).__name__, flush=True)
    try:
        print("TradingView:", fetch_tv(symbols, cache), "Ratings", flush=True)
    except Exception as e:  # noqa: BLE001
        print("TradingView nicht geladen:", type(e).__name__, flush=True)
    save(path, cache)
    return cache
