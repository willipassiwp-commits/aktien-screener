"""Täglicher Lauf: Kurse für alle Aktien laden, einen Teil der Fundamentaldaten auffrischen, alles bewerten.

Ergebnis für die Website: docs/data.json (Bewertung), docs/charts.json (Wochenkurse, fünf Jahre) und
docs/history.json (Gesamtscore der letzten Läufe, für die Veränderung). Fundamentaldaten liegen im Cache
data/fundamentals.json und werden reihum erneuert (die ältesten zuerst), damit Yahoo Finance pro Lauf nur wenige
Anfragen bekommt.

Umgebungsvariablen: FUND_LIMIT (Standard 150) = wie viele Aktien pro Lauf neue Fundamentaldaten bekommen.
"""
import bisect
import json
import re
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_universe import SUBUNIT, fx_rates  # noqa: E402
from scoring import (consensus_refs, horizon_scores,  # noqa: E402
                     FIN_PE_FACTOR, FIN_TH, G_CAP, HOT_MIN, TH, TH_ALL, VERSION, WEIGHTS, finalize, num, pct,  # noqa: E402
                     price_metrics, score, short_analysis)
import external  # noqa: E402
import market  # noqa: E402
import shortplus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, DOCS = os.path.join(ROOT, "data"), os.path.join(ROOT, "docs")
FUND_PATH = os.path.join(DATA, "fundamentals.json")
OUT_PATH = os.path.join(DOCS, "data.json")
CHART_PATH = os.path.join(DOCS, "charts.json")
DAILY_PATH = os.path.join(DOCS, "daily.json")          # Tageskurse der letzten Monate für den Kurzfrist-Chart
EXT_PATH = os.path.join(DATA, "ext.json")              # Cache der externen Chartanalysen
KURZ_PATH = os.path.join(DOCS, "kurz.json")            # ausführliche Kurzfristanalyse je Aktie (lädt die Seite erst beim Öffnen)
MARKET_PATH = os.path.join(DATA, "market.json")        # Marktampel, Angst-/Gier-Index (Zustand und letzter Stand)
MEGA = [["AAPL", "Apple"], ["MSFT", "Microsoft"], ["NVDA", "NVIDIA"], ["AMZN", "Amazon"], ["GOOGL", "Alphabet"],
        ["META", "Meta"], ["AVGO", "Broadcom"], ["TSLA", "Tesla"]]
HIST_PATH = os.path.join(DOCS, "history.json")
TRACK_PATH = os.path.join(DATA, "track.json")          # altes Format, wird nur noch einmal gelesen
TRACK_PATHS = {"track": os.path.join(DATA, "track"), "cases": os.path.join(DOCS, "cases"),
               "summary": os.path.join(DOCS, "track_summary.json"), "meta": os.path.join(DATA, "track", "meta.json"),
               "legacy": TRACK_PATH}
FUND_LIMIT = int(os.environ.get("FUND_LIMIT") or 0)  # 0 = automatisch: fehlende und veraltete zuerst, sonst 150
BATCH = 40
HISTORY_YEARS = 21
HISTORY_KEEP = 120
FUND_VERSION = 3  # Einträge im Cache mit älterem Format werden zuerst neu geladen
Y_KEYS = ["trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "returnOnAssets", "operatingMargins",
          "profitMargins", "ebitda", "totalDebt", "totalCash", "totalRevenue", "payoutRatio", "beta",
          "recommendationMean", "numberOfAnalystOpinions", "targetMeanPrice", "targetHighPrice", "targetLowPrice",
          "targetMedianPrice", "shortPercentOfFloat", "shortRatio", "heldPercentInsiders"]
BENCH = {"^GSPC": "S&P 500", "URTH": "MSCI World (ETF)", "^STOXX": "STOXX Europe 600", "^GDAXI": "DAX",
         "^N225": "Nikkei 225"}
# Börsenkürzel am Ende des Symbols -> Land der Börse (für die Wahl der Heimatnotiz bei Doppelnotierungen)
SUFFIX_COUNTRY = {
    "": ("United States",), "DE": ("Germany",), "F": ("Germany",), "PA": ("France",), "L": ("United Kingdom",),
    "SW": ("Switzerland",), "AS": ("Netherlands",), "TO": ("Canada",), "HK": ("Hong Kong", "China"),
    "SS": ("China",), "SZ": ("China",), "T": ("Japan",), "ST": ("Sweden",), "CO": ("Denmark",), "OL": ("Norway",),
    "HE": ("Finland",), "MI": ("Italy",), "MC": ("Spain",), "BR": ("Belgium",), "VI": ("Austria",),
    "LS": ("Portugal",), "IR": ("Ireland",), "AX": ("Australia",), "NS": ("India",), "BO": ("India",),
    "KS": ("South Korea",), "KQ": ("South Korea",), "TW": ("Taiwan",), "TWO": ("Taiwan",), "SA": ("Brazil",),
    "MX": ("Mexico",), "SR": ("Saudi Arabia",), "JO": ("South Africa",), "SI": ("Singapore",), "JK": ("Indonesia",),
    "BK": ("Thailand",), "KL": ("Malaysia",), "TA": ("Israel",), "WA": ("Poland",), "IS": ("Turkey",),
    "SN": ("Chile",), "NZ": ("New Zealand",), "AT": ("Greece",), "PR": ("Czech Republic",), "BD": ("Hungary",),
    "QA": ("Qatar",), "AE": ("United Arab Emirates",), "PS": ("Philippines",),
}


def row(df, names):
    if df is None or getattr(df, "empty", True):
        return []
    for n in names:
        if n in df.index:
            return [num(x) for x in df.loc[n].tolist()][:5]
    return []


def dates(df):
    if df is None or getattr(df, "empty", True):
        return []
    return [str(pd.Timestamp(c).date()) for c in df.columns][:5]


def est_growth(df):
    try:
        vals = [num(df.loc[k, "growth"]) for k in ("0y", "+1y") if k in df.index]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    except Exception:
        return None


def est_years(df):
    """Erwartetes Umsatzwachstum laut Analysten für das laufende und das nächste Geschäftsjahr, je einzeln."""
    try:
        out = [num(df.loc[k, "growth"]) if k in df.index else None for k in ("0y", "+1y")]
        return out if any(v is not None for v in out) else None
    except Exception:
        return None


def long_term_growth(df):
    """Langfristige Gewinnwachstumsschätzung der Analysten, falls Yahoo sie liefert."""
    try:
        for k in ("LTG", "+5y"):
            if k in df.index:
                return num(df.loc[k, "stockTrend"])
    except Exception:
        pass
    return None


def rec_counts(df):
    """Analystenempfehlungen je Zeitraum: [klar kaufen, kaufen, halten, verkaufen, klar verkaufen]"""
    out = {}
    try:
        for _, r in df.iterrows():
            out[str(r["period"])] = [int(num(r.get(k)) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")]
    except Exception:
        pass
    return out


def eps_trend(df):
    out = {}
    try:
        for k in ("0y", "+1y"):
            if k in df.index:
                out[k] = {"cur": num(df.loc[k, "current"]), "d30": num(df.loc[k, "30daysAgo"]),
                          "d90": num(df.loc[k, "90daysAgo"])}
    except Exception:
        pass
    return out


def eps_revisions(df):
    try:
        k = "+1y" if "+1y" in df.index else "0y"
        return {"up30": num(df.loc[k, "upLast30days"]), "down30": num(df.loc[k, "downLast30days"])}
    except Exception:
        return {}


def up_downgrades(df):
    """Änderungen von Analystenurteilen der letzten 120 Tage: [Datum, Haus, von, auf, Art]"""
    out = []
    try:
        df = df.sort_index(ascending=False)
        limit = pd.Timestamp.now() - pd.Timedelta(days=120)
        for date, r in df.iterrows():
            if date < limit or len(out) >= 10:
                break
            act = str(r.get("Action") or "").lower()
            kind = "up" if act.startswith("up") else "down" if act.startswith("down") else "init" if act.startswith("init") else "keep"
            out.append([str(date.date()), str(r.get("Firm") or "")[:40], str(r.get("FromGrade") or "")[:24],
                        str(r.get("ToGrade") or "")[:24], kind])
    except Exception:
        pass
    return out


def insider_items(df):
    """Insidergeschäfte der letzten zwölf Monate: [Datum, Name, Position, "B"/"S", Stückzahl, Wert]. Nur echte Käufe
    und Verkäufe am Markt; Zuteilungen, Optionsausübungen und Schenkungen zählen nicht. Yahoo liefert das fast nur
    für US-Aktien. None = Yahoo liefert nichts, [] = keine Käufe oder Verkäufe im Zeitraum."""
    try:
        if df is None or not hasattr(df, "iterrows") or df.empty:
            return None
        cut = datetime.now(timezone.utc) - timedelta(days=365)
        out = []
        for _, r in df.iterrows():
            text = str(r.get("Text") or "") + " " + str(r.get("Transaction") or "")
            low = text.lower()
            if any(w in low for w in ("exercise", "award", "grant", "gift", "conversion", "option")):
                continue
            kind = "B" if ("purchase" in low or "buy" in low) else "S" if ("sale" in low or "sell" in low) else None
            if kind is None:
                continue
            d = r.get("Start Date")
            try:
                d = pd.Timestamp(d)
                d = d.tz_localize("UTC") if d.tzinfo is None else d
            except Exception:
                continue
            if d < cut:
                continue
            sh, val = num(r.get("Shares")), num(r.get("Value"))
            out.append([d.strftime("%Y-%m-%d"), str(r.get("Insider") or "")[:60].title(),
                        str(r.get("Position") or "")[:60], kind, int(sh) if sh else None,
                        float("%.4g" % val) if val else None])
        out.sort(key=lambda v: v[0], reverse=True)
        return out[:40]
    except Exception:
        return None


def inst_items(df):
    """Größte Fondsbestände laut den Pflichtmeldungen an die US-Börsenaufsicht (13F, bis zu 45 Tage verzögert):
    [Fonds, Anteil an der Firma, Veränderung zur Vormeldung, Wert in $, Meldedatum]. Nur Anzeige. None, wenn Yahoo
    nichts liefert."""
    try:
        if df is None or not hasattr(df, "iterrows") or df.empty:
            return None
        out = []
        for _, r in df.iterrows():
            name = str(r.get("Holder") or r.get("Organization") or "").strip()
            if not name:
                continue
            ph, pc, val = num(r.get("pctHeld") or r.get("% Out")), num(r.get("pctChange")), num(r.get("Value"))
            d = r.get("Date Reported") or r.get("Reported")
            try:
                d = pd.Timestamp(d).strftime("%Y-%m-%d")
            except Exception:
                d = None
            out.append([name[:60], None if ph is None else round(ph, 4), None if pc is None else round(pc, 4),
                        float("%.4g" % val) if val else None, d])
        return out[:10] or None
    except Exception:
        return None


def surprises(hist, dates_fn=None):
    """Die letzten vier gemeldeten Quartale: [Datum, Schätzung, Ergebnis je Aktie, Abweichung in %]."""
    out = []
    try:
        if hist is not None and hasattr(hist, "iterrows") and not hist.empty:
            for idx, r in hist.iterrows():
                est, act = num(r.get("epsEstimate")), num(r.get("epsActual"))
                if est is None or act is None:
                    continue
                out.append([pd.Timestamp(idx).strftime("%Y-%m-%d"), est, act])
    except Exception:
        out = []
    if not out and dates_fn is not None:
        try:
            dates_df = dates_fn()
            if dates_df is not None and hasattr(dates_df, "iterrows") and not dates_df.empty:
                for idx, r in dates_df.iterrows():
                    est, act = num(r.get("EPS Estimate")), num(r.get("Reported EPS"))
                    if est is None or act is None:
                        continue
                    out.append([pd.Timestamp(idx).strftime("%Y-%m-%d"), est, act])
        except Exception:
            out = []
    out.sort(key=lambda v: v[0], reverse=True)
    res = []
    for d, est, act in out[:4]:
        sp = (act - est) / abs(est) * 100 if est else None
        res.append([d, float("%.4g" % est), float("%.4g" % act), None if sp is None else round(sp, 1)])
    return res or None


def news_items(items):
    """Schlagzeilen der letzten 30 Tage: [Titel, Quelle, Datum, Link]"""
    out = []
    limit = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    for it in items or []:
        try:
            c = it.get("content")
            if isinstance(c, dict):
                title, date = c.get("title"), str(c.get("pubDate") or c.get("displayTime") or "")[:10]
                src = (c.get("provider") or {}).get("displayName")
                url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
            else:
                title, src, url = it.get("title"), it.get("publisher"), it.get("link")
                ts = it.get("providerPublishTime")
                date = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") if ts else ""
            if title and url and str(url).startswith("http") and (not date or date >= limit):
                out.append([str(title)[:170], str(src or "")[:40], date, str(url)])
        except Exception:
            continue
    return out[:8]


def calendar_dates(cal):
    earn, exdiv = None, None
    try:
        today = datetime.now(timezone.utc).date()
        ds = sorted(d for d in (cal.get("Earnings Date") or []) if d >= today - timedelta(days=1))
        earn = str(ds[0]) if ds else None
        x = cal.get("Ex-Dividend Date")
        exdiv = str(x) if x and x >= today - timedelta(days=1) else None
    except Exception:
        pass
    return earn, exdiv


def safe(fn):
    try:
        return fn()
    except YFRateLimitError:
        raise
    except Exception:
        return None


_ISIN_T = [0.0]                                     # Zeit für ISIN-Abfragen in diesem Lauf (Sekunden)


def isin_of(t):
    """ISIN der Aktie (nur für den Depot-Import auf der Seite, keine Bewertung). Eigene Abfrage über yfinance; nach
    insgesamt 120 Sekunden in einem Lauf wird nicht mehr gefragt, damit der Lauf nicht hängt."""
    if _ISIN_T[0] > 120:
        return None
    t0 = time.time()
    v = safe(lambda: t.isin)
    _ISIN_T[0] += time.time() - t0
    return v if isinstance(v, str) and re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", v) else None


def fetch_fund(sym):
    t = yf.Ticker(sym)
    info = safe(lambda: t.info) or {}
    inc = safe(lambda: t.income_stmt)
    bal = safe(lambda: t.balance_sheet)
    rev_est = safe(lambda: t.revenue_estimate)
    eps_est = safe(lambda: t.earnings_estimate)
    cf = safe(lambda: t.cashflow)
    earn, exdiv = calendar_dates(safe(lambda: t.calendar) or {})
    rev = row(inc, ["Total Revenue", "Operating Revenue"])
    gp = row(inc, ["Gross Profit"])
    gross = num(info.get("grossMargins"))
    if gross is None and gp and rev and gp[0] is not None and rev[0]:
        gross = gp[0] / rev[0]
    return {
        "ts": time.time(),
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"), "industry": info.get("industry"), "country": info.get("country"),
        "currency": info.get("currency"), "finCurrency": info.get("financialCurrency"),
        "mcap": num(info.get("marketCap")),
        "priceAtFetch": num(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")),
        "trailingEps": num(info.get("trailingEps")), "forwardEps": num(info.get("forwardEps")),
        "grossMargin": gross,
        "incDates": dates(inc),
        "revenue": rev,
        "ebit": row(inc, ["Operating Income", "EBIT", "Pretax Income"]),
        "netIncome": row(inc, ["Net Income", "Net Income Common Stockholders"]),
        "dilutedEps": row(inc, ["Diluted EPS", "Basic EPS"]),
        "ebitda": row(inc, ["EBITDA", "Normalized EBITDA"]),
        "equity": row(bal, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
        "assets": row(bal, ["Total Assets"]),
        "totalDebt": row(bal, ["Total Debt"]),
        "cash": row(bal, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]),
        "revGrowthEst": est_growth(rev_est), "epsGrowthEst": est_growth(eps_est),
        "revEstYrs": est_years(rev_est),
        "ltg": None,  # Yahoo liefert keine Langfristschätzung mehr
        "y": {k: num(info.get(k)) for k in Y_KEYS},
        "web": (str(info.get("website") or "")[:120] or None),     # nur für das Logo in der Rangliste
        "fv": FUND_VERSION,
        "ocf": row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
        "fcf": row(cf, ["Free Cash Flow"]),
        "interest": row(inc, ["Interest Expense", "Interest Expense Non Operating"]),
        "sharesDil": row(inc, ["Diluted Average Shares", "Basic Average Shares"]),
        "curAssets": row(bal, ["Current Assets"]),
        "curLiab": row(bal, ["Current Liabilities"]),
        "rec": rec_counts(safe(lambda: t.recommendations)),
        "epsTrend": eps_trend(safe(lambda: t.eps_trend)),
        "epsRev": eps_revisions(safe(lambda: t.eps_revisions)),
        "updown": up_downgrades(safe(lambda: t.upgrades_downgrades)),
        "earn": earn, "exDiv": exdiv,
        "news": news_items(safe(lambda: t.news)),
        "ins": insider_items(safe(lambda: t.insider_transactions)),
        "inst": inst_items(safe(lambda: t.institutional_holders)) if "." not in sym else None,
        "isin": isin_of(t),
        "surp": surprises(safe(lambda: t.earnings_history), lambda: safe(lambda: t.get_earnings_dates(limit=12))),
    }


def refresh_fundamentals(symbols, cache):
    def stamp(s):  # fehlende Einträge und solche im alten Format zuerst
        c = cache.get(s)
        return 0 if not c or c.get("fv") != FUND_VERSION else c.get("ts", 0)

    order = sorted(symbols, key=stamp)
    missing = sum(1 for s in symbols if stamp(s) == 0)
    limit = FUND_LIMIT or max(170, missing)
    limits, done = 0, 0
    for s in order[:limit]:
        for attempt in range(2):
            try:
                rec = fetch_fund(s)
                if not rec.get("isin") and cache.get(s, {}).get("isin"):
                    rec["isin"] = cache[s]["isin"]          # ISIN ändert sich nicht: alten Wert behalten
                # Leere Antwort: alten Stand behalten, nur den Zeitstempel weiterdrehen
                if not rec["revenue"] and cache.get(s, {}).get("revenue"):
                    cache[s]["ts"] = rec["ts"]
                else:
                    cache[s] = rec
                done += 1
                break
            except YFRateLimitError:
                limits += 1
                print("Rate-Limit bei", s, "- Pause", flush=True)
                time.sleep(120)
            except Exception as e:  # noqa: BLE001
                print("Fehler bei", s, type(e).__name__, flush=True)
                break
        if limits >= 4:
            print("Zu viele Rate-Limits, Fundamentaldaten werden beim nächsten Lauf fortgesetzt.", flush=True)
            break
        time.sleep(0.5)
        if done and done % 50 == 0:
            print("Fundamentaldaten:", done, flush=True)
            json.dump(cache, open(FUND_PATH, "w"), default=plain)
    json.dump(cache, open(FUND_PATH, "w"), default=plain)
    return done


# Ungefährer Börsenschluss in UTC je Börsenendung (Winterzeit); dient nur dem Hinweis „laufender Handelstag“
CLOSE_UTC = {"": 21.0, "TO": 21.0, "L": 16.5, "DE": 16.5, "PA": 16.5, "AS": 16.5, "BR": 16.5, "MI": 16.5, "MC": 16.5,
             "SW": 16.5, "ST": 16.5, "CO": 16.0, "HE": 16.5, "OL": 15.3, "IR": 16.5, "LS": 16.5, "VI": 16.5,
             "T": 6.5, "HK": 8.2, "TW": 5.6, "KS": 6.6, "SS": 7.0, "SZ": 7.0, "NS": 10.0, "BO": 10.0, "AX": 6.2,
             "SI": 9.2, "NZ": 4.0}


def live_bar(symbol, df):
    """True, wenn der letzte Tageskurs von heute ist und die Börse vermutlich noch handelt."""
    try:
        now = datetime.now(timezone.utc)
        last = df.index[-1]
        last = last.date() if hasattr(last, "date") else last
        if last != now.date():
            return False
        suf = symbol.rsplit(".", 1)[1] if "." in symbol else ""
        close = CLOSE_UTC.get(suf, 21.0)
        return now.hour + now.minute / 60 < close + 0.25
    except Exception:
        return False


def download_prices(symbols):
    start = (datetime.now(timezone.utc) - timedelta(days=int(HISTORY_YEARS * 365.25))).strftime("%Y-%m-%d")
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        px = None
        for attempt in range(3):
            try:
                # auto_adjust=False: Schlusskurse wie im Chart (um Splits bereinigt, nicht um Dividenden)
                px = yf.download(batch, start=start, auto_adjust=False, actions=True, group_by="ticker",
                                 progress=False, threads=True)
                if px is not None and not px.empty:
                    break
            except Exception as e:  # noqa: BLE001
                print("Kursabruf-Fehler", type(e).__name__, flush=True)
            time.sleep(30 * (attempt + 1))
        for s in batch:
            try:
                df = px[s].dropna(how="all") if px is not None else None
            except Exception:
                df = None
            yield s, df
        print("Kurse:", min(i + BATCH, len(symbols)), "/", len(symbols), flush=True)
        time.sleep(1.5)


def benchmarks(charts):
    """Vergleichsindizes: Wochenkurse für die Depotauswertung und Kursentwicklung über 1, 3, 6 und 12 Monate."""
    out = []
    try:
        start = (datetime.now(timezone.utc) - timedelta(days=int(6 * 365.25))).strftime("%Y-%m-%d")
        px = yf.download(list(BENCH), start=start, auto_adjust=False, group_by="ticker", progress=False, threads=True)
        for sym, name in BENCH.items():
            try:
                pm = price_metrics(px[sym].dropna(how="all"))
            except Exception:
                pm = None
            if pm:
                charts["_" + sym] = pm["weekly"]
                out.append([sym, name, [pct(pm["p1m"]), pct(pm["p3m"]), pct(pm["p6m"]), pct(pm["p1y"])]])
    except Exception as e:  # noqa: BLE001
        print("Vergleichsindizes nicht geladen:", type(e).__name__, flush=True)
    # Wechselkurs Dollar je Euro: Damit lässt sich der MSCI-World-ETF (in Dollar) in Euro umrechnen.
    try:
        start = (datetime.now(timezone.utc) - timedelta(days=int(5 * 365.25))).strftime("%Y-%m-%d")
        fx = yf.download("EURUSD=X", start=start, auto_adjust=False, progress=False)
        c = fx["Close"]
        if hasattr(c, "columns"):
            c = c.iloc[:, 0]
        w = c.dropna().resample("W").last().dropna()
        if len(w) >= 100 and 0.5 < float(w.iloc[-1]) < 2.5:
            charts["_EURUSD"] = [float("%.5g" % v) for v in w.values]
    except Exception as e:  # noqa: BLE001
        print("Wechselkurs nicht geladen:", type(e).__name__, flush=True)
    return out


def plain(o):
    """numpy-Werte für JSON in gewöhnliche Python-Werte umwandeln."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


def find_dupes(symbols, cache):
    """Doppelnotierungen desselben Unternehmens. Behalten wird die Notiz an der Heimatbörse (Land der Börse =
    Sitz des Unternehmens), bei Gleichstand die in der Bilanzwährung, sonst die erste."""
    by_name, dupes = {}, set()
    for s in symbols:
        f = cache.get(s) or {}
        if f.get("name"):
            by_name.setdefault(f["name"], []).append(s)

    def rank(s):
        f = cache[s]
        suffix = s.rsplit(".", 1)[1] if "." in s else ""
        home = f.get("country") in SUFFIX_COUNTRY.get(suffix, ())
        cur = SUBUNIT.get(f.get("currency"), (f.get("currency"), 1))[0]
        return (0 if home else 1, 0 if cur == f.get("finCurrency") else 1)

    for syms in by_name.values():
        if len(syms) > 1:
            keep = min(syms, key=lambda s: (rank(s), syms.index(s)))
            dupes.update(s for s in syms if s != keep)
    return dupes


def adj_lookup(df):
    """Funktion Datum -> dividendenbereinigter Schlusskurs (Adjusted Close) am Datum oder am letzten Handelstag davor."""
    if df is None or "Adj Close" not in df:
        return None
    a = df["Adj Close"].dropna().iloc[-400:]
    if a.empty:
        return None
    idx = [d.strftime("%Y-%m-%d") for d in a.index]
    vals = [float(v) for v in a.values]

    def at(day):
        i = bisect.bisect_right(idx, day) - 1
        return vals[i] if i >= 0 else None
    return at


def run_tracking(stocks, adj, rates, meta):
    """Trefferquote fortschreiben (höchstens einmal je Kalendertag, siehe tracking.py)."""
    import tracking
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not rates.get("EUR"):
        print("Kein Euro-Kurs, Trefferquote ausgelassen.")
        return None
    fx = {c: v / rates["EUR"] for c, v in rates.items() if v}
    cases = tracking.load_cases(TRACK_PATHS["cases"])
    have = {x["s"] for x in stocks}
    extra = sorted({c["s"] for c in cases if c["x"] is None or len(c["r"]) < len(tracking.HORIZONS)} - have)
    px = None
    try:
        px = yf.download(["URTH"] + extra, period="1y", auto_adjust=False, group_by="ticker", progress=False,
                         threads=True)
    except Exception as e:  # noqa: BLE001
        print("Zusatzkurse nicht geladen:", type(e).__name__, flush=True)

    def pick(sym):
        try:
            return adj_lookup(px[sym].dropna(how="all"))
        except Exception:
            return None
    mw = pick("URTH") if px is not None else None
    extra_adj = {s: f for s in extra for f in [pick(s) if px is not None else None] if f}
    info = {}
    for x in stocks:
        base, factor = SUBUNIT.get(x.get("cur"), (x.get("cur"), 1))
        m = meta.get(x["s"]) or {}
        info[x["s"]] = {"cur": base, "px": x["p"] * factor if x.get("p") else None, "rg": m.get("region"),
                        "sc": x.get("sec") or None, "ta": x.get("tm"), "cd": 1 if x.get("tc") else 0,
                        "kl": x["kl"][1] if x.get("kl") else None}
    return tracking.run(today, VERSION, stocks, adj, fx, mw, info, TRACK_PATHS, extra_adj)


def add_history(stocks):
    """Hält den Gesamtscore je Tag fest und trägt die Veränderung zum Vortag und zu vor sieben Tagen ein."""
    try:
        hist = json.load(open(HIST_PATH))
    except Exception:
        hist = {"dates": [], "tot": {}}
    if hist.get("v") != VERSION:  # neues Regelwerk: alte Scores sind nicht vergleichbar
        hist = {"v": VERSION, "dates": [], "tot": {}}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dates, tot = hist["dates"], hist["tot"]
    if dates and dates[-1] == today:
        dates.pop()
        for v in tot.values():
            if v:
                v.pop()
    n = len(dates)
    now = {x["s"]: x.get("tot") for x in stocks}
    for s in set(tot) | set(now):
        row = tot.setdefault(s, [])
        row.extend([None] * (n - len(row)))
        row.append(now.get(s))
    dates.append(today)
    if len(dates) > HISTORY_KEEP:
        cut = len(dates) - HISTORY_KEEP
        del dates[:cut]
        for s in tot:
            del tot[s][:cut]

    def back(days):
        limit = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        idx = [i for i, d in enumerate(dates[:-1]) if d <= limit]
        return idx[-1] if idx else None

    i1, i7, i30 = (len(dates) - 2 if len(dates) > 1 else None), back(7), back(30)
    for x in stocks:
        row = tot.get(x["s"]) or []
        cur = x.get("tot")
        x["c1"] = round(cur - row[i1], 1) if i1 is not None and cur is not None and row[i1] is not None else None
        x["c7"] = round(cur - row[i7], 1) if i7 is not None and cur is not None and row[i7] is not None else None
        x["c30"] = round(cur - row[i30], 1) if i30 is not None and cur is not None and row[i30] is not None else None
    with open(HIST_PATH, "w") as f:
        json.dump(hist, f, separators=(",", ":"), default=plain)


def market_context(stocks):
    """Marktbreite, Lage je Sektor und Rang jeder Aktie in ihrem Sektor. Reine Anzeige, ohne Einfluss auf den Score.
    Rückgabe: Zusammenfassung für data.json; je Aktie werden "sv" (Branchenvergleich) und "sr" ergänzt."""
    def med(v):
        v = sorted(x for x in v if x is not None)
        return None if not v else round((v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2), 1)

    def share(v):
        v = [x for x in v if x is not None]
        return round(100 * sum(1 for x in v if x) / len(v)) if v else None
    ok = [x for x in stocks if x.get("p") and x.get("s200") and x.get("s50")]
    pf = lambda x, i: (x.get("pf") or [None] * 4)[i]   # noqa: E731
    mkt = {"n": len(ok), "a200": share([x["p"] > x["s200"] for x in ok]), "a50": share([x["p"] > x["s50"] for x in ok]),
           "nh5": share([(x.get("d52") or 99) <= 5 for x in ok]), "med1": med([pf(x, 0) for x in ok]),
           "med3": med([pf(x, 1) for x in ok]), "med12": med([pf(x, 3) for x in ok]), "sec": {}}
    by = {}
    for x in ok:
        by.setdefault(x.get("sec") or "", []).append(x)
    for sec, xs in by.items():
        if not sec or len(xs) < 5:
            continue
        mkt["sec"][sec] = [len(xs), share([x["p"] > x["s200"] for x in xs]), med([pf(x, 0) for x in xs]),
                           med([pf(x, 1) for x in xs]), med([pf(x, 3) for x in xs])]
    # Branchenvergleich: Anteil der Sektorkollegen, die schlechter abschneiden (KGV: niedriger ist besser).
    # Banken und Versicherer werden nur untereinander verglichen, weil sie andere Kennzahlen haben.
    keys = [("revH", True), ("margin", True), ("nm", True), ("roic", True), ("roe", True), ("pe", False)]

    def val(x, k):
        if k == "pe":
            return x.get("pe") if x.get("pe") and x["pe"] > 0 else None
        return next((c[2] for c in x.get("F") or [] if c[1] == k), None)
    groups = {}
    for x in ok:
        groups.setdefault((x.get("sec") or "", bool(x.get("fin"))), []).append(x)
    for (sec, fin), xs in groups.items():
        for x in xs:
            x["sv"] = []
        for k, hi in keys:
            vals = [(x, val(x, k)) for x in xs]
            vals = [(x, v) for x, v in vals if v is not None]
            if len(vals) < 8:
                continue
            for x, v in vals:
                worse = sum(1 for _, w in vals if (w < v if hi else w > v))
                x["sv"].append([k, v, round(100 * worse / (len(vals) - 1)), len(vals)])
        r3 = [(x, pf(x, 1)) for x in xs if pf(x, 1) is not None]
        for x, v in r3:
            x["sr"] = round(100 * sum(1 for _, w in r3 if w < v) / max(len(r3) - 1, 1)) if len(r3) >= 5 else None
    return mkt


def main():
    uni = json.load(open(os.path.join(DATA, "universe.json")))["stocks"]
    symbols = [u["symbol"] for u in uni]
    if len(symbols) < 300:
        print("Universum unvollständig (%d Aktien). Abbruch." % len(symbols))
        sys.exit(1)
    meta = {u["symbol"]: u for u in uni}
    cache = json.load(open(FUND_PATH)) if os.path.exists(FUND_PATH) else {}
    print("Universum:", len(symbols), "| Fundamentaldaten im Cache:", len(cache), flush=True)
    refresh_fundamentals(symbols, cache)

    old, old_version = {}, None
    if os.path.exists(OUT_PATH):
        try:
            prev = json.load(open(OUT_PATH))
            old, old_version = {x["s"]: x for x in prev["stocks"]}, prev.get("v")
        except Exception:
            old = {}

    rates = fx_rates({(cache.get(s) or {}).get(k) for s in symbols for k in ("currency", "finCurrency")})

    # Vergleichswerte für den Analystenkonsens je Land (siehe scoring.consensus_refs)
    def _country(s):
        return (cache.get(s) or {}).get("country") or meta[s].get("country") or ""
    citems = []
    for s in symbols:
        y = (cache.get(s) or {}).get("y") or {}
        r, n = num(y.get("recommendationMean")), num(y.get("numberOfAnalystOpinions"))
        citems.append((_country(s), meta[s].get("region") or "", r if (r and 1 <= r <= 5 and n and n >= 3) else None))
    crefs = consensus_refs(citems)
    print("Konsens-Vergleich:", {k: round(v[0], 2) for k, v in crefs.items() if v[2] == k}, flush=True)

    dupes = find_dupes(symbols, cache)
    print("Doppelnotierungen entfernt:", len(dupes), flush=True)
    charts, adj = {}, {}
    stale_kept = 0
    try:
        old_charts = json.load(open(CHART_PATH))
    except Exception:
        old_charts = {}
    try:
        old_daily = json.load(open(DAILY_PATH))
    except Exception:
        old_daily = {}
    daily = {}
    MD = market.fetch()                            # Indizes, VIX, Sektor-ETFs (für Marktampel und Beta je Aktie)
    print("Marktdaten:", len(MD), "Reihen", flush=True)

    def mclose(sym):
        d = MD.get(sym)
        return d["Close"].astype(float) if d is not None else None
    us_closes = {}
    stocks, fresh, priced, skipped = [], [], 0, 0
    for s, df in download_prices(symbols):
        adj[s] = adj_lookup(df)
        if s in dupes:
            continue
        fund = cache.get(s)
        pm = price_metrics(df)
        if (pm is not None and fund and s in old and old_version == VERSION and old_charts.get(s)
                and str(old[s].get("d") or "") > str(pm["date"])):
            # Yahoo liefert manchmal vorübergehend Kurse, die älter sind als die schon veröffentlichten (z. B. nachts
            # am Wochenende für europäische Börsen). Dann bleibt der neuere, bereits veröffentlichte Stand stehen.
            stocks.append(old[s])
            charts[s] = old_charts[s]
            if old_daily.get(s):
                daily[s] = old_daily[s]
            stale_kept += 1
            continue
        if pm is None or not fund:
            # alten Stand weiterzeigen, die Seite markiert ihn über das Kursdatum als veraltet
            if s in old and fund and old_version == VERSION:
                stocks.append(old[s])
                if old_daily.get(s):
                    daily[s] = old_daily[s]
            else:
                skipped += 1
            continue
        priced += 1
        mcap_usd = None
        cur = fund.get("currency")
        base = SUBUNIT.get(cur, (cur, 1))[0]
        if fund.get("mcap") and fund.get("priceAtFetch") and base in rates:
            mcap_usd = fund["mcap"] * pm["last"] / fund["priceAtFetch"] * rates[base]
        if mcap_usd is None:
            mcap_usd = meta[s].get("mcap_usd")
        fin_cur = fund.get("finCurrency")
        fx_fin = rates[base] / rates[fin_cur] if base in rates and fin_cur in rates else None
        rec = score(fund, pm, mcap_usd, fx_fin, symbol=s, cons_ref=crefs.get(_country(s)))
        if rec is None:
            skipped += 1
            continue
        if (time.time() - fund["ts"]) > 21 * 86400:
            rec["qn"].append("Fundamentaldaten seit über drei Wochen nicht aktualisiert")
            rec["q"] = "C" if rec["q"] == "B" else "B" if rec["q"] == "A" else rec["q"]
        charts[s] = pm["weekly"]
        try:                                       # Mini-Chart für die Rangliste: 26 Wochen, auf 0–99 skaliert
            wk = [v for v in pm["weekly"][-26:] if v]
            if len(wk) >= 8:
                lo, hi = min(wk), max(wk)
                rec["sp"] = [round((v - lo) / (hi - lo) * 99) if hi > lo else 50 for v in wk]
                rec["spc"] = round(wk[-1] / wk[0] - 1, 3)
        except Exception:
            pass
        web = (fund or {}).get("web")
        if web:
            dom = re.sub(r"^https?://(www\.)?", "", web).split("/")[0].lower()
            if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", dom):
                rec["web"] = dom
        try:                                       # Kurzfristige Chartanalyse, nur Anzeige
            sa = short_analysis(df, live=live_bar(s, df))
        except Exception as e:  # noqa: BLE001
            print("Kurzfristanalyse fehlgeschlagen:", s, type(e).__name__, flush=True)
            sa = None
        if sa:
            rec["ka"], daily[s] = sa
        try:                                       # Erweiterte Kurzfristanalyse (Markt, Wende-Treppe, frühere Fälle)
            hname, hsym = market.home_index(s)
            secname = rec.get("sec") or fund.get("sector") or ""
            etf = market.SECTOR_ETF.get(fund.get("sector") or "") if "." not in s else None
            kx, rsl = shortplus.extend(df, mclose(hsym), mclose(etf) if etf else None, pm.get("sym"), rec.get("ql"),
                                       None, hname, market.NAMES.get(etf) if etf else None, live=live_bar(s, df))
            if kx:
                rec["kx"] = kx
                if rsl and s in daily:
                    daily[s]["rs"] = rsl
                    daily[s]["rsn"] = hname
        except Exception as e:  # noqa: BLE001
            print("Erweiterte Kurzfristanalyse fehlgeschlagen:", s, type(e).__name__, flush=True)
        if "." not in s:
            try:
                cc = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if getattr(cc.index, "tz", None) is not None:
                    cc.index = cc.index.tz_localize(None)
                us_closes[s] = cc.iloc[-1100:]
            except Exception:
                pass
        if fund.get("isin"):                       # nur für den Depot-Import
            rec["isin"] = fund["isin"]
        if fund.get("inst"):                       # Großinvestoren (13F), nur Anzeige
            rec["iv"] = fund["inst"]
        rec.update(s=s, n=fund.get("name") or meta[s]["name"], c=fund.get("country") or meta[s].get("country") or "",
                   sec=fund.get("sector") or meta[s].get("sector") or "", ind=fund.get("industry") or "",
                   cur=cur or "", fd=datetime.fromtimestamp(fund["ts"], timezone.utc).strftime("%Y-%m-%d"))
        # Kurs in Euro (Börsenkurs mal Wechselkurs; Pence und andere Untereinheiten zuerst umgerechnet)
        sub = SUBUNIT.get(cur, (cur, 1))
        rec["pxe"] = (float("%.5g" % (rec["p"] * sub[1] * rates[sub[0]] / rates["EUR"]))
                      if rec.get("p") and sub[0] in rates and rates.get("EUR") else None)
        fresh.append(rec)

    stocks.extend(finalize(fresh))
    bench = benchmarks(charts)
    print("Bewertet:", priced, "| ohne ausreichende Daten:", skipped, "| neuerer Stand behalten:", stale_kept, flush=True)
    if priced < 0.5 * len(symbols):
        print("Weniger als die Hälfte der Kurse geladen. data.json bleibt unverändert.")
        sys.exit(1 if priced == 0 else 0)
    os.makedirs(DOCS, exist_ok=True)
    add_history(stocks)
    try:
        mkt = market_context(stocks)
    except Exception as e:  # noqa: BLE001  Zusatzanzeige darf den Lauf nie aufhalten
        print("Marktumfeld nicht berechnet:", type(e).__name__, flush=True)
        mkt = None
    for x in stocks:                               # Score je Zeitraum (3 Monate, 2 Jahre+); 1 Jahr = Gesamtscore
        try:
            x["hz"] = horizon_scores(x)
        except Exception:
            x.pop("hz", None)
    try:
        tk = run_tracking(stocks, adj, rates, meta)
        print("Trefferquote:", tk or "heute schon erfasst", flush=True)
    except Exception as e:  # noqa: BLE001  Die Bewertung soll nie an der Trefferquote scheitern
        import traceback
        traceback.print_exc()
        print("Trefferquote nicht aktualisiert:", type(e).__name__, flush=True)
    for x in stocks:                               # Qualität steht erst nach finalize fest
        pb = (x.get("kx") or {}).get("pb")
        if pb:
            pb["c"][1][1] = bool(x.get("ql") is not None and x["ql"] >= 70)
            shortplus.pb_text(pb)
    mk = None
    try:                                           # Marktampel, Angst-/Gier-Index, Marktupdate
        mega = [[n, (cache.get(t) or {}).get("earn")] for t, n in MEGA if (cache.get(t) or {}).get("earn")]
        mk, _ = market.run(us_closes, MARKET_PATH, mega, D=MD)
    except Exception as e:  # noqa: BLE001
        print("Marktlage übersprungen:", type(e).__name__, e, flush=True)
    if mk and mk.get("light"):
        code = mk["light"]["c"]
        for x in stocks:
            kx = x.get("kx")
            if not kx:
                continue
            tl = kx.get("tl")
            state = "trend" if not tl else ("boden" if tl["prog"] >= 2 else "faellt")
            ml = kx.get("mk") or {}
            rel = bool(ml.get("rs") and ml["rs"]["20"][2] is not None and ml["rs"]["20"][2] > 0.03)
            c_, hm = code, None
            if "." in x["s"]:                     # außerhalb der USA zählt der Heimatmarkt
                hname, hsym = market.home_index(x["s"])
                hc = market.simple_code(MD[hsym]) if hsym in MD else None
                if hc:
                    c_, hm = hc, [hname, market.LIGHT[hc][0]]
            kx["mx"] = {"s": state, "c": c_, "hm": hm,
                        "t": shortplus.matrix_text(c_, state, ml.get("b250") or ml.get("b60"), rel)}
    try:                                           # Externe Chartanalysen, nur Anzeige
        ext = external.run([x["s"] for x in stocks], EXT_PATH)
        for x in stocks:
            xt = external.view(x["s"], ext)
            if xt:
                x["xt"] = xt
            else:
                x.pop("xt", None)
    except Exception as e:  # noqa: BLE001
        print("Externe Analysen übersprungen:", type(e).__name__, flush=True)
    # Ausführliche Kurzfristanalyse in eine eigene Datei, in data.json nur das, was Rangliste und Schnellauswahl brauchen
    try:
        old_kurz = json.load(open(KURZ_PATH))
    except Exception:
        old_kurz = {}
    kurz = {}
    for x in stocks:
        s_, ka_, kx_ = x["s"], x.get("ka"), x.get("kx")
        if ka_ and "g" in ka_:
            kurz.setdefault(s_, {})["ka"] = ka_
        if kx_:
            kurz.setdefault(s_, {})["kx"] = kx_
        if not kurz.get(s_) and old_kurz.get(s_):
            kurz[s_] = old_kurz[s_]
        K = kurz.get(s_) or {}
        ka_, kx_ = K.get("ka"), K.get("kx") or {}
        if ka_:
            x["ka"] = {"n": ka_["n"], "lb": ka_["lb"], "live": ka_.get("live")}
        else:
            x.pop("ka", None)
        x.pop("kx", None)
        ks = {}
        if kx_.get("tl"):
            ks["tp"] = kx_["tl"]["prog"]
        if kx_.get("pb"):
            ks["pm"] = kx_["pb"]["met"]
        if (kx_.get("mk") or {}).get("rs"):
            ks["rs"] = kx_["mk"]["rs"].get("20")
        if kx_.get("ex"):
            ks["en"] = kx_["ex"]["n"]
            if kx_["ex"].get("cap"):
                ks["cap"] = kx_["ex"]["cap"]
        if ks:
            x["ks"] = ks
        else:
            x.pop("ks", None)
    out = {"v": VERSION, "mx": mk, "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "universe": len(symbols), "dupes": len(dupes), "skipped": skipped,
           "th": TH, "thAll": TH_ALL, "finTh": FIN_TH, "finPe": FIN_PE_FACTOR, "gCap": G_CAP, "weights": WEIGHTS,
           "hotMin": HOT_MIN, "bench": bench, "mkt": mkt, "stocks": stocks}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"), default=plain)
    if charts:
        with open(CHART_PATH, "w") as f:
            json.dump(charts, f, separators=(",", ":"), default=plain)
    if kurz:
        with open(KURZ_PATH, "w") as f:
            json.dump(kurz, f, separators=(",", ":"), ensure_ascii=False, default=plain)
    if daily:
        with open(DAILY_PATH, "w") as f:
            json.dump(daily, f, separators=(",", ":"), default=plain)
    q = {k: sum(1 for x in stocks if x.get("q") == k) for k in "ABC"}
    print("Geschrieben:", OUT_PATH, len(stocks), "Aktien | Datenqualität", q)


if __name__ == "__main__":
    main()
