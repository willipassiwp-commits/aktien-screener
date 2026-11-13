"""Täglicher Lauf: Kurse für alle Aktien laden, einen Teil der Fundamentaldaten auffrischen, alles bewerten.

Ergebnis für die Website: docs/data.json (Bewertung), docs/charts.json (Wochenkurse, fünf Jahre) und
docs/history.json (Gesamtscore der letzten Läufe, für die Veränderung). Fundamentaldaten liegen im Cache
data/fundamentals.json und werden reihum erneuert (die ältesten zuerst), damit Yahoo Finance pro Lauf nur wenige
Anfragen bekommt.

Umgebungsvariablen: FUND_LIMIT (Standard 150) = wie viele Aktien pro Lauf neue Fundamentaldaten bekommen.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_universe import SUBUNIT, fx_rates  # noqa: E402
from scoring import FIN_PE_FACTOR, FIN_TH, G_CAP, TH, VERSION, num, price_metrics, score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, DOCS = os.path.join(ROOT, "data"), os.path.join(ROOT, "docs")
FUND_PATH = os.path.join(DATA, "fundamentals.json")
OUT_PATH = os.path.join(DOCS, "data.json")
CHART_PATH = os.path.join(DOCS, "charts.json")
HIST_PATH = os.path.join(DOCS, "history.json")
FUND_LIMIT = int(os.environ.get("FUND_LIMIT") or 0)  # 0 = automatisch: fehlende und veraltete zuerst, sonst 150
BATCH = 40
HISTORY_YEARS = 21
HISTORY_KEEP = 120
Y_KEYS = ["trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "returnOnAssets", "operatingMargins",
          "profitMargins", "ebitda", "totalDebt", "totalCash", "totalRevenue", "payoutRatio", "beta"]
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


def long_term_growth(df):
    """Langfristige Gewinnwachstumsschätzung der Analysten, falls Yahoo sie liefert."""
    try:
        for k in ("LTG", "+5y"):
            if k in df.index:
                return num(df.loc[k, "stockTrend"])
    except Exception:
        pass
    return None


def safe(fn):
    try:
        return fn()
    except YFRateLimitError:
        raise
    except Exception:
        return None


def fetch_fund(sym):
    t = yf.Ticker(sym)
    info = safe(lambda: t.info) or {}
    inc = safe(lambda: t.income_stmt)
    bal = safe(lambda: t.balance_sheet)
    rev_est = safe(lambda: t.revenue_estimate)
    eps_est = safe(lambda: t.earnings_estimate)
    growth = safe(lambda: t.growth_estimates)
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
        "ltg": long_term_growth(growth),
        "y": {k: num(info.get(k)) for k in Y_KEYS},
    }


def refresh_fundamentals(symbols, cache):
    def stamp(s):  # fehlende Einträge und solche im alten Format zuerst
        c = cache.get(s)
        return 0 if not c or "y" not in c else c.get("ts", 0)

    order = sorted(symbols, key=stamp)
    missing = sum(1 for s in symbols if stamp(s) == 0)
    limit = FUND_LIMIT or max(150, missing)
    limits, done = 0, 0
    for s in order[:limit]:
        for attempt in range(2):
            try:
                rec = fetch_fund(s)
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
            json.dump(cache, open(FUND_PATH, "w"))
    json.dump(cache, open(FUND_PATH, "w"))
    return done


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


def add_history(stocks):
    """Hält den Gesamtscore je Tag fest und trägt die Veränderung zum Vortag und zu vor sieben Tagen ein."""
    try:
        hist = json.load(open(HIST_PATH))
    except Exception:
        hist = {"dates": [], "tot": {}}
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

    i1, i7 = (len(dates) - 2 if len(dates) > 1 else None), back(7)
    for x in stocks:
        row = tot.get(x["s"]) or []
        cur = x.get("tot")
        x["c1"] = round(cur - row[i1], 1) if i1 is not None and cur is not None and row[i1] is not None else None
        x["c7"] = round(cur - row[i7], 1) if i7 is not None and cur is not None and row[i7] is not None else None
    with open(HIST_PATH, "w") as f:
        json.dump(hist, f, separators=(",", ":"))


def main():
    uni = json.load(open(os.path.join(DATA, "universe.json")))["stocks"]
    symbols = [u["symbol"] for u in uni]
    if len(symbols) < 500:
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

    dupes = find_dupes(symbols, cache)
    print("Doppelnotierungen entfernt:", len(dupes), flush=True)
    charts = {}
    stocks, priced, skipped = [], 0, 0
    for s, df in download_prices(symbols):
        if s in dupes:
            continue
        fund = cache.get(s)
        pm = price_metrics(df)
        if pm is None or not fund:
            # alten Stand weiterzeigen, die Seite markiert ihn über das Kursdatum als veraltet
            if s in old and fund and old_version == VERSION:
                stocks.append(old[s])
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
        rec = score(fund, pm, mcap_usd, fx_fin)
        if rec is None:
            skipped += 1
            continue
        if (time.time() - fund["ts"]) > 21 * 86400:
            rec["qn"].append("Fundamentaldaten seit über drei Wochen nicht aktualisiert")
            rec["q"] = "C" if rec["q"] == "B" else "B" if rec["q"] == "A" else rec["q"]
        charts[s] = pm["weekly"]
        rec.update(s=s, n=fund.get("name") or meta[s]["name"], c=fund.get("country") or meta[s].get("country") or "",
                   sec=fund.get("sector") or meta[s].get("sector") or "", ind=fund.get("industry") or "",
                   cur=cur or "", fd=datetime.fromtimestamp(fund["ts"], timezone.utc).strftime("%Y-%m-%d"))
        stocks.append(rec)

    print("Bewertet:", priced, "| ohne ausreichende Daten:", skipped, flush=True)
    if priced < 0.5 * len(symbols):
        print("Weniger als die Hälfte der Kurse geladen. data.json bleibt unverändert.")
        sys.exit(1 if priced == 0 else 0)
    os.makedirs(DOCS, exist_ok=True)
    add_history(stocks)
    out = {"v": VERSION, "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "universe": len(symbols), "dupes": len(dupes), "skipped": skipped,
           "th": TH, "finTh": FIN_TH, "finPe": FIN_PE_FACTOR, "gCap": G_CAP, "stocks": stocks}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    if charts:
        with open(CHART_PATH, "w") as f:
            json.dump(charts, f, separators=(",", ":"))
    q = {k: sum(1 for x in stocks if x.get("q") == k) for k in "ABC"}
    print("Geschrieben:", OUT_PATH, len(stocks), "Aktien | Datenqualität", q)


if __name__ == "__main__":
    main()
