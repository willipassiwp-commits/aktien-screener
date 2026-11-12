"""Täglicher Lauf: Kurse für alle Aktien laden, einen Teil der Fundamentaldaten auffrischen, alles bewerten.

Ergebnis: docs/data.json (liest die Website). Fundamentaldaten liegen im Cache data/fundamentals.json und werden
reihum erneuert (die ältesten zuerst), damit Yahoo Finance pro Lauf nur wenige Anfragen bekommt.

Umgebungsvariablen: FUND_LIMIT (Standard 200) = wie viele Aktien pro Lauf neue Fundamentaldaten bekommen.
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
from scoring import TH, num, price_metrics, score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, DOCS = os.path.join(ROOT, "data"), os.path.join(ROOT, "docs")
FUND_PATH = os.path.join(DATA, "fundamentals.json")
OUT_PATH = os.path.join(DOCS, "data.json")
FUND_LIMIT = int(os.environ.get("FUND_LIMIT") or 0)  # 0 = automatisch: beim ersten Lauf alle, danach 200
BATCH = 40
HISTORY_YEARS = 21


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
    }


def refresh_fundamentals(symbols, cache):
    order = sorted(symbols, key=lambda s: cache.get(s, {}).get("ts", 0))
    missing = sum(1 for s in symbols if s not in cache)
    limit = FUND_LIMIT or max(200, missing)
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
                px = yf.download(batch, start=start, auto_adjust=True, actions=True, group_by="ticker",
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

    old = {}
    if os.path.exists(OUT_PATH):
        try:
            old = {x["s"]: x for x in json.load(open(OUT_PATH))["stocks"]}
        except Exception:
            old = {}

    rates = fx_rates({(cache.get(s) or {}).get("currency") for s in symbols})
    stocks, priced, skipped = [], 0, 0
    for s, df in download_prices(symbols):
        fund = cache.get(s)
        pm = price_metrics(df)
        if pm is None or not fund:
            if s in old:  # alten Stand weiterzeigen, die Seite markiert ihn über das Kursdatum als veraltet
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
        rec = score(fund, pm, mcap_usd)
        if rec is None:
            skipped += 1
            continue
        rec.update(s=s, n=fund.get("name") or meta[s]["name"], c=fund.get("country") or meta[s].get("country") or "",
                   sec=fund.get("sector") or meta[s].get("sector") or "", ind=fund.get("industry") or "",
                   cur=cur or "", fd=datetime.fromtimestamp(fund["ts"], timezone.utc).strftime("%Y-%m-%d"))
        stocks.append(rec)

    print("Bewertet:", priced, "| ohne ausreichende Daten:", skipped, flush=True)
    if priced < 0.5 * len(symbols):
        print("Weniger als die Hälfte der Kurse geladen. data.json bleibt unverändert.")
        sys.exit(1 if priced == 0 else 0)
    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "universe": len(symbols),
           "skipped": skipped, "th": TH, "stocks": stocks}
    os.makedirs(DOCS, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("Geschrieben:", OUT_PATH, len(stocks), "Aktien")


if __name__ == "__main__":
    main()
