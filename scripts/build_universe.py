"""Baut das Aktienuniversum: die rund 1000 größten börsennotierten Unternehmen weltweit.

Schritt 1: Kandidaten aus der offenen FinanceDatabase (Mega und Large Cap), je Unternehmen nur die Heimatnotiz.
Schritt 2: Aktuelle Marktkapitalisierung von Yahoo Finance holen, in USD umrechnen, die größten auswählen.

Aufruf: python scripts/build_universe.py <Pfad zu FinanceDatabase/database/equities> [--candidates-only]
"""
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TOP_N = 1000

US = ["NYQ", "NMS", "NGM", "NCM", "ASE"]
HOME = {
    "United States": US, "Canada": ["TOR"], "Germany": ["GER"], "United Kingdom": ["LSE"], "France": ["PAR"],
    "Netherlands": ["AMS"], "Switzerland": ["EBS"], "Italy": ["MIL"], "Spain": ["MCE"], "Sweden": ["STO"],
    "Denmark": ["CPH"], "Norway": ["OSL"], "Finland": ["HEL"], "Belgium": ["BRU"], "Austria": ["VIE"],
    "Portugal": ["LIS"], "Ireland": ["ISE"], "Japan": ["JPX"], "China": ["HKG", "SHH", "SHZ"], "Hong Kong": ["HKG"],
    "Taiwan": ["TAI", "TWO"], "South Korea": ["KSC", "KOE"], "India": ["NSI", "NSE", "BSE"], "Australia": ["ASX"],
    "Singapore": ["SES"], "Brazil": ["SAO"], "Mexico": ["MEX"], "Saudi Arabia": ["SAU"], "South Africa": ["JNB"],
    "Indonesia": ["JKT"], "Thailand": ["SET"], "Malaysia": ["KLS"], "Israel": ["TLV"], "Poland": ["WSE"],
    "Turkey": ["IST"], "Chile": ["SGO"], "New Zealand": ["NZE"], "Greece": ["ATH"], "Czech Republic": ["PRA"],
    "Hungary": ["BUD"], "Qatar": ["DOH"], "United Arab Emirates": ["DFM", "ADX"], "Philippines": ["PHS"],
}
# Für Länder ohne eigene Zuordnung und für Firmen, die nur im Ausland notieren (z. B. Linde, Accenture)
FALLBACK = US + ["LSE", "HKG", "AMS", "PAR", "EBS", "GER", "TOR", "ASX", "SES"]
SUBUNIT = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}


def norm(name):
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    stop = {"inc", "corp", "corporation", "co", "ltd", "limited", "plc", "sa", "se", "ag", "nv", "the", "company",
            "holdings", "holding", "group", "class", "a", "b", "c", "adr", "spa", "s", "asa", "ab", "oyj", "as"}
    w = [x for x in s.split() if x not in stop]
    return " ".join(w[:3])


def candidates(fd_dir):
    files = sorted(glob.glob(os.path.join(fd_dir, "*.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df = df[df["market_cap"].isin(["Mega Cap", "Large Cap"])]
    if "delisted" in df.columns:
        df = df[df["delisted"].astype(str).str.lower() != "true"]
    df = df.dropna(subset=["symbol", "name", "exchange"])
    junk = r"\bCDR\b|\bGDR\b|\bADR\b|\bADS\b|Depositar|Sponsored|Preferred|Series [A-Z]\b|Warrant|\bUnits?\b|\bNotes\b|%"
    df = df[~df["name"].str.contains(junk, case=False, regex=True, na=False)]
    df["key"] = df["name"].map(norm)
    df = df[df["key"] != ""]

    def rank(r):
        home = HOME.get(r["country"], [])
        if r["exchange"] in home:
            return home.index(r["exchange"])
        if r["exchange"] in FALLBACK:
            return 100 + FALLBACK.index(r["exchange"])
        return 999

    df["rank"] = df.apply(rank, axis=1)
    df = df[df["rank"] < 999]
    df["len"] = df["symbol"].str.len()
    df = df.sort_values(["key", "rank", "len", "symbol"]).drop_duplicates("key", keep="first")
    # Dasselbe Papier unter zwei Namen (gleiche ISIN): nur die Heimatnotiz behalten
    df = df.sort_values(["rank", "len", "symbol"])
    df = pd.concat([df[df["isin"].isna()], df[df["isin"].notna()].drop_duplicates("isin", keep="first")])
    out = []
    for r in df.itertuples():
        out.append({"symbol": r.symbol, "name": r.name, "country": r.country if isinstance(r.country, str) else "",
                    "sector": r.sector if isinstance(r.sector, str) else "",
                    "isin": r.isin if isinstance(r.isin, str) else ""})
    return out


def fx_rates(currencies):
    import yfinance as yf
    rates = {"USD": 1.0}
    need = sorted({SUBUNIT.get(c, (c, 1))[0] for c in currencies if c} - {"USD"})
    if need:
        px = yf.download([c + "USD=X" for c in need], period="7d", progress=False, group_by="ticker", threads=False)
        for c in need:
            try:
                rates[c] = float(px[c + "USD=X"]["Close"].dropna().iloc[-1])
            except Exception:
                pass
    return rates


def to_usd(value, cur, rates, subunit_value=True):
    base, f = SUBUNIT.get(cur, (cur, 1.0))
    if base not in rates or value is None:
        return None
    return value * rates[base] * (f if subunit_value else 1.0)


def universe_ok():
    try:
        return len(json.load(open(os.path.join(DATA, "universe.json")))["stocks"]) >= 500
    except Exception:
        return False


def main():
    fd_dir = sys.argv[1]
    cands = candidates(fd_dir)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "candidates.json"), "w") as f:
        json.dump(cands, f, ensure_ascii=False)
    print(len(cands), "Kandidaten")
    if "--candidates-only" in sys.argv:
        return

    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    cache_path = os.path.join(DATA, "mcap.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    now = time.time()
    limits = 0
    for i, c in enumerate(cands):
        s = c["symbol"]
        if s in cache and cache[s].get("mc") and now - cache[s].get("ts", 0) < 25 * 86400:
            continue
        try:
            fi = yf.Ticker(s).fast_info
            mc, cur = fi["marketCap"], fi["currency"]
            cache[s] = {"mc": float(mc) if mc else None, "cur": cur, "ts": now}
        except YFRateLimitError:
            limits += 1
            print("Rate-Limit, Pause", flush=True)
            if limits > 5:
                break
            time.sleep(120)
        except Exception:
            cache[s] = {"mc": None, "cur": None, "ts": now}
        time.sleep(0.35)
        if i % 100 == 0:
            print(i, "/", len(cands), flush=True)
            json.dump(cache, open(cache_path, "w"))
    json.dump(cache, open(cache_path, "w"))

    rates = fx_rates({v.get("cur") for v in cache.values()})
    rows = []
    for c in cands:
        v = cache.get(c["symbol"]) or {}
        usd = to_usd(v.get("mc"), v.get("cur"), rates)
        if usd and usd > 0:
            rows.append(dict(c, mcap_usd=usd))
    rows.sort(key=lambda r: -r["mcap_usd"])
    have = len(rows)
    print("Marktkapitalisierungen gefunden:", have, "von", len(cands))
    if have < TOP_N * 0.8:
        print("Zu wenige Marktkapitalisierungen. Das Universum wird nicht überschrieben.")
        sys.exit(0 if universe_ok() else 1)
    top = rows[:TOP_N]
    with open(os.path.join(DATA, "universe.json"), "w") as f:
        json.dump({"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stocks": top}, f, ensure_ascii=False)
    print("Universum:", len(top), "Aktien, kleinste Mrd. USD:", round(top[-1]["mcap_usd"] / 1e9, 1) if top else None)


if __name__ == "__main__":
    main()
