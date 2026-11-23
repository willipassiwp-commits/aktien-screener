"""Schnelle Kurse und Tagesnachrichten für die Titelseite. Nur Anzeige, fließt in keinen Score ein.

1. Kurse großer US- und kanadischer Aktien an deutschen Handelsplätzen (Frankfurt, Xetra, Berlin, Stuttgart).
   Dort wird von 8 bis 22 Uhr gehandelt, also auch vor dem US-Börsenstart um 15:30 Uhr. Gezeigt wird die
   Veränderung seit dem letzten US-Schlusskurs, gerechnet mit dem deutschen Kurs zum Zeitpunkt des US-Schlusses
   als Bezug (so fällt der Unterschied zwischen Euro und Dollar heraus; die Kursänderung des Euro wird
   herausgerechnet).
2. US-Terminkontrakte (S&P 500, Nasdaq 100, Dow Jones) seit dem US-Schluss.
3. Die wichtigsten Schlagzeilen der letzten 24 Stunden, gewichtet nach Thema, Quelle und Kursbewegung.

Ergebnis: docs/vorb.json. Zuordnung US-Kürzel → deutsches Kürzel im Cache data/de_map.json.
Jeder Fehler wird abgefangen; ohne Ergebnis bleibt die alte Datei stehen und die Seite zeigt den Abschnitt nicht.
"""
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "docs", "data.json")
FUND_PATH = os.path.join(ROOT, "data", "fundamentals.json")
MAP_PATH = os.path.join(ROOT, "data", "de_map.json")
OUT_PATH = os.path.join(ROOT, "docs", "vorb.json")

N_STOCKS = 70                       # so viele der größten US- und kanadischen Werte
SUFFIX = [["F", "Frankfurt"], ["DE", "Xetra"], ["BE", "Berlin"], ["SG", "Stuttgart"]]
FUTURES = [["ES=F", "S&P 500"], ["NQ=F", "Nasdaq 100"], ["YM=F", "Dow Jones"]]
MAX_SEARCH = 25                     # neue Zuordnungen je Lauf
MAX_MOVE = 0.25                     # größere Abweichungen gelten als Datenfehler
LEVEL_TOL = 0.12                    # deutscher Kurs (in Dollar) muss so nah am US-Schluss liegen, sonst falsche Zuordnung

# Bekannte Kürzel an deutschen Börsen (gleich in Frankfurt, Xetra, Berlin, Stuttgart). Jede Zuordnung wird bei jedem
# Lauf am Kursniveau geprüft; passt sie nicht, wird sie nicht angezeigt und neu gesucht.
SEED = {"AAPL": "APC", "MSFT": "MSF", "NVDA": "NVD", "AMZN": "AMZ", "GOOGL": "ABEA", "GOOG": "ABEC", "META": "FB2A",
        "TSLA": "TL0", "AVGO": "1YD", "LLY": "LLY", "JPM": "CMC", "V": "3V64", "MA": "M4I", "WMT": "WMT",
        "XOM": "XONA", "JNJ": "JNJ", "AMD": "AMD", "NFLX": "NFC", "COST": "CTO", "KO": "CCC3", "PG": "PRG",
        "ORCL": "ORC", "BRK-B": "BRYN", "UNH": "UNH", "HD": "HDI", "BAC": "NCB", "CVX": "CHV", "ABBV": "4AB",
        "MU": "MTE", "INTC": "INL", "CSCO": "CIS", "PEP": "PEP", "MRK": "6MK", "ADBE": "ADB", "CRM": "FOO",
        "QCOM": "QCI", "PLTR": "PTX", "IBM": "IBM", "DIS": "WDP", "MCD": "MDO", "GS": "GOS", "PFE": "PFE",
        "NKE": "NKE", "BA": "BCO", "CAT": "CAT1", "TXN": "TII", "AMAT": "AP2", "UBER": "UT8", "INTU": "ITU",
        "NOW": "4S0", "ISRG": "IUI1", "AMGN": "AMG", "GE": "GEC", "MS": "DWD", "WFC": "NWT", "AXP": "AEC1",
        "LRCX": "LAR", "KLAC": "KLA", "ANET": "1A1", "TMO": "TN8", "ABT": "ABL", "LIN": "LIN", "HON": "ALD",
        "SBUX": "SRB", "BKNG": "PCE1", "CMCSA": "CTP2", "T": "SOBA", "VZ": "BAC", "PM": "4I1", "SHOP": "307"}

# Schlagzeilen: Themen mit deutschem Namen (Reihenfolge = Vorrang bei der Anzeige)
TOPICS = [
    ["Trump / Politik", r"\btrump\b|white house|executive order|truth social|\bcongress\b|\bsenate\b"],
    ["Zölle / Handel", r"tariff|trade war|trade deal|export (ban|curb|control|restriction)|\bsanction"],
    ["Notenbank / Zinsen", r"\bfed\b|federal reserve|powell|rate (cut|hike)|interest rate|\becb\b|lagarde|treasury yield|bond yield"],
    ["Konjunkturdaten", r"\binflation\b|\bcpi\b|\bpce\b|jobs report|payroll|unemployment|\bgdp\b|recession|retail sales"],
    ["Chips / KI", r"\bai\b|artificial intelligence|\bchips?\b|semiconductor|data cent"],
    ["Terminkontrakte", r"futures"],
    ["Börse insgesamt", r"wall street|stock market|s&p 500|\bnasdaq\b|\bdow\b|sell-?off|\brout\b|\brally\b|\bcrash\b|correction|record high"],
    ["Rohstoffe", r"\boil\b|\bopec\b|\bgold\b|crude"],
    ["Quartalszahlen", r"earnings|quarterly|\bresults\b|guidance|\bbeats?\b|\bmiss(es)?\b"],
]
GOOD_SRC = r"^(reuters|bloomberg|cnbc|the wall street journal|wsj|financial times|associated press|ap|marketwatch|barron'?s|yahoo finance|investor'?s business daily|fortune|the new york times|axios|business insider|euronews)$"
WEAK_SRC = r"prediction markets|simply wall|insider monkey|motley fool|24/7 wall|zacks|investorplace|tipranks|gurufocus|tikr|benzinga|beincrypto|stocktwits|finbold|coincentral|seeking alpha|validea|kiplinger"
PROMO = r"should you buy|\bbuy now\b|top \d+ |\d+ (stocks?|reasons)|is it (a|too late)|millionaire|forever|no-brainer|best .*stocks? to|could soar|prediction:|\bvs\.? "
ALIAS = {"GOOGL": ["Google", "Alphabet"], "GOOG": ["Google", "Alphabet"], "META": ["Meta", "Facebook"],
         "2330.TW": ["TSMC", "Taiwan Semi"], "BRK-B": ["Berkshire"], "BRK-A": ["Berkshire"], "JNJ": ["J&J"],
         "005930.KS": ["Samsung"], "000660.KS": ["SK Hynix"], "0700.HK": ["Tencent"], "9988.HK": ["Alibaba"],
         "LLY": ["Eli Lilly", "Lilly"], "JPM": ["JPMorgan"], "XOM": ["Exxon"], "WMT": ["Walmart"], "V": ["Visa"],
         "MA": ["Mastercard"], "KO": ["Coca-Cola"], "PG": ["Procter"], "BAC": ["Bank of America"]}
NOISE = {"american", "general", "united", "first", "international", "national", "bank", "global", "china", "japan",
         "the", "new", "royal", "advanced", "applied", "texas", "southern", "union", "capital", "home", "booking",
         "target", "visa", "shell", "intuitive", "progressive", "public", "constellation", "deutsche", "canadian"}


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ") if t else None


def _r(v, n=2):
    return None if v is None or not math.isfinite(v) else round(float(v), n)


# ------------------------------------------------------------------ Rechnung (im Selbsttest abgesichert)
def last_at(series, t, window_h=3):
    """Letzter Wert einer Zeitreihe [(Zeit, Wert)] bis einschließlich t, höchstens window_h Stunden davor."""
    best = None
    for ts, v in series:
        if ts <= t and (t - ts) <= timedelta(hours=window_h):
            best = (ts, v)
    return best


def calc_move(home, ger, fx, tol_min=10):
    """home, ger, fx: Listen [(Zeitpunkt, Kurs)] aufsteigend (fx = Dollar je Euro, kann leer sein).
    Ergebnis: {"c": Veränderung seit US-Schluss (Anteil), "t": Zeit des deutschen Kurses, "ref": US-Schluss,
    "th": Zeit US-Schluss} oder None, wenn es nach dem US-Schluss keinen deutschen Kurs gibt oder die Daten nicht
    zusammenpassen."""
    if not home or not ger:
        return None
    th, ph = home[-1]
    tg, pg = ger[-1]
    if tg <= th + timedelta(minutes=tol_min) or not ph or not pg:
        return None                                 # kein neuerer deutscher Kurs (US-Handel läuft oder Nacht)
    then = last_at(ger, th)
    if not then:
        return None                                 # zum US-Schluss gab es keinen deutschen Kurs
    f_now, f_then = (last_at(fx, tg, 12) or (None, None))[1], (last_at(fx, th, 12) or (None, None))[1]
    if f_now and f_then:
        if abs(then[1] * f_then / ph - 1) > LEVEL_TOL:
            return None                             # Kursniveau passt nicht: falsche Zuordnung
        chg = (pg * f_now) / (then[1] * f_then) - 1
    else:
        chg = pg / then[1] - 1
    if abs(chg) > MAX_MOVE:
        return None
    return {"c": chg, "t": tg, "ref": ph, "th": th}


def short_names(stocks):
    """Namen, unter denen Unternehmen in Schlagzeilen vorkommen: {Suchwort: Kürzel}."""
    out = {}
    for x in stocks:
        s, n = x["s"], x.get("n") or ""
        names = list(ALIAS.get(s, []))
        n = re.sub(r"[,.]? (Inc|Corp|Corporation|Ltd|Limited|AG|SE|S\.A|N\.V|plc|Co|Company|Holdings?|Incorporated|Group|"
                   r"Platforms|Technologies|Systems|Class [A-C])\.?$", "", n).strip()
        first = re.sub(r"\.com$", "", n.split(" ")[0]) if n else ""
        if len(first) >= 4 and first.lower() not in NOISE:
            names.append(first)
        if "." not in s and len(s) >= 3 and s.isalpha():
            names.append(s)                         # Börsenkürzel, nur in Großbuchstaben gesucht
        for k in names:
            out.setdefault(k, s)
    return out


def rank_news(items, names, moves, now):
    """items: [{t,d,u,p}]. names: {Suchwort: Kürzel}. moves: {Kürzel: Veränderung in %}.
    Gibt die wichtigsten (höchstens 6) mit Thema k und betroffenen Kürzeln s zurück."""
    scored = []
    for it in items:
        t = (it.get("t") or "").strip()
        if not t or not it.get("u"):
            continue
        try:
            d = datetime.fromisoformat(str(it.get("d")).replace("Z", "+00:00"))
        except Exception:
            continue
        age = (now - d).total_seconds() / 3600
        if age > 24 or age < -1:
            continue
        w = 2.0 * max(0.0, 24 - age) / 24
        src = (it.get("p") or "").strip().lower()
        w += 2 if re.search(GOOD_SRC, src) else (-2 if re.search(WEAK_SRC, src) else 0)
        if re.search(PROMO, t, re.I):
            w -= 3
        if t.endswith("?"):
            w -= 1
        k = [name for name, rx in TOPICS if re.search(rx, t, re.I)]
        w += 2.5 * min(2, len(k))
        hit = []
        for word, s in names.items():
            flags = 0 if word == s else re.I        # Börsenkürzel nur in Großbuchstaben
            if s not in hit and re.search(r"(?<![\w$])" + re.escape(word) + r"(?![\w])", t, flags):
                hit.append(s)
        for s in hit[:3]:
            w += 1 + min(3.0, abs(moves.get(s) or 0) / 2)
        scored.append({"t": t[:200], "d": it.get("d"), "u": it["u"], "p": it.get("p") or "", "k": k[0] if k else None,
                       "s": hit[:3], "w": round(w, 2)})
    scored.sort(key=lambda x: -x["w"])
    out = []
    for x in scored:
        a = set(re.findall(r"[a-z0-9]{3,}", x["t"].lower()))
        if any(len(a & b) / max(1, len(a | b)) > 0.45 for b in (set(re.findall(r"[a-z0-9]{3,}", y["t"].lower())) for y in out)):
            continue                                # fast gleiche Schlagzeile
        if x["w"] < 2:
            break
        out.append(x)
        if len(out) >= 6:
            break
    return out


# ------------------------------------------------------------------ Abruf
def _series(df, sym, tz_shift=15):
    """[(Bar-Ende in UTC, Schlusskurs)] aus einem yfinance-Intraday-Ergebnis."""
    try:
        c = df[sym]["Close"] if sym in getattr(df.columns, "levels", [[]])[0] else None
    except Exception:
        c = None
    if c is None:
        return []
    out = []
    for ts, v in c.dropna().items():
        try:
            t = ts.tz_convert("UTC").to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if v and math.isfinite(float(v)):
            out.append((t + timedelta(minutes=tz_shift), float(v)))
    return out


def download(symbols):
    import yfinance as yf
    res = {}
    for i in range(0, len(symbols), 100):
        part = symbols[i:i + 100]
        try:
            df = yf.download(part, period="5d", interval="15m", group_by="ticker", progress=False, threads=True,
                             auto_adjust=False, prepost=False)
            for s in part:
                res[s] = _series(df, s)
        except Exception as e:  # noqa: BLE001
            print("Schnelle Kurse: Abruf fehlgeschlagen", type(e).__name__, flush=True)
        time.sleep(1)
    return res


def search_base(sym, name, isin):
    """Deutsches Kürzel über die Yahoo-Suche (ISIN, sonst Name)."""
    import yfinance as yf
    qs = [q for q in (isin, re.sub(r"[,.]? (Inc|Corp|Corporation|Ltd|Limited|Co|Company|Holdings?)\.?$", "", name or ""))
          if q]
    for q in qs:
        try:
            r = yf.Search(q, max_results=20, news_count=0, lists_count=0, recommended=0, timeout=10,
                          raise_errors=False)
            for qt in r.quotes or []:
                sy = qt.get("symbol") or ""
                if qt.get("quoteType") == "EQUITY" and "." in sy and sy.rsplit(".", 1)[1] in ("F", "DE", "BE", "SG", "MU", "DU"):
                    return sy.rsplit(".", 1)[0]
        except Exception:
            continue
    return None


def fetch_news(syms):
    import yfinance as yf
    items = []
    for s in syms:
        try:
            for n in (yf.Ticker(s).news or [])[:15]:
                c = n.get("content") or n
                ts = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")
                if isinstance(ts, (int, float)):
                    t = datetime.fromtimestamp(ts, timezone.utc)
                else:
                    try:
                        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    except Exception:
                        continue
                url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                       or c.get("link"))
                pub = (c.get("provider") or {}).get("displayName") or c.get("publisher")
                if c.get("title") and url:
                    items.append({"t": c["title"].strip(), "d": _iso(t.astimezone(timezone.utc)), "u": url, "p": pub})
        except Exception:
            continue
    return items


def run(data_path=DATA_PATH, map_path=MAP_PATH, out_path=OUT_PATH, fund_path=FUND_PATH):
    now = datetime.now(timezone.utc)
    d = json.load(open(data_path))
    stocks = d.get("stocks") or []
    try:
        fund = json.load(open(fund_path))
    except Exception:
        fund = {}
    try:
        mp = json.load(open(map_path))
    except Exception:
        mp = {}
    am = [x for x in stocks if ("." not in x["s"] or x["s"].endswith(".TO"))]
    am.sort(key=lambda x: -(x.get("mc") or 0))
    am = am[:N_STOCKS]
    # Zuordnung ergänzen
    n_search = 0
    for x in am:
        s = x["s"]
        e = mp.get(s) or {}
        if e.get("b") and e.get("bad", 0) < 3:
            continue
        if s in SEED and not e.get("seedbad"):
            mp[s] = {"b": SEED[s], "t": now.strftime("%Y-%m-%d"), "bad": 0}
            continue
        if e.get("t") and e.get("t") > (now - timedelta(days=14)).strftime("%Y-%m-%d") and not e.get("b"):
            continue                                # erst nach 14 Tagen wieder suchen
        if n_search >= MAX_SEARCH:
            continue
        n_search += 1
        b = search_base(s, x.get("n"), (fund.get(s) or {}).get("isin"))
        if b and e.get("b") == b:
            b = None                                # dieselbe falsche Zuordnung nicht wieder nehmen
        mp[s] = {"b": b, "t": now.strftime("%Y-%m-%d"), "bad": 0}
    cur = {x["s"]: (x.get("cur") or "USD") for x in am}
    fxs = sorted({"EUR%s=X" % c for c in cur.values() if c != "EUR"})
    ger = []
    for x in am:
        b = (mp.get(x["s"]) or {}).get("b")
        if b:
            ger += ["%s.%s" % (b, sf) for sf, _ in SUFFIX]
    syms = [x["s"] for x in am] + ger + fxs + [f for f, _ in FUTURES]
    D = download(syms)
    for k in D:                                     # laufende Kerze: Ende nicht in der Zukunft
        D[k] = [(min(t, now), v) for t, v in D[k]]
    st, us_close = {}, None
    for x in am:
        s = x["s"]
        home = D.get(s) or []
        if home and "." not in s:
            us_close = max(us_close, home[-1][0]) if us_close else home[-1][0]
        b = (mp.get(s) or {}).get("b")
        if not b or not home:
            continue
        fx = D.get("EUR%s=X" % cur[s]) or []
        best = None
        for sf, lab in SUFFIX:
            g = D.get("%s.%s" % (b, sf)) or []
            r = calc_move(home, g, fx)
            if r and (not best or r["t"] > best["t"]):
                best = dict(r, x=lab)
        # Kursniveau-Prüfung für die Zuordnung (auch wenn gerade nichts Neueres da ist)
        lv = None
        for sf, _ in SUFFIX:
            g = D.get("%s.%s" % (b, sf)) or []
            then = last_at(g, home[-1][0], 6) if g else None
            f = last_at(fx, home[-1][0], 12)
            if then and f:
                lv = abs(then[1] * f[1] / home[-1][1] - 1) <= LEVEL_TOL
                break
        if lv is False:
            mp[s]["bad"] = mp[s].get("bad", 0) + 1
            if mp[s]["bad"] >= 3:
                if s in SEED:
                    mp[s]["seedbad"] = 1
                mp[s]["b"] = None
            best = None
        elif lv:
            mp[s]["bad"] = 0
        if best:
            st[s] = {"c": _r(best["c"] * 100, 1), "p": _r(best["ref"] * (1 + best["c"]), 2), "ref": _r(best["ref"], 2),
                     "t": _iso(best["t"]), "x": best["x"]}
    fut = []
    for f, lab in FUTURES:
        sr = D.get(f) or []
        if sr and us_close:
            then = last_at(sr, us_close, 3)
            if then and sr[-1][0] > us_close + timedelta(minutes=10):
                c = sr[-1][1] / then[1] - 1
                if abs(c) < 0.15:
                    fut.append([lab, _r(c * 100, 2), _iso(sr[-1][0])])
    us_open = bool(us_close and now - us_close < timedelta(minutes=40))
    # Nachrichten: Markt (aus data.json) + größte Bewegungen + Terminkontrakte
    moves = {x["s"]: x.get("p1d") for x in stocks}
    for s, v in st.items():
        moves[s] = v["c"]
    big = sorted(st, key=lambda s: -abs(st[s]["c"] or 0))[:6]
    big += [x["s"] for x in sorted(stocks, key=lambda x: -abs(x.get("p1d") or 0) * (x.get("mc") or 0) ** 0.5)[:4]]
    items = list((d.get("mx") or {}).get("news") or [])
    items += fetch_news(["ES=F", "NQ=F", "^GSPC"] + list(dict.fromkeys(big)))
    top = sorted(stocks, key=lambda x: -(x.get("mc") or 0))[:150]
    news = rank_news(items, short_names(top), moves, now)
    out = {"ts": _iso(now), "us": {"open": us_open, "close": _iso(us_close)}, "fut": fut, "st": st, "news": news}
    if not st and not fut and not news:
        print("Schnelle Kurse: nichts geladen, alte Datei bleibt.", flush=True)
    else:
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    with open(map_path, "w") as f:
        json.dump(mp, f, separators=(",", ":"), sort_keys=True)
    print("Schnelle Kurse:", len(st), "Aktien in Deutschland,", len(fut), "Terminkontrakte,", len(news),
          "Schlagzeilen", "| US-Handel läuft" if us_open else "", flush=True)
    return out


if __name__ == "__main__":
    run()
