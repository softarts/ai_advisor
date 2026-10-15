"""解析 PortfolioAnalyst 自开户报告（多区块 CSV，区块名,Header/Data/MetaInfo）。"""
from __future__ import annotations
import csv, hashlib, io, re
from dataclasses import dataclass, field
from datetime import datetime, date


def num(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--", "N/A", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _ym(s):   # 201708 -> 2017-08
    s = str(s).strip()
    return f"{s[:4]}-{s[4:6]}"


def _mon(s):  # "Aug 2017" -> 2017-08
    return datetime.strptime(s.strip(), "%b %Y").strftime("%Y-%m")


def _ymd(s):  # 20180215 -> 2018-02-15
    s = str(s).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _mdy(s):  # 08/10/17 or 10/09/2026
    m, d, y = s.strip().split("/")
    y = int(y)
    y = 2000 + y if y < 100 else y
    return f"{y:04d}-{int(m):02d}-{int(d):02d}"


@dataclass
class PAReport:
    account: str
    base_ccy: str
    period_start: str
    period_end: str
    file_hash: str
    benchmarks: list
    nav_monthly: list        # {month, nav, commodities, equities, etfs, futures_options, options, stocks, cash}
    returns_monthly: list    # {month, account, bm1, bm2, bm3}
    contrib_monthly: list    # {month, etfs, fs_options, futures, futures_options, options, stocks, cash}  (百分点)
    flows: list              # {date, type, description, amount}
    interest: list           # {date, description, amount}
    fees: list
    dividends: list          # {date, symbol, amount}
    official: dict           # 报告日的官方区间收益 {MTD,QTD,YTD,1 Year,3 Year,5 Year,Since Inception}
    key_stats: dict
    risk: list
    symbol_perf: list
    trade_summary: list
    open_positions: list
    concentration: dict      # {holdings:[...], asset_class:[...]}
    year_returns: list       # {year, account, bm1, bm2, bm3}
    checks: list = field(default_factory=list)


def adapt_portfolio(rep: PAReport) -> list[dict]:
    """Map source-specific PA fields to canonical portfolio observations."""
    return [{"term_month": row["month"], "value": row["nav"], "currency": rep.base_ccy,
             "etfs": row.get("etfs"), "stocks": row.get("stocks"), "options": row.get("options"),
             "futures_options": row.get("futures_options"), "cash": row.get("cash"),
             "commodities": row.get("commodities"), "equities": row.get("equities")}
            for row in rep.nav_monthly]


def _groups(rows):
    groups, cur = [], None
    for r in rows:
        if len(r) < 2:
            continue
        sec, kind = r[0], r[1]
        if kind == "Header":
            cur = {"sec": sec, "h": r[2:], "d": []}
            groups.append(cur)
        elif kind == "Data" and cur and cur["sec"] == sec:
            cur["d"].append(dict(zip(cur["h"], r[2:])))
    return groups


def parse_report(path: str) -> PAReport:
    with open(path, "rb") as fh:
        raw = fh.read()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    G = _groups(rows)

    def first(sec, col=None):
        for g in G:
            if g["sec"] == sec and (col is None or col in g["h"]):
                return g["d"]
        return None

    need = ["Introduction", "Key Statistics", "Allocation by Financial Instrument", "Allocation by Asset Class",
            "Time Period Benchmark Comparison", "Performance by Financial Instrument", "Deposits And Withdrawals"]
    missing = [s for s in need if first(s) is None]
    if missing:
        raise ValueError(f"不是有效的 PA 自开户报告，缺少区块: {missing}")

    intro = first("Introduction")[0]
    m = re.search(r"([A-Za-z]+ \d+, \d{4}) to ([A-Za-z]+ \d+, \d{4})", intro["AnalysisPeriod"])
    if not m:
        raise ValueError(f"无法解析分析区间: {intro['AnalysisPeriod']}")
    pstart, pend = (datetime.strptime(x, "%B %d, %Y").date().isoformat() for x in m.groups())
    account, base = intro["Account"], intro["BaseCurrency"]

    # 月度净值
    ac = {_ym(d["Date"]): d for d in first("Allocation by Asset Class")}
    nav = []
    for d in first("Allocation by Financial Instrument"):
        k = _ym(d["Date"]); a = ac.get(k, {})
        nav.append({"month": k, "nav": num(d["NAV"]), "etfs": num(d["ETFs"]), "futures_options": num(d["Futures Options"]),
                    "options": num(d["Options"]), "stocks": num(d["Stocks"]), "cash": num(d["Cash"]),
                    "commodities": num(a.get("Commodities")), "equities": num(a.get("Equities"))})
    nav.sort(key=lambda r: r["month"])

    # 月度收益与基准
    tp = first("Time Period Benchmark Comparison")
    benchmarks = [tp[0]["BM1"], tp[0]["BM2"], tp[0]["BM3"]]
    rets = sorted([{"month": _mon(d["Date"]), "account": num(d[f"{account}Return"]), "bm1": num(d["BM1Return"]),
                    "bm2": num(d["BM2Return"]), "bm3": num(d["BM3Return"])} for d in tp], key=lambda r: r["month"])

    contrib = sorted([{"month": _ym(d["Date"]), "etfs": num(d["ETFs"]) or 0, "fs_options": num(d["Future-Style Options"]) or 0,
                       "futures": num(d["Futures"]) or 0, "futures_options": num(d["Futures Options"]) or 0,
                       "options": num(d["Options"]) or 0, "stocks": num(d["Stocks"]) or 0, "cash": num(d["Cash"]) or 0}
                      for d in first("Performance by Financial Instrument")], key=lambda r: r["month"])

    flows = [{"date": _mdy(d["Date"]), "type": d["Type"], "description": d["Description"], "amount": num(d["Amount"]) or 0}
             for d in first("Deposits And Withdrawals")]
    flows.sort(key=lambda r: r["date"])
    interest = [{"date": _ymd(d["Date"]), "description": d["Description"], "amount": num(d["Amount"]) or 0}
                for d in (first("Interest Details") or [])]
    fees = [{"date": _ymd(d["Date"]), "description": d["Description"], "amount": num(d["Amount"]) or 0}
            for d in (first("Fee Summary") or [])]
    divs = [{"date": _ymd(d["PayDate"]), "symbol": d["Symbol"], "amount": num(d["Amount"]) or 0}
            for d in (first("Dividends") or [])]

    # 官方区间收益（含账户行）与年度收益
    official = {}
    for d in first("Historical Performance Benchmark Comparison", "MTD") or []:
        if d["Account"] == account:
            official = {k: num(v) for k, v in d.items() if k != "Account"}
    years = [{"year": int(d["Year"]) if d["Year"].strip().isdigit() else int(pend[:4]), "account": num(d["AccountReturn"]), "bm1": num(d["BM1Return"]),
              "bm2": num(d["BM2Return"]), "bm3": num(d["BM3Return"])}
             for d in (first("Historical Performance Benchmark Comparison", "Year") or [])]

    ks = first("Key Statistics")[0]
    key_stats = {k: (num(v) if num(v) is not None else v) for k, v in ks.items()}

    risk = []
    for d in first("Risk Measures Benchmark Comparison", "Account Value") or []:
        risk.append({"measure": d["Risk Measure"], "account": num(d["Account Value"]) if num(d["Account Value"]) is not None else d["Account Value"],
                     "bm1": num(d["BM1 Value"]), "bm2": num(d["BM2 Value"]), "bm3": num(d["BM3 Value"])})

    sp = [{"symbol": d["Symbol"], "description": d["Description"], "instrument": d["FinancialInstrument"],
           "avg_weight": num(d["AvgWeight"]), "ret_pct": num(d["Return"]), "contribution": num(d["Contribution"]),
           "unrealized": num(d["Unrealized_P&L"]) or 0, "realized": num(d["Realized_P&L"]) or 0, "open": d["Open"]}
          for d in (first("Performance by Symbol") or []) if d["FinancialInstrument"]]

    ts = [{"instrument": d["Financial Instrument"], "currency": d["Currency"], "symbol": d["Symbol"],
           "qty_bought": num(d["Quantity Bought"]), "avg_price_bought": num(d["Average Price Bought"]),
           "proceeds_bought_base": num(d["Proceeds Bought in Base"]), "qty_sold": num(d["Quantity Sold"]),
           "avg_price_sold": num(d["Average Price Sold"]), "proceeds_sold_base": num(d["Proceeds Sold in Base"])}
          for d in (first("Trade Summary") or []) if d["Symbol"]]

    op = [{"date": _mdy(d["Date"]), "instrument": d["FinancialInstrument"], "currency": d["Currency"], "symbol": d["Symbol"],
           "description": d["Description"], "quantity": num(d["Quantity"]), "close_price": num(d["ClosePrice"]),
           "value": num(d["Value"]), "cost_basis": num(d["Cost Basis"]), "unrealized": num(d["UnrealizedP&L"]),
           "fx_to_base": num(d["FXRateToBase"])} for d in (first("Open Position Summary") or []) if "/" in d["Date"]]

    conc = {"holdings": [], "asset_class": []}
    for g in G:
        if g["sec"] != "Concentration":
            continue
        for d in g["d"]:
            if "NetValue" in d and d.get("SubSection") == "Holdings":
                conc["holdings"].append({"symbol": d["Symbol"], "description": d["Description"], "long_value": num(d["LongValue"]),
                                         "short_value": num(d["ShortValue"]), "net_value": num(d["NetValue"]),
                                         "net_weight_pct": num(d["NetParsedWeight"])})
            elif "AssetClass" in d:
                conc["asset_class"].append({"asset_class": d["AssetClass"], "long_weight_pct": num(d["LongParsedWeight"]),
                                            "short_weight_pct": num(d["ShortParsedWeight"])})

    rep = PAReport(account, base, pstart, pend, hashlib.sha256(raw).hexdigest(), benchmarks, nav, rets, contrib, flows,
                   interest, fees, divs, official, key_stats, risk, sp, ts, op, conc, years)
    rep.checks = validate(rep)
    return rep


def validate(rep: PAReport) -> list:
    """导入时的自动校验：报告内部的恒等式，失败说明格式或口径可能变了。"""
    ks, out = rep.key_stats, []

    def chk(cid, desc, a, b, tol):
        ok = a is not None and b is not None and abs(a - b) <= tol
        out.append({"id": cid, "desc": desc, "ok": bool(ok), "a": a, "b": b})

    t = 1.0
    for r in rep.returns_monthly:
        t *= 1 + (r["account"] or 0) / 100
    chk("V1", "月度收益链接 = 官方累计收益(%)", (t - 1) * 100, ks.get("CumulativeReturn"), 0.01)
    parts = [ks.get(k) or 0 for k in ("MTM", "Deposits & Withdrawals", "Dividends", "Interest", "Fees & Commissions", "Other")]
    chk("V2", "MTM+出入金+股息+利息+费用+其他 = 净值变动", sum(parts), ks.get("ChangeInNAV"), 1.0)
    chk("V3", "出入金明细合计 = 关键统计", sum(f["amount"] for f in rep.flows), ks.get("Deposits & Withdrawals"), 1.0)
    chk("V4", "利息明细合计 = 关键统计", sum(f["amount"] for f in rep.interest), ks.get("Interest"), 1.0)
    chk("V5", "股息明细合计 = 关键统计", sum(f["amount"] for f in rep.dividends), ks.get("Dividends"), 1.0)
    chk("V6", "最后月末净值 = 期末净值", rep.nav_monthly[-1]["nav"], ks.get("EndingNAV"), 1.0)
    return out
