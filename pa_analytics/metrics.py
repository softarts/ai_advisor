"""核心计算（只读 PAReport，返回 Result）。窗口按“月份标签”定义，含当月至今：
1M=当月至今，3M=最近3个月标签，YTD=今年各月，ALL=自开户。收益率为月度 TWR 链接。"""
from __future__ import annotations
import calendar
from collections import defaultdict
from datetime import date, timedelta
from .constraints import CONSTRAINTS
from .models import Result

WINDOWS = ("YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y", "ALL")
WINDOW_LABEL = {"YTD": "YTD", "1M": "近1个月", "3M": "近3个月", "6M": "近6个月", "1Y": "近1年",
                "3Y": "近3年", "5Y": "近5年", "ALL": "自开户"}
_N = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60}
_OFFICIAL = {"YTD": "YTD", "1Y": "1 Year", "3Y": "3 Year", "5Y": "5 Year", "ALL": "Since Inception"}
VALIDATED_FROM = "2025-01"   # 归因估算只与 2025 年报、2026 Flex 对照验证过；更早的月份未验证
MAX_CONTRIBUTORS = 20


def _d(s):
    return date.fromisoformat(s)


def _months(rep):
    return [r["month"] for r in rep.nav_monthly]


def _month_end(rep, m):
    if m == _months(rep)[-1]:
        return _d(rep.period_end)
    y, mo = int(m[:4]), int(m[5:])
    return date(y, mo, calendar.monthrange(y, mo)[1])


def resolve_months(rep, window: str) -> list:
    months = _months(rep)
    if window not in WINDOWS:
        raise ValueError(f"不支持的窗口 {window!r}；可用窗口：{', '.join(WINDOWS)}")
    if window == "ALL":
        return months
    if window == "YTD":
        return [m for m in months if m[:4] == months[-1][:4]]
    n = _N[window]
    if len(months) < n:
        raise ValueError(f"数据不足：{WINDOW_LABEL[window]}需要 {n} 个月，现有 {len(months)} 个月")
    return months[-n:]


def _prev(rep, m):
    months = _months(rep)
    i = months.index(m)
    return months[i - 1] if i > 0 else None


def _period(rep, inc: list) -> dict:
    navs = {r["month"]: r for r in rep.nav_monthly}
    rets = {r["month"]: r for r in rep.returns_monthly}
    anchor = _prev(rep, inc[0])
    start = _month_end(rep, anchor) if anchor else _d(rep.period_start) - timedelta(days=1)
    end = _month_end(rep, inc[-1])
    v0 = navs[anchor]["nav"] if anchor else 0.0
    v1 = navs[inc[-1]]["nav"]
    fl = [f for f in rep.flows if start < _d(f["date"]) <= end]
    dep = sum(f["amount"] for f in fl if f["amount"] > 0)
    wd = sum(f["amount"] for f in fl if f["amount"] < 0)
    cum = {"account": 1.0, "bm1": 1.0, "bm2": 1.0, "bm3": 1.0}
    curve = {k: [[start.isoformat(), 0.0]] for k in cum}
    for m in inc:
        for k in cum:
            cum[k] *= 1 + (rets[m][k] or 0) / 100
            curve[k].append([_month_end(rep, m).isoformat(), (cum[k] - 1) * 100])
    series = ([[start.isoformat(), v0]] if anchor else []) + [[_month_end(rep, m).isoformat(), navs[m]["nav"]] for m in inc]
    alloc = [{"date": _month_end(rep, m).isoformat(), **{k: navs[m][k] or 0 for k in
              ("etfs", "futures_options", "options", "stocks", "cash")}, "nav": navs[m]["nav"]} for m in inc]
    return {"months": inc, "start_date": start.isoformat(), "end_date": end.isoformat(), "start_nav": v0, "end_nav": v1,
            "deposits": dep, "withdrawals": wd, "net_flow": dep + wd, "profit": v1 - v0 - (dep + wd),
            "twr_pct": (cum["account"] - 1) * 100, "benchmark_twr_pct": {rep.benchmarks[i]: (cum[f"bm{i+1}"] - 1) * 100 for i in range(3)},
            "nav_series": series, "return_curves": {"account": curve["account"], **{rep.benchmarks[i]: curve[f"bm{i+1}"] for i in range(3)}},
            "allocation": alloc, "flows": [[f["date"], f["amount"]] for f in fl]}


def window_summary(rep, window: str = "YTD") -> Result:
    base = dict(basis="NAV + 月度TWR链接", currency=rep.base_ccy, sources=["Allocation by Financial Instrument", "Time Period Benchmark Comparison", "Deposits And Withdrawals"])
    try:
        inc = resolve_months(rep, window)
    except ValueError as e:
        return Result(None, status="insufficient", warnings=[str(e)], **base)
    p = _period(rep, inc)
    ks, off, note = rep.key_stats, None, ""
    if window == "1M":
        off, note = ks.get("1MonthReturn"), "PA 关键统计（同口径）"
    elif window == "3M":
        off, note = ks.get("3MonthReturn"), "PA 关键统计（同口径）"
    elif window in _OFFICIAL:
        off = rep.official.get(_OFFICIAL[window])
        note = "PA 官方（同口径）" if window in ("YTD", "ALL") else "PA 官方（按日滚动，口径略有不同）"
    p.update({"window": window, "official_pct": off, "official_note": note})
    warns = [f"{WINDOW_LABEL[window]}按月份标签对齐：{p['start_date']} ~ {p['end_date']}（含当月至今）"]
    if window == "ALL":
        warns.append("自开户首月无期初净值，期初资产按 0 处理")
    return Result(p, period=(p["start_date"], p["end_date"]), warnings=warns, **base)


def monthly_return_extremes(rep, window: str = "YTD") -> Result:
    """Return highest/lowest monthly TWR rows within a supported reporting window."""
    base = dict(basis="报告月度账户收益；月度 TWR", currency=rep.base_ccy,
                sources=["Time Period Benchmark Comparison"])
    try:
        inc = resolve_months(rep, window)
    except ValueError as e:
        return Result(None, status="insufficient", warnings=[str(e)], **base)
    by_month = {row["month"]: row for row in rep.returns_monthly}
    rows = [{"month": m, "account_return_pct": by_month[m]["account"],
             "benchmark_returns_pct": {rep.benchmarks[i]: by_month[m][f"bm{i+1}"] for i in range(3)}}
            for m in inc]
    valid = [row for row in rows if row["account_return_pct"] is not None]
    if not valid:
        return Result(None, status="insufficient", warnings=["窗口内没有可用的月度收益"], **base)
    best = max(valid, key=lambda row: row["account_return_pct"])
    worst = min(valid, key=lambda row: row["account_return_pct"])
    warns = [f"窗口 {window} 包含 {len(rows)} 个报告月份（{inc[0]} 至 {inc[-1]}）。"]
    if inc[-1] == _months(rep)[-1]:
        warns.append(f"最后一个报告月 {inc[-1]} 截至 {_d(rep.period_end).isoformat()}，可能是部分月。")
    return Result({"window": window, "month_count": len(rows), "best_month": best,
                   "worst_month": worst, "monthly_returns": rows},
                  period=(inc[0], inc[-1]), warnings=warns, **base)


def _month_profit(rep, m):
    """某月精确盈亏 = 月末净值 - 上月末净值 - 当月出入金（来自净值和出入金明细，不依赖贡献）。"""
    navs = {r["month"]: r["nav"] for r in rep.nav_monthly}
    prev = _prev(rep, m)
    start = _month_end(rep, prev) if prev else _d(rep.period_start) - timedelta(days=1)
    end = _month_end(rep, m)
    flow = sum(f["amount"] for f in rep.flows if start < _d(f["date"]) <= end)
    return navs[m] - (navs[prev] if prev else 0.0) - flow


def _sum_in(rows, start, end):
    return sum(r["amount"] for r in rows if start < _d(r["date"]) <= end)


_KEYS = ("etfs", "stocks", "options", "futures", "futures_options", "fs_options", "cash")


def _attr(rep, inc, p) -> dict:
    """每个月的精确盈亏，按各品种月度贡献占比分摊；利息、费用取精确明细；期货/期货期权/外汇为剩余值。"""
    contrib = {r["month"]: r for r in rep.contrib_monthly}
    stocks = opts = 0.0
    pa, skipped = defaultdict(float), []
    for m in inc:
        c = contrib.get(m)
        S = sum(c[k] for k in _KEYS) if c else 0.0
        if abs(S) < 0.05:           # 贡献合计接近 0，无法分摊，该月盈亏进入剩余项
            skipped.append(m)
            continue
        k = _month_profit(rep, m) / S
        stocks += (c["etfs"] + c["stocks"]) * k
        opts += c["options"] * k
        for key in ("futures", "futures_options", "fs_options", "cash"):
            pa[key] += c[key] * k
    s, e = _d(p["start_date"]), _d(p["end_date"])
    interest, fees, divs = _sum_in(rep.interest, s, e), _sum_in(rep.fees, s, e), _sum_in(rep.dividends, s, e)
    resid = p["profit"] - stocks - opts - interest - fees
    validated = all(m >= VALIDATED_FROM for m in inc)
    q = "estimate" if validated else "unverified"
    return {"buckets": [
        {"key": "stocks_etf", "label": "股票/ETF", "amount": stocks, "quality": q},
        {"key": "stock_options", "label": "股票期权", "amount": opts, "quality": q},
        {"key": "deriv_fx", "label": "期货/期货期权/外汇（合并）", "amount": resid, "quality": "residual"},
        {"key": "interest", "label": "利息", "amount": interest, "quality": "exact"},
        {"key": "fees", "label": "其他费用", "amount": fees, "quality": "exact"}],
        "validated": validated, "skipped_months": skipped, "memo_dividends": divs,
        "pa_split_reference": {"futures": pa["futures"], "futures_options": pa["futures_options"],
                               "future_style_options": pa["fs_options"], "cash": pa["cash"]},
        "profit": p["profit"]}


def window_attribution(rep, window: str = "YTD") -> Result:
    base = dict(basis="PA 月度贡献折算 + 精确的利息/费用", currency=rep.base_ccy,
                sources=["Performance by Financial Instrument", "Interest Details", "Fee Summary"])
    try:
        inc = resolve_months(rep, window)
    except ValueError as e:
        return Result(None, status="insufficient", warnings=[str(e)], **base)
    p = _period(rep, inc)
    a = _attr(rep, inc, p)
    a.update({"window": window, "start_date": p["start_date"], "end_date": p["end_date"]})
    warns = ["股票/ETF、股票期权为估算（K2/K5）；期货、期货期权、外汇合并展示（K3）；利息、费用精确（K4）",
             "各项之和等于净值口径的区间盈亏（合并项为剩余值）"]
    if not a["validated"]:
        warns.append(f"窗口包含 {VALIDATED_FROM} 之前的月份，归因估算未经验证，且与 PA 累计按品种盈亏相差较大（见 K5），仅供参考")
    if a["skipped_months"]:
        warns.append(f"{len(a['skipped_months'])} 个月贡献合计接近 0，无法分摊，盈亏已计入合并项")
    return Result(a, period=(p["start_date"], p["end_date"]), warnings=warns, **base)


def yearly_table(rep) -> Result:
    months = _months(rep)
    off = {y["year"]: y["account"] for y in rep.year_returns}
    rows = []
    for y in sorted({m[:4] for m in months}):
        inc = [m for m in months if m[:4] == y]
        p = _period(rep, inc)
        a = _attr(rep, inc, p)
        rows.append({"year": int(y), "start_nav": p["start_nav"], "end_nav": p["end_nav"], "net_flow": p["net_flow"],
                     "profit": p["profit"], "twr_pct": p["twr_pct"], "official_pct": off.get(int(y)),
                     "partial": inc[0][5:] != "01" or y == months[-1][:4], "validated": a["validated"],
                     "buckets": {b["key"]: b["amount"] for b in a["buckets"]}})
    return Result(rows, basis="NAV + 月度TWR链接；归因同 window_attribution", currency=rep.base_ccy,
                  warnings=["2017 年从 8 月开户，期初资产为 0；今年为 YTD"])


def cash_interest(rep, window: str = "YTD") -> Result:
    base = dict(basis="利息明细（精确）÷ 月末现金平均（近似）", currency=rep.base_ccy)
    try:
        inc = resolve_months(rep, window)
    except ValueError as e:
        return Result(None, status="insufficient", warnings=[str(e)], **base)
    p = _period(rep, inc)
    s, e = _d(p["start_date"]), _d(p["end_date"])
    interest, fees = _sum_in(rep.interest, s, e), _sum_in(rep.fees, s, e)
    cash = [r["cash"] or 0 for r in rep.nav_monthly if r["month"] in inc]
    avg = sum(cash) / len(cash)
    days = max((e - s).days, 1)
    return Result({"window": window, "interest": interest, "other_fees": fees, "avg_month_end_cash": avg,
                   "implied_annual_yield_pct": interest / avg * 365 / days * 100 if avg else None,
                   "start_date": p["start_date"], "end_date": p["end_date"]},
                  warnings=["现金含 SGD 与 USD 且按月末值平均，隐含收益率只是近似；利息含借款利息（可能为负）（K6）"], **base)


def top_contributors(rep, n: int = 10) -> Result:
    if isinstance(n, bool) or not isinstance(n, int):
        return Result(None, status="insufficient", warnings=[f"n 必须是整数，范围为 1 到 {MAX_CONTRIBUTORS}"],
                      basis="累计已实现+未实现盈亏（合约级，不可按时间切片）", currency=rep.base_ccy)
    if n < 1 or n > MAX_CONTRIBUTORS:
        return Result(None, status="insufficient", warnings=[f"n 超出允许范围 1 到 {MAX_CONTRIBUTORS}"],
                      basis="累计已实现+未实现盈亏（合约级，不可按时间切片）", currency=rep.base_ccy)
    rows = [{**r, "pnl": r["unrealized"] + r["realized"]} for r in rep.symbol_perf]
    by_inst = defaultdict(float)
    for r in rows:
        by_inst[r["instrument"]] += r["pnl"]
    rows.sort(key=lambda r: r["pnl"])
    cut = lambda r: {k: r[k] for k in ("symbol", "instrument", "pnl", "open")}
    return Result({"by_instrument": dict(by_inst), "total": sum(by_inst.values()),
                   "losers": [cut(r) for r in rows[:n]], "winners": [cut(r) for r in rows[::-1][:n]]},
                  basis="累计已实现+未实现盈亏（合约级，不可按时间切片）", currency=rep.base_ccy,
                  warnings=["合计与 MTM 口径不同（K10）；合约级，同一标的的不同合约未合并"])


def risk_measures(rep) -> Result:
    return Result(rep.risk, basis="PA 月度数据计算", warnings=["基于月度数据，会低估月内回撤（K1）"])


def open_positions(rep) -> Result:
    total = sum(r["value"] or 0 for r in rep.open_positions)
    return Result({"positions": rep.open_positions, "market_value_total": total,
                   "asset_class_weights": rep.concentration["asset_class"],
                   "cash_net_weight_pct": next((h["net_weight_pct"] for h in rep.concentration["holdings"] if h["symbol"] == "CASH"), None)},
                  basis="市值口径", currency=rep.base_ccy,
                  warnings=["这是市值口径，不是风险口径：卖出期权的名义敞口未体现（K9）"])


ANSWERABLE = ["区间收益与 TWR", "与 SPXTR/EFA/VT 的比较", "资产配置历史（市值口径）", "月度回撤与风险指标",
              "利息、费用与现金拖累", "按合约/品种的累计盈亏", "出入金历史"]
WITH_CAVEATS = ["按资产类别归因（见 K2~K5）"]
NOT_ANSWERABLE = ["杠杆与保证金", "存入 SGD 能否释放期货账户的 USD", "证券/期货分段的现金分布", "卖出期权持仓多久、delta 退出点",
                  "月度成交笔数与交易频率", "每日净值与月内回撤", "名义风险敞口（需另行设计）"]


def get_capabilities(rep, snapshots: list | None = None) -> dict:
    months = _months(rep)
    return {"source": "PortfolioAnalyst 自开户报告（月度，TWR）", "account": rep.account, "base_currency": rep.base_ccy,
            "period": [rep.period_start, rep.period_end], "months": len(months), "last_month_is_partial": True,
            "windows": {w: ("ok" if (w in ("ALL", "YTD") or len(months) >= _N[w]) else "insufficient") for w in WINDOWS},
            "answerable": ANSWERABLE, "answerable_with_caveats": WITH_CAVEATS, "not_answerable": NOT_ANSWERABLE,
            "constraints": CONSTRAINTS, "import_checks": rep.checks, "snapshots": snapshots or []}
