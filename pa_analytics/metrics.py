"""核心计算（优先通过 Store 规范数据层读取，亦兼容直接传入 PAReport，返回 Result）。
窗口按“月份标签”定义，含当月至今：
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
CONSISTENCY_THRESHOLD_PP = 1.0
_KEYS = ("etfs", "stocks", "options", "futures", "futures_options", "fs_options", "cash")


def _d(s):
    return date.fromisoformat(s)


def _months(source):
    if hasattr(source, "get_account_info"):
        return source.get_account_info()["months"]
    return [r["month"] for r in source.nav_monthly]


def _month_end(source, m):
    months = _months(source)
    if m == months[-1]:
        if hasattr(source, "get_account_info"):
            return _d(source.get_account_info()["period_end"])
        return _d(source.period_end)
    y, mo = int(m[:4]), int(m[5:])
    return date(y, mo, calendar.monthrange(y, mo)[1])


def resolve_months(source, window: str) -> list:
    months = _months(source)
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


def _prev(source, m):
    months = _months(source)
    i = months.index(m)
    return months[i - 1] if i > 0 else None


def _sum_in(rows, start, end):
    return sum(r["amount"] for r in rows if start < _d(r["date"]) <= end)


def _period(source, inc: list) -> dict:
    if hasattr(source, "get_account_info"):
        info = source.get_account_info()
        anchor = _prev(source, inc[0])
        start = _month_end(source, anchor) if anchor else _d(info["period_start"]) - timedelta(days=1)
        end = _month_end(source, inc[-1])
        start_query_m = anchor or inc[0]
        breakdowns = source.get_portfolio_breakdown(start_month=start_query_m, end_month=inc[-1])
        navs = {r["term_month"]: r for r in breakdowns}
        v0 = navs[anchor]["nav"] if anchor else 0.0
        v1 = navs[inc[-1]]["nav"]
        rets_rows = source.get_monthly_returns_data(start_month=inc[0], end_month=inc[-1])
        rets = {r["term_month"]: {"account": r["account_return"], "bm1": r["bm1_return"],
                                  "bm2": r["bm2_return"], "bm3": r["bm3_return"]} for r in rets_rows}
        fl = source.get_flows_data(start_date=start.isoformat(), end_date=end.isoformat())
        dep = sum(f["amount"] for f in fl if f["amount"] > 0)
        wd = sum(f["amount"] for f in fl if f["amount"] < 0)
        benchmarks = info["benchmarks"]
        cum = {"account": 1.0, "bm1": 1.0, "bm2": 1.0, "bm3": 1.0}
        curve = {k: [[start.isoformat(), 0.0]] for k in cum}
        for m in inc:
            for k in cum:
                cum[k] *= 1 + (rets[m][k] or 0) / 100
                curve[k].append([_month_end(source, m).isoformat(), (cum[k] - 1) * 100])
        series = ([[start.isoformat(), v0]] if anchor else []) + [[_month_end(source, m).isoformat(), navs[m]["nav"]] for m in inc]
        alloc = [{"date": _month_end(source, m).isoformat(), **{k: navs[m][k] or 0 for k in
                  ("etfs", "futures_options", "options", "stocks", "cash")}, "nav": navs[m]["nav"]} for m in inc]
        return {"months": inc, "start_date": start.isoformat(), "end_date": end.isoformat(), "start_nav": v0, "end_nav": v1,
                "deposits": dep, "withdrawals": wd, "net_flow": dep + wd, "profit": v1 - v0 - (dep + wd),
                "twr_pct": (cum["account"] - 1) * 100, "benchmark_twr_pct": {benchmarks[i]: (cum[f"bm{i+1}"] - 1) * 100 for i in range(len(benchmarks))},
                "nav_series": series, "return_curves": {"account": curve["account"], **{benchmarks[i]: curve[f"bm{i+1}"] for i in range(len(benchmarks))}},
                "allocation": alloc, "flows": [[f["date"], f["amount"]] for f in fl]}
    else:
        rep = source
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


def window_summary(source, window: str = "YTD") -> Result:
    if hasattr(source, "get_account_info"):
        info = source.get_account_info()
        base = dict(basis="NAV + 月度TWR链接", currency=info["base_currency"],
                    sources=["Allocation by Financial Instrument", "Time Period Benchmark Comparison", "Deposits And Withdrawals"])
        try:
            inc = resolve_months(source, window)
        except ValueError as e:
            return Result(None, status="insufficient", warnings=[str(e)], **base)
        p = _period(source, inc)
        ks = info["key_stats"]
        off, note = None, ""
        if window == "1M":
            off, note = ks.get("1MonthReturn"), "PA 关键统计（同口径）"
        elif window == "3M":
            off, note = ks.get("3MonthReturn"), "PA 关键统计（同口径）"
        elif window in _OFFICIAL:
            official_dict = source.get_official_returns_data()
            off = official_dict.get(_OFFICIAL[window])
            note = "PA 官方（同口径）" if window in ("YTD", "ALL") else "PA 官方（按日滚动，口径略有不同）"
        p.update({"window": window, "official_pct": off, "official_note": note})
        warns = [f"{WINDOW_LABEL[window]}按月份标签对齐：{p['start_date']} ~ {p['end_date']}（含当月至今）"]
        if window == "ALL":
            warns.append("自开户首月无期初净值，期初资产按 0 处理")
        return Result(p, period=(p["start_date"], p["end_date"]), warnings=warns, **base)
    else:
        rep = source
        base = dict(basis="NAV + 月度TWR链接", currency=rep.base_ccy,
                    sources=["Allocation by Financial Instrument", "Time Period Benchmark Comparison", "Deposits And Withdrawals"])
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


def monthly_return_extremes(source, window: str = "YTD") -> Result:
    """Return highest/lowest monthly TWR rows within a supported reporting window."""
    if hasattr(source, "get_account_info"):
        info = source.get_account_info()
        base = dict(basis="报告月度账户收益；月度 TWR", currency=info["base_currency"],
                    sources=["Time Period Benchmark Comparison"])
        try:
            inc = resolve_months(source, window)
        except ValueError as e:
            return Result(None, status="insufficient", warnings=[str(e)], **base)
        ret_rows = source.get_monthly_returns_data(start_month=inc[0], end_month=inc[-1])
        by_month = {row["term_month"]: row for row in ret_rows}
        benchmarks = info["benchmarks"]
        rows = [{"month": m, "account_return_pct": by_month[m]["account_return"],
                 "benchmark_returns_pct": {benchmarks[i]: by_month[m][f"bm{i+1}_return"] for i in range(len(benchmarks))}}
                for m in inc]
        valid = [row for row in rows if row["account_return_pct"] is not None]
        if not valid:
            return Result(None, status="insufficient", warnings=["窗口内没有可用的月度收益"], **base)
        best = max(valid, key=lambda row: row["account_return_pct"])
        worst = min(valid, key=lambda row: row["account_return_pct"])
        warns = [f"窗口 {window} 包含 {len(rows)} 个报告月份（{inc[0]} 至 {inc[-1]}）。"]
        if inc[-1] == info["months"][-1]:
            warns.append(f"最后一个报告月 {inc[-1]} 截至 {_d(info['period_end']).isoformat()}，可能是部分月。")
        return Result({"window": window, "month_count": len(rows), "best_month": best,
                       "worst_month": worst, "monthly_returns": rows},
                      period=(inc[0], inc[-1]), warnings=warns, **base)
    else:
        rep = source
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


def _month_profit(source, m):
    """某月精确盈亏 = 月末净值 - 上月末净值 - 当月出入金（来自净值和出入金明细，不依赖贡献）。"""
    if hasattr(source, "get_account_info"):
        info = source.get_account_info()
        prev = _prev(source, m)
        start = _month_end(source, prev) if prev else _d(info["period_start"]) - timedelta(days=1)
        end = _month_end(source, m)
        nav_rows = source.get_portfolio_series(start_month=(prev or m), end_month=m)
        navs = {r["term_month"]: r["value"] for r in nav_rows}
        flow_rows = source.get_flows_data(start_date=start.isoformat(), end_date=end.isoformat())
        flow = sum(f["amount"] for f in flow_rows)
        return navs[m] - (navs[prev] if prev else 0.0) - flow
    else:
        rep = source
        navs = {r["month"]: r["nav"] for r in rep.nav_monthly}
        prev = _prev(rep, m)
        start = _month_end(rep, prev) if prev else _d(rep.period_start) - timedelta(days=1)
        end = _month_end(rep, m)
        flow = sum(f["amount"] for f in rep.flows if start < _d(f["date"]) <= end)
        return navs[m] - (navs[prev] if prev else 0.0) - flow


def _attr(source, inc, p) -> dict:
    """每个月的精确盈亏，按各品种月度贡献占比分摊；利息、费用取精确明细；期货/期货期权/外汇为剩余值。"""
    if hasattr(source, "get_account_info"):
        contrib_rows = source.get_monthly_contributions(start_month=inc[0], end_month=inc[-1])
        contrib = {r["term_month"]: r for r in contrib_rows}
        stocks = opts = 0.0
        pa, skipped = defaultdict(float), []
        for m in inc:
            c = contrib.get(m)
            S = sum((c[k] or 0.0) for k in _KEYS) if c else 0.0
            if abs(S) < 0.05:           # 贡献合计接近 0，无法分摊，该月盈亏进入剩余项
                skipped.append(m)
                continue
            k = _month_profit(source, m) / S
            stocks += ((c["etfs"] or 0.0) + (c["stocks"] or 0.0)) * k
            opts += (c["options"] or 0.0) * k
            for key in ("futures", "futures_options", "fs_options", "cash"):
                pa[key] += (c[key] or 0.0) * k
        s, e = p["start_date"], p["end_date"]
        interest = sum(r["amount"] for r in source.get_cash_transactions(start_date=s, end_date=e, tx_type="interest"))
        fees = sum(r["amount"] for r in source.get_cash_transactions(start_date=s, end_date=e, tx_type="fee"))
        divs = sum(r["amount"] for r in source.get_cash_transactions(start_date=s, end_date=e, tx_type="dividend"))
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
    else:
        rep = source
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


def window_attribution(source, window: str = "YTD") -> Result:
    currency = source.get_account_info()["base_currency"] if hasattr(source, "get_account_info") else source.base_ccy
    base = dict(basis="PA 月度贡献折算 + 精确的利息/费用", currency=currency,
                sources=["Performance by Financial Instrument", "Interest Details", "Fee Summary"])
    try:
        inc = resolve_months(source, window)
    except ValueError as e:
        return Result(None, status="insufficient", warnings=[str(e)], **base)
    p = _period(source, inc)
    a = _attr(source, inc, p)
    a.update({"window": window, "start_date": p["start_date"], "end_date": p["end_date"]})
    warns = ["股票/ETF、股票期权为估算（K2/K5）；期货、期货期权、外汇合并展示（K3）；利息、费用精确（K4）",
             "各项之和等于净值口径的区间盈亏（合并项为剩余值）"]
    if not a["validated"]:
        warns.append(f"窗口包含 {VALIDATED_FROM} 之前的月份，归因估算未经验证，且与 PA 累计按品种盈亏相差较大（见 K5），仅供参考")
    if a["skipped_months"]:
        warns.append(f"{len(a['skipped_months'])} 个月贡献合计接近 0，无法分摊，盈亏已计入合并项")
    return Result(a, period=(p["start_date"], p["end_date"]), warnings=warns, **base)


def yearly_table(source) -> Result:
    months = _months(source)
    if hasattr(source, "get_account_info"):
        currency = source.get_account_info()["base_currency"]
        year_rows = source.get_yearly_returns_data()
        off = {y["year"]: y["account"] for y in year_rows}
    else:
        currency = source.base_ccy
        off = {y["year"]: y["account"] for y in source.year_returns}
    rows = []
    for y in sorted({m[:4] for m in months}):
        inc = [m for m in months if m[:4] == y]
        p = _period(source, inc)
        a = _attr(source, inc, p)
        rows.append({"year": int(y), "start_nav": p["start_nav"], "end_nav": p["end_nav"], "net_flow": p["net_flow"],
                     "profit": p["profit"], "twr_pct": p["twr_pct"], "official_pct": off.get(int(y)),
                     "partial": inc[0][5:] != "01" or y == months[-1][:4], "validated": a["validated"],
                     "buckets": {b["key"]: b["amount"] for b in a["buckets"]}})
    return Result(rows, basis="NAV + 月度TWR链接；归因同 window_attribution", currency=currency,
                  warnings=["2017 年从 8 月开户，期初资产为 0；今年为 YTD"])


def cash_interest(source, window: str = "YTD") -> Result:
    currency = source.get_account_info()["base_currency"] if hasattr(source, "get_account_info") else source.base_ccy
    base = dict(basis="利息明细（精确）÷ 月末现金平均（近似）", currency=currency)
    try:
        inc = resolve_months(source, window)
    except ValueError as e:
        return Result(None, status="insufficient", warnings=[str(e)], **base)
    p = _period(source, inc)
    s, e = p["start_date"], p["end_date"]
    if hasattr(source, "get_account_info"):
        interest = sum(r["amount"] for r in source.get_cash_transactions(start_date=s, end_date=e, tx_type="interest"))
        fees = sum(r["amount"] for r in source.get_cash_transactions(start_date=s, end_date=e, tx_type="fee"))
        breakdowns = source.get_portfolio_breakdown(start_month=inc[0], end_month=inc[-1])
        cash = [r["cash"] or 0 for r in breakdowns if r["term_month"] in inc]
    else:
        interest = _sum_in(source.interest, _d(s), _d(e))
        fees = _sum_in(source.fees, _d(s), _d(e))
        cash = [r["cash"] or 0 for r in source.nav_monthly if r["month"] in inc]
    avg = sum(cash) / len(cash) if cash else 0.0
    days = max((_d(e) - _d(s)).days, 1)
    return Result({"window": window, "interest": interest, "other_fees": fees, "avg_month_end_cash": avg,
                   "implied_annual_yield_pct": interest / avg * 365 / days * 100 if avg else None,
                   "start_date": p["start_date"], "end_date": p["end_date"]},
                  warnings=["现金含 SGD 与 USD 且按月末值平均，隐含收益率只是近似；利息含借款利息（可能为负）（K6）"], **base)


def top_contributors(source, n: int = 10) -> Result:
    if isinstance(n, bool) or not isinstance(n, int):
        return Result(None, status="insufficient", warnings=[f"n 必须是整数，范围为 1 到 {MAX_CONTRIBUTORS}"],
                      basis="累计已实现+未实现盈亏（合约级，不可按时间切片）",
                      currency=source.get_account_info()["base_currency"] if hasattr(source, "get_account_info") else source.base_ccy)
    if n < 1 or n > MAX_CONTRIBUTORS:
        return Result(None, status="insufficient", warnings=[f"n 超出允许范围 1 到 {MAX_CONTRIBUTORS}"],
                      basis="累计已实现+未实现盈亏（合约级，不可按时间切片）",
                      currency=source.get_account_info()["base_currency"] if hasattr(source, "get_account_info") else source.base_ccy)
    if hasattr(source, "get_top_contributors"):
        pnl_data = source.get_top_contributors(n)
        return Result(pnl_data, basis="累计已实现+未实现盈亏（合约级，不可按时间切片；来自 table_symbol_pnl）",
                      currency=source.get_account_info()["base_currency"],
                      warnings=["合计与 MTM 口径不同（K10）；合约级，同一标的的不同合约未合并"])
    else:
        rep = source
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


def risk_measures(source) -> Result:
    if hasattr(source, "get_risk_measures_data"):
        data = source.get_risk_measures_data()
    else:
        data = source.risk
    return Result(data, basis="PA 月度数据计算", warnings=["基于月度数据，会低估月内回撤（K1）"])


def open_positions(source) -> Result:
    if hasattr(source, "get_open_positions_summary"):
        info = source.get_account_info()
        summary = source.get_open_positions_summary()
        return Result({"positions": summary["positions"], "market_value_total": summary["market_value_total"],
                       "asset_class_weights": summary["asset_class_weights"],
                       "cash_net_weight_pct": summary["cash_net_weight_pct"]},
                      basis="市值口径", currency=info["base_currency"],
                      warnings=["这是市值口径，不是风险口径：卖出期权的名义敞口未体现（K9）"])
    else:
        rep = source
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
                  "月度成交笔数与交易频率（无逐笔成交时间，且不得用到期月份代替，见 K8）", "每日净值与月内回撤", "名义风险敞口（需另行设计）"]


def get_capabilities(source, snapshots: list | None = None) -> dict:
    if hasattr(source, "get_account_info"):
        info = source.get_account_info()
        account = info["account"]
        base_ccy = info["base_currency"]
        period = [info["period_start"], info["period_end"]]
        months = info["months"]
        checks = info["checks"]
        snaps = snapshots if snapshots is not None else source.snapshots()
    else:
        months = _months(source)
        account = source.account
        base_ccy = source.base_ccy
        period = [source.period_start, source.period_end]
        checks = source.checks
        snaps = snapshots or []
    return {"source": "PortfolioAnalyst 自开户报告（月度，TWR）", "account": account, "base_currency": base_ccy,
            "period": period, "months": len(months), "last_month_is_partial": True,
            "windows": {w: ("ok" if (w in ("ALL", "YTD") or len(months) >= _N[w]) else "insufficient") for w in WINDOWS},
            "answerable": ANSWERABLE, "answerable_with_caveats": WITH_CAVEATS, "not_answerable": NOT_ANSWERABLE,
            "constraints": CONSTRAINTS, "import_checks": checks, "snapshots": snaps}


def month_detail(source, month: str) -> dict:
    """Return comprehensive single-month analysis with deterministic boundary checks and warnings."""
    import re
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        return {
            "status": "error",
            "error": {"code": "INVALID_MONTH", "message": "月份格式错误，必须为 YYYY-MM"},
            "value": None,
            "warnings": ["参数 month 格式无效，必须为 YYYY-MM"]
        }

    months = _months(source)
    if month not in months:
        min_m, max_m = months[0], months[-1]
        return {
            "status": "error",
            "error": {
                "code": "MONTH_OUT_OF_RANGE",
                "message": f"月份 {month} 超出报告可用范围（{min_m} 至 {max_m}）",
                "details": {"available_range": [min_m, max_m], "min_month": min_m, "max_month": max_m}
            },
            "value": None,
            "warnings": [f"月份超出可用范围：{min_m} ~ {max_m}"]
        }

    is_partial = (month == months[-1])
    prev_m = _prev(source, month)

    if hasattr(source, "get_account_info"):
        info = source.get_account_info()
        base_ccy = info["base_currency"]
        as_of_date = info["period_end"]
        benchmarks = info["benchmarks"]
        opening_date = _month_end(source, prev_m).isoformat() if prev_m else _d(info["period_start"]).isoformat()
        closing_date = _d(info["period_end"]).isoformat() if is_partial else _month_end(source, month).isoformat()

        nav_rows = source.get_portfolio_breakdown(start_month=(prev_m or month), end_month=month)
        navs = {r["term_month"]: r for r in nav_rows}
        v0 = navs[prev_m]["nav"] if prev_m else 0.0
        v1 = navs[month]["nav"]
        nav_change = v1 - v0

        fl = source.get_flows_data(start_date=opening_date, end_date=closing_date)
        dep = sum(f["amount"] for f in fl if f["amount"] > 0)
        wd = sum(f["amount"] for f in fl if f["amount"] < 0)
        net_flow = dep + wd
        profit_sgd = nav_change - net_flow
        flow_items = [{"date": f["date"], "type": f.get("type"), "description": f.get("description"), "amount": f["amount"]} for f in fl]

        all_rets = source.get_monthly_returns_data()
        rets_map = {r["term_month"]: r for r in all_rets}
        m_ret = rets_map.get(month, {})
        twr_pct = m_ret.get("account_return")
        bms = {benchmarks[i]: m_ret.get(f"bm{i+1}_return") for i in range(len(benchmarks))}
        spxtr = bms.get("SPXTR")
        excess_vs_spxtr = (twr_pct - spxtr) if (twr_pct is not None and spxtr is not None) else None

        complete_months = months[:-1]
        sorted_twrs = sorted([(m, rets_map[m]["account_return"]) for m in complete_months if rets_map.get(m, {}).get("account_return") is not None], key=lambda x: x[1], reverse=True)
        if month in complete_months:
            rank_idx = next(i + 1 for i, (m, _) in enumerate(sorted_twrs) if m == month)
            rank_info = {"rank": rank_idx, "of": len(sorted_twrs)}
        else:
            rank_info = {"rank": None, "of": len(sorted_twrs), "note": "部分月不参与完整月排名"}

        cur_nav = navs[month]
        holdings_month_end = {
            "etfs": cur_nav.get("etfs") or 0.0,
            "stocks": cur_nav.get("stocks") or 0.0,
            "options": cur_nav.get("options") or 0.0,
            "futures_options": cur_nav.get("futures_options") or 0.0,
            "cash": cur_nav.get("cash") or 0.0,
            "nav": cur_nav.get("nav") or 0.0,
        }

        c_rows = source.get_monthly_contributions(start_month=month, end_month=month)
        c_row = c_rows[0] if c_rows else {}
        c_pp = {
            "etfs": c_row.get("etfs") or 0.0,
            "stocks": c_row.get("stocks") or 0.0,
            "options": c_row.get("options") or 0.0,
            "futures": c_row.get("futures") or 0.0,
            "futures_options": c_row.get("futures_options") or 0.0,
            "future_style_options": c_row.get("fs_options") or 0.0,
            "cash": c_row.get("cash") or 0.0,
        }
        interest_exact = sum(r["amount"] for r in source.get_cash_transactions(start_date=opening_date, end_date=closing_date, tx_type="interest"))
        fees_exact = sum(r["amount"] for r in source.get_cash_transactions(start_date=opening_date, end_date=closing_date, tx_type="fee"))
        div_amount = sum(r["amount"] for r in source.get_cash_transactions(start_date=opening_date, end_date=closing_date, tx_type="dividend"))
    else:
        rep = source
        base_ccy = rep.base_ccy
        as_of_date = rep.period_end
        benchmarks = rep.benchmarks
        opening_date = _month_end(rep, prev_m).isoformat() if prev_m else _d(rep.period_start).isoformat()
        closing_date = _d(rep.period_end).isoformat() if is_partial else _month_end(rep, month).isoformat()
        s_date, e_date = _d(opening_date), _d(closing_date)

        navs = {r["month"]: r for r in rep.nav_monthly}
        v0 = navs[prev_m]["nav"] if prev_m else 0.0
        v1 = navs[month]["nav"]
        nav_change = v1 - v0

        fl = [f for f in rep.flows if s_date < _d(f["date"]) <= e_date]
        dep = sum(f["amount"] for f in fl if f["amount"] > 0)
        wd = sum(f["amount"] for f in fl if f["amount"] < 0)
        net_flow = dep + wd
        profit_sgd = nav_change - net_flow
        flow_items = [{"date": f["date"], "type": f.get("type"), "description": f.get("description"), "amount": f["amount"]} for f in fl]

        rets = {r["month"]: r for r in rep.returns_monthly}
        m_ret = rets.get(month, {})
        twr_pct = m_ret.get("account")
        bms = {benchmarks[i]: m_ret.get(f"bm{i+1}") for i in range(len(benchmarks))}
        spxtr = bms.get("SPXTR")
        excess_vs_spxtr = (twr_pct - spxtr) if (twr_pct is not None and spxtr is not None) else None

        complete_months = months[:-1]
        sorted_twrs = sorted([(m, rets[m]["account"]) for m in complete_months if rets.get(m, {}).get("account") is not None], key=lambda x: x[1], reverse=True)
        if month in complete_months:
            rank_idx = next(i + 1 for i, (m, _) in enumerate(sorted_twrs) if m == month)
            rank_info = {"rank": rank_idx, "of": len(sorted_twrs)}
        else:
            rank_info = {"rank": None, "of": len(sorted_twrs), "note": "部分月不参与完整月排名"}

        cur_nav = navs[month]
        holdings_month_end = {
            "etfs": cur_nav.get("etfs") or 0.0,
            "stocks": cur_nav.get("stocks") or 0.0,
            "options": cur_nav.get("options") or 0.0,
            "futures_options": cur_nav.get("futures_options") or 0.0,
            "cash": cur_nav.get("cash") or 0.0,
            "nav": cur_nav.get("nav") or 0.0,
        }

        c_map = {r["month"]: r for r in rep.contrib_monthly}
        c_row = c_map.get(month, {})
        c_pp = {
            "etfs": c_row.get("etfs", 0.0),
            "stocks": c_row.get("stocks", 0.0),
            "options": c_row.get("options", 0.0),
            "futures": c_row.get("futures", 0.0),
            "futures_options": c_row.get("futures_options", 0.0),
            "future_style_options": c_row.get("fs_options", 0.0),
            "cash": c_row.get("cash", 0.0),
        }
        interest_exact = _sum_in(rep.interest, s_date, e_date)
        fees_exact = _sum_in(rep.fees, s_date, e_date)
        div_amount = _sum_in(rep.dividends, s_date, e_date)

    sum_pp = sum(c_pp.values())
    twr_gap_pp = sum_pp - (twr_pct or 0.0)
    consistency = "ok" if abs(twr_gap_pp) <= CONSISTENCY_THRESHOLD_PP else "inconsistent"
    quality = "estimate" if month >= VALIDATED_FROM else "unverified"

    if consistency == "ok" and abs(sum_pp) >= 0.05:
        composition_unavailable_reason = None
        k_factor = profit_sgd / sum_pp
        stocks_etf_pp = c_pp["etfs"] + c_pp["stocks"]
        options_pp = c_pp["options"]
        derivs_pp = c_pp["futures"] + c_pp["futures_options"] + c_pp["future_style_options"]
        cash_pp = c_pp["cash"]

        comp_items = [
            {
                "category": "stocks_etf",
                "label": "股票/ETF",
                "pp": stocks_etf_pp,
                "share_pct": (stocks_etf_pp / sum_pp) * 100,
                "sgd_estimate": stocks_etf_pp * k_factor,
                "quality": quality,
            },
            {
                "category": "stock_options",
                "label": "股票期权",
                "pp": options_pp,
                "share_pct": (options_pp / sum_pp) * 100,
                "sgd_estimate": options_pp * k_factor,
                "quality": quality,
            },
            {
                "category": "derivatives_merged",
                "label": "期货/期货期权/未来风格期权（合并）",
                "pp": derivs_pp,
                "share_pct": (derivs_pp / sum_pp) * 100,
                "sgd_estimate": derivs_pp * k_factor,
                "quality": quality,
                "note": "行权交叉影响导致期货与期货期权不可单拆，默认合并展示（见 K3）",
            },
            {
                "category": "cash_category",
                "label": "现金类",
                "pp": cash_pp,
                "share_pct": (cash_pp / sum_pp) * 100,
                "sgd_estimate": cash_pp * k_factor,
                "quality": quality,
            },
        ]
        cash_sgd_est = cash_pp * k_factor
        unexplained = cash_sgd_est - interest_exact - fees_exact
        cash_category = {
            "pp": cash_pp,
            "sgd_estimate": cash_sgd_est,
            "interest_exact": interest_exact,
            "fees_exact": fees_exact,
            "unexplained_sgd_estimate": unexplained,
            "note": "现金类含利息、其他费用、外汇折算及未明确归类部分，未解释金额不可断言为利息、费用或外汇（见 K13）。",
        }
    else:
        comp_items = None
        cash_category = None
        if consistency != "ok":
            composition_unavailable_reason = (
                f"各品种贡献之和（{sum_pp:.2f}%）与月度 TWR（{twr_pct:.2f}%）相差 {twr_gap_pp:+.2f} 个百分点，"
                f"超出允许门禁（±{CONSISTENCY_THRESHOLD_PP} 个百分点），品种拆分不自洽，不提供品种拆分。"
            )
        else:
            composition_unavailable_reason = (
                f"各品种贡献之和（{sum_pp:.2f}%）接近 0，无法按比例分摊盈亏。"
            )

    dividends_memo = {
        "amount": div_amount,
        "currency": base_ccy,
        "note": "股息明细精确，但 PA 未指明计入哪一项贡献，仅作备注，不单独扣减（见 K4）。",
    }

    can_say = [
        f"本月（{month}）TWR 收益率（{twr_pct:+.2f}%）、同月基准表现及在 110 个完整月中的历史排名",
        "净值变动、当月出入金流水与精确投资盈亏金额（入金不计入收益）",
        "各类资产对当月收益的贡献占比（PA 口径，属于笼统类别占比）",
        f"利息与其它费用的当月精确发生金额",
    ]
    cannot_say = [
        "具体是哪笔交易、哪个合约、什么方向、什么时间拉高了收益（PA 不含逐笔成交时间，见 K8）",
        "期货与期货期权各自独立的收益贡献（因行权搬家互相关联，必须合并展示，见 K3）",
        "现金类中未解释部分的来源（不得断言为美元升值、利息或外汇，见 K13）",
        "杠杆倍数、占用保证金、剩余流动性或名义风险敞口（无相关数据，见 K7、K9）",
    ]

    if is_partial:
        can_say.append(f"该月份为截至 {closing_date} 的部分月数据")
        cannot_say.append(f"不得断言为完整全月表现（数据仅截至 {closing_date}）")
    if consistency == "inconsistent":
        cannot_say.append(f"本月贡献之和与 TWR 差距超出阈值（差 {twr_gap_pp:+.2f} 个百分点），严禁引用任何品种拆分与估算（见 K3）")
    if quality == "unverified":
        cannot_say.append(f"本月处于 {VALIDATED_FROM} 验证起点之前，归因估算未经验证，仅供参考（见 K5）")

    caveats = ["K1", "K2", "K3", "K4", "K9", "K13", "K14"]
    if quality == "unverified":
        caveats.append("K5")

    warnings = []
    if is_partial:
        warnings.append(f"{month} 截至 {closing_date}，是部分月。")
    if consistency != "ok":
        warnings.append(f"{month} 品种贡献之和（{sum_pp:.2f}%）与 TWR（{twr_pct:.2f}%）相差 {twr_gap_pp:+.2f} 个百分点，自洽门禁未通过，不提供品种拆分。")
    if quality == "unverified":
        warnings.append(f"{month} 处于 {VALIDATED_FROM} 之前，归因估算未经验证。")

    res = {
        "status": "ok",
        "month": month,
        "is_partial": is_partial,
        "opening_date": opening_date,
        "closing_date": closing_date,
        "return": {
            "twr_pct": twr_pct,
            "benchmarks_pct": bms,
            "excess_vs_spxtr_pct": excess_vs_spxtr,
            "rank": rank_info,
        },
        "nav": {
            "opening": v0,
            "closing": v1,
            "change": nav_change,
        },
        "flows": {
            "items": flow_items,
            "deposits": dep,
            "withdrawals": wd,
            "net": net_flow,
        },
        "profit_sgd": profit_sgd,
        "holdings_month_end": holdings_month_end,
        "contribution_pp": {
            **c_pp,
            "sum": sum_pp,
            "twr_gap_pp": twr_gap_pp,
            "consistency": consistency,
        },
        "composition": comp_items,
        "composition_unavailable_reason": composition_unavailable_reason,
        "cash_category": cash_category,
        "dividends_memo": dividends_memo,
        "boundary": {
            "can_say": can_say,
            "cannot_say": cannot_say,
        },
        "caveats": sorted(set(caveats), key=lambda x: int(x[1:])),
        "quality": quality,
        "as_of": as_of_date,
        "basis": "PA 月度净值与资金流精确盈亏 + 各品种月度贡献折算",
        "period": [opening_date, closing_date],
        "currency": base_ccy,
        "warnings": warnings,
    }
    res["value"] = res
    return res
