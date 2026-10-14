"""LLM 可调用的 tool 清单。每个 tool 返回 Result.to_dict() 或规范 dict（带口径与警告）。"""
from . import metrics as M

_WIN = {"type": "string", "enum": list(M.WINDOWS), "description": "窗口，默认 YTD。1M=当月至今，ALL=自开户"}

TOOLS = [
    {"name": "get_capabilities", "description": "查询数据来源、可回答/不可回答的问题和已知约束。回答前先调用，遇到杠杆、保证金、持仓时间等问题要据此拒绝并说明缺什么。",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_portfolio_detail", "description": "查询指定月份区间的月末投资组合资产价值与计价币种（含首尾月份）。",
     "parameters": {"type": "object", "properties": {
         "start_month": {"type": "string", "format": "year-month", "description": "起始月份 (YYYY-MM)"},
         "end_month": {"type": "string", "format": "year-month", "description": "结束月份 (YYYY-MM)"}},
         "required": ["start_month", "end_month"]}},
    {"name": "get_window_summary", "description": "某窗口的期初/期末资产、转入、转出、区间盈亏、月度链接 TWR 与月度链接基准收益。6M 没有 PA 官方滚动参考值；ALL 期初净值为 0，自开户盈亏必须结合累计资金流解读，优先引用 TWR。",
     "parameters": {"type": "object", "properties": {"window": _WIN}}},
    {"name": "get_monthly_returns", "description": "返回指定报告窗口内全部月度账户收益率、最高收益月份和最低收益月份，并附同期基准月收益。窗口使用受支持的通用窗口标签；最后一个报告月可能是部分月。",
     "parameters": {"type": "object", "properties": {"window": _WIN}}},
    {"name": "get_month_detail", "description": "用于回答某月收益高或低的原因，只给类别层面的收益占比，不能指明具体交易。包含当月精确盈亏、出入金排除、各品种贡献占比与自洽检查、现金类未解释部分以及严谨的回答边界（can_say / cannot_say）。",
     "parameters": {"type": "object", "properties": {
         "month": {"type": "string", "format": "year-month", "description": "指定月份 (YYYY-MM)"}},
         "required": ["month"]}},
    {"name": "get_attribution", "description": "某窗口按资产类别的盈亏归因（股票/ETF、股票期权、期货及期货期权及外汇合并、利息、费用），每项带精确/估算/未验证标记；必须附带返回的口径警告。不得用窗口归因推断某个月的原因。",
     "parameters": {"type": "object", "properties": {"window": _WIN}}},
    {"name": "get_yearly_table", "description": "按自然年的收益、TWR、净转入和归因。", "parameters": {"type": "object", "properties": {}}},
    {"name": "get_cash_interest", "description": "某窗口的利息、费用、平均现金和隐含年化利率（近似），可用于现金拖累分析。",
     "parameters": {"type": "object", "properties": {"window": _WIN}}},
    {"name": "get_top_contributors", "description": "按合约的累计盈亏排名（赢家/输家）及按品种合计。",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}}}},
    {"name": "get_risk_measures", "description": "最大回撤、夏普等月度风险指标，含基准对比。", "parameters": {"type": "object", "properties": {}}},
    {"name": "get_open_positions", "description": "当前持仓与资产类别权重（市值口径，不是风险口径）。", "parameters": {"type": "object", "properties": {}}},
]


def call(target, name: str, args: dict | None = None, snapshots: list | None = None) -> dict:
    from .store import Store
    from .loader import PAReport
    if isinstance(target, Store):
        store = target
        rep = store.current()
        snaps = snapshots if snapshots is not None else store.snapshots()
    elif isinstance(target, PAReport):
        store = None
        rep = target
        snaps = snapshots or []
    else:
        raise TypeError(f"Unsupported target type for tool execution: {type(target)}")

    a = args or {}
    if name == "get_capabilities":
        value = M.get_capabilities(rep, snaps)
        return {"value": value, "basis": "imported PA report capabilities", "period": value.get("period"),
                "currency": value.get("base_currency"), "status": "ok", "warnings": []}
    if name == "get_portfolio_detail":
        if store is not None:
            return store.portfolio_detail(a["start_month"], a["end_month"])
        rows = [{"term_month": r["month"], "value": r["nav"], "currency": rep.base_ccy}
                for r in rep.nav_monthly if a["start_month"] <= r["month"] <= a["end_month"]]
        return {"account_id": rep.account, "start_month": a["start_month"], "end_month": a["end_month"],
                "observations": rows, "basis": "月末资产价值；月份范围含首尾"}
    if name == "get_top_contributors" and store is not None:
        n = a.get("n", 10)
        pnl_data = store.get_top_contributors(n)
        return {"value": pnl_data, "basis": "累计已实现+未实现盈亏（合约级，不可按时间切片；来自 table_symbol_pnl）",
                "period": [None, None], "currency": rep.base_ccy, "status": "ok",
                "warnings": ["合计与 MTM 口径不同（K10）；合约级，同一标的的不同合约未合并"]}
    if name == "get_month_detail":
        return M.month_detail(rep, a.get("month", ""))

    fn = {"get_window_summary": lambda: M.window_summary(rep, a.get("window", "YTD")),
          "get_monthly_returns": lambda: M.monthly_return_extremes(rep, a.get("window", "YTD")),
          "get_attribution": lambda: M.window_attribution(rep, a.get("window", "YTD")),
          "get_yearly_table": lambda: M.yearly_table(rep),
          "get_cash_interest": lambda: M.cash_interest(rep, a.get("window", "YTD")),
          "get_top_contributors": lambda: M.top_contributors(rep, a.get("n", 10)),
          "get_risk_measures": lambda: M.risk_measures(rep),
          "get_open_positions": lambda: M.open_positions(rep)}.get(name)
    if fn is None:
        raise KeyError(f"未知 tool: {name}")
    r = fn().to_dict()
    if name == "get_window_summary" and r["value"]:       # 曲线数据不给 LLM，节省上下文
        r["value"] = {k: v for k, v in r["value"].items() if k not in ("nav_series", "return_curves", "allocation", "flows")}
    return r

