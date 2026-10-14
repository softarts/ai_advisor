import os
import re
import pytest
from pathlib import Path

from pa_analytics import Store, parse_report, window_summary, get_capabilities
from pa_analytics.constraints import CONSTRAINTS, render_markdown
from pa_analytics.metrics import month_detail, NOT_ANSWERABLE
from advisor.tools import create_tools, execute


@pytest.fixture(scope="session")
def report_path():
    p = os.environ.get("PA_REPORT_PATH", "data/report_since_inception.plain")
    assert Path(p).exists(), f"PA report file not found at {p}"
    return p


@pytest.fixture(scope="session")
def report(report_path):
    return parse_report(report_path)


@pytest.fixture(scope="session")
def store(report_path, tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test_pa.db"
    raw_dir = tmp_path_factory.mktemp("raw")
    s = Store(str(db_file), str(raw_dir))
    res = s.ingest(report_path)
    assert res["action"] in ("added", "duplicate")
    return s


# 1. 解析：6 项导入校验全部通过；各区块行数非零
def test_parse_and_checks(report):
    assert len(report.checks) == 6
    for c in report.checks:
        assert c["ok"] is True, f"Check {c['id']} failed: {c['desc']}"
    assert len(report.nav_monthly) > 0
    assert len(report.returns_monthly) > 0
    assert len(report.contrib_monthly) > 0
    assert len(report.flows) > 0
    assert len(report.symbol_perf) > 0
    assert len(report.open_positions) > 0


# 2. get_month_detail 验收值（容差：百分比 ±0.01，金额 ±2，估算项 ±150）
def test_month_detail_2025_03(report):
    d = month_detail(report, "2025-03")
    assert d["status"] == "ok"
    assert d["month"] == "2025-03"

    # TWR, SPXTR, Rank
    assert pytest.approx(4.495, abs=0.01) == d["return"]["twr_pct"]
    assert pytest.approx(-5.63, abs=0.01) == d["return"]["benchmarks_pct"]["SPXTR"]
    assert d["return"]["rank"]["rank"] == 6
    assert d["return"]["rank"]["of"] == 110

    # Contrib sum and consistency
    assert pytest.approx(4.46, abs=0.01) == d["contribution_pp"]["sum"]
    assert d["contribution_pp"]["consistency"] == "ok"
    assert pytest.approx(-0.04, abs=0.01) == d["contribution_pp"]["twr_gap_pp"]

    # NAV & Profit
    assert pytest.approx(420878, abs=2) == d["nav"]["opening"]
    assert pytest.approx(499814, abs=2) == d["nav"]["closing"]
    assert pytest.approx(18935, abs=2) == d["profit_sgd"]

    # Flows
    assert len(d["flows"]["items"]) == 2
    flow_dates = [f["date"] for f in d["flows"]["items"]]
    assert "2025-03-24" in flow_dates
    assert "2025-03-25" in flow_dates
    for f in d["flows"]["items"]:
        assert pytest.approx(30000, abs=1) == f["amount"]

    # Holdings
    assert pytest.approx(-7453, abs=2) == d["holdings_month_end"]["futures_options"]
    assert pytest.approx(507267, abs=2) == d["holdings_month_end"]["cash"]
    assert pytest.approx(0, abs=1) == d["holdings_month_end"]["etfs"]
    assert pytest.approx(0, abs=1) == d["holdings_month_end"]["stocks"]
    assert pytest.approx(0, abs=1) == d["holdings_month_end"]["options"]

    # Composition & Cash category
    assert d["composition"] is not None
    derivs = next(x for x in d["composition"] if x["category"] == "derivatives_merged")
    assert pytest.approx(3.32, abs=0.01) == derivs["pp"]

    cc = d["cash_category"]
    assert cc is not None
    assert pytest.approx(406, abs=2) == cc["interest_exact"]
    assert pytest.approx(-33, abs=2) == cc["fees_exact"]
    assert pytest.approx(4843, abs=150) == cc["sgd_estimate"]
    assert pytest.approx(4470, abs=150) == cc["unexplained_sgd_estimate"]


def test_month_detail_2025_04(report):
    d = month_detail(report, "2025-04")
    assert d["status"] == "ok"
    assert pytest.approx(-3.747, abs=0.01) == d["return"]["twr_pct"]
    assert pytest.approx(-0.68, abs=0.01) == d["return"]["benchmarks_pct"]["SPXTR"]
    assert d["return"]["rank"]["rank"] == 99
    assert d["return"]["rank"]["of"] == 110

    assert pytest.approx(-17.91, abs=0.01) == d["contribution_pp"]["sum"]
    assert d["contribution_pp"]["consistency"] == "inconsistent"
    assert pytest.approx(-14.16, abs=0.01) == d["contribution_pp"]["twr_gap_pp"]

    assert pytest.approx(499814, abs=2) == d["nav"]["opening"]
    assert pytest.approx(481087, abs=2) == d["nav"]["closing"]
    assert pytest.approx(-18726, abs=2) == d["profit_sgd"]

    # Must be null
    assert d["composition"] is None
    assert d["cash_category"] is None
    assert d["composition_unavailable_reason"] is not None
    assert "-14.16" in d["composition_unavailable_reason"] or "门禁" in d["composition_unavailable_reason"]


def test_month_detail_2025_05(report):
    d = month_detail(report, "2025-05")
    assert d["status"] == "ok"
    assert pytest.approx(-4.015, abs=0.01) == d["return"]["twr_pct"]
    assert pytest.approx(6.29, abs=0.01) == d["return"]["benchmarks_pct"]["SPXTR"]
    assert d["return"]["rank"]["rank"] == 100
    assert d["return"]["rank"]["of"] == 110

    assert pytest.approx(-7.23, abs=0.01) == d["contribution_pp"]["sum"]
    assert d["contribution_pp"]["consistency"] == "inconsistent"
    assert pytest.approx(-3.21, abs=0.01) == d["contribution_pp"]["twr_gap_pp"]

    assert pytest.approx(481087, abs=2) == d["nav"]["opening"]
    assert pytest.approx(461770, abs=2) == d["nav"]["closing"]
    assert pytest.approx(-19317, abs=2) == d["profit_sgd"]

    assert d["composition"] is None
    assert d["cash_category"] is None
    assert d["composition_unavailable_reason"] is not None


def test_month_detail_2026_03(report):
    d = month_detail(report, "2026-03")
    assert d["status"] == "ok"
    assert pytest.approx(4.485, abs=0.01) == d["return"]["twr_pct"]
    assert pytest.approx(-4.98, abs=0.01) == d["return"]["benchmarks_pct"]["SPXTR"]
    assert d["return"]["rank"]["rank"] == 7
    assert d["return"]["rank"]["of"] == 110

    assert pytest.approx(4.43, abs=0.01) == d["contribution_pp"]["sum"]
    assert d["contribution_pp"]["consistency"] == "ok"

    assert pytest.approx(536358, abs=2) == d["nav"]["opening"]
    assert pytest.approx(553325, abs=2) == d["nav"]["closing"]
    assert pytest.approx(23966, abs=2) == d["profit_sgd"]

    assert d["composition"] is not None
    derivs = next(x for x in d["composition"] if x["category"] == "derivatives_merged")
    assert pytest.approx(3.25, abs=0.01) == derivs["pp"]

    cc = d["cash_category"]
    assert cc is not None
    assert pytest.approx(515, abs=2) == cc["interest_exact"]
    assert pytest.approx(-7, abs=2) == cc["fees_exact"]
    assert pytest.approx(6666, abs=150) == cc["sgd_estimate"]
    assert pytest.approx(6157, abs=150) == cc["unexplained_sgd_estimate"]


def test_month_detail_2022_10(report):
    d = month_detail(report, "2022-10")
    assert d["status"] == "ok"
    assert pytest.approx(16.504, abs=0.01) == d["return"]["twr_pct"]
    assert pytest.approx(8.10, abs=0.01) == d["return"]["benchmarks_pct"]["SPXTR"]
    assert d["return"]["rank"]["rank"] == 1
    assert d["return"]["rank"]["of"] == 110

    assert pytest.approx(15.71, abs=0.01) == d["contribution_pp"]["sum"]
    assert d["contribution_pp"]["consistency"] == "ok"
    assert pytest.approx(-0.79, abs=0.01) == d["contribution_pp"]["twr_gap_pp"]

    assert pytest.approx(196184, abs=2) == d["nav"]["opening"]
    assert pytest.approx(228563, abs=2) == d["nav"]["closing"]
    assert pytest.approx(32378, abs=2) == d["profit_sgd"]

    # quality must be unverified
    assert d["quality"] == "unverified"
    assert "K5" in d["caveats"]


# 3. composition 各项 sgd_estimate 之和 == profit_sgd（±0.01）；inconsistent 月份 composition 为 null 且给出原因
def test_composition_properties(report):
    for m in ["2025-03", "2026-03", "2022-10"]:
        d = month_detail(report, m)
        assert d["composition"] is not None
        sgd_sum = sum(item["sgd_estimate"] for item in d["composition"])
        assert pytest.approx(d["profit_sgd"], abs=0.01) == sgd_sum

    for m in ["2025-04", "2025-05"]:
        d = month_detail(report, m)
        assert d["composition"] is None
        assert d["composition_unavailable_reason"] is not None
        assert len(d["composition_unavailable_reason"]) > 0


# 4. 月份回显与错误路径
def test_month_echo_and_errors(report):
    d25 = month_detail(report, "2025-03")
    d26 = month_detail(report, "2026-03")
    assert d25["month"] == "2025-03"
    assert d26["month"] == "2026-03"
    assert d25["opening_date"] != d26["opening_date"]
    assert d25["closing_date"] != d26["closing_date"]

    # INVALID_MONTH
    err1 = month_detail(report, "2025/03")
    assert err1["status"] == "error"
    assert err1["error"]["code"] == "INVALID_MONTH"

    err2 = month_detail(report, "invalid")
    assert err2["status"] == "error"
    assert err2["error"]["code"] == "INVALID_MONTH"

    # MONTH_OUT_OF_RANGE
    err3 = month_detail(report, "1999-01")
    assert err3["status"] == "error"
    assert err3["error"]["code"] == "MONTH_OUT_OF_RANGE"
    assert "available_range" in err3["error"]["details"]


# 5. 约束：清单含 K1–K14；docs 文件与源文件一致；get_capabilities 返回的约束与源文件一致；not_answerable 含月度成交笔数；无代理词
def test_constraints_and_no_proxy(report):
    assert len(CONSTRAINTS) == 14
    ids = [c["id"] for c in CONSTRAINTS]
    for i in range(1, 15):
        assert f"K{i}" in ids

    # docs markdown matches render_markdown
    docs_file = Path("PA_KNOWN_CONSTRAINTS.md")
    assert docs_file.exists()
    assert docs_file.read_text(encoding="utf-8") == render_markdown()

    # get_capabilities constraints
    caps = get_capabilities(report)
    assert caps["constraints"] == CONSTRAINTS

    # not_answerable has monthly trade volume constraint mentioning K8
    matched_na = [item for item in caps["not_answerable"] if "月度成交笔数与交易频率" in item]
    assert len(matched_na) == 1
    assert "无逐笔成交时间" in matched_na[0]
    assert "K8" in matched_na[0]

    # No get_trading_activity_proxy in repo python or markdown files
    for root, _, files in os.walk("."):
        if any(ign in root for ign in [".git", "__pycache__", ".pytest_cache"]):
            continue
        for f in files:
            if f == "test_pa.py":
                continue
            if f.endswith((".py", ".md", ".json")):
                p = Path(root) / f
                content = p.read_text(encoding="utf-8", errors="ignore")
                assert "get_trading_activity_proxy" not in content, f"Found get_trading_activity_proxy in {p}"


# 6. PA 版基础测试补齐：窗口 TWR 与官方值一致；快照存储
def test_pa_baseline_and_store(report, store, report_path):
    # YTD: 16.726, 1M: 0.849, 3M: 3.358, ALL: 8.474 (±0.01)
    ytd = window_summary(report, "YTD")
    assert pytest.approx(16.726, abs=0.01) == ytd.value["twr_pct"]

    m1 = window_summary(report, "1M")
    assert pytest.approx(0.849, abs=0.01) == m1.value["twr_pct"]

    m3 = window_summary(report, "3M")
    assert pytest.approx(3.358, abs=0.01) == m3.value["twr_pct"]

    all_win = window_summary(report, "ALL")
    assert pytest.approx(8.474, abs=0.01) == all_win.value["twr_pct"]

    # Duplicate ingest
    dup = store.ingest(report_path)
    assert dup["action"] == "duplicate"

    # Current snapshot
    cur = store.current()
    assert cur.account == report.account
    assert cur.period_end == report.period_end
