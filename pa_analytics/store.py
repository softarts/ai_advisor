"""快照存储：每次导入的 PA 报告是一份完整快照（自开户起累计），按报告截止日保存，不做合并。
当前视图 = 截止日最新的快照。旧快照保留，可用于“和上次相比变了什么”。"""
from __future__ import annotations
import json, shutil, sqlite3
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from .loader import parse_report, PAReport, adapt_portfolio

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots(
  id INTEGER PRIMARY KEY, account TEXT, period_start TEXT, period_end TEXT, file_hash TEXT UNIQUE,
  file_name TEXT, imported_at TEXT, checks_ok INT, payload TEXT);

CREATE TABLE IF NOT EXISTS table_portfolio(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, term_month TEXT NOT NULL,
  value REAL, currency TEXT NOT NULL,
  PRIMARY KEY(snapshot_id, account_id, term_month));
CREATE INDEX IF NOT EXISTS idx_portfolio_period ON table_portfolio(account_id, term_month);

CREATE TABLE IF NOT EXISTS table_monthly_returns(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, term_month TEXT NOT NULL,
  account_return REAL, bm1_return REAL, bm2_return REAL, bm3_return REAL,
  PRIMARY KEY(snapshot_id, account_id, term_month));
CREATE INDEX IF NOT EXISTS idx_returns_period ON table_monthly_returns(account_id, term_month);

CREATE TABLE IF NOT EXISTS table_flows(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, flow_date TEXT NOT NULL,
  flow_type TEXT, description TEXT, amount REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_flows_date ON table_flows(account_id, flow_date);

CREATE TABLE IF NOT EXISTS table_symbol_pnl(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, symbol TEXT NOT NULL,
  description TEXT, instrument TEXT, avg_weight REAL,
  ret_pct REAL, contribution REAL, unrealized REAL,
  realized REAL, total_pnl REAL, open_status TEXT);
CREATE INDEX IF NOT EXISTS idx_symbol_pnl ON table_symbol_pnl(snapshot_id, total_pnl);

CREATE TABLE IF NOT EXISTS table_open_positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, as_of_date TEXT NOT NULL,
  instrument TEXT, currency TEXT, symbol TEXT NOT NULL,
  description TEXT, quantity REAL, close_price REAL,
  value REAL, cost_basis REAL, unrealized REAL, fx_to_base REAL);
CREATE INDEX IF NOT EXISTS idx_open_positions ON table_open_positions(snapshot_id, account_id);
"""


class Store:
    def __init__(self, db_path: str, raw_dir: str | None = None):
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._backfill()

    def _backfill(self):
        """Populate canonical tables for snapshots created before new tables existed."""
        rows = self.conn.execute("SELECT id, account, payload FROM snapshots").fetchall()
        with self.conn:
            for row in rows:
                snap_id, account = row["id"], row["account"]
                report = PAReport(**json.loads(row["payload"]))
                if not self.conn.execute("SELECT 1 FROM table_portfolio WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO table_portfolio(snapshot_id,account_id,term_month,value,currency) VALUES(?,?,?,?,?)",
                        [(snap_id, account, item["term_month"], item["value"], item["currency"])
                         for item in adapt_portfolio(report)])
                if not self.conn.execute("SELECT 1 FROM table_monthly_returns WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO table_monthly_returns(snapshot_id,account_id,term_month,account_return,bm1_return,bm2_return,bm3_return) VALUES(?,?,?,?,?,?,?)",
                        [(snap_id, account, item["month"], item["account"], item["bm1"], item["bm2"], item["bm3"])
                         for item in report.returns_monthly])
                if not self.conn.execute("SELECT 1 FROM table_flows WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_flows(snapshot_id,account_id,flow_date,flow_type,description,amount) VALUES(?,?,?,?,?,?)",
                        [(snap_id, account, f["date"], f.get("type"), f.get("description"), f["amount"])
                         for f in report.flows])
                if not self.conn.execute("SELECT 1 FROM table_symbol_pnl WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_symbol_pnl(snapshot_id,account_id,symbol,description,instrument,avg_weight,ret_pct,contribution,unrealized,realized,total_pnl,open_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(snap_id, account, s["symbol"], s.get("description"), s.get("instrument"), s.get("avg_weight"), s.get("ret_pct"), s.get("contribution"), s["unrealized"], s["realized"], s["unrealized"] + s["realized"], s.get("open"))
                         for s in report.symbol_perf])
                if not self.conn.execute("SELECT 1 FROM table_open_positions WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_open_positions(snapshot_id,account_id,as_of_date,instrument,currency,symbol,description,quantity,close_price,value,cost_basis,unrealized,fx_to_base) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(snap_id, account, p["date"], p.get("instrument"), p.get("currency"), p["symbol"], p.get("description"), p.get("quantity"), p.get("close_price"), p.get("value"), p.get("cost_basis"), p.get("unrealized"), p.get("fx_to_base"))
                         for p in report.open_positions])

    def ingest(self, path: str, force: bool = False) -> dict:
        try:
            rep = parse_report(path)
        except (ValueError, KeyError) as e:
            return {"action": "rejected", "file": path, "reason": f"解析失败: {e}"}
        if self.conn.execute("SELECT 1 FROM snapshots WHERE file_hash=?", (rep.file_hash,)).fetchone():
            return {"action": "duplicate", "file": path}
        failed = [c for c in rep.checks if not c["ok"]]
        if failed and not force:
            return {"action": "rejected", "file": path, "checks": failed,
                    "reason": "校验未通过: " + "; ".join(f"{c['id']} {c['desc']}" for c in failed) + "（可用 force 强制导入）"}
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO snapshots(account,period_start,period_end,file_hash,file_name,imported_at,checks_ok,payload) VALUES(?,?,?,?,?,?,?,?)",
                (rep.account, rep.period_start, rep.period_end, rep.file_hash, Path(path).name,
                 datetime.now().isoformat(timespec="seconds"), int(not failed), json.dumps(asdict(rep), ensure_ascii=False)))
            snap_id = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO table_portfolio(snapshot_id,account_id,term_month,value,currency) VALUES(?,?,?,?,?)",
                [(snap_id, rep.account, row["term_month"], row["value"], row["currency"])
                 for row in adapt_portfolio(rep)])
            self.conn.executemany(
                "INSERT INTO table_monthly_returns(snapshot_id,account_id,term_month,account_return,bm1_return,bm2_return,bm3_return) VALUES(?,?,?,?,?,?,?)",
                [(snap_id, rep.account, item["month"], item["account"], item["bm1"], item["bm2"], item["bm3"])
                 for item in rep.returns_monthly])
            self.conn.executemany(
                "INSERT INTO table_flows(snapshot_id,account_id,flow_date,flow_type,description,amount) VALUES(?,?,?,?,?,?)",
                [(snap_id, rep.account, f["date"], f.get("type"), f.get("description"), f["amount"])
                 for f in rep.flows])
            self.conn.executemany(
                "INSERT INTO table_symbol_pnl(snapshot_id,account_id,symbol,description,instrument,avg_weight,ret_pct,contribution,unrealized,realized,total_pnl,open_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                [(snap_id, rep.account, s["symbol"], s.get("description"), s.get("instrument"), s.get("avg_weight"), s.get("ret_pct"), s.get("contribution"), s["unrealized"], s["realized"], s["unrealized"] + s["realized"], s.get("open"))
                 for s in rep.symbol_perf])
            self.conn.executemany(
                "INSERT INTO table_open_positions(snapshot_id,account_id,as_of_date,instrument,currency,symbol,description,quantity,close_price,value,cost_basis,unrealized,fx_to_base) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(snap_id, rep.account, p["date"], p.get("instrument"), p.get("currency"), p["symbol"], p.get("description"), p.get("quantity"), p.get("close_price"), p.get("value"), p.get("cost_basis"), p.get("unrealized"), p.get("fx_to_base"))
                 for p in rep.open_positions])
        if self.raw_dir:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self.raw_dir / f"{rep.account}_{rep.period_end}_{rep.file_hash[:8]}.plain")
        return {"action": "added", "file": path, "snapshot_id": snap_id, "period_end": rep.period_end,
                "checks_ok": not failed}

    def snapshots(self) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT id,account,period_start,period_end,file_name,imported_at,checks_ok FROM snapshots ORDER BY period_end,id")]

    def load(self, snapshot_id: int) -> PAReport:
        row = self.conn.execute("SELECT payload FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if not row:
            raise KeyError(f"没有快照 {snapshot_id}")
        return PAReport(**json.loads(row["payload"]))

    def current(self, account: str | None = None) -> PAReport:
        q = "SELECT id FROM snapshots " + ("WHERE account=? " if account else "") + "ORDER BY period_end DESC, id DESC LIMIT 1"
        row = self.conn.execute(q, (account,) if account else ()).fetchone()
        if not row:
            raise ValueError("库中没有快照，请先导入 PA 报告")
        return self.load(row["id"])

    def _current_snap_id(self, account: str | None = None) -> tuple[int, str]:
        q = "SELECT id, account FROM snapshots " + ("WHERE account=? " if account else "") + "ORDER BY period_end DESC, id DESC LIMIT 1"
        row = self.conn.execute(q, (account,) if account else ()).fetchone()
        if not row:
            raise ValueError("库中没有快照，请先导入 PA 报告")
        return row["id"], row["account"]

    def get_portfolio_series(self, start_month: str | None = None, end_month: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query portfolio observations from table_portfolio."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if start_month:
            conditions.append("term_month >= ?")
            params.append(start_month)
        if end_month:
            conditions.append("term_month <= ?")
            params.append(end_month)
        sql = f"SELECT term_month, value, currency FROM table_portfolio WHERE {' AND '.join(conditions)} ORDER BY term_month"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_monthly_returns_data(self, start_month: str | None = None, end_month: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query monthly return observations from table_monthly_returns."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if start_month:
            conditions.append("term_month >= ?")
            params.append(start_month)
        if end_month:
            conditions.append("term_month <= ?")
            params.append(end_month)
        sql = f"SELECT term_month, account_return, bm1_return, bm2_return, bm3_return FROM table_monthly_returns WHERE {' AND '.join(conditions)} ORDER BY term_month"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_top_contributors(self, n: int = 10, account: str | None = None) -> dict:
        """Data API: Efficiently query symbol PnL rankings and totals using table_symbol_pnl."""
        snap_id, _ = self._current_snap_id(account)
        winners = [dict(r) for r in self.conn.execute(
            "SELECT symbol, instrument, total_pnl AS pnl, open_status AS open FROM table_symbol_pnl WHERE snapshot_id=? ORDER BY total_pnl DESC LIMIT ?",
            (snap_id, n)).fetchall()]
        losers = [dict(r) for r in self.conn.execute(
            "SELECT symbol, instrument, total_pnl AS pnl, open_status AS open FROM table_symbol_pnl WHERE snapshot_id=? ORDER BY total_pnl ASC LIMIT ?",
            (snap_id, n)).fetchall()]
        by_inst = {r["instrument"]: r["total"] for r in self.conn.execute(
            "SELECT instrument, SUM(total_pnl) AS total FROM table_symbol_pnl WHERE snapshot_id=? GROUP BY instrument",
            (snap_id,)).fetchall()}
        return {"by_instrument": by_inst, "total": sum(by_inst.values()), "winners": winners, "losers": losers}

    def get_flows_data(self, start_date: str | None = None, end_date: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query cash flows from table_flows."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if start_date:
            conditions.append("flow_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("flow_date <= ?")
            params.append(end_date)
        sql = f"SELECT flow_date, flow_type, description, amount FROM table_flows WHERE {' AND '.join(conditions)} ORDER BY flow_date"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_open_positions_data(self, account: str | None = None) -> list[dict]:
        """Data API: Query open positions from table_open_positions."""
        snap_id, _ = self._current_snap_id(account)
        sql = "SELECT as_of_date, instrument, currency, symbol, description, quantity, close_price, value, cost_basis, unrealized, fx_to_base FROM table_open_positions WHERE snapshot_id=?"
        return [dict(r) for r in self.conn.execute(sql, (snap_id,)).fetchall()]

    def portfolio_detail(self, start_month: str, end_month: str, account: str | None = None) -> dict:
        """Query canonical portfolio observations for an inclusive month range."""
        try:
            if not isinstance(start_month, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", start_month):
                raise ValueError("invalid YYYY-MM")
            if not isinstance(end_month, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", end_month):
                raise ValueError("invalid YYYY-MM")
            datetime.strptime(start_month, "%Y-%m")
            datetime.strptime(end_month, "%Y-%m")
        except (TypeError, ValueError) as exc:
            raise ValueError("月份必须使用 YYYY-MM 格式") from exc
        if start_month > end_month:
            raise ValueError("start_month 必须小于或等于 end_month")
        snap_id, account_id = self._current_snap_id(account)
        rows = self.get_portfolio_series(start_month=start_month, end_month=end_month, account=account)
        return {"account_id": account_id, "start_month": start_month, "end_month": end_month,
                "observations": rows, "basis": "月末资产价值；月份范围含首尾"}
