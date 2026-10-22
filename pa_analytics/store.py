"""快照存储：每次导入的 PA 报告是一份完整快照（自开户起累计），按报告截止日保存，不做合并。
当前视图 = 截止日最新的快照。旧快照保留，可用于“和上次相比变了什么”。
提供规范的数据层抽象 API，业务逻辑与工具均通过规范数据层读取数据。"""
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

CREATE TABLE IF NOT EXISTS table_account_metadata(
  snapshot_id INTEGER PRIMARY KEY REFERENCES snapshots(id),
  account_id TEXT NOT NULL, base_currency TEXT NOT NULL,
  period_start TEXT NOT NULL, period_end TEXT NOT NULL,
  benchmarks TEXT NOT NULL, key_stats TEXT, checks TEXT);

CREATE TABLE IF NOT EXISTS table_portfolio(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, term_month TEXT NOT NULL,
  value REAL, currency TEXT NOT NULL,
  etfs REAL, stocks REAL, options REAL, futures_options REAL, cash REAL,
  commodities REAL, equities REAL,
  PRIMARY KEY(snapshot_id, account_id, term_month));
CREATE INDEX IF NOT EXISTS idx_portfolio_period ON table_portfolio(account_id, term_month);

CREATE TABLE IF NOT EXISTS table_monthly_returns(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, term_month TEXT NOT NULL,
  account_return REAL, bm1_return REAL, bm2_return REAL, bm3_return REAL,
  PRIMARY KEY(snapshot_id, account_id, term_month));
CREATE INDEX IF NOT EXISTS idx_returns_period ON table_monthly_returns(account_id, term_month);

CREATE TABLE IF NOT EXISTS table_monthly_contributions(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, term_month TEXT NOT NULL,
  etfs REAL, stocks REAL, options REAL, futures REAL,
  futures_options REAL, fs_options REAL, cash REAL,
  PRIMARY KEY(snapshot_id, account_id, term_month));
CREATE INDEX IF NOT EXISTS idx_contrib_period ON table_monthly_contributions(account_id, term_month);

CREATE TABLE IF NOT EXISTS table_flows(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, flow_date TEXT NOT NULL,
  flow_type TEXT, description TEXT, amount REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_flows_date ON table_flows(account_id, flow_date);

CREATE TABLE IF NOT EXISTS table_cash_transactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, tx_date TEXT NOT NULL,
  tx_type TEXT NOT NULL, symbol TEXT, description TEXT, amount REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_cash_tx_date ON table_cash_transactions(account_id, tx_date);
CREATE INDEX IF NOT EXISTS idx_cash_tx_type ON table_cash_transactions(snapshot_id, tx_type);

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

CREATE TABLE IF NOT EXISTS table_risk_measures(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, measure_name TEXT NOT NULL,
  account_value TEXT, bm1_value REAL, bm2_value REAL, bm3_value REAL);
CREATE INDEX IF NOT EXISTS idx_risk_measures ON table_risk_measures(snapshot_id);

CREATE TABLE IF NOT EXISTS table_yearly_returns(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, year INTEGER NOT NULL,
  account_return REAL, bm1_return REAL, bm2_return REAL, bm3_return REAL,
  PRIMARY KEY(snapshot_id, account_id, year));

CREATE TABLE IF NOT EXISTS table_official_returns(
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, period_label TEXT NOT NULL,
  return_pct REAL,
  PRIMARY KEY(snapshot_id, account_id, period_label));

CREATE TABLE IF NOT EXISTS table_asset_weights(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  account_id TEXT NOT NULL, category TEXT NOT NULL,
  name TEXT NOT NULL, description TEXT,
  long_value REAL, short_value REAL, net_value REAL,
  long_weight_pct REAL, short_weight_pct REAL, net_weight_pct REAL);
CREATE INDEX IF NOT EXISTS idx_asset_weights ON table_asset_weights(snapshot_id, category);
"""


class Store:
    def __init__(self, db_path: str, raw_dir: str | None = None):
        self.raw_dir = Path(raw_dir) if raw_dir else None
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._ensure_columns()
        self._backfill()

    def _ensure_columns(self):
        """Ensure backward-compatible schema evolutions for existing databases."""
        cur = self.conn.execute("PRAGMA table_info(table_portfolio)")
        existing = {r[1] for r in cur.fetchall()}
        for col in ("etfs", "stocks", "options", "futures_options", "cash", "commodities", "equities"):
            if col not in existing:
                self.conn.execute(f"ALTER TABLE table_portfolio ADD COLUMN {col} REAL")

    def _backfill(self):
        """Populate canonical tables for snapshots created before new tables existed."""
        rows = self.conn.execute("SELECT id, account, payload FROM snapshots").fetchall()
        with self.conn:
            for row in rows:
                snap_id, account = row["id"], row["account"]
                report = PAReport(**json.loads(row["payload"]))

                # table_account_metadata
                if not self.conn.execute("SELECT 1 FROM table_account_metadata WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.execute(
                        "INSERT OR REPLACE INTO table_account_metadata(snapshot_id,account_id,base_currency,period_start,period_end,benchmarks,key_stats,checks) VALUES(?,?,?,?,?,?,?,?)",
                        (snap_id, account, report.base_ccy, report.period_start, report.period_end,
                         json.dumps(report.benchmarks, ensure_ascii=False),
                         json.dumps(report.key_stats, ensure_ascii=False),
                         json.dumps(report.checks, ensure_ascii=False)))

                # table_portfolio
                needs_portfolio_update = not self.conn.execute(
                    "SELECT 1 FROM table_portfolio WHERE snapshot_id=? AND etfs IS NOT NULL LIMIT 1", (snap_id,)).fetchone()
                if needs_portfolio_update:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO table_portfolio(snapshot_id,account_id,term_month,value,currency,etfs,stocks,options,futures_options,cash,commodities,equities) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(snap_id, account, item["term_month"], item["value"], item["currency"],
                          item.get("etfs"), item.get("stocks"), item.get("options"),
                          item.get("futures_options"), item.get("cash"),
                          item.get("commodities"), item.get("equities"))
                         for item in adapt_portfolio(report)])

                # table_monthly_returns
                if not self.conn.execute("SELECT 1 FROM table_monthly_returns WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO table_monthly_returns(snapshot_id,account_id,term_month,account_return,bm1_return,bm2_return,bm3_return) VALUES(?,?,?,?,?,?,?)",
                        [(snap_id, account, item["month"], item["account"], item["bm1"], item["bm2"], item["bm3"])
                         for item in report.returns_monthly])

                # table_monthly_contributions
                if not self.conn.execute("SELECT 1 FROM table_monthly_contributions WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO table_monthly_contributions(snapshot_id,account_id,term_month,etfs,stocks,options,futures,futures_options,fs_options,cash) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        [(snap_id, account, item["month"], item.get("etfs"), item.get("stocks"),
                          item.get("options"), item.get("futures"), item.get("futures_options"),
                          item.get("fs_options"), item.get("cash"))
                         for item in report.contrib_monthly])

                # table_flows
                if not self.conn.execute("SELECT 1 FROM table_flows WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_flows(snapshot_id,account_id,flow_date,flow_type,description,amount) VALUES(?,?,?,?,?,?)",
                        [(snap_id, account, f["date"], f.get("type"), f.get("description"), f["amount"])
                         for f in report.flows])

                # table_cash_transactions
                if not self.conn.execute("SELECT 1 FROM table_cash_transactions WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    txs = []
                    for item in report.interest:
                        txs.append((snap_id, account, item["date"], "interest", None, item.get("description"), item["amount"]))
                    for item in report.fees:
                        txs.append((snap_id, account, item["date"], "fee", None, item.get("description"), item["amount"]))
                    for item in report.dividends:
                        txs.append((snap_id, account, item["date"], "dividend", item.get("symbol"), None, item["amount"]))
                    if txs:
                        self.conn.executemany(
                            "INSERT INTO table_cash_transactions(snapshot_id,account_id,tx_date,tx_type,symbol,description,amount) VALUES(?,?,?,?,?,?,?)",
                            txs)

                # table_symbol_pnl
                if not self.conn.execute("SELECT 1 FROM table_symbol_pnl WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_symbol_pnl(snapshot_id,account_id,symbol,description,instrument,avg_weight,ret_pct,contribution,unrealized,realized,total_pnl,open_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(snap_id, account, s["symbol"], s.get("description"), s.get("instrument"), s.get("avg_weight"), s.get("ret_pct"), s.get("contribution"), s["unrealized"], s["realized"], s["unrealized"] + s["realized"], s.get("open"))
                         for s in report.symbol_perf])

                # table_open_positions
                if not self.conn.execute("SELECT 1 FROM table_open_positions WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_open_positions(snapshot_id,account_id,as_of_date,instrument,currency,symbol,description,quantity,close_price,value,cost_basis,unrealized,fx_to_base) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(snap_id, account, p["date"], p.get("instrument"), p.get("currency"), p["symbol"], p.get("description"), p.get("quantity"), p.get("close_price"), p.get("value"), p.get("cost_basis"), p.get("unrealized"), p.get("fx_to_base"))
                         for p in report.open_positions])

                # table_risk_measures
                if not self.conn.execute("SELECT 1 FROM table_risk_measures WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT INTO table_risk_measures(snapshot_id,account_id,measure_name,account_value,bm1_value,bm2_value,bm3_value) VALUES(?,?,?,?,?,?,?)",
                        [(snap_id, account, r["measure"], str(r["account"]) if r["account"] is not None else None,
                          r.get("bm1"), r.get("bm2"), r.get("bm3"))
                         for r in report.risk])

                # table_yearly_returns
                if not self.conn.execute("SELECT 1 FROM table_yearly_returns WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO table_yearly_returns(snapshot_id,account_id,year,account_return,bm1_return,bm2_return,bm3_return) VALUES(?,?,?,?,?,?,?)",
                        [(snap_id, account, y["year"], y.get("account"), y.get("bm1"), y.get("bm2"), y.get("bm3"))
                         for y in report.year_returns])

                # table_official_returns
                if not self.conn.execute("SELECT 1 FROM table_official_returns WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO table_official_returns(snapshot_id,account_id,period_label,return_pct) VALUES(?,?,?,?)",
                        [(snap_id, account, k, v) for k, v in report.official.items()])

                # table_asset_weights
                if not self.conn.execute("SELECT 1 FROM table_asset_weights WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone():
                    weights = []
                    for ac in report.concentration.get("asset_class", []):
                        weights.append((snap_id, account, "asset_class", ac.get("asset_class"), None,
                                        None, None, None, ac.get("long_weight_pct"), ac.get("short_weight_pct"), None))
                    for h in report.concentration.get("holdings", []):
                        weights.append((snap_id, account, "holding", h.get("symbol"), h.get("description"),
                                        h.get("long_value"), h.get("short_value"), h.get("net_value"),
                                        None, None, h.get("net_weight_pct")))
                    if weights:
                        self.conn.executemany(
                            "INSERT INTO table_asset_weights(snapshot_id,account_id,category,name,description,long_value,short_value,net_value,long_weight_pct,short_weight_pct,net_weight_pct) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            weights)

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

            # table_account_metadata
            self.conn.execute(
                "INSERT OR REPLACE INTO table_account_metadata(snapshot_id,account_id,base_currency,period_start,period_end,benchmarks,key_stats,checks) VALUES(?,?,?,?,?,?,?,?)",
                (snap_id, rep.account, rep.base_ccy, rep.period_start, rep.period_end,
                 json.dumps(rep.benchmarks, ensure_ascii=False),
                 json.dumps(rep.key_stats, ensure_ascii=False),
                 json.dumps(rep.checks, ensure_ascii=False)))

            # table_portfolio
            self.conn.executemany(
                "INSERT OR REPLACE INTO table_portfolio(snapshot_id,account_id,term_month,value,currency,etfs,stocks,options,futures_options,cash,commodities,equities) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                [(snap_id, rep.account, row["term_month"], row["value"], row["currency"],
                  row.get("etfs"), row.get("stocks"), row.get("options"),
                  row.get("futures_options"), row.get("cash"),
                  row.get("commodities"), row.get("equities"))
                 for row in adapt_portfolio(rep)])

            # table_monthly_returns
            self.conn.executemany(
                "INSERT INTO table_monthly_returns(snapshot_id,account_id,term_month,account_return,bm1_return,bm2_return,bm3_return) VALUES(?,?,?,?,?,?,?)",
                [(snap_id, rep.account, item["month"], item["account"], item["bm1"], item["bm2"], item["bm3"])
                 for item in rep.returns_monthly])

            # table_monthly_contributions
            self.conn.executemany(
                "INSERT INTO table_monthly_contributions(snapshot_id,account_id,term_month,etfs,stocks,options,futures,futures_options,fs_options,cash) VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(snap_id, rep.account, item["month"], item.get("etfs"), item.get("stocks"),
                  item.get("options"), item.get("futures"), item.get("futures_options"),
                  item.get("fs_options"), item.get("cash"))
                 for item in rep.contrib_monthly])

            # table_flows
            self.conn.executemany(
                "INSERT INTO table_flows(snapshot_id,account_id,flow_date,flow_type,description,amount) VALUES(?,?,?,?,?,?)",
                [(snap_id, rep.account, f["date"], f.get("type"), f.get("description"), f["amount"])
                 for f in rep.flows])

            # table_cash_transactions
            txs = []
            for item in rep.interest:
                txs.append((snap_id, rep.account, item["date"], "interest", None, item.get("description"), item["amount"]))
            for item in rep.fees:
                txs.append((snap_id, rep.account, item["date"], "fee", None, item.get("description"), item["amount"]))
            for item in rep.dividends:
                txs.append((snap_id, rep.account, item["date"], "dividend", item.get("symbol"), None, item["amount"]))
            if txs:
                self.conn.executemany(
                    "INSERT INTO table_cash_transactions(snapshot_id,account_id,tx_date,tx_type,symbol,description,amount) VALUES(?,?,?,?,?,?,?)",
                    txs)

            # table_symbol_pnl
            self.conn.executemany(
                "INSERT INTO table_symbol_pnl(snapshot_id,account_id,symbol,description,instrument,avg_weight,ret_pct,contribution,unrealized,realized,total_pnl,open_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                [(snap_id, rep.account, s["symbol"], s.get("description"), s.get("instrument"), s.get("avg_weight"), s.get("ret_pct"), s.get("contribution"), s["unrealized"], s["realized"], s["unrealized"] + s["realized"], s.get("open"))
                 for s in rep.symbol_perf])

            # table_open_positions
            self.conn.executemany(
                "INSERT INTO table_open_positions(snapshot_id,account_id,as_of_date,instrument,currency,symbol,description,quantity,close_price,value,cost_basis,unrealized,fx_to_base) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(snap_id, rep.account, p["date"], p.get("instrument"), p.get("currency"), p["symbol"], p.get("description"), p.get("quantity"), p.get("close_price"), p.get("value"), p.get("cost_basis"), p.get("unrealized"), p.get("fx_to_base"))
                 for p in rep.open_positions])

            # table_risk_measures
            self.conn.executemany(
                "INSERT INTO table_risk_measures(snapshot_id,account_id,measure_name,account_value,bm1_value,bm2_value,bm3_value) VALUES(?,?,?,?,?,?,?)",
                [(snap_id, rep.account, r["measure"], str(r["account"]) if r["account"] is not None else None,
                  r.get("bm1"), r.get("bm2"), r.get("bm3"))
                 for r in rep.risk])

            # table_yearly_returns
            self.conn.executemany(
                "INSERT INTO table_yearly_returns(snapshot_id,account_id,year,account_return,bm1_return,bm2_return,bm3_return) VALUES(?,?,?,?,?,?,?)",
                [(snap_id, rep.account, y["year"], y.get("account"), y.get("bm1"), y.get("bm2"), y.get("bm3"))
                 for y in rep.year_returns])

            # table_official_returns
            self.conn.executemany(
                "INSERT INTO table_official_returns(snapshot_id,account_id,period_label,return_pct) VALUES(?,?,?,?)",
                [(snap_id, rep.account, k, v) for k, v in rep.official.items()])

            # table_asset_weights
            weights = []
            for ac in rep.concentration.get("asset_class", []):
                weights.append((snap_id, rep.account, "asset_class", ac.get("asset_class"), None,
                                None, None, None, ac.get("long_weight_pct"), ac.get("short_weight_pct"), None))
            for h in rep.concentration.get("holdings", []):
                weights.append((snap_id, rep.account, "holding", h.get("symbol"), h.get("description"),
                                h.get("long_value"), h.get("short_value"), h.get("net_value"),
                                None, None, h.get("net_weight_pct")))
            if weights:
                self.conn.executemany(
                    "INSERT INTO table_asset_weights(snapshot_id,account_id,category,name,description,long_value,short_value,net_value,long_weight_pct,short_weight_pct,net_weight_pct) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    weights)

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

    # ================= Canonical Data Layer APIs =================

    def get_account_info(self, account: str | None = None) -> dict:
        """Query snapshot account metadata and valid month labels."""
        snap_id, account_id = self._current_snap_id(account)
        meta = self.conn.execute(
            "SELECT account_id, base_currency, period_start, period_end, benchmarks, key_stats, checks "
            "FROM table_account_metadata WHERE snapshot_id=?", (snap_id,)).fetchone()
        months = [r[0] for r in self.conn.execute(
            "SELECT term_month FROM table_portfolio WHERE snapshot_id=? ORDER BY term_month", (snap_id,)).fetchall()]
        if meta:
            return {
                "snapshot_id": snap_id,
                "account": meta["account_id"],
                "base_currency": meta["base_currency"],
                "period_start": meta["period_start"],
                "period_end": meta["period_end"],
                "benchmarks": json.loads(meta["benchmarks"]) if meta["benchmarks"] else ["SPXTR", "EFA", "VT"],
                "key_stats": json.loads(meta["key_stats"]) if meta["key_stats"] else {},
                "checks": json.loads(meta["checks"]) if meta["checks"] else [],
                "months": months,
            }
        snap = self.conn.execute("SELECT account, period_start, period_end FROM snapshots WHERE id=?", (snap_id,)).fetchone()
        port = self.conn.execute("SELECT currency FROM table_portfolio WHERE snapshot_id=? LIMIT 1", (snap_id,)).fetchone()
        return {
            "snapshot_id": snap_id,
            "account": snap["account"],
            "base_currency": port["currency"] if port else "SGD",
            "period_start": snap["period_start"],
            "period_end": snap["period_end"],
            "benchmarks": ["SPXTR", "EFA", "VT"],
            "key_stats": {},
            "checks": [],
            "months": months,
        }

    def get_portfolio_series(self, start_month: str | None = None, end_month: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query portfolio observations (month, value, currency) from table_portfolio."""
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

    def get_portfolio_breakdown(self, start_month: str | None = None, end_month: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query portfolio observations with full asset class values."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if start_month:
            conditions.append("term_month >= ?")
            params.append(start_month)
        if end_month:
            conditions.append("term_month <= ?")
            params.append(end_month)
        sql = f"SELECT term_month, value AS nav, currency, etfs, stocks, options, futures_options, cash, commodities, equities FROM table_portfolio WHERE {' AND '.join(conditions)} ORDER BY term_month"
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

    def get_monthly_contributions(self, start_month: str | None = None, end_month: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query monthly asset contributions in pp from table_monthly_contributions."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if start_month:
            conditions.append("term_month >= ?")
            params.append(start_month)
        if end_month:
            conditions.append("term_month <= ?")
            params.append(end_month)
        sql = f"SELECT term_month, etfs, stocks, options, futures, futures_options, fs_options, cash FROM table_monthly_contributions WHERE {' AND '.join(conditions)} ORDER BY term_month"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_flows_data(self, start_date: str | None = None, end_date: str | None = None, account: str | None = None, inclusive_start: bool = False) -> list[dict]:
        """Data API: Query cash flows from table_flows."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if start_date:
            conditions.append("flow_date >= ?" if inclusive_start else "flow_date > ?")
            params.append(start_date)
        if end_date:
            conditions.append("flow_date <= ?")
            params.append(end_date)
        sql = f"SELECT flow_date AS date, flow_type AS type, description, amount FROM table_flows WHERE {' AND '.join(conditions)} ORDER BY flow_date"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_cash_transactions(self, start_date: str | None = None, end_date: str | None = None, tx_type: str | None = None, account: str | None = None, inclusive_start: bool = False) -> list[dict]:
        """Data API: Query interest, fees, or dividends from table_cash_transactions."""
        snap_id, _ = self._current_snap_id(account)
        conditions, params = ["snapshot_id=?"], [snap_id]
        if tx_type:
            conditions.append("tx_type = ?")
            params.append(tx_type)
        if start_date:
            conditions.append("tx_date >= ?" if inclusive_start else "tx_date > ?")
            params.append(start_date)
        if end_date:
            conditions.append("tx_date <= ?")
            params.append(end_date)
        sql = f"SELECT tx_date AS date, tx_type AS type, symbol, description, amount FROM table_cash_transactions WHERE {' AND '.join(conditions)} ORDER BY tx_date"
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

    def get_risk_measures_data(self, account: str | None = None) -> list[dict]:
        """Data API: Query risk measures from table_risk_measures."""
        snap_id, _ = self._current_snap_id(account)
        rows = self.conn.execute(
            "SELECT measure_name AS measure, account_value AS account, bm1_value AS bm1, bm2_value AS bm2, bm3_value AS bm3 FROM table_risk_measures WHERE snapshot_id=? ORDER BY id",
            (snap_id,)).fetchall()
        out = []
        for r in rows:
            val = r["account"]
            try:
                val = float(val) if val is not None else None
            except ValueError:
                pass
            out.append({"measure": r["measure"], "account": val, "bm1": r["bm1"], "bm2": r["bm2"], "bm3": r["bm3"]})
        return out

    def get_yearly_returns_data(self, account: str | None = None) -> list[dict]:
        """Data API: Query yearly returns from table_yearly_returns."""
        snap_id, _ = self._current_snap_id(account)
        rows = self.conn.execute(
            "SELECT year, account_return AS account, bm1_return AS bm1, bm2_return AS bm2, bm3_return AS bm3 FROM table_yearly_returns WHERE snapshot_id=? ORDER BY year",
            (snap_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_official_returns_data(self, account: str | None = None) -> dict:
        """Data API: Query official period returns from table_official_returns."""
        snap_id, _ = self._current_snap_id(account)
        rows = self.conn.execute(
            "SELECT period_label, return_pct FROM table_official_returns WHERE snapshot_id=?",
            (snap_id,)).fetchall()
        return {r["period_label"]: r["return_pct"] for r in rows}

    def get_asset_weights(self, category: str | None = None, account: str | None = None) -> list[dict]:
        """Data API: Query asset class and holding weights from table_asset_weights."""
        snap_id, _ = self._current_snap_id(account)
        cond, params = ["snapshot_id=?"], [snap_id]
        if category:
            cond.append("category=?")
            params.append(category)
        sql = f"SELECT category, name, description, long_value, short_value, net_value, long_weight_pct, short_weight_pct, net_weight_pct FROM table_asset_weights WHERE {' AND '.join(cond)} ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_open_positions_data(self, account: str | None = None) -> list[dict]:
        """Data API: Query open positions from table_open_positions.

        The report's ``value`` column is already expressed in the account's
        base currency (for example USD cash of 365,251.95 is reported as
        467,778.17 SGD, and the two cash rows sum to the report total), so
        ``value_base`` repeats that figure under an explicit name. It lets a
        reader see that the amount is in the account currency instead of
        having to infer it from the sibling ``currency`` field.
        """
        snap_id, _ = self._current_snap_id(account)
        sql = "SELECT as_of_date, instrument, currency, symbol, description, quantity, close_price, value, cost_basis, unrealized, fx_to_base FROM table_open_positions WHERE snapshot_id=?"
        rows = [dict(r) for r in self.conn.execute(sql, (snap_id,)).fetchall()]
        for row in rows:
            row["value_base"] = row.get("value")
            row["unrealized_base"] = row.get("unrealized")
        return rows

    def get_open_positions_summary(self, account: str | None = None) -> dict:
        """Data API: Query aggregated open positions and concentration weights."""
        positions = self.get_open_positions_data(account=account)
        total = sum(r["value"] or 0 for r in positions)
        classes = [{"asset_class": r["name"], "long_weight_pct": r["long_weight_pct"], "short_weight_pct": r["short_weight_pct"]}
                   for r in self.get_asset_weights(category="asset_class", account=account)]
        cash_row = next((r for r in self.get_asset_weights(category="holding", account=account) if r["name"] == "CASH"), None)
        cash_net_weight = cash_row["net_weight_pct"] if cash_row else None
        return {
            "positions": positions,
            "market_value_total": total,
            "asset_class_weights": classes,
            "cash_net_weight_pct": cash_net_weight
        }

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
        report = self.current()
        return {"account_id": account_id, "start_month": start_month, "end_month": end_month,
                "as_of_date": report.period_end,
                "observations": rows, "basis": "月末资产价值；月份范围含首尾"}
