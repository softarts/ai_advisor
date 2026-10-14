"""统一的结果信封：数字 + 口径 + 警告。dashboard 与 AI advisor 共用。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Result:
    value: Any
    basis: str = ""                 # 口径说明，如 "NAV+TWR(月度)"
    period: tuple = (None, None)
    currency: str = "SGD"
    status: str = "ok"              # ok | insufficient
    warnings: list = field(default_factory=list)
    sources: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["period"] = list(self.period)
        return d
