"""Per-hospital windows over the scored stream: latency, throughput, positive rate, drift,
and the rule that turns drift into an alert.

Windows are counted, not timed: a hospital's window closes after `window_events` events.
PSI needs a few hundred events to tell drift from sampling noise, and a small hospital
fills a timed window with too few events to judge.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stroke.drift import MONITORED, drift_report

PSI_WARN = 0.10  # above this: a moderate shift, logged
PSI_ALERT = 0.25  # above this: a major shift
ALERT_CONSECUTIVE = 2  # windows in a row above PSI_ALERT before an alert fires


@dataclass
class Window:
    """One hospital's events since its last window closed. The feature values live only
    here, in memory: they are never published or written anywhere."""

    start: float
    latencies_ms: list[float] = field(default_factory=list)
    positives: int = 0
    rows: list[dict] = field(default_factory=list)
    model_versions: set[str] = field(default_factory=set)

    def add(self, row: dict, latency_ms: float, high_risk: bool, model_version: str) -> None:
        self.rows.append({col: row[col] for col in MONITORED})
        self.latencies_ms.append(latency_ms)
        self.positives += int(high_risk)
        self.model_versions.add(model_version)


def summarize(w: Window, now: float) -> dict:
    count = len(w.rows)
    elapsed = now - w.start
    p50 = p95 = None
    if w.latencies_ms:
        p50, p95 = (round(float(v), 1) for v in np.percentile(w.latencies_ms, [50, 95]))
    return {
        "count": count,
        "throughput_eps": round(count / elapsed, 1) if elapsed > 0 else None,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "positive_rate": round(w.positives / count, 4) if count else None,
    }


class AlertRule:
    """An alert fires when a column's PSI stays above `alert` for `consecutive` windows in
    a row. One noisy window is not enough, and a shift that persists fires once, not on
    every window after it."""

    def __init__(self, alert: float = PSI_ALERT, consecutive: int = ALERT_CONSECUTIVE) -> None:
        self.alert, self.consecutive = alert, consecutive
        self.streak: dict[str, int] = {}

    def update(self, psi_by_column: dict[str, float]) -> list[str]:
        """Feed one window's PSI values; returns the columns that fire an alert now."""
        fired = []
        for col, value in psi_by_column.items():
            self.streak[col] = self.streak.get(col, 0) + 1 if value > self.alert else 0
            if self.streak[col] == self.consecutive:
                fired.append(col)
        return fired


class Monitor:
    """One window and one alert rule per hospital, each compared with its own baseline."""

    def __init__(
        self,
        baselines: dict[str, dict],
        window_events: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.baselines = baselines
        self.window_events = window_events
        self.clock = clock
        self.windows: dict[str, Window] = {}
        self.rules = {hospital: AlertRule() for hospital in baselines}

    def observe(
        self, hospital_id: str, row: dict, latency_ms: float, high_risk: bool, model_version: str
    ) -> dict | None:
        """Add one scored event. Returns the window's report if this event closed it."""
        now = self.clock()
        window = self.windows.setdefault(hospital_id, Window(start=now))
        window.add(row, latency_ms, high_risk, model_version)
        if len(window.rows) < self.window_events:
            return None
        del self.windows[hospital_id]  # the hospital's next event opens a fresh window
        # min_count is the window size itself, so a full window is never skipped
        psi = drift_report(
            self.baselines[hospital_id], pd.DataFrame(window.rows), min_count=self.window_events
        )
        return {
            "hospital_id": hospital_id,
            "window_start": window.start,
            "window_end": now,
            **summarize(window, now),
            "psi": {col: round(value, 4) for col, value in psi.items()},
            "warnings": sorted(col for col, value in psi.items() if value > PSI_WARN),
            "alerts": self.rules[hospital_id].update(psi),
            "model_versions": sorted(window.model_versions),
        }
