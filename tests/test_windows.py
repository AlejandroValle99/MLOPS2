"""services.stream_worker.windows: window summaries, the alert rule, the per-hospital
monitor, and the whole drift path on a replayed stream of the real dataset (no broker)."""

import json
from itertools import count
from pathlib import Path

import pytest

from services.producer.main import events
from services.stream_worker.windows import AlertRule, Monitor, Window, summarize

BASELINES = json.loads(
    (Path(__file__).parent.parent / "data/drift/reference_stats.json").read_text()
)["hospitals"]

ROW = {
    "age": 61.0,
    "avg_glucose_level": 120.0,
    "bmi": 28.0,
    "work_type": "Private",
    "smoking_status": "never smoked",
}


def test_summary_of_a_window():
    w = Window(start=100.0)
    for i, latency in enumerate([10.0, 20.0, 30.0, 40.0]):
        w.add(ROW, latency, high_risk=i == 0, model_version="3")
    summary = summarize(w, now=102.0)
    assert summary["count"] == 4
    assert summary["throughput_eps"] == 2.0  # 4 events in 2 seconds
    assert summary["latency_p50_ms"] == 25.0
    assert summary["latency_p95_ms"] == pytest.approx(38.5)
    assert summary["positive_rate"] == 0.25


def test_a_window_keeps_only_the_monitored_columns():
    w = Window(start=0.0)
    w.add({**ROW, "gender": "Male", "patient_id": "p-1"}, 1.0, False, "1")
    assert set(w.rows[0]) == set(ROW)


def test_one_breach_is_not_an_alert_but_two_in_a_row_are():
    rule = AlertRule(alert=0.25, consecutive=2)
    assert rule.update({"bmi": 0.3}) == []
    assert rule.update({"bmi": 0.3}) == ["bmi"]


def test_the_monitors_rule_needs_two_windows_above_0_25():
    rule = Monitor(BASELINES, window_events=300).rules["rural-60plus"]
    assert rule.update({"bmi": 0.24}) == []
    assert rule.update({"bmi": 0.26}) == []
    assert rule.update({"bmi": 0.26}) == ["bmi"]


def test_a_persisting_shift_alerts_once_and_a_quiet_window_resets_it():
    rule = AlertRule(alert=0.25, consecutive=2)
    fired = [rule.update({"bmi": value}) for value in [0.3, 0.3, 0.3, 0.05, 0.3, 0.3]]
    assert fired == [[], ["bmi"], [], [], [], ["bmi"]]


def test_each_hospital_closes_its_own_window():
    clock = count()  # 0, 1, 2, ... one tick per event
    monitor = Monitor(BASELINES, window_events=3, clock=lambda: float(next(clock)))
    reports = [
        monitor.observe(hospital, ROW, 5.0, False, "1")
        for hospital in ["rural-60plus", "urban-60plus", "rural-60plus", "rural-60plus"]
    ]
    assert reports[:3] == [None, None, None]
    report = reports[3]
    assert report["hospital_id"] == "rural-60plus" and report["count"] == 3
    assert (report["window_start"], report["window_end"]) == (0.0, 3.0)
    assert set(report["psi"]) <= set(ROW)
    assert "urban-60plus" in monitor.windows  # still open, with its one event
    assert "rural-60plus" not in monitor.windows  # closed; its next event opens a new one


def replay(X, **kwargs) -> list[dict]:
    """Every closed window's report for a replay of X through a 300-event monitor."""
    monitor = Monitor(BASELINES, window_events=300)
    reports = []
    for event in events(X, passes=3, **kwargs):
        report = monitor.observe(event["hospital_id"], event["patient"], 5.0, False, "1")
        if report is not None:
            reports.append(report)
    return reports


def alerts(reports: list[dict]) -> set[tuple[str, str]]:
    return {(r["hospital_id"], feature) for r in reports for feature in r["alerts"]}


def test_a_stable_stream_raises_no_alert(dataset):
    X, _ = dataset
    reports = replay(X)
    assert len(reports) > 40  # every hospital closed several windows
    assert alerts(reports) == set()
    assert max(value for r in reports for value in r["psi"].values()) < 0.25


def test_a_glucose_shift_at_one_hospital_alerts_that_hospital_only(dataset):
    X, _ = dataset
    reports = replay(X, drift_hospital="rural-60plus", drift_from=0, factor=1.2)
    assert alerts(reports) == {("rural-60plus", "avg_glucose_level")}
