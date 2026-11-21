"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def _raise_alert(self, metric: str, value: float, threshold: float, message: str):
        """Append an alert, or refresh the existing one for the same metric."""
        for existing in self.alerts:
            if existing.metric == metric:
                existing.value = value  # refresh instead of duplicating
                existing.message = message
                return None
        alert = Alert(metric=metric, value=value, threshold=threshold, message=message)
        self.alerts.append(alert)
        return alert

    def check_metrics(self) -> list[Alert]:
        """Compute rates and raise an Alert per metric that exceeds its threshold.

        Idempotent: calling it repeatedly never duplicates an alert for a metric.
        Returns the full list of active alerts.
        """
        snap = self.snapshot()
        if snap["block_rate"] > self.block_rate_threshold:
            self._raise_alert(
                "block_rate", snap["block_rate"], self.block_rate_threshold,
                f"Block rate {snap['block_rate']:.0%} exceeds "
                f"{self.block_rate_threshold:.0%} "
                f"({self.blocked_requests}/{self.total_requests} requests blocked)",
            )
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            self._raise_alert(
                "rate_limit_hits", float(self.rate_limit_hits),
                float(self.rate_limit_hit_threshold),
                f"{self.rate_limit_hits} rate-limit hits "
                f"(threshold {self.rate_limit_hit_threshold})",
            )
        if snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            self._raise_alert(
                "judge_fail_rate", snap["judge_fail_rate"],
                self.judge_fail_rate_threshold,
                f"Judge fail rate {snap['judge_fail_rate']:.0%} exceeds "
                f"{self.judge_fail_rate_threshold:.0%}",
            )
        return self.alerts

    def export_json(self, filepath: str | None = None) -> str:
        """Write metrics + alerts to ``outputs/metrics.json`` (repo root) by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
