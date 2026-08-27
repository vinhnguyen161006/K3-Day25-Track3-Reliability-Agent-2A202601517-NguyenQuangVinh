from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a circuit is open and calls should fail fast."""


@dataclass(slots=True)
class CircuitBreaker:
    """Per-provider circuit breaker with a CLOSED / OPEN / HALF_OPEN state machine.

    CLOSED: calls pass through; failures are counted.
    OPEN: calls fail fast until reset_timeout_seconds elapses.
    HALF_OPEN: a single probe is allowed; success closes the circuit,
    failure re-opens it immediately.
    """

    name: str
    failure_threshold: int
    reset_timeout_seconds: float
    success_threshold: int = 1
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    opened_at: float | None = None
    transition_log: list[dict[str, str | float]] = field(default_factory=list)

    def allow_request(self) -> bool:
        """Return whether a request should be attempted in the current state."""
        if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            return True
        # OPEN: only allow once the reset timeout has elapsed, and treat that
        # first allowed request as a probe by moving to HALF_OPEN.
        timeout_elapsed = (
            self.opened_at is not None
            and time.monotonic() - self.opened_at >= self.reset_timeout_seconds
        )
        if timeout_elapsed:
            self._transition(CircuitState.HALF_OPEN, "reset_timeout_elapsed")
            return True
        return False

    def call(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Invoke ``fn`` through the breaker, recording the outcome."""
        if not self.allow_request():
            raise CircuitOpenError(f"Circuit {self.name} is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def record_success(self) -> None:
        """Reset the failure streak and close the circuit after a successful probe."""
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN and self.success_count >= self.success_threshold:
            self._transition(CircuitState.CLOSED, "probe_success")
            self.success_count = 0

    def record_failure(self) -> None:
        """Track a failure and open the circuit if warranted.

        HALF_OPEN and threshold breaches are kept as separate branches
        (not combined with ``or``) because they report different reasons
        in the transition log: a failed probe vs. a run of failures.
        """
        self.failure_count += 1
        self.success_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.opened_at = time.monotonic()
            self._transition(CircuitState.OPEN, "probe_failure")
        elif self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()
            self._transition(CircuitState.OPEN, "failure_threshold_reached")

    def _transition(self, new_state: CircuitState, reason: str) -> None:
        if self.state == new_state:
            return
        self.transition_log.append(
            {"from": self.state.value, "to": new_state.value, "reason": reason, "ts": time.time()}
        )
        self.state = new_state
