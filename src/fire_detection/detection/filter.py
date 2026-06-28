"""
Temporal filter and alert state machine for fire/smoke detection.

Converts noisy, frame-level YOLO detections into a reliable alert signal by
requiring a minimum number of detections within a sliding time window before
confirming a fire event. This eliminates single-frame false positives (glare,
compression artifacts, brief misclassifications) while remaining responsive
to genuine, persistent fire.

Tuning guide:
    conf_threshold  : lower → more sensitive, more false positives
    window_seconds  : longer → more evidence required, slower confirmation
    min_detections  : higher → more evidence required, fewer false alerts
    cooldown_seconds: longer → fewer repeated alerts for ongoing fires
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from fire_detection.inference.schema import Detection


class AlertState(Enum):
    IDLE      = "idle"       # no evidence of fire in the current window
    SUSPECTED = "suspected"  # detections building; below confirmation threshold
    CONFIRMED = "confirmed"  # threshold crossed; alert active; fire detected
    COOLDOWN  = "cooldown"   # fire dropped below threshold post-confirmation; timer running


@dataclass(frozen=True)
class FilterConfig:
    conf_threshold: float = 0.45    # minimum per-frame confidence to count as a detection
    window_seconds: float = 1.0     # sliding look-back window for detection counting
    min_detections: int   = 3       # detections needed within window to confirm
    cooldown_seconds: float = 30.0  # minimum seconds between successive alert emissions


@dataclass(frozen=True)
class FilterResult:
    state: AlertState
    alert_emitted: bool        # True exactly when an alert fires this update cycle
    recent_detections: int     # detections counted in the current window
    smoothed_confidence: float # mean confidence of detections in window (0.0 if none)
    cooldown_remaining: float  # seconds until next alert is allowed; 0.0 when unrestricted


class TemporalFilter:
    """
    Sliding-window temporal filter with a four-state alert machine.

    Call update() once per inference frame. The returned FilterResult carries
    the current state and a flag indicating whether an alert fired this cycle.

    Thread safety: not thread-safe. Call exclusively from the inference thread.
    """

    def __init__(self, config: FilterConfig | None = None):
        self._cfg = config or FilterConfig()
        self._state = AlertState.IDLE

        # (timestamp, confidence) for each frame with a detection above threshold.
        # Pruned to window_seconds on every update().
        self._detections: deque[tuple[float, float]] = deque()

        self._alert_time: float = 0.0      # monotonic time of the most recent alert emission
        self._cooldown_start: float = 0.0  # monotonic time when COOLDOWN state began

    @property
    def state(self) -> AlertState:
        return self._state

    def update(
        self,
        detections: Sequence[Detection],
        now: float | None = None,
    ) -> FilterResult:
        """
        Advance the filter by one frame.

        Args:
            detections: raw detections from this inference frame (may be empty).
            now:        current monotonic timestamp; uses time.monotonic() if None.
                        Pass explicitly in tests to control time without sleeping.
        """
        if now is None:
            now = time.monotonic()

        # Record this frame if any detection exceeds the confidence threshold.
        best_conf = max((d.confidence for d in detections), default=0.0)
        if best_conf >= self._cfg.conf_threshold:
            self._detections.append((now, best_conf))

        # Prune detections that have aged out of the look-back window.
        cutoff = now - self._cfg.window_seconds
        while self._detections and self._detections[0][0] < cutoff:
            self._detections.popleft()

        recent_count = len(self._detections)
        smoothed_conf = (
            sum(c for _, c in self._detections) / recent_count
            if recent_count > 0 else 0.0
        )

        alert_emitted = self._step(now, recent_count)

        cooldown_remaining = max(
            0.0, self._cfg.cooldown_seconds - (now - self._alert_time)
        )

        return FilterResult(
            state=self._state,
            alert_emitted=alert_emitted,
            recent_detections=recent_count,
            smoothed_confidence=smoothed_conf,
            cooldown_remaining=cooldown_remaining,
        )

    def _step(self, now: float, recent_count: int) -> bool:
        """Advance the state machine one step. Returns True if an alert fires."""
        confirmed       = recent_count >= self._cfg.min_detections
        cooldown_elapsed = (now - self._alert_time) >= self._cfg.cooldown_seconds

        alert = False

        if self._state == AlertState.IDLE:
            if confirmed:
                self._state = AlertState.CONFIRMED
                self._alert_time = now
                alert = True
            elif recent_count > 0:
                self._state = AlertState.SUSPECTED

        elif self._state == AlertState.SUSPECTED:
            if confirmed:
                self._state = AlertState.CONFIRMED
                self._alert_time = now
                alert = True
            elif recent_count == 0:
                self._state = AlertState.IDLE

        elif self._state == AlertState.CONFIRMED:
            if not confirmed:
                # Detections dropped — fire may have stopped or been briefly occluded.
                self._state = AlertState.COOLDOWN
                self._cooldown_start = now
            elif cooldown_elapsed:
                # Fire still burning; re-alert at the configured interval.
                self._alert_time = now
                alert = True

        elif self._state == AlertState.COOLDOWN:
            if confirmed and cooldown_elapsed:
                # Enough evidence has re-accumulated and the re-alert timer has cleared.
                self._state = AlertState.CONFIRMED
                self._alert_time = now
                alert = True
            elif not confirmed:
                # No active detections; decide whether to exit COOLDOWN.
                # Exit is gated on cooldown_seconds from COOLDOWN entry, not from last
                # alert, so a brief post-fire window gives time for temporary occlusion.
                post_quiet = (now - self._cooldown_start) >= self._cfg.cooldown_seconds
                if post_quiet:
                    self._state = AlertState.SUSPECTED if recent_count > 0 else AlertState.IDLE

        return alert
