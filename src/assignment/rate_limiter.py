"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import threading
import time

from google.adk.plugins import base_plugin
from google.genai import types

# Prefix of the block message; the pipeline uses it to attribute concurrent requests.
RATE_LIMIT_MESSAGE = "Rate limit exceeded."


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        # The check-and-append below must be atomic when requests arrive from
        # several threads at once (a flood client does not wait for answers).
        self._lock = threading.Lock()

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        with self._lock:
            self.total_count += 1
            # Monotonic clock: immune to wall-clock jumps (NTP, DST, manual changes)
            now = time.monotonic()
            window = self.user_windows[user_id]

            # Drop timestamps that fell out of the sliding window
            while window and window[0] <= now - self.window_seconds:
                window.popleft()

            if len(window) >= self.max_requests:
                wait = max(0.0, self.window_seconds - (now - window[0]))
                self.blocked_count += 1
                return self._block_response(
                    f"{RATE_LIMIT_MESSAGE} Try again in {wait:.0f}s."
                )

            window.append(now)
            return None
