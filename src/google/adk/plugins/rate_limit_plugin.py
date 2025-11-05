# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import time
from typing import Optional

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from .base_plugin import BasePlugin


class RateLimitPlugin(BasePlugin):
  """Plugin that enforces global rate limiting on LLM requests.

  This plugin implements a sliding window rate limiter that restricts the
  total number of LLM requests across all models within a one-minute window.
  When the rate limit is exceeded, the plugin blocks (waits) until a slot
  becomes available.

  Example:
    ```python
    from google.adk import Agent, Runner
    from google.adk.plugins.rate_limit_plugin import RateLimitPlugin

    agent = Agent(name="assistant", model="gemini-2.5-flash", ...)

    runner = Runner(
        agents=[agent],
        plugins=[
            RateLimitPlugin(max_requests_per_minute=15)
        ]
    )
    ```

  Attributes:
    max_requests_per_minute: Maximum number of requests allowed per minute
        globally across all models.
  """

  def __init__(
      self,
      max_requests_per_minute: int = 15,
      name: str = 'rate_limit_plugin',
  ):
    """Initialize the rate limit plugin.

    Args:
      max_requests_per_minute: Maximum requests allowed per minute globally.
      name: Name of the plugin instance.
    """
    super().__init__(name)
    self.max_requests = max_requests_per_minute

    # Track request timestamps globally (all models)
    # List of timestamps (in seconds since epoch)
    self._request_timestamps: list[float] = []

    # Lock for thread-safe access to timestamps
    self._lock = asyncio.Lock()

  def _clean_old_timestamps(
      self, timestamps: list[float], current_time: float
  ) -> list[float]:
    """Remove timestamps older than 1 minute from the tracking list.

    Args:
      timestamps: List of request timestamps.
      current_time: Current time in seconds since epoch.

    Returns:
      Filtered list containing only timestamps from the last minute.
    """
    # Keep only timestamps within the last 60 seconds
    cutoff_time = current_time - 60.0
    return [ts for ts in timestamps if ts > cutoff_time]

  async def _wait_for_rate_limit(self, current_time: float) -> None:
    """Wait until a request slot becomes available.

    Args:
      current_time: Current time in seconds since epoch.
    """
    while True:
      async with self._lock:
        timestamps = self._clean_old_timestamps(
            self._request_timestamps, time.time()
        )
        self._request_timestamps = timestamps

        if len(timestamps) < self.max_requests:
          # Slot available, exit loop
          return

        # Calculate wait time until the oldest request falls outside the window
        oldest_timestamp = timestamps[0]
        wait_seconds = 60.0 - (time.time() - oldest_timestamp) + 0.1

      # Wait outside the lock to allow other operations
      if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
      else:
        # Re-check immediately
        await asyncio.sleep(0.01)

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Check and enforce rate limits before each LLM request.

    This callback is invoked before every LLM request. It checks whether
    the request would exceed the configured global rate limit across all models.
    If so, it blocks (waits) until the rate limit allows the request.

    Args:
      callback_context: Context containing agent, user, and session information.
      llm_request: The LLM request that is about to be sent.

    Returns:
      None to allow the request to proceed (after waiting if necessary).
    """
    current_time = time.time()

    async with self._lock:
      # Clean old timestamps
      timestamps = self._clean_old_timestamps(
          self._request_timestamps, current_time
      )
      self._request_timestamps = timestamps

      # Check if rate limit would be exceeded
      if len(timestamps) >= self.max_requests:
        # Need to wait
        pass
      else:
        # Slot available, record and proceed
        self._request_timestamps.append(current_time)
        return None

    # Wait for availability if limit exceeded
    await self._wait_for_rate_limit(current_time)

    # Record this request after waiting
    async with self._lock:
      current_time = time.time()
      self._request_timestamps.append(current_time)

    # Allow request to proceed
    return None
