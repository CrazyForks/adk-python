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

import asyncio
import time
from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.rate_limit_plugin import RateLimitPlugin
from google.genai import types
import pytest


@pytest.fixture
def callback_context():
  """Create a callback context for tests."""
  return Mock(spec=CallbackContext)


@pytest.fixture
def llm_request():
  """Create a basic LLM request for tests."""
  return LlmRequest(
      model='gemini-2.5-flash',
      contents=[
          types.Content(role='user', parts=[types.Part(text='Hello, world!')])
      ],
  )


class TestRateLimitPlugin:
  """Tests for the RateLimitPlugin."""

  @pytest.mark.asyncio
  async def test_plugin_allows_requests_within_limit(
      self, callback_context, llm_request
  ):
    """Test that requests within the rate limit are allowed."""
    plugin = RateLimitPlugin(max_requests_per_minute=5)

    # Send 5 requests (within limit)
    for _ in range(5):
      result = await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      assert result is None  # Request allowed

  @pytest.mark.asyncio
  async def test_plugin_tracks_rate_limit_globally(self, callback_context):
    """Test that rate limits are tracked globally across all models."""
    plugin = RateLimitPlugin(max_requests_per_minute=5)

    # Create requests for two different models
    request_model1 = LlmRequest(
        model='gemini-2.5-flash',
        contents=[types.Content(role='user', parts=[types.Part(text='Hello')])],
    )
    request_model2 = LlmRequest(
        model='gemini-2.0-flash',
        contents=[types.Content(role='user', parts=[types.Part(text='Hello')])],
    )

    # Send 3 requests to model1
    for _ in range(3):
      result = await plugin.before_model_callback(
          callback_context=callback_context, llm_request=request_model1
      )
      assert result is None

    # Send 2 requests to model2 (fills global quota of 5)
    for _ in range(2):
      result = await plugin.before_model_callback(
          callback_context=callback_context, llm_request=request_model2
      )
      assert result is None

    # Verify all 5 requests were tracked globally
    async with plugin._lock:
      assert len(plugin._request_timestamps) == 5

  @pytest.mark.asyncio
  async def test_plugin_sliding_window_allows_requests_after_time(
      self, callback_context, llm_request
  ):
    """Test that requests are allowed after sliding window expires."""
    plugin = RateLimitPlugin(max_requests_per_minute=2)

    # Send 2 requests (fill quota)
    for _ in range(2):
      result = await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      assert result is None

    # Simulate that requests happened 61 seconds ago (past the 60-second window)
    async with plugin._lock:
      plugin._request_timestamps = [
          time.time() - 61.0,
          time.time() - 61.0,
      ]

    # New request should be allowed immediately (old timestamps expired)
    result = await plugin.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_plugin_waits_for_availability(
      self, callback_context, llm_request
  ):
    """Test that plugin waits when rate limit is exceeded."""
    plugin = RateLimitPlugin(max_requests_per_minute=2)

    # Send 2 requests (fill quota)
    for _ in range(2):
      result = await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      assert result is None

    # Simulate old timestamps to avoid long wait
    async with plugin._lock:
      plugin._request_timestamps = [
          time.time() - 59.5,  # Will expire in ~0.5 seconds
          time.time() - 59.5,
      ]

    # 3rd request should block briefly and then succeed
    start_time = time.time()
    result = await plugin.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    elapsed_time = time.time() - start_time

    assert result is None  # Request eventually allowed
    assert elapsed_time >= 0.5  # Should have waited at least 0.5 seconds
    assert elapsed_time < 2.0  # But not too long

  @pytest.mark.asyncio
  async def test_plugin_cleans_old_timestamps(
      self, callback_context, llm_request
  ):
    """Test that old timestamps are properly cleaned up."""
    plugin = RateLimitPlugin(max_requests_per_minute=5)

    # Manually add old timestamps
    async with plugin._lock:
      plugin._request_timestamps = [
          time.time() - 120.0,  # 2 minutes ago
          time.time() - 90.0,  # 1.5 minutes ago
          time.time() - 70.0,  # 70 seconds ago
      ]

    # New request should trigger cleanup and be allowed
    result = await plugin.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    assert result is None

    # Verify old timestamps were cleaned
    async with plugin._lock:
      timestamps = plugin._request_timestamps
      # Should only have the new timestamp
      assert len(timestamps) == 1

  @pytest.mark.asyncio
  async def test_plugin_concurrent_requests(self, callback_context):
    """Test that plugin handles concurrent requests safely."""
    plugin = RateLimitPlugin(max_requests_per_minute=5)

    # Create multiple requests (more than the limit)
    requests = [
        LlmRequest(
            model='gemini-2.5-flash',
            contents=[
                types.Content(
                    role='user', parts=[types.Part(text=f'Request {i}')]
                )
            ],
        )
        for i in range(7)
    ]

    # Simulate old timestamps for the first 5 requests to fill the quota
    async with plugin._lock:
      plugin._request_timestamps = [time.time() - 59.5 for _ in range(5)]

    # Send 2 more requests concurrently (should both wait)
    tasks = [
        plugin.before_model_callback(
            callback_context=callback_context, llm_request=requests[i]
        )
        for i in range(2)
    ]

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time

    # All requests should eventually succeed (after waiting)
    assert all(r is None for r in results)
    # Should have waited for at least one slot to become available
    assert elapsed_time >= 0.5

  @pytest.mark.asyncio
  async def test_plugin_default_parameters(self, callback_context, llm_request):
    """Test plugin works with default parameters."""
    plugin = RateLimitPlugin()  # All defaults

    # Should allow up to 15 requests (default max_requests_per_minute)
    for _ in range(15):
      result = await plugin.before_model_callback(
          callback_context=callback_context, llm_request=llm_request
      )
      assert result is None

    # 16th request should block and wait
    # Simulate old timestamps to avoid long wait
    async with plugin._lock:
      plugin._request_timestamps = [time.time() - 59.8 for _ in range(15)]

    # This should block briefly and succeed
    result = await plugin.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    assert result is None
