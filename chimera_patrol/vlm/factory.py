"""build_adapter：按 ``vlm.provider`` 选择 VLMAdapter 实现（需求 5.3）。

密钥由调用方读取后传入。本模块不读取环境变量。
"""

from __future__ import annotations

import asyncio

from chimera_patrol.config import VlmSection
from chimera_patrol.vlm.mock import MockAdapter
from chimera_patrol.vlm.openai_compat import OpenAICompatibleAdapter
from chimera_patrol.vlm.protocol import VLMAdapter


def build_adapter(vlm_section: VlmSection, api_key: str | None) -> VLMAdapter:
    """Create the adapter named by ``vlm_section.provider``.

    Args:
        vlm_section: The ``vlm`` configuration section.
        api_key: API key resolved by the caller from ``vlm.openai_compatible.api_key_env``.
            Required for ``openai_compatible``, ignored for ``mock``.

    Raises:
        ValueError: If the provider needs a key and ``api_key`` is missing or empty.
    """
    if vlm_section.provider == "mock":
        return MockAdapter(
            default_verdict=vlm_section.mock.default_verdict,
            by_camera=vlm_section.mock.by_camera,
        )
    if not api_key:
        raise ValueError(
            f"API key is required for provider openai_compatible "
            f"(set environment variable {vlm_section.openai_compatible.api_key_env})"
        )
    return OpenAICompatibleAdapter(
        vlm_section.openai_compatible,
        api_key,
        semaphore=asyncio.Semaphore(vlm_section.max_concurrency),
    )
