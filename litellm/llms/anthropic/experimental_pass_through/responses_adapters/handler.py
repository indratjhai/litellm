"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI or Azure model.
"""

import json
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union

import litellm
from litellm.exceptions import ContextWindowExceededError
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse

from .streaming_iterator import AnthropicResponsesStreamWrapper
from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()
_CHATGPT_COMPACTION_INPUT_BUDGET = 100_000
_CHATGPT_COMPACTION_RECENT_BUDGET = 35_000
_CHATGPT_COMPACTION_CHUNK_BUDGET = 45_000


def _count_compaction_tokens(value: Any) -> int:
    serialized = json.dumps(value, ensure_ascii=False)
    try:
        return litellm.token_counter(model="gpt-4o", text=serialized)
    except Exception:
        return len(serialized) // 4


def _text_from_response(response: ResponsesAPIResponse) -> str:
    translated = _ADAPTER.translate_response(response)
    content = (
        translated.get("content", [])
        if isinstance(translated, dict)
        else translated.content
    )
    return "\n".join(
        block.get("text", "")
        for block in content
        if block.get("type") == "text" and block.get("text")
    ).strip()


async def _reduce_oversized_compaction(
    messages: List[Dict],
    responses_kwargs: Dict[str, Any],
    model: str,
    provider_kwargs: Dict[str, Any],
) -> None:
    """Hierarchically summarize old compact input without dropping it."""
    if not _ADAPTER._is_claude_code_compaction_request(messages):
        return

    input_items = responses_kwargs.get("input")
    if not isinstance(input_items, list) or _count_compaction_tokens(
        {"input": input_items, "instructions": responses_kwargs.get("instructions", "")}
    ) <= _CHATGPT_COMPACTION_INPUT_BUDGET:
        return

    recent_items: List[Dict] = []
    recent_tokens = 0
    for item in reversed(input_items):
        item_tokens = _count_compaction_tokens(item)
        if recent_items and recent_tokens + item_tokens > _CHATGPT_COMPACTION_RECENT_BUDGET:
            break
        recent_items.append(item)
        recent_tokens += item_tokens
    recent_items.reverse()
    old_items = input_items[: len(input_items) - len(recent_items)]

    chunks: List[List[Any]] = []
    current_chunk: List[Any] = []
    current_tokens = 0
    for item in old_items:
        item_parts: List[Any] = [item]
        if _count_compaction_tokens(item) > _CHATGPT_COMPACTION_CHUNK_BUDGET:
            serialized_item = json.dumps(item, ensure_ascii=False)
            item_parts = [
                {
                    "fragment": fragment_index,
                    "content": serialized_item[offset : offset + 120_000],
                }
                for fragment_index, offset in enumerate(
                    range(0, len(serialized_item), 120_000), start=1
                )
            ]
        for item_part in item_parts:
            item_tokens = _count_compaction_tokens(item_part)
            if (
                current_chunk
                and current_tokens + item_tokens > _CHATGPT_COMPACTION_CHUNK_BUDGET
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(item_part)
            current_tokens += item_tokens
    if current_chunk:
        chunks.append(current_chunk)

    summaries: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        summary_prompt = (
            "Create a precise continuity summary of this earlier conversation "
            "section for a later summarization pass. Preserve decisions, user "
            "requests, completed work, open work, constraints, and identifiers. "
            "Do not call tools.\n\n"
            f"Section {index} of {len(chunks)}:\n{json.dumps(chunk, ensure_ascii=False)}"
        )
        summary_response = await litellm.aresponses(
            model=responses_kwargs["model"],
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": summary_prompt}],
                }
            ],
            max_output_tokens=2_000,
            **{
                key: provider_kwargs[key]
                for key in ("custom_llm_provider", "api_key", "api_base", "api_version")
                if provider_kwargs.get(key) is not None
            },
        )
        if not isinstance(summary_response, ResponsesAPIResponse):
            raise ValueError(
                f"Expected ResponsesAPIResponse, got {type(summary_response)}"
            )
        summary_text = _text_from_response(summary_response)
        if not summary_text:
            raise ContextWindowExceededError(
                message="Unable to summarize an earlier compaction section.",
                model=model,
                llm_provider="chatgpt",
            )
        summaries.append(summary_text)

    responses_kwargs["input"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Earlier conversation sections:\n\n"
                    + "\n\n".join(summaries),
                }
            ],
        },
        *recent_items,
    ]


def _build_responses_kwargs(
    *,
    max_tokens: int,
    messages: List[Dict],
    model: str,
    context_management: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    output_config: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    output_format: Optional[Dict] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the kwargs dict to pass directly to litellm.responses() / litellm.aresponses().
    """
    # Build a typed AnthropicMessagesRequest for the adapter
    request_data: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if context_management:
        request_data["context_management"] = context_management
    if output_config:
        request_data["output_config"] = output_config
    if metadata:
        request_data["metadata"] = metadata
    if system:
        request_data["system"] = system
    if temperature is not None:
        request_data["temperature"] = temperature
    if thinking:
        request_data["thinking"] = thinking
    if tool_choice:
        request_data["tool_choice"] = tool_choice
    if tools:
        request_data["tools"] = tools
    if top_p is not None:
        request_data["top_p"] = top_p
    if output_format:
        request_data["output_format"] = output_format

    anthropic_request = AnthropicMessagesRequest(**request_data)  # type: ignore[typeddict-item]
    custom_llm_provider = (extra_kwargs or {}).get("custom_llm_provider")
    if custom_llm_provider == "chatgpt" and tools:
        # ChatGPT subscription Responses rejects LiteLLM's web_search_preview
        # translation. Claude Code can continue with its local tools.
        tools = [
            tool
            for tool in tools
            if not (
                str(tool.get("type", "")).startswith("web_search")
                or tool.get("name") == "web_search"
            )
        ]
        if not tools:
            request_data.pop("tools", None)
        else:
            request_data["tools"] = tools
        anthropic_request = AnthropicMessagesRequest(**request_data)  # type: ignore[typeddict-item]
    responses_kwargs = _ADAPTER.translate_request(
        anthropic_request,
        use_developer_role_for_system=custom_llm_provider == "chatgpt",
    )

    # Normalize reasoning effort based on model capabilities
    # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
    reasoning = responses_kwargs.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        from litellm.llms.anthropic.experimental_pass_through.utils import (
            normalize_reasoning_effort_value,
        )

        effort = reasoning["effort"]
        normalized = normalize_reasoning_effort_value(
            effort,
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
        if normalized != effort:
            responses_kwargs["reasoning"] = {**reasoning, "effort": normalized}

    if stream:
        responses_kwargs["stream"] = True

    # Forward litellm-specific kwargs (api_key, api_base, logging obj, etc.)
    excluded = {"anthropic_messages"}
    for key, value in (extra_kwargs or {}).items():
        if key == "litellm_logging_obj" and value is not None:
            from litellm.litellm_core_utils.litellm_logging import (
                Logging as LiteLLMLoggingObject,
            )
            from litellm.types.utils import CallTypes

            if isinstance(value, LiteLLMLoggingObject):
                # The success handler receives the raw Responses object, not the
                # translated Anthropic response, so log it as a completion.
                setattr(value, "call_type", CallTypes.acompletion.value)
            responses_kwargs[key] = value
        elif key not in excluded and key not in responses_kwargs and value is not None:
            responses_kwargs[key] = value

    return responses_kwargs


class LiteLLMMessagesToResponsesAPIHandler:
    """
    Handles Anthropic /v1/messages requests for OpenAI / Azure models by
    calling litellm.responses() / litellm.aresponses() directly and translating
    the response back to Anthropic format.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        context_management: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        output_config: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        responses_kwargs = _build_responses_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            context_management=context_management,
            metadata=metadata,
            output_config=output_config,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )
        await _reduce_oversized_compaction(messages, responses_kwargs, model, kwargs)

        result = await litellm.aresponses(**responses_kwargs)

        if stream:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=model
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        context_management: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        output_config: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        AnthropicMessagesResponse,
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
    ]:
        if _is_async:
            return (
                LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    context_management=context_management,
                    metadata=metadata,
                    output_config=output_config,
                    stop_sequences=stop_sequences,
                    stream=stream,
                    system=system,
                    temperature=temperature,
                    thinking=thinking,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_k=top_k,
                    top_p=top_p,
                    output_format=output_format,
                    **kwargs,
                )
            )

        # Sync path
        responses_kwargs = _build_responses_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            context_management=context_management,
            metadata=metadata,
            output_config=output_config,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )
        result = litellm.responses(**responses_kwargs)

        if stream:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=model
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)
