from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Optional, Union, cast

from pydantic import BaseModel, ConfigDict


class UsageInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ToolUseContent(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, list[dict[str, Any]]]
    is_error: Optional[bool] = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"]
    thinking: str


ContentItem = Union[TextContent, ToolUseContent, ToolResultContent, ThinkingContent]


class UserMessage(BaseModel):
    role: Literal["user"]
    content: Union[str, list[ContentItem]]


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[ContentItem]
    stop_reason: Optional[str] = None
    usage: Optional[UsageInfo] = None


class BaseTranscriptEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parentUuid: Optional[str] = None
    isSidechain: bool = False
    userType: str = "external"
    cwd: str = ""
    sessionId: str
    uuid: str
    timestamp: str
    isMeta: Optional[bool] = None


class UserTranscriptEntry(BaseTranscriptEntry):
    type: Literal["user"]
    message: UserMessage


class AssistantTranscriptEntry(BaseTranscriptEntry):
    type: Literal["assistant"]
    message: AssistantMessage
    requestId: Optional[str] = None


class SummaryTranscriptEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["summary"]
    summary: str
    leafUuid: str
    cwd: Optional[str] = None


class SystemTranscriptEntry(BaseTranscriptEntry):
    type: Literal["system"]
    content: str = ""
    level: Optional[str] = None


class FunctionTranscriptEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function"]
    role: Literal["function"]
    name: str
    content: Union[str, dict[str, Any]]
    parentUuid: Optional[str] = None
    isSidechain: bool = False
    userType: str = "unknown"
    cwd: Optional[str] = None
    sessionId: Optional[str] = None
    version: Optional[str] = None
    uuid: Optional[str] = None
    timestamp: Optional[str] = None
    isMeta: Optional[bool] = None


TranscriptEntry = Union[
    UserTranscriptEntry,
    AssistantTranscriptEntry,
    SummaryTranscriptEntry,
    SystemTranscriptEntry,
    FunctionTranscriptEntry,
]


class LLMCall(BaseModel):
    event_type: Literal["llm_call"] = "llm_call"
    timestamp: datetime
    session_id: str
    request_id: Optional[str] = None
    message_id: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    has_tool_use: bool = False
    num_tools_called: int = 0


class ToolCall(BaseModel):
    event_type: Literal["tool_call"] = "tool_call"
    timestamp: datetime
    session_id: str
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    associated_llm_call_id: str
    has_result: bool = False
    is_error: Optional[bool] = None
    execution_time_ms: Optional[float] = None
    result_timestamp: Optional[datetime] = None


Event = Union[LLMCall, ToolCall]


def parse_content_item(item_data: dict[str, Any]) -> ContentItem:
    content_type = item_data.get("type", "")
    if content_type == "tool_use":
        return ToolUseContent.model_validate(item_data)
    if content_type == "tool_result":
        return ToolResultContent.model_validate(item_data)
    if content_type == "thinking":
        return ThinkingContent.model_validate(item_data)
    return TextContent.model_validate(item_data)


def parse_message_content(content_data: Any) -> Union[str, list[ContentItem]]:
    if isinstance(content_data, str):
        return content_data
    if isinstance(content_data, list):
        content_list = cast(list[dict[str, Any]], content_data)
        return [parse_content_item(item) for item in content_list]
    return str(content_data)


def parse_transcript_entry(data: dict[str, Any]) -> TranscriptEntry:
    entry_type = data.get("type")

    if entry_type == "user":
        data_copy = data.copy()
        if "message" in data_copy and "content" in data_copy["message"]:
            data_copy["message"] = data_copy["message"].copy()
            data_copy["message"]["content"] = parse_message_content(
                data_copy["message"]["content"]
            )
        return UserTranscriptEntry.model_validate(data_copy)

    if entry_type == "assistant":
        data_copy = data.copy()
        if "message" in data_copy and "content" in data_copy["message"]:
            message_copy = data_copy["message"].copy()
            message_copy["content"] = parse_message_content(message_copy["content"])
            if "usage" in message_copy and message_copy["usage"] is not None:
                message_copy["usage"] = UsageInfo.model_validate(message_copy["usage"])
            data_copy["message"] = message_copy
        return AssistantTranscriptEntry.model_validate(data_copy)

    if entry_type == "summary":
        return SummaryTranscriptEntry.model_validate(data)

    if entry_type == "system":
        return SystemTranscriptEntry.model_validate(data)

    if entry_type == "function":
        data_copy = data.copy()
        if "content" in data_copy and isinstance(data_copy["content"], str):
            try:
                parsed_content = json.loads(data_copy["content"])
                data_copy["content"] = parsed_content
                if isinstance(parsed_content, dict) and "sessionId" in parsed_content:
                    if data_copy.get("sessionId") is None:
                        data_copy["sessionId"] = parsed_content["sessionId"]
            except (json.JSONDecodeError, ValueError):
                pass
        return FunctionTranscriptEntry.model_validate(data_copy)

    raise ValueError(f"Unknown transcript entry type: {entry_type}")


def _parse_timestamp(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def extract_events_from_transcript(transcript_entries: list[TranscriptEntry]) -> list[Event]:
    events: list[Event] = []
    tool_results: dict[str, tuple[bool, Optional[bool], Optional[datetime]]] = {}

    for entry in transcript_entries:
        if isinstance(entry, UserTranscriptEntry):
            result_timestamp = _parse_timestamp(entry.timestamp)
            message_content = entry.message.content
            if isinstance(message_content, list):
                for item in message_content:
                    if isinstance(item, ToolResultContent):
                        tool_results[item.tool_use_id] = (
                            True,
                            item.is_error,
                            result_timestamp,
                        )

    for entry_idx, entry in enumerate(transcript_entries):
        if isinstance(entry, AssistantTranscriptEntry):
            timestamp = _parse_timestamp(entry.timestamp)
            tool_calls: list[ToolCall] = []

            for item in entry.message.content:
                if isinstance(item, ToolUseContent):
                    has_result, is_error, result_timestamp = tool_results.get(
                        item.id, (False, None, None)
                    )
                    execution_time_ms = None
                    if result_timestamp:
                        execution_time_ms = (
                            result_timestamp - timestamp
                        ).total_seconds() * 1000

                    tool_calls.append(
                        ToolCall(
                            timestamp=timestamp,
                            session_id=entry.sessionId,
                            tool_use_id=item.id,
                            tool_name=item.name,
                            tool_input=item.input,
                            associated_llm_call_id=entry.message.id,
                            has_result=has_result,
                            is_error=is_error,
                            execution_time_ms=execution_time_ms,
                            result_timestamp=result_timestamp,
                        )
                    )

            usage = entry.message.usage
            events.append(
                LLMCall(
                    timestamp=timestamp,
                    session_id=entry.sessionId,
                    request_id=entry.requestId,
                    message_id=entry.message.id,
                    model=entry.message.model,
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    cache_creation_input_tokens=(
                        usage.cache_creation_input_tokens if usage else None
                    ),
                    cache_read_input_tokens=(
                        usage.cache_read_input_tokens if usage else None
                    ),
                    stop_reason=entry.message.stop_reason,
                    has_tool_use=len(tool_calls) > 0,
                    num_tools_called=len(tool_calls),
                )
            )
            events.extend(tool_calls)

        elif isinstance(entry, FunctionTranscriptEntry):
            if not entry.timestamp:
                continue
            timestamp = _parse_timestamp(entry.timestamp)

            tool_input: dict[str, Any]
            if isinstance(entry.content, dict):
                tool_input = entry.content
            elif isinstance(entry.content, str):
                try:
                    tool_input = json.loads(entry.content)
                except (json.JSONDecodeError, ValueError):
                    tool_input = {"content": entry.content}
            else:
                tool_input = {}

            tool_use_id = f"toucan_{entry.name}_{timestamp.isoformat()}"
            associated_llm_call_id = "unknown"
            for prev_entry in reversed(transcript_entries[:entry_idx]):
                if isinstance(prev_entry, AssistantTranscriptEntry):
                    associated_llm_call_id = prev_entry.message.id
                    break

            events.append(
                ToolCall(
                    timestamp=timestamp,
                    session_id=entry.sessionId or "unknown",
                    tool_use_id=tool_use_id,
                    tool_name=entry.name,
                    tool_input=tool_input,
                    associated_llm_call_id=associated_llm_call_id,
                    has_result=True,
                    is_error=None,
                    execution_time_ms=None,
                    result_timestamp=timestamp,
                )
            )

    events.sort(key=lambda e: e.timestamp)
    return events


def events_to_dict(events: list[Event]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in events:
        event_dict = event.model_dump()
        if event_dict.get("timestamp"):
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()
        if event_dict.get("result_timestamp"):
            event_dict["result_timestamp"] = event_dict["result_timestamp"].isoformat()
        result.append(event_dict)
    return result
