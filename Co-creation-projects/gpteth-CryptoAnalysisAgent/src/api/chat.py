"""Chat API with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .deps import agent_init_error, get_agent

router = APIRouter(prefix="/chat", tags=["chat"])
log = logging.getLogger(__name__)

# Comment lines keep proxies and the browser from treating a long tool call as a dead connection.
HEARTBEAT_SECONDS = 10
_STREAM_EVENTS = {
    "session": ("session_id",),
    "step_start": ("step", "max_steps"),
    "thought": ("content",),
    "chunk": ("content",),
    "tool_start": ("tool", "args"),
    "tool_finish": ("tool", "result"),
    "step_finish": ("step",),
    "memory": ("saved",),
    "done": ("content", "session_id"),
    "error": ("error",),
}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    content: str
    session_id: Optional[str] = None
    error: Optional[str] = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _load_agent():
    try:
        return get_agent()
    except Exception as exc:
        log.exception("chat agent init failed")
        raise HTTPException(status_code=503, detail=agent_init_error(exc)) from exc


def _validate(request: ChatRequest) -> None:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")


@router.post("/send/sync", response_model=ChatResponse)
def send_sync(request: ChatRequest):
    _validate(request)
    agent = _load_agent()
    session_id = request.session_id
    content = ""
    error = None
    try:
        for event in agent.iter_events(request.message, session_id):
            if event["type"] == "session":
                session_id = event.get("session_id") or session_id
            elif event["type"] == "chunk":
                content += event.get("content") or ""
            elif event["type"] == "done":
                content = event.get("content") or content
                session_id = event.get("session_id") or session_id
            elif event["type"] == "error":
                error = event.get("error") or "未知错误"
    except Exception as exc:
        log.exception("chat turn failed")
        error = str(exc) or type(exc).__name__
    return ChatResponse(content=content, session_id=session_id, error=error)


@router.post("/send/stream")
async def send_stream(request: ChatRequest):
    _validate(request)
    agent = _load_agent()

    async def generate():
        yield ": connected\n\n"
        iterator = agent.iter_events(request.message, request.session_id)
        try:
            while True:
                pending = asyncio.ensure_future(asyncio.to_thread(next, iterator, None))
                while True:
                    done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_SECONDS)
                    if done:
                        break
                    yield ": ping\n\n"
                event = pending.result()
                if event is None:
                    break
                kind = event.get("type")
                fields = _STREAM_EVENTS.get(kind)
                if fields is None:
                    continue
                yield _sse(kind, {name: event.get(name) for name in fields})
        except Exception as exc:
            log.exception("chat stream failed")
            yield _sse("error", {"error": str(exc) or type(exc).__name__})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/send")
def send_compat(request: ChatRequest):
    return send_sync(request)
