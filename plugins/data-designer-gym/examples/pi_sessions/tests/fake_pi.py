# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import time
from pathlib import Path


def emit(value):
    print(json.dumps(value), flush=True)


def option(name):
    return sys.argv[sys.argv.index(name) + 1]


def main():
    if "--version" in sys.argv:
        print("1.2.0")
        return
    session = (
        Path(option("--session"))
        if "--session" in sys.argv
        else Path(option("--session-dir")) / (option("--session-id") + ".jsonl")
    )
    messages = json.loads(session.read_text()) if session.exists() else []
    for line in sys.stdin:
        command = json.loads(line)
        kind = command["type"]
        data = {}
        if kind == "get_messages":
            data = {"messages": messages}
        elif kind == "get_state":
            data = {"sessionFile": str(session), "sessionId": session.stem}
        emit({"type": "response", "id": command["id"], "success": True, "data": data})
        if kind != "prompt":
            continue
        text = command["message"]
        if text == "hang":
            time.sleep(30)
        if text == "crash":
            os._exit(7)
        user = {"role": "user", "content": [{"type": "text", "text": text}]}
        call_id = f"call-{len(messages)}"
        call = {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": call_id, "name": "write", "arguments": {"path": "solution.py"}}],
            "stopReason": "toolUse",
        }
        Path("solution.py").write_text(text)
        tool = {
            "role": "toolResult",
            "toolCallId": call_id,
            "content": [{"type": "text", "text": "Saved"}],
            "isError": False,
        }
        assistant = {
            "role": "assistant",
            "content": [{"type": "text", "text": "" if text == "empty" else "Saved."}],
            "stopReason": "length" if text == "incomplete" else "stop",
            "usage": {"input": 10, "output": 5},
        }
        messages += [user, call, tool, assistant]
        session.write_text(json.dumps(messages))
        emit({"type": "tool_execution_start", "toolCallId": call_id, "toolName": "write"})
        for message in (call, tool, assistant):
            emit({"type": "message_end", "message": message})
        emit({"type": "tool_execution_end", "toolCallId": call_id, "toolName": "write", "isError": False})
        emit({"type": "agent_end", "messages": [assistant]})
        emit({"type": "agent_settled", "aborted": False})


if __name__ == "__main__":
    main()
