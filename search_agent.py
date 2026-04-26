"""
LangGraph Search Engine Agent
- Supports Claude, OpenAI, Gemini (set LLM_PROVIDER in .env)
- Uses DuckDuckGo for web search (no API key needed)
- Whisper: transcribe audio file or record from mic (set WHISPER_MODE in .env)
- ReAct loop: Think → Search → Evaluate → Answer

Usage:
    python3 search_agent.py "text query"          # text
    python3 search_agent.py query.wav             # audio file
    python3 search_agent.py --listen              # mic (press Enter to stop)
    python3 search_agent.py --listen --seconds 10 # mic fixed duration
"""

import os
import time
import tempfile
import threading
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from ddgs import DDGS

from llm_provider import get_llm, current_model
from db import save_run

load_dotenv()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".webm"}


# ─── Whisper ──────────────────────────────────────────────────────────────────

def transcribe_file(file_path: str) -> str:
    """Transcribe an audio file using Whisper (API or local based on .env)."""
    mode = os.getenv("WHISPER_MODE", "api").lower()
    print(f"\n[WHISPER] Transcribing {Path(file_path).name} ({mode} mode)...")

    if mode == "local":
        import whisper
        model_name = os.getenv("WHISPER_MODEL", "base")
        model = whisper.load_model(model_name)
        result = model.transcribe(file_path)
        text = result["text"].strip()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with open(file_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = result.text.strip()

    print(f"[WHISPER] Transcribed: \"{text}\"")
    return text


def record_audio(seconds: int | None = None) -> str:
    """Record from microphone. Returns path to a temp WAV file.
    - seconds=None  → records until user presses Enter
    - seconds=N     → records for N seconds
    """
    import sounddevice as sd
    import scipy.io.wavfile as wav

    sample_rate = 16000
    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=callback)

    if seconds:
        print(f"\n[MIC] Recording for {seconds}s... speak now")
        with stream:
            sd.sleep(seconds * 1000)
    else:
        print("\n[MIC] Recording... press Enter to stop")
        with stream:
            input()                          # blocks until Enter

    import numpy as np
    audio = np.concatenate(frames, axis=0)
    tmp = tempfile.mktemp(suffix=".wav")
    wav.write(tmp, sample_rate, audio)
    print(f"[MIC] Saved recording ({len(audio)/sample_rate:.1f}s)")
    return tmp


def listen_and_transcribe(seconds: int | None = None) -> str:
    """Record from mic then transcribe."""
    audio_path = record_audio(seconds)
    try:
        return transcribe_file(audio_path)
    finally:
        os.unlink(audio_path)               # clean up temp file


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information. Returns top results with title, URL, and snippet."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r['title']}\n    URL: {r['href']}\n    {r['body']}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


TOOLS = [web_search]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a smart research assistant with web search capability.

When answering questions:
1. Search for relevant information using web_search
2. You can search multiple times with refined queries if needed
3. Once you have enough information, provide a clear, concise answer
4. Cite sources (URLs) in your final answer

Always search before answering factual questions — do not rely on training data alone."""


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── Nodes ────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    from langchain_core.messages import SystemMessage
    llm = get_llm(tools=TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    results = []
    for tool_call in last_msg.tool_calls:
        fn = TOOL_MAP[tool_call["name"]]
        output = fn.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": results}


# ─── Routing ──────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "__end__"


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─── Search ───────────────────────────────────────────────────────────────────

def search(query: str, verbose: bool = True) -> str:
    """Run a text query through the search agent."""
    app = build_graph()
    state = {"messages": [HumanMessage(content=query)]}
    t0 = time.monotonic()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Query    : {query}")
        print(f"Provider : {current_model()}")
        print(f"{'='*60}")

    for step in app.stream(state, stream_mode="values"):
        last = step["messages"][-1]
        if verbose:
            if isinstance(last, AIMessage):
                if hasattr(last, "tool_calls") and last.tool_calls:
                    for tc in last.tool_calls:
                        print(f"\n[SEARCH] → {tc['args'].get('query', '')}")
                else:
                    print(f"\n[ANSWER]\n{last.content}")
            elif isinstance(last, ToolMessage):
                preview = last.content[:200].replace("\n", " ")
                print(f"[RESULTS] {preview}...")

    final = step["messages"][-1].content
    duration = int((time.monotonic() - t0) * 1000)

    run_id = save_run(
        agent_name="search_agent",
        query=query,
        messages=step["messages"],
        answer=final,
        model=current_model(),
        duration_ms=duration,
    )
    if verbose:
        print(f"\n[DB] Saved run #{run_id} ({duration}ms)")

    return final


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args:
        query = "What is LangGraph and when was it released?"

    elif args[0] == "--listen":
        # mic recording mode
        seconds = None
        if "--seconds" in args:
            idx = args.index("--seconds")
            seconds = int(args[idx + 1])
        query = listen_and_transcribe(seconds)

    elif Path(args[0]).suffix.lower() in AUDIO_EXTENSIONS:
        # audio file mode
        query = transcribe_file(args[0])

    else:
        # plain text mode
        query = " ".join(args)

    search(query)
