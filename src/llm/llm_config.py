from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from llama_index.core.llms import LLM

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
# Groq Llama 3.x production models use 128k context; override via GROQ_CONTEXT_WINDOW if needed.
DEFAULT_GROQ_CONTEXT_WINDOW = 131_072


@dataclass(frozen=True)
class LLMInfo:
    provider: str
    model: str


def get_llm() -> tuple[LLM, LLMInfo]:
    """
    Auto-select an LLM.

    Priority:
    1) Groq if GROQ_API_KEY is set  — uses llama-index-llms-groq (OpenAILike subclass)
    2) OpenAI if OPENAI_API_KEY is set
    3) Ollama local otherwise
    """
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        from llama_index.llms.groq import Groq

        model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
        context_window_raw = os.getenv("GROQ_CONTEXT_WINDOW", "").strip()
        context_window = int(context_window_raw) if context_window_raw else DEFAULT_GROQ_CONTEXT_WINDOW
        llm: LLM = Groq(
            model=model,
            api_key=groq_key,
            temperature=0.2,
            request_timeout=120.0,
            context_window=context_window,
        )
        return llm, LLMInfo(provider="groq", model=model)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        from llama_index.llms.openai import OpenAI as OpenAILlm

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        llm = OpenAILlm(model=model, api_key=openai_key, temperature=0.2)
        return llm, LLMInfo(provider="openai", model=model)

    from llama_index.llms.ollama import Ollama

    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip() or "llama3.2:3b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
    llm = Ollama(model=model, base_url=base_url, temperature=0.2, request_timeout=120.0)
    return llm, LLMInfo(provider="ollama", model=model)
