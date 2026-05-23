from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load secrets: Streamlit Cloud uses st.secrets; local dev uses .env.
# Injecting into os.environ means the rest of the codebase (llm_config, etc.)
# can use os.getenv() uniformly without knowing which source provided the value.
load_dotenv(ROOT / ".env")
_SECRET_KEYS = ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_CONTEXT_WINDOW", "OPENAI_API_KEY", "OPENAI_MODEL")
for _k in _SECRET_KEYS:
    if _k not in os.environ:
        try:
            _v = st.secrets.get(_k)
            if _v:
                os.environ[_k] = str(_v)
        except Exception:
            pass

from config import settings
from src.llm.llm_config import get_llm
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.vector_retriever import VectorRetriever


def build_user_prompt(*, question: str, graph_context: str | None, vector_context: str | None) -> str:
    return textwrap.dedent(
        f"""
        Answer in a way that fits this question only (overview vs deep dive vs single topic)—do not use a fixed report layout unless the user asks for a full walkthrough.

        User question:
        {question}

        Graph context (authoritative, complete):
        {graph_context or "(none)"}

        Vector context (supporting excerpts):
        {vector_context or "(none)"}
        """
    ).strip()


# Cap prior turns so context stays bounded (each message is one user or assistant turn).
_MAX_PRIOR_CHAT_MESSAGES = 24


def _chat_messages_for_llm(*, user_prompt: str, session_messages: list[dict]) -> list[ChatMessage]:
    """System + recent history (plain text) + current user message (with retrieval context)."""
    out: list[ChatMessage] = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    prior = session_messages[:-1]
    if len(prior) > _MAX_PRIOR_CHAT_MESSAGES:
        prior = prior[-_MAX_PRIOR_CHAT_MESSAGES :]
    for msg in prior:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append(ChatMessage(role=role, content=content))
    out.append(ChatMessage(role="user", content=user_prompt))
    return out


@st.cache_resource
def get_retrievers() -> tuple[GraphRetriever, VectorRetriever]:
    return GraphRetriever(settings.GRAPH_JSON_PATH), VectorRetriever(persist_dir=settings.VECTOR_STORE_DIR)


def _llm_config_key() -> str:
    """Cache key so changing .env (e.g. GROQ_MODEL) picks up the new model after refresh."""
    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq:" + (os.getenv("GROQ_MODEL", "") or "llama-3.1-8b-instant")
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai:" + (os.getenv("OPENAI_MODEL", "") or "default")
    return "ollama:" + (os.getenv("OLLAMA_MODEL", "") or "llama3.2:3b")


@st.cache_resource
def get_cached_llm(_config_key: str):
    return get_llm()


def main() -> None:
    st.set_page_config(page_title="Oak Training Assistant", page_icon="🌳", layout="wide")
    st.title("Oak Training Assistant")
    # st.caption("Hybrid Retrieval: GraphRAG (complete project) + Vector RAG (supporting excerpts)")

    graph_ret, vec_ret = get_retrievers()
    llm, llm_info = get_cached_llm(_llm_config_key())

    with st.sidebar:
        st.subheader("Status")
        st.write(f"**LLM provider**: {llm_info.provider}")
        st.write(f"**LLM model**: {llm_info.model}")
        # st.write(f"**Graph store**: `{settings.GRAPH_JSON_PATH}`")
        # st.write(f"**Vector store**: `{settings.VECTOR_STORE_DIR}`")

        st.subheader("Known projects")
        projects = graph_ret.list_project_names()
        if projects:
            st.write("\n".join([f"- {p}" for p in projects]))
        else:
            st.warning("No projects found. Add YAML docs to `data/projects/` and re-run ingestion.")

        # if st.button("Reload graph"):
        #     graph_ret.reload()
        #     st.success("Reloaded graph.")

        if st.button("Clear chat"):
            st.session_state.pop("messages", None)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about a project (e.g., 'Explain project Raman Drug Detection').")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            detected = graph_ret.detect_project(question)

            graph_context = None
            if detected:
                graph_context = graph_ret.get_project_context_markdown(detected)

            top_k = settings.VECTOR_TOP_K_WHEN_PROJECT_KNOWN if detected else settings.VECTOR_TOP_K
            snippets = vec_ret.retrieve(question, top_k=top_k, project_name=detected)
            vector_context = vec_ret.snippets_to_context(snippets)

            user_prompt = build_user_prompt(question=question, graph_context=graph_context, vector_context=vector_context)

            response = llm.chat(_chat_messages_for_llm(user_prompt=user_prompt, session_messages=st.session_state.messages))

            answer = response.message.content.strip()
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

