from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings


@dataclass(frozen=True)
class RetrievedSnippet:
    text: str
    score: float | None
    metadata: dict


def _has_persisted_vectors(persist_dir: Path, collection_name: str) -> bool:
    """Return True only when the ChromaDB sqlite file exists and has data."""
    db_file = persist_dir / "chroma.sqlite3"
    return db_file.exists() and db_file.stat().st_size > 0


def _build_in_memory_index(
    embed_model: HuggingFaceEmbedding,
    graph_json_path: Path,
) -> VectorStoreIndex:
    """
    Fallback for Streamlit Cloud (or any environment where the persisted
    vector store was not committed).  Builds a transient in-memory index
    directly from the graph JSON so the app is still usable.
    """
    from llama_index.core import Document
    from src.utils.helpers import load_json, format_project_as_markdown, ProjectDoc

    graph = load_json(graph_json_path)
    documents: list[Document] = []
    for key, rec in graph.get("projects", {}).items():
        doc = ProjectDoc.from_dict(rec, source_path=str(rec.get("source_path", "")))
        documents.append(
            Document(
                text=format_project_as_markdown(doc),
                metadata={"project_name": doc.project_name, "section": "full"},
            )
        )
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)


class VectorRetriever:
    def __init__(
        self,
        *,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_json_path: Path | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.VECTOR_STORE_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self._graph_json_path = graph_json_path or settings.GRAPH_JSON_PATH

        embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        if _has_persisted_vectors(self.persist_dir, self.collection_name):
            # Normal path: use the pre-built ChromaDB store
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = client.get_or_create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            self._mode = "persisted"
        else:
            # Cloud / first-run fallback: build a transient in-memory index
            self.index = _build_in_memory_index(embed_model, self._graph_json_path)
            self._mode = "in-memory"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        project_name: str | None = None,
    ) -> list[RetrievedSnippet]:
        top_k = top_k or settings.VECTOR_TOP_K

        filters = None
        # Metadata filters only work reliably with the persisted Chroma store;
        # the in-memory index has no filter support, so skip them there.
        if project_name and self._mode == "persisted":
            filters = MetadataFilters(
                filters=[MetadataFilter(key="project_name", value=project_name, operator=FilterOperator.EQ)]
            )

        retriever = self.index.as_retriever(similarity_top_k=top_k, filters=filters)
        nodes: list[NodeWithScore] = retriever.retrieve(query)

        snippets: list[RetrievedSnippet] = []
        for n in nodes:
            node = n.node
            snippets.append(
                RetrievedSnippet(
                    text=node.get_content(metadata_mode="none"),
                    score=float(n.score) if n.score is not None else None,
                    metadata=dict(node.metadata or {}),
                )
            )
        return snippets

    @staticmethod
    def snippets_to_context(snippets: list[RetrievedSnippet], *, max_chars: int = 6000) -> str:
        parts: list[str] = []
        total = 0
        for s in snippets:
            header = []
            if s.metadata.get("project_name"):
                header.append(f"project={s.metadata.get('project_name')}")
            if s.metadata.get("section"):
                header.append(f"section={s.metadata.get('section')}")
            header_str = f"[{', '.join(header)}]" if header else "[snippet]"

            block = f"{header_str}\n{s.text.strip()}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n---\n".join(parts).strip()
