from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import settings

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RetrievedSnippet:
    text: str
    score: float | None
    metadata: dict


def _has_persisted_vectors(persist_dir: Path) -> bool:
    """Return True only when the ChromaDB sqlite file exists and has data."""
    db_file = persist_dir / "chroma.sqlite3"
    return db_file.exists() and db_file.stat().st_size > 0


def _build_persisted_index(
    persist_dir: Path,
    collection_name: str,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Load the pre-built ChromaDB vector store (local / self-hosted only)."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def _build_in_memory_index(
    embed_model: HuggingFaceEmbedding,
    graph_json_path: Path,
) -> VectorStoreIndex:
    """
    Fallback for Streamlit Cloud (or any environment without a persisted
    vector store). Builds a transient in-memory index from graph_store/oak_graph.json.
    chromadb is NOT imported here, so this path works even if chromadb is absent.
    """
    from llama_index.core import Document
    from src.utils.helpers import ProjectDoc, format_project_as_markdown, load_json

    graph = load_json(graph_json_path)
    documents: list[Document] = []
    for _key, rec in graph.get("projects", {}).items():
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

        if _has_persisted_vectors(self.persist_dir):
            try:
                self.index = _build_persisted_index(self.persist_dir, self.collection_name, embed_model)
                self._mode = "persisted"
            except Exception as exc:
                # Stale collection ID, protobuf conflict, or any other ChromaDB
                # failure — fall back to in-memory so the app keeps running.
                import warnings
                warnings.warn(
                    f"ChromaDB load failed ({exc}); falling back to in-memory index. "
                    "Re-run 'python scripts/run_ingestion.py' to rebuild the vector store.",
                    stacklevel=2,
                )
                self.index = _build_in_memory_index(embed_model, self._graph_json_path)
                self._mode = "in-memory"
        else:
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
        # Metadata filters only work reliably with the persisted Chroma store
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
