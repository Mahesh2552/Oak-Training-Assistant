from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Project root: .../oak_training_assistant
    ROOT_DIR: Path = Path(__file__).resolve().parent

    DATA_DIR: Path = ROOT_DIR / "data"
    VECTOR_STORE_DIR: Path = ROOT_DIR / "vector_store"
    GRAPH_STORE_DIR: Path = ROOT_DIR / "graph_store"

    # Subpaths
    PROJECT_DOCS_DIR: Path = DATA_DIR / "projects"
    GRAPH_JSON_PATH: Path = GRAPH_STORE_DIR / "oak_graph.json"

    # Chroma
    CHROMA_COLLECTION: str = "oak_projects"

    # Retrieval
    VECTOR_TOP_K: int = 6
    VECTOR_TOP_K_WHEN_PROJECT_KNOWN: int = 10


settings = Settings()
