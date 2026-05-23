from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work when running as a script: python scripts/run_ingestion.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.ingestion.build_graph import build_graph
from src.ingestion.build_vector_db import build_vector_db
from src.utils.helpers import ensure_dir


def main() -> None:
    ensure_dir(settings.PROJECT_DOCS_DIR)
    ensure_dir(settings.VECTOR_STORE_DIR)
    ensure_dir(settings.GRAPH_STORE_DIR)

    print("Building knowledge graph...")
    graph_path = build_graph()
    print(f"Graph written to: {graph_path}")

    print("Building vector database (ChromaDB)...")
    build_vector_db()
    print("Vector DB build complete.")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()

