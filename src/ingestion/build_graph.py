from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import settings
from src.utils.helpers import ProjectDoc, dump_json, ensure_dir, load_project_docs


def _normalize_project_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def build_graph_json(*, projects_dir: Path, out_path: Path) -> dict[str, Any]:
    """
    Build a lightweight knowledge graph that guarantees full project coverage.

    We store:
    - canonical project records (structured fields)
    - simple triples for navigation and debug visibility
    """
    project_docs: list[ProjectDoc] = load_project_docs(projects_dir)

    projects: dict[str, Any] = {}
    triples: list[dict[str, str]] = []

    for doc in project_docs:
        key = _normalize_project_name(doc.project_name)

        projects[key] = {
            "project_name": doc.project_name,
            "overview": doc.overview,
            "problem_statement": doc.problem_statement,
            "architecture": doc.architecture,
            "workflow": doc.workflow,
            "technologies_used": doc.technologies_used,
            "challenges": doc.challenges,
            "use_cases": doc.use_cases,
            "source_path": doc.source_path,
        }

        subj = doc.project_name
        if doc.overview:
            triples.append({"s": subj, "p": "HAS_OVERVIEW", "o": "Overview"})
        if doc.problem_statement:
            triples.append({"s": subj, "p": "HAS_PROBLEM_STATEMENT", "o": "Problem Statement"})

        for i, a in enumerate(doc.architecture, start=1):
            triples.append({"s": subj, "p": "HAS_ARCHITECTURE_ITEM", "o": f"{i}. {a}"})

        for wf in doc.workflow:
            step = wf.get("step")
            title = str(wf.get("title", "")).strip()
            details = str(wf.get("details", "")).strip()
            step_label = f"Step {step}" if step not in (None, "", "None") else "Step"
            o = f"{step_label}: {title}".strip()
            if details:
                o = f"{o} — {details}"
            triples.append({"s": subj, "p": "HAS_WORKFLOW_STEP", "o": o})

        for t in doc.technologies_used:
            triples.append({"s": subj, "p": "USES_TECH", "o": t})
        for c in doc.challenges:
            triples.append({"s": subj, "p": "HAS_CHALLENGE", "o": c})
        for u in doc.use_cases:
            triples.append({"s": subj, "p": "HAS_USE_CASE", "o": u})

    graph = {
        "version": 1,
        "projects": projects,
        "triples": triples,
    }
    return graph


def build_graph(*, projects_dir: Path | None = None, out_path: Path | None = None) -> Path:
    projects_dir = projects_dir or settings.PROJECT_DOCS_DIR
    out_path = out_path or settings.GRAPH_JSON_PATH

    ensure_dir(out_path.parent)
    graph = build_graph_json(projects_dir=projects_dir, out_path=out_path)
    dump_json(out_path, graph)
    return out_path


if __name__ == "__main__":
    path = build_graph()
    print(f"Wrote graph to: {path}")
