from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from config import settings
from src.utils.helpers import ProjectDoc, format_project_as_markdown, load_json


def _normalize(name: str) -> str:
    return " ".join(name.strip().split()).lower()


@dataclass(frozen=True)
class GraphProject:
    key: str
    record: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.record.get("project_name", "")).strip()

    def to_project_doc(self) -> ProjectDoc:
        return ProjectDoc.from_dict(self.record, source_path=str(self.record.get("source_path", "")))


class GraphRetriever:
    def __init__(self, graph_json_path: Path | None = None) -> None:
        self.graph_json_path = graph_json_path or settings.GRAPH_JSON_PATH
        self._graph: dict[str, Any] = {}
        self._projects: dict[str, dict[str, Any]] = {}
        self._keys: list[str] = []
        self.reload()

    def reload(self) -> None:
        self._graph = load_json(self.graph_json_path)
        self._projects = dict(self._graph.get("projects", {}))
        self._keys = sorted(self._projects.keys())

    def list_project_names(self) -> list[str]:
        out: list[str] = []
        for k in self._keys:
            rec = self._projects[k]
            name = str(rec.get("project_name", "")).strip()
            if name:
                out.append(name)
        return out

    def detect_project(self, query: str, *, score_threshold: int = 78) -> str | None:
        """
        Fuzzy-detect a project name. Returns canonical project_name or None.
        """
        q = query.strip()
        if not q:
            return None

        # Fast path: direct substring match against canonical names
        for k in self._keys:
            name = str(self._projects[k].get("project_name", "")).strip()
            if name and name.lower() in q.lower():
                return name

        choices = [(k, str(self._projects[k].get("project_name", "")).strip()) for k in self._keys]
        labels = [c[1] for c in choices if c[1]]
        if not labels:
            return None

        best = process.extractOne(q, labels, scorer=fuzz.WRatio)
        if not best:
            return None
        label, score, _idx = best
        if score < score_threshold:
            return None
        return str(label)

    def get_project(self, project_name: str) -> GraphProject | None:
        needle = _normalize(project_name)
        if needle in self._projects:
            return GraphProject(key=needle, record=self._projects[needle])

        for k in self._keys:
            rec = self._projects[k]
            if _normalize(str(rec.get("project_name", ""))) == needle:
                return GraphProject(key=k, record=rec)
        return None

    def get_project_context_markdown(self, project_name: str) -> str | None:
        """
        GraphRAG context: **complete** project record (including all workflow steps).
        """
        p = self.get_project(project_name)
        if not p:
            return None
        doc = p.to_project_doc()
        return format_project_as_markdown(doc)

    def get_all_project_contexts_for_comparison(self, project_names: list[str]) -> str:
        blocks: list[str] = []
        for name in project_names:
            md = self.get_project_context_markdown(name)
            if md:
                blocks.append(md)
        return "\n\n---\n\n".join(blocks).strip()

