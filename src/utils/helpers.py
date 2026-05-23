from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_yaml_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.y*ml") if p.is_file()])


def list_project_files(folder: Path) -> list[Path]:
    """Return all supported project files: YAML, TXT, PDF."""
    if not folder.exists():
        return []
    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml", "*.txt", "*.pdf"):
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stem_to_name(stem: str) -> str:
    """Convert a filename stem like 'my_project_name' to 'My Project Name'."""
    return re.sub(r"[_\-]+", " ", stem).strip().title()


def _load_txt(path: Path) -> dict[str, Any]:
    """
    Load a plain-text project file.
    Optional first line: 'project_name: Some Name'
    Otherwise the filename stem is used as the project name.
    Full text becomes the overview.
    """
    text = path.read_text(encoding="utf-8").strip()
    name: str | None = None
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("project_name:"):
        name = lines[0].split(":", 1)[1].strip()
        text = "\n".join(lines[1:]).strip()
    elif lines and lines[0].startswith("#"):
        name = lines[0].lstrip("#").strip()
        text = "\n".join(lines[1:]).strip()
    if not name:
        name = _stem_to_name(path.stem)
    return {"project_name": name, "overview": text}


def _load_pdf(path: Path) -> dict[str, Any]:
    """
    Extract text from a PDF and treat it as an unstructured project doc.
    Requires pypdf (pip install pypdf).
    Project name is derived from the filename stem.
    Full extracted text becomes the overview.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required to load PDF files. "
            "Install it with: pip install pypdf"
        )
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    name = _stem_to_name(path.stem)
    return {"project_name": name, "overview": text}


@dataclass(frozen=True)
class ProjectDoc:
    project_name: str
    overview: str
    problem_statement: str
    architecture: list[str]
    workflow: list[dict[str, Any]]
    technologies_used: list[str]
    challenges: list[str]
    use_cases: list[str]
    source_path: str

    @staticmethod
    def from_dict(d: dict[str, Any], *, source_path: str) -> "ProjectDoc":
        def as_str(key: str) -> str:
            val = d.get(key, "")
            return "" if val is None else str(val).strip()

        def as_list_str(key: str) -> list[str]:
            val = d.get(key, [])
            if val is None:
                return []
            if isinstance(val, str):
                return [val.strip()]
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            return [str(val).strip()]

        workflow_val = d.get("workflow", [])
        workflow: list[dict[str, Any]] = []
        if isinstance(workflow_val, list):
            for item in workflow_val:
                if isinstance(item, dict):
                    workflow.append(item)
                else:
                    workflow.append({"step": None, "title": str(item), "details": ""})
        elif isinstance(workflow_val, dict):
            workflow = [workflow_val]
        elif isinstance(workflow_val, str) and workflow_val.strip():
            workflow = [{"step": None, "title": "Workflow", "details": workflow_val.strip()}]

        name = as_str("project_name")
        if not name:
            raise ValueError(f"Missing required field project_name in {source_path}")

        return ProjectDoc(
            project_name=name,
            overview=as_str("overview"),
            problem_statement=as_str("problem_statement"),
            architecture=as_list_str("architecture"),
            workflow=workflow,
            technologies_used=as_list_str("technologies_used"),
            challenges=as_list_str("challenges"),
            use_cases=as_list_str("use_cases"),
            source_path=source_path,
        )


def load_project_docs(projects_dir: Path) -> list[ProjectDoc]:
    """Load all project docs from YAML, TXT, and PDF files."""
    docs: list[ProjectDoc] = []
    for path in list_project_files(projects_dir):
        try:
            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                raw = load_yaml(path)
            elif suffix == ".txt":
                raw = _load_txt(path)
            elif suffix == ".pdf":
                raw = _load_pdf(path)
            else:
                continue
            docs.append(ProjectDoc.from_dict(raw, source_path=str(path)))
        except Exception as exc:
            print(f"Warning: skipping {path.name} — {exc}")
    return docs


def format_project_as_markdown(doc: ProjectDoc) -> str:
    wf_lines: list[str] = []
    for item in doc.workflow:
        step = item.get("step")
        title = str(item.get("title", "")).strip()
        details = str(item.get("details", "")).strip()
        prefix = f"Step {step}" if step not in (None, "", "None") else "Step"
        if title:
            wf_lines.append(f"- **{prefix}: {title}**")
        else:
            wf_lines.append(f"- **{prefix}**")
        if details:
            wf_lines.append(f"  - {details}")

    def bullets(xs: Iterable[str]) -> str:
        xs = [x for x in xs if str(x).strip()]
        return "\n".join([f"- {x}" for x in xs]) if xs else "- (Not specified)"

    md = f"""# {doc.project_name}

## Overview
{doc.overview or "(Not specified)"}

## Problem Statement
{doc.problem_statement or "(Not specified)"}

## Architecture
{bullets(doc.architecture)}

## Workflow (step-by-step)
{("\n".join(wf_lines) if wf_lines else "- (Not specified)")}

## Technologies Used
{bullets(doc.technologies_used)}

## Challenges
{bullets(doc.challenges)}

## Use Cases
{bullets(doc.use_cases)}
"""
    return md.strip() + "\n"
