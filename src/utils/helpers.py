from __future__ import annotations

import json
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
    docs: list[ProjectDoc] = []
    for path in list_yaml_files(projects_dir):
        raw = load_yaml(path)
        docs.append(ProjectDoc.from_dict(raw, source_path=str(path)))
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
