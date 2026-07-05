
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .obsidian_read_adapter import ObsidianReadAdapter, ObsidianReadError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = ROOT / "data" / "knowledge_vault_index.json"


@dataclass
class KnowledgeNoteRecord:
    path: str
    folder: str
    filename: str
    title: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    preview: str = ""
    content_hash: str = ""
    size_chars: int = 0


@dataclass
class KnowledgeVaultIndex:
    version: str
    built_at: float
    note_count: int
    folder_count: int
    notes: list[KnowledgeNoteRecord]
    folders: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "built_at": self.built_at,
            "note_count": self.note_count,
            "folder_count": self.folder_count,
            "notes": [asdict(n) for n in self.notes],
            "folders": list(self.folders),
        }


def _clean_join(base: str, child: str) -> str:
    base = str(base or "").strip("/").replace("\\", "/")
    child = str(child or "").strip().replace("\\", "/")
    if not base:
        return child
    return f"{base}/{child}"


def _strip_markdown_noise(text: str, limit: int = 600) -> str:
    s = str(text or "").replace("\r\n", "\n")
    s = re.sub(r"^---\n.*?\n---\n", "", s, flags=re.S)
    s = re.sub(r"`{3}.*?`{3}", "", s, flags=re.S)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()[:limit]


def _parse_frontmatter(text: str) -> dict[str, Any]:
    s = str(text or "").replace("\r\n", "\n")
    if not s.startswith("---\n"):
        return {}

    end = s.find("\n---", 4)
    if end == -1:
        return {}

    raw = s[4:end].strip("\n")
    meta: dict[str, Any] = {}
    current_key: str | None = None

    for line in raw.splitlines():
        if not line.strip():
            continue

        if re.match(r"^\s*-\s+", line) and current_key:
            item = re.sub(r"^\s*-\s+", "", line).strip()
            meta.setdefault(current_key, [])
            if isinstance(meta[current_key], list):
                meta[current_key].append(item)
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key

            if not value:
                meta[key] = []
            elif value.startswith("[") and value.endswith("]"):
                parts = [x.strip().strip('"').strip("'") for x in value[1:-1].split(",") if x.strip()]
                meta[key] = parts
            else:
                meta[key] = value.strip('"').strip("'")

    return meta


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        if value.strip():
            return [value.strip()]
    return []


def _extract_note_record(path: str, content: str) -> KnowledgeNoteRecord:
    clean_path = path.replace("\\", "/").strip("/")
    folder = clean_path.rsplit("/", 1)[0] if "/" in clean_path else ""
    filename = clean_path.rsplit("/", 1)[-1]
    stem = filename[:-3] if filename.lower().endswith(".md") else filename

    meta = _parse_frontmatter(content)
    headings = []
    for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", content, flags=re.M):
        headings.append(m.group(2).strip())

    title = ""
    if isinstance(meta.get("title"), str) and meta.get("title").strip():
        title = str(meta.get("title")).strip()
    elif headings:
        title = headings[0]
    else:
        title = stem

    tags = []
    tags.extend(_listify(meta.get("tags")))
    tags.extend(_listify(meta.get("tag")))
    tags = sorted({t.strip("#").strip() for t in tags if t.strip()})

    aliases = []
    aliases.extend(_listify(meta.get("aliases")))
    aliases.extend(_listify(meta.get("alias")))
    aliases = sorted({a for a in aliases if a})

    wikilinks = sorted({m.group(1).split("|", 1)[0].strip() for m in re.finditer(r"\[\[([^\]]+)\]\]", content)})

    return KnowledgeNoteRecord(
        path=clean_path,
        folder=folder,
        filename=filename,
        title=title,
        aliases=aliases,
        tags=tags,
        headings=headings[:40],
        wikilinks=wikilinks[:80],
        preview=_strip_markdown_noise(content),
        content_hash=hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
        size_chars=len(content),
    )


def build_structural_vault_index(
    adapter: ObsidianReadAdapter | None = None,
    *,
    start_path: str = "",
    max_depth: int = 8,
    max_notes: int = 1000,
    cache_path: Path | None = DEFAULT_INDEX_PATH,
) -> KnowledgeVaultIndex:
    adapter = adapter or ObsidianReadAdapter()

    folders: set[str] = set()
    notes: list[KnowledgeNoteRecord] = []
    visited: set[str] = set()

    def walk(folder: str, depth: int) -> None:
        if depth > max_depth:
            return

        clean_folder = folder.strip("/").replace("\\", "/")
        if clean_folder in visited:
            return
        visited.add(clean_folder)

        try:
            entries = adapter.list_dir(clean_folder)
        except ObsidianReadError:
            return

        if clean_folder:
            folders.add(clean_folder + "/")
        else:
            folders.add("/")

        for entry in entries:
            if len(notes) >= max_notes:
                return

            entry = str(entry)
            child = _clean_join(clean_folder, entry)

            if entry.endswith("/"):
                walk(child.strip("/"), depth + 1)
            elif entry.lower().endswith(".md"):
                try:
                    content = adapter.read_note(child)
                    notes.append(_extract_note_record(child, content))
                except ObsidianReadError:
                    continue

    walk(start_path, 0)

    index = KnowledgeVaultIndex(
        version="v0.16-E1-structural",
        built_at=time.time(),
        note_count=len(notes),
        folder_count=len(folders),
        notes=notes,
        folders=sorted(folders),
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(index.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    return index


def load_cached_vault_index(cache_path: Path | None = DEFAULT_INDEX_PATH) -> KnowledgeVaultIndex | None:
    if not cache_path or not cache_path.exists():
        return None

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    notes = [KnowledgeNoteRecord(**n) for n in data.get("notes", [])]
    return KnowledgeVaultIndex(
        version=str(data.get("version") or ""),
        built_at=float(data.get("built_at") or 0),
        note_count=int(data.get("note_count") or len(notes)),
        folder_count=int(data.get("folder_count") or len(data.get("folders", []))),
        notes=notes,
        folders=[str(x) for x in data.get("folders", [])],
    )


def _tokens(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\-]+", str(query or "").lower()) if len(t) >= 2]


def search_structural_vault_index(index: KnowledgeVaultIndex, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    toks = _tokens(query)
    scored: list[tuple[int, KnowledgeNoteRecord, list[str]]] = []

    for note in index.notes:
        haystacks = {
            "path": note.path.lower(),
            "title": note.title.lower(),
            "aliases": " ".join(note.aliases).lower(),
            "tags": " ".join(note.tags).lower(),
            "headings": " ".join(note.headings).lower(),
            "preview": note.preview.lower(),
            "wikilinks": " ".join(note.wikilinks).lower(),
        }

        score = 0
        reasons = []

        for tok in toks:
            if tok in haystacks["path"]:
                score += 8
                reasons.append(f"path:{tok}")
            if tok in haystacks["title"]:
                score += 10
                reasons.append(f"title:{tok}")
            if tok in haystacks["aliases"]:
                score += 8
                reasons.append(f"alias:{tok}")
            if tok in haystacks["tags"]:
                score += 7
                reasons.append(f"tag:{tok}")
            if tok in haystacks["headings"]:
                score += 5
                reasons.append(f"heading:{tok}")
            if tok in haystacks["wikilinks"]:
                score += 4
                reasons.append(f"wikilink:{tok}")
            if tok in haystacks["preview"]:
                score += 2
                reasons.append(f"content:{tok}")

        if score > 0:
            scored.append((score, note, reasons))

    scored.sort(key=lambda x: (-x[0], x[1].path.lower()))

    results = []
    for score, note, reasons in scored[:limit]:
        results.append({
            "score": score,
            "path": note.path,
            "title": note.title,
            "folder": note.folder,
            "tags": note.tags,
            "headings": note.headings[:8],
            "preview": note.preview[:300],
            "match_reasons": sorted(set(reasons))[:12],
        })

    return results
