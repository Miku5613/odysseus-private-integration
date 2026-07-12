
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .knowledge_vault_index import (
    KnowledgeVaultIndex,
    build_structural_vault_index,
    load_cached_vault_index,
)
from .context_vault_read_adapter import ContextVaultReadAdapter
from .obsidian_read_adapter import ObsidianReadAdapter, ObsidianReadError
from src.chroma_client import get_chroma_client
from src.embeddings import EmbeddingClient


COLLECTION_NAME = "odysseus_knowledge_vault_nomic"
INDEX_NAMESPACE = "ideaverse_vault"
DEFAULT_EMBEDDING_URL = "http://localhost:11434/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"


@dataclass
class SemanticIndexStats:
    collection_name: str
    embedding_model: str
    note_count: int
    chunk_count: int
    indexed_at: float
    rebuilt: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "note_count": self.note_count,
            "chunk_count": self.chunk_count,
            "indexed_at": self.indexed_at,
            "rebuilt": self.rebuilt,
        }


def _embedding_client() -> EmbeddingClient:
    return EmbeddingClient(
        url=os.getenv("KNOWLEDGE_EMBEDDING_URL", DEFAULT_EMBEDDING_URL),
        model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        api_key=os.getenv("KNOWLEDGE_EMBEDDING_API_KEY", ""),
    )


def _collection():
    client = get_chroma_client()
    return client.get_or_create_collection(COLLECTION_NAME)


def _safe_meta_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(x) for x in value if str(x).strip())
    return str(value)


def _content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def _chunk_id(path: str, note_hash: str, chunk_index: int, chunk_text: str) -> str:
    raw = f"{INDEX_NAMESPACE}|{path}|{note_hash}|{chunk_index}|{_content_hash(chunk_text)}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _strip_frontmatter(text: str) -> str:
    s = str(text or "").replace("\r\n", "\n")
    if s.startswith("---\n"):
        end = s.find("\n---", 4)
        if end != -1:
            return s[end + 4:].lstrip("\n")
    return s


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    s = _strip_frontmatter(text)
    lines = s.splitlines()

    sections: list[tuple[str, list[str]]] = []
    current_heading = "Document"
    current_lines: list[str] = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    if not sections:
        return [("Document", s)]

    return [(heading, "\n".join(body).strip()) for heading, body in sections if "\n".join(body).strip()]


def chunk_markdown_note(
    path: str,
    title: str,
    content: str,
    *,
    max_chars: int = 1200,
    overlap: int = 160,
) -> list[dict[str, Any]]:
    note_hash = _content_hash(content)
    chunks: list[dict[str, Any]] = []

    for heading, section_text in _split_markdown_sections(content):
        text = section_text.strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "path": path,
                    "title": title,
                    "heading": heading,
                    "chunk_index": len(chunks),
                    "text": chunk_text,
                    "note_hash": note_hash,
                })

            if end >= len(text):
                break
            start = max(0, end - overlap)

    return chunks


def _delete_existing_namespace(collection) -> int:
    try:
        existing = collection.get(
            where={"index_namespace": INDEX_NAMESPACE},
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def build_semantic_vault_index(
    *,
    structural_index: KnowledgeVaultIndex | None = None,
    adapter: ObsidianReadAdapter | ContextVaultReadAdapter | None = None,
    rebuild: bool = True,
    max_notes: int = 1000,
    batch_size: int = 16,
) -> SemanticIndexStats:
    adapter = adapter or ContextVaultReadAdapter()
    index = structural_index or load_cached_vault_index() or build_structural_vault_index(adapter=adapter, max_notes=max_notes)

    emb = _embedding_client()
    # Force health/dimension probe early so failures are clear.
    emb.get_sentence_embedding_dimension()

    col = _collection()

    if rebuild:
        _delete_existing_namespace(col)

    all_docs: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict[str, Any]] = []

    notes = index.notes[:max_notes]

    for note in notes:
        try:
            content = adapter.read_note(note.path)
        except ObsidianReadError:
            continue

        chunks = chunk_markdown_note(note.path, note.title, content)
        for ch in chunks:
            chunk_text = ch["text"]
            chunk_id = _chunk_id(note.path, ch["note_hash"], ch["chunk_index"], chunk_text)

            metadata = {
                "index_namespace": INDEX_NAMESPACE,
                "source": "obsidian",
                "vault": "Ideaverse",
                "path": note.path,
                "folder": note.folder,
                "filename": note.filename,
                "title": note.title,
                "heading": ch["heading"],
                "chunk_index": int(ch["chunk_index"]),
                "note_hash": ch["note_hash"],
                "tags": _safe_meta_value(note.tags),
                "aliases": _safe_meta_value(note.aliases),
                "wikilinks": _safe_meta_value(note.wikilinks),
                "embedding_model": emb.model,
                "indexed_at": float(time.time()),
            }

            all_ids.append(chunk_id)
            all_docs.append(chunk_text)
            all_metas.append(metadata)

    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_metas = all_metas[i:i + batch_size]

        vectors = emb.encode(batch_docs).tolist()

        col.upsert(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    return SemanticIndexStats(
        collection_name=COLLECTION_NAME,
        embedding_model=emb.model,
        note_count=len(notes),
        chunk_count=len(all_docs),
        indexed_at=time.time(),
        rebuilt=rebuild,
    )


def search_semantic_vault_index(query: str, *, k: int = 8) -> list[dict[str, Any]]:
    query = str(query or "").strip()
    if not query:
        return []

    emb = _embedding_client()
    emb.get_sentence_embedding_dimension()

    col = _collection()
    qvec = emb.encode([query]).tolist()[0]

    results = col.query(
        query_embeddings=[qvec],
        n_results=max(1, int(k)),
        where={"index_namespace": INDEX_NAMESPACE},
        include=["documents", "metadatas", "distances"],
    )

    out: list[dict[str, Any]] = []
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for idx, doc_id in enumerate(ids):
        meta = metas[idx] if idx < len(metas) else {}
        distance = distances[idx] if idx < len(distances) else None
        doc = docs[idx] if idx < len(docs) else ""

        similarity = None
        if isinstance(distance, (int, float)):
            similarity = round(1.0 - float(distance), 4)

        out.append({
            "id": doc_id,
            "path": meta.get("path", ""),
            "title": meta.get("title", ""),
            "heading": meta.get("heading", ""),
            "folder": meta.get("folder", ""),
            "tags": meta.get("tags", ""),
            "similarity": similarity,
            "distance": distance,
            "preview": str(doc or "")[:500],
            "metadata": meta,
        })

    return out


def get_semantic_vault_stats() -> dict[str, Any]:
    col = _collection()
    try:
        total = col.count()
    except Exception:
        total = 0

    namespace_count = 0
    try:
        existing = col.get(where={"index_namespace": INDEX_NAMESPACE}, include=["metadatas"])
        namespace_count = len(existing.get("ids") or [])
    except Exception:
        namespace_count = 0

    return {
        "collection_name": COLLECTION_NAME,
        "total_collection_count": total,
        "namespace_count": namespace_count,
        "index_namespace": INDEX_NAMESPACE,
        "embedding_model": os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_url": os.getenv("KNOWLEDGE_EMBEDDING_URL", DEFAULT_EMBEDDING_URL),
    }
