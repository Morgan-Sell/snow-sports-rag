from __future__ import annotations

from ..config.loader import AppConfig
from .models import SourceDocument
from .parse import extract_headings, extract_title, normalize_doc_id


class KnowledgeBaseLoader:
    """Discover and load all ``*.md`` files under a configured root directory."""

    def __init__(self, config: AppConfig) -> None:
        """Remember the corpus root from application config.

        Parameters
        ----------
        config : AppConfig
            Must expose ``knowledge_base_path`` pointing at the Markdown tree.
        """
        self._root = config.knowledge_base_path

    def load_all(self) -> list[SourceDocument]:
        """Walk the tree and build one :class:`SourceDocument` per Markdown file.

        Returns
        -------
        list[SourceDocument]
            Sorted by ``doc_id`` (lexicographic path order).

        Raises
        ------
        FileNotFoundError
            If the configured knowledge base path is not a directory.
        """
        root = self._root
        if not root.is_dir():
            raise FileNotFoundError(f"Knowledge base directory not found: {root}")

        docs: list[SourceDocument] = []
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root)
            rel_posix = normalize_doc_id(str(rel))
            entity_type = rel_posix.split("/", 1)[0] if "/" in rel_posix else ""

            text = path.read_text(encoding="utf-8")
            title = extract_title(text) or path.stem
            headings = extract_headings(text)

            docs.append(
                SourceDocument(
                    doc_id=rel_posix,
                    entity_type=entity_type,
                    title=title,
                    raw_markdown=text,
                    headings=headings,
                )
            )
        return docs
