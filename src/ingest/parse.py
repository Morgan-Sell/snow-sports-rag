from __future__ import annotations

import re
from pathlib import PurePosixPath


def normalize_doc_id(relative_path: str) -> str:
    """Normalize a relative path to a stable POSIX ``doc_id`` string.

    Backslashes become forward slashes; a leading ``./`` is removed.

    Parameters
    ----------
    relative_path : str
        Path as returned by the OS or joined from path components.

    Returns
    -------
    str
        POSIX-style relative path, or ``""`` if ``relative_path`` is empty after
        normalization.
    """
    p = PurePosixPath(relative_path.replace("\\", "/"))
    parts = p.parts
    if parts and parts[0] == ".":
        parts = parts[1:]
    return str(PurePosixPath(*parts)) if parts else ""


_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")
_HEADING2_RE = re.compile(r"^##\s+(.+?)\s*$")
_HEADING3_RE = re.compile(r"^###\s+(.+?)\s*$")


def extract_title(markdown: str) -> str:
    """Return the first ATX H1 title line in the document.

    Lines starting with ``##`` are not treated as H1. If no H1 exists,
    returns an empty string (callers may fall back to the file stem).

    Parameters
    ----------
    markdown : str
        Full Markdown source.

    Returns
    -------
    str
        Stripped H1 text without the leading ``# `` markers.
    """
    for line in markdown.splitlines():
        if _TITLE_RE.match(line) and not line.startswith("##"):
            m = _TITLE_RE.match(line)
            return m.group(1).strip() if m else ""
    return ""


def extract_headings(markdown: str) -> tuple[str, ...]:
    """Collect ``##`` and ``###`` heading texts in document order.

    Parameters
    ----------
    markdown : str
        Full Markdown source.

    Returns
    -------
    tuple[str, ...]
        Tuple of stripped heading texts (no ``#`` markers).
    """
    out: list[str] = []
    for line in markdown.splitlines():
        m2 = _HEADING2_RE.match(line)
        if m2:
            out.append(m2.group(1).strip())
            continue
        m3 = _HEADING3_RE.match(line)
        if m3:
            out.append(m3.group(1).strip())
    return tuple(out)
