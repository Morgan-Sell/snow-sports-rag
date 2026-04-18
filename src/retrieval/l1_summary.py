from __future__ import annotations

import re

from ..ingest.models import SourceDocument

__all__ = ["l1_summary_text"]

__doc__ = """Heuristic document summaries for Phase 2.1 L1 embeddings."""

_OVERVIEW_HEADING = re.compile(
    r"^(Overview|Summary|Introduction|About|Background)\b",
    re.IGNORECASE,
)


def _strip_first_h1_block(lines: list[str]) -> str:
    """Return body after the first ATX ``#`` title line.

    Parameters
    ----------
    lines : list of str
        Full document split into lines (typically ``raw_markdown.splitlines()``).

    Returns
    -------
    str
        Remaining Markdown after the H1 block and following blank lines.
    """
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("# "):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def _parse_h2_sections(body: str) -> list[tuple[str, str]]:
    """Split ``body`` on ``##`` headings into ``(heading, section_text)`` pairs.

    Parameters
    ----------
    body : str
        Markdown after the document H1 (may be empty).

    Returns
    -------
    list of tuple[str, str]
        In-order sections; ``section_text`` excludes the heading line and is
        stripped of outer blank lines.
    """
    if not body.strip():
        return []
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    if not matches:
        return []
    out: list[tuple[str, str]] = []
    for j, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[j + 1].start() if j + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        out.append((heading, chunk))
    return out


def _first_matching_or_first_h2(sections: list[tuple[str, str]]) -> str:
    """Return overview-style section text, else the first H2 body.

    Parameters
    ----------
    sections : list of tuple[str, str]
        ``(heading, body)`` pairs from :func:`_parse_h2_sections`.

    Returns
    -------
    str
        Excerpt text, possibly empty when ``sections`` is empty.
    """
    for heading, text in sections:
        if _OVERVIEW_HEADING.match(heading) and text:
            return text
    if sections:
        return sections[0][1]
    return ""


def _fallback_prose(body: str, max_chars: int) -> str:
    """Collect non-heading prose from ``body`` when no H2 sections exist.

    Parameters
    ----------
    body : str
        Markdown after the document title.
    max_chars : int
        Hard cap on returned character length.

    Returns
    -------
    str
        Whitespace-normalized excerpt, truncated to ``max_chars``.
    """
    lines = body.splitlines()
    buf: list[str] = []
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        buf.append(line)
    return "\n".join(buf).strip()[:max_chars]


def l1_summary_text(
    document: SourceDocument,
    *,
    max_section_chars: int = 1500,
) -> str:
    """Build L1 text: title plus a key overview excerpt (Phase 2.1 heuristic).

    Chooses, in order:

    #. The first ``##`` section whose heading starts with one of
       Overview, Summary, Introduction, About, Background (case-insensitive).
    #. Otherwise the body of the first ``##`` section.
    #. Otherwise the first lines of prose after the H1 (no ``##`` present).

    Parameters
    ----------
    document : SourceDocument
        Ingested Markdown source.
    max_section_chars : int, optional
        Truncate the overview excerpt to this many characters.

    Returns
    -------
    str
        Text embedded for the document-level (L1) vector.
    """
    title = document.title.strip()
    after_h1 = _strip_first_h1_block(document.raw_markdown.splitlines())
    sections = _parse_h2_sections(after_h1)
    overview = _first_matching_or_first_h2(sections)
    if not overview.strip():
        overview = _fallback_prose(after_h1, max_section_chars)
    overview = overview.strip()[:max_section_chars]
    if overview:
        return f"{title}\n\n{overview}"
    return title
