from snow_sports_rag.ingest.parse import (
    extract_headings,
    extract_title,
    normalize_doc_id,
)


def test_normalize_doc_id_posix() -> None:
    assert normalize_doc_id(r"athletes\Foo.md") == "athletes/Foo.md"
    assert normalize_doc_id("./circuits/fis.md") == "circuits/fis.md"


def test_extract_title_first_h1_only() -> None:
    md = """# Main Title

## Section
body
"""
    assert extract_title(md) == "Main Title"


def test_extract_title_prefers_h1_over_later_hashes() -> None:
    md = """## Early section
# Real Title
"""
    assert extract_title(md) == "Real Title"


def test_extract_title_empty_when_no_h1() -> None:
    assert extract_title("## Only h2\n") == ""


def test_extract_headings_h2_and_h3_order() -> None:
    md = """# Title

## A
### B
## C
"""
    assert extract_headings(md) == ("A", "B", "C")
