"""Read visually complex PDF pages with a multimodal model.

Designed documents (product reports, brochures, slide-like pages) put meaning in the
layout: side-by-side cards, untitled tables, charts whose numbers only make sense next
to their labels, and photos. Text extraction scrambles these, so such pages are rendered
to an image and the model transcribes them into text, tables and figure descriptions.
Plain text pages stay with the free layout parser.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Literal

import pymupdf
from google.genai import types
from pydantic import BaseModel

from app.pdf_layout import Segment

logger = logging.getLogger("uvicorn.error")

# Bump when the prompt or schema changes so cached pages are read again
VISION_VERSION = "2"

# Thresholds measured on the sample reports: plain text pages have 0-2 vector drawings,
# while charts, table rules and card layouts have 10 or more
MIN_DRAWINGS = 10
MIN_IMAGE_AREA = 0.15
MIN_TEXT_COLUMNS = 4

PROMPT = """This is page {page} of a PDF document. Transcribe everything a reader could ask about into blocks, in reading order.

- "text": prose, bullet lists, callouts and stat cards. Keep each card or column separate; never merge lines of side-by-side columns. Put a big stat with its caption ("76%" + "recycled content in fiber packaging" -> "76% recycled content in fiber packaging").
- "table": every table, with its column headers and every row. The first cell of a row is its label. Copy values exactly, with units.
- "figure": charts, diagrams and informative images. Give the title, what it shows, and every value printed on it with its label (e.g. "Baseline emissions: 445 kg; Emissions reductions: -99 kg; iMac emissions: 346 kg"). Explain how the parts relate. Never estimate values that are not printed.
- Skip logos, decorative photos, running headers and footers, and page numbers.
- Skip small navigation graphics that only mark which section of the document this page belongs to (e.g. a mini diagram with the current stage highlighted); describe such a diagram only when it is the page's main content.
- Drop footnote reference markers ("content1" -> "content"), but keep footnote text itself as a text block.
- Restore subscripts and symbols (CO2e, not COe).
- "heading" is the nearest section heading above the block on this page, or "" if none.

Text extracted from this page, possibly out of order, to help with exact spelling and numbers:
{text}"""


class Block(BaseModel):
    kind: Literal["text", "table", "figure"]
    heading: str
    title: str
    text: str
    columns: list[str]
    rows: list[list[str]]


class PageContent(BaseModel):
    blocks: list[Block]


def visual_pages(path: Path) -> list[int]:
    """1-based numbers of pages that need the vision model."""
    pages = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            area = page.rect.width * page.rect.height
            image_area = sum(
                (r.width * r.height) for r in (pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info())
            )
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and len(b[4].strip()) > 30]
            # Distinct left edges of text blocks, in twelfths of the page: 4+ means a card layout
            columns = {round(b[0] / (page.rect.width / 12)) for b in blocks}
            if (
                len(page.get_drawings()) >= MIN_DRAWINGS
                or image_area / area >= MIN_IMAGE_AREA
                or len(columns) >= MIN_TEXT_COLUMNS
            ):
                pages.append(page.number + 1)
    return pages


def read_pages(
    path: Path,
    pages: list[int],
    generate: Callable[[list, type[BaseModel]], BaseModel],
    cache_dir: Path,
    dpi: int,
    workers: int,
) -> dict[int, PageContent]:
    """Transcribe pages, reusing cached results; pages that fail are left out."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache = cache_dir / digest
    cache.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(str(path)) as doc:
        inputs = {n: (doc[n - 1].get_pixmap(dpi=dpi).tobytes("png"), doc[n - 1].get_text()) for n in pages}

    def read(number: int) -> tuple[int, PageContent | None]:
        cached = cache / f"p{number}-v{VISION_VERSION}.json"
        if cached.exists():
            return number, PageContent.model_validate_json(cached.read_text(encoding="utf-8"))
        image, text = inputs[number]
        contents = [
            types.Part.from_bytes(data=image, mime_type="image/png"),
            PROMPT.format(page=number, text=text[:6000]),
        ]
        try:
            content = generate(contents, PageContent)
        except Exception as exc:
            # Fall back to the layout parser for this page rather than failing the upload
            logger.warning("Vision read of page %d failed, using text extraction: %s", number, exc)
            return number, None
        cached.write_text(content.model_dump_json(), encoding="utf-8")
        return number, content

    results: dict[int, PageContent] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (number, content) in enumerate(pool.map(read, pages), start=1):
            if content is not None:
                results[number] = content
            logger.info("Vision read %d/%d pages", done, len(pages))
    return results


def to_segments(page: int, content: PageContent, skip_tables: bool) -> list[Segment]:
    """Turn one page's blocks into segments shaped like the layout parser's output."""
    segments = []
    for i, block in enumerate(content.blocks):
        section = block.heading.strip()
        if block.kind == "text":
            if block.text.strip():
                segments.append(Segment(block.text.strip(), "text", page, page, section))
        elif block.kind == "figure":
            body = "\n".join(p for p in [block.title.strip(), block.text.strip()] if p)
            if body:
                segments.append(Segment(body, "figure", page, page, section, standalone=True))
        elif not skip_tables and block.rows:
            segments.extend(_table_segments(page, i, block, section))
    return segments


def _table_segments(page: int, index: int, block: Block, section: str) -> list[Segment]:
    # Same shape as parsed tables: an overview, then one row per segment with every value
    # labelled by its column, so retrieval can match single rows and still expand the table
    name = block.title.strip() or block.heading.strip() or f"Table on page {page}"
    columns = [c.strip() for c in block.columns]
    common = dict(page_start=page, page_end=page, section=section, table_id=f"v{page}-{index}", standalone=True)

    overview = name
    if columns:
        overview += ". Columns: " + ", ".join(c for c in columns if c)
    labels = [row[0].strip() for row in block.rows if row and row[0].strip()]
    if labels:
        overview += ". Rows: " + ", ".join(dict.fromkeys(labels))
    if block.text.strip():
        overview += ". " + block.text.strip()
    segments = [Segment(overview, "table", **common)]

    for row in block.rows:
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        label, values = cells[0], cells[1:]
        headers = columns[1:] if len(columns) == len(cells) else [""] * len(values)
        body = "; ".join(f"{h}: {v}" if h else v for h, v in zip(headers, values) if v)
        segments.append(Segment(f"{name} — {label}: {body}" if body else f"{name} — {label}", "table", **common))
    return segments
