"""Layout-aware PDF parsing.

Reports such as the World Bank's Global Economic Prospects mix two-column prose,
shaded boxes that run across pages, tables, and charts on the same page. Plain text
extraction interleaves all of these, so each page is split into regions and every
region is read in its own order:

* tables  -> one segment per row, every value labelled with its column header
* figures -> title, summary, panel titles and notes; axis tick labels are dropped
* boxes   -> their own stream, stitched across "(continued)" pages
* prose   -> left column, then right column, with running headers and footers removed
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

TITLE_RE = re.compile(r"^(TABLE|FIGURE|BOX)\s+([A-Z]{0,2}\d+(?:\.\d+)*)\.?\s*(.*)$")
CONTINUED_RE = re.compile(r"\(\s*continued\s*\)", re.I)
SOURCE_RE = re.compile(r"^Sources?\s*:")
NOTES_START_RE = re.compile(r"^(Sources?|Notes?)\s*:")
# Column headers of time-series tables: 2023, 2025e, 2026f, 24Q4, 26Q1e, FY2026/27
PERIOD_RE = re.compile(r"^(?:(?:19|20)\d{2}[a-z]{0,2}|\d{2}Q[1-4][a-z]?|FY\d{2,4}(?:/\d{2,4})?[a-z]?)$")
VALUE_RE = re.compile(r"^[-−–]?\$?\d[\d,]*(?:\.\d+)?%?$|^\.\.$|^n\.?a\.?$|^…$")
TOC_LEADER_RE = re.compile(r"\.{4,}|(?:\. ){4,}")
REFERENCES_RE = re.compile(r"^(references|bibliography)$", re.I)
# Date ticks on chart axes ("Feb-26", "Jan 2024") carry no information without the plot
AXIS_TICK_RE = re.compile(r"^((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s']?\d{2,4}\s*)+$|^Percent$")
TERMINAL_PUNCT =(".", "?", "!", ":", "”", '"', ")")


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    font: str
    block: int
    col: int = -1  # 0 left, 1 right, -1 spans both columns

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Segment:
    """A piece of the document with one kind and location, ready for chunking."""

    text: str
    kind: str  # text | box | table | figure | footnote | references
    page_start: int
    page_end: int
    section: str
    chapter: str = ""
    table_id: str = ""
    # Table rows and figures are retrieved on their own; prose is packed into larger chunks
    standalone: bool = False


@dataclass
class _Page:
    number: int
    width: float
    height: float
    lines: list[Line]
    words: list[Word]
    fills: list[tuple[pymupdf.Rect, tuple]]


@dataclass
class _Region:
    rect: pymupdf.Rect
    kind: str  # table | figure
    title: Line | None
    label: str


@dataclass
class _Stream:
    """Prose flowing across pages: the main text, or one box."""

    kind: str
    base: list[str]
    headings: list[tuple[float, str]] = field(default_factory=list)
    paras: list[Segment] = field(default_factory=list)
    footnotes: list[Segment] = field(default_factory=list)
    last_was_body: bool = False
    in_references: bool = False

    def section(self) -> str:
        return " > ".join(self.base + [text for _, text in self.headings])


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("­", "-")
    return re.sub(r"\s+", " ", text).strip()


def _in_rect(rect: pymupdf.Rect, x: float, y: float) -> bool:
    return rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1


def _family(font: str) -> str:
    return font.split("-")[0].split(",")[0]


def _is_bold(font: str) -> bool:
    return bool(re.search(r"bold|bol\b|-bd|black|heavy", font, re.I))


def _read_page(page: pymupdf.Page, number: int) -> _Page:
    lines: list[Line] = []
    for block_no, block in enumerate(page.get_text("dict")["blocks"]):
        if block["type"] != 0:
            continue
        for raw in block["lines"]:
            spans = [s for s in raw["spans"] if s["text"].strip()]
            if not spans:
                continue
            # Leading spaces are often set in another font; style comes from the longest span
            main = max(spans, key=lambda s: len(s["text"].strip()))
            parts = []
            for span in raw["spans"]:
                # Superscript footnote markers would glue onto words ("growth3"); drop them
                if span["size"] < main["size"] * 0.75 and re.fullmatch(r"\s*(?:[\d,*\s]+|[a-z])\s*", span["text"]):
                    continue
                parts.append(span["text"])
            text = _clean("".join(parts))
            if text:
                x0, y0, x1, y1 = raw["bbox"]
                lines.append(Line(x0, y0, x1, y1, text, round(main["size"], 1), main["font"], block_no))

    words = [Word(w[0], w[1], w[2], w[3], _clean(w[4])) for w in page.get_text("words")]
    fills = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        rect = drawing["rect"]
        # Shaded panels (boxes); white fills are just chart backgrounds
        if fill and min(fill) < 0.97 and rect.width > page.rect.width * 0.4 and rect.height > 40:
            fills.append((rect, tuple(round(c, 2) for c in fill)))
    return _Page(number, page.rect.width, page.rect.height, lines, words, fills)


class _DocParser:
    def __init__(self, pages: list[_Page]) -> None:
        self.pages = pages
        all_lines = [line for page in pages for line in page.lines]

        size_chars: Counter = Counter()
        font_chars: Counter = Counter()
        for line in all_lines:
            size_chars[line.size] += len(line.text)
            font_chars[_family(line.font)] += len(line.text)
        self.body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
        self.body_family = font_chars.most_common(1)[0][0] if font_chars else ""

        self.vocab: Counter = Counter()
        for line in all_lines:
            self.vocab.update(t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", line.text))

        self.margin_keys = self._repeating_margin_lines()
        self.segments: list[Segment] = []
        self.main = _Stream("text", [])
        self.boxes: dict[str, _Stream] = {}
        self.open_box: tuple[str, tuple] | None = None
        self.chapter = ""
        self.table_count = 0

    # ----- text helpers -------------------------------------------------------

    def _repeating_margin_lines(self) -> set[str]:
        counts: Counter = Counter()
        for page in self.pages:
            keys = {self._margin_key(l) for l in page.lines if self._in_margin(l, page)}
            counts.update(keys)
        return {key for key, count in counts.items() if count >= 3}

    @staticmethod
    def _margin_key(line: Line) -> str:
        return re.sub(r"\d+", "#", line.text.lower())

    @staticmethod
    def _in_margin(line: Line, page: _Page) -> bool:
        return line.y1 < page.height * 0.08 or line.y0 > page.height * 0.93

    def _fix_words(self, text: str) -> str:
        # Some fonts drop the "Th" ligature: "Te global economy", "Tis is why"
        def repair(match: re.Match) -> str:
            word = match.group(0)
            candidate = "th" + word[1:].lower()
            if self.vocab[candidate] >= 3 and self.vocab[candidate] > 1.5 * self.vocab[word.lower()]:
                return "Th" + word[1:]
            return word

        return re.sub(r"\bT[a-z]+\b", repair, text)

    def _join(self, parts: list[str]) -> str:
        text = ""
        for part in parts:
            if not text:
                text = part
            elif text.endswith("-") and not text.endswith("--") and part[:1].isalpha():
                head = re.search(r"([A-Za-z]+)-$", text)
                tail = re.match(r"[A-Za-z]+", part)
                if head and tail:
                    joined = (head.group(1) + tail.group(0)).lower()
                    hyphenated = f"{head.group(1)}-{tail.group(0)}".lower()
                    # Typesetting hyphen ("com- modity") vs real compound ("AI-related")
                    keep = self.vocab[hyphenated] > self.vocab[joined] or (
                        not self.vocab[joined] and not head.group(1).islower()
                    )
                    text = text + part if keep else text[:-1] + part
                else:
                    text += part
            else:
                text += " " + part
        return self._fix_words(text)

    def _heading_level(self, para: list[Line]) -> float | None:
        """Return the heading's font size (used as its level), or None for body text."""
        text = " ".join(l.text for l in para)
        if len(para) > 3 or len(text) > 120 or text.endswith((".", ",", ";")):
            return None
        sizes = [l.size for l in para]
        size = max(sizes)
        if size > self.body_size * 3:  # cover-page lettering
            return None
        if size >= self.body_size * 1.2:
            return size
        styled = all(
            (_family(l.font) != self.body_family or _is_bold(l.font)) and l.size >= self.body_size * 0.85
            for l in para
        )
        return self.body_size if styled else None

    # ----- page layout --------------------------------------------------------

    @staticmethod
    def _column_split(lines: list[Line], x0: float, x1: float) -> float | None:
        width = x1 - x0
        narrow = [l for l in lines if 30 < l.x1 - l.x0 < width * 0.6]
        if len(narrow) < 6:
            return None
        coverage = {}
        for x in range(int(x0 + width * 0.3), int(x0 + width * 0.7), 2):
            coverage[x] = sum(1 for l in narrow if l.x0 < x < l.x1)
        lowest = min(coverage.values())
        if lowest > len(narrow) * 0.1:
            return None
        gap = [x for x, c in coverage.items() if c == lowest]
        split = gap[len(gap) // 2]
        left = sum(1 for l in narrow if l.x1 <= split)
        right = sum(1 for l in narrow if l.x0 >= split)
        return split if left >= 3 and right >= 3 else None

    def _reading_order(self, lines: list[Line], x0: float, x1: float) -> list[Line]:
        """Top-to-bottom bands split by full-width lines; within a band, left column then right."""
        lines = sorted(lines, key=lambda l: (l.y0, l.x0))
        split = self._column_split(lines, x0, x1)
        if split is None:
            for line in lines:
                line.col = 0
            return lines
        ordered: list[Line] = []
        left: list[Line] = []
        right: list[Line] = []
        for line in lines:
            if line.x0 < split - 5 and line.x1 > split + 5:
                line.col = -1
                ordered += left + right + [line]
                left, right = [], []
            elif line.x1 <= split + 5:
                line.col = 0
                left.append(line)
            else:
                line.col = 1
                right.append(line)
        return ordered + left + right

    @staticmethod
    def _paragraphs(lines: list[Line]) -> list[list[Line]]:
        paras: list[list[Line]] = []
        for line in lines:
            if paras:
                prev = paras[-1][-1]
                gap = line.y0 - prev.y1
                same = (
                    line.col == prev.col
                    and -2 < gap < prev.size * 0.45
                    and abs(line.size - prev.size) < 0.6
                    and _is_bold(line.font) == _is_bold(prev.font)
                )
                if same:
                    paras[-1].append(line)
                    continue
            paras.append([line])
        return paras

    def _is_toc(self, page: _Page) -> bool:
        leaders = sum(1 for l in page.lines if TOC_LEADER_RE.search(l.text))
        return leaders >= 5

    # ----- regions: tables and figures ---------------------------------------

    def _find_regions(self, page: _Page, lines: list[Line]) -> list[_Region]:
        regions = []
        titles = [
            l for l in lines
            if (m := TITLE_RE.match(l.text)) and m.group(1) in ("TABLE", "FIGURE") and _is_bold(l.font)
        ]
        for title in titles:
            kind = TITLE_RE.match(title.text).group(1).lower()
            sources = sorted(
                (l for l in lines if SOURCE_RE.match(l.text) and l.y0 > title.y0 and abs(l.x0 - title.x0) < 30),
                key=lambda l: l.y0,
            )
            if sources:
                source = sources[0]
                bottom, right = source.y1, max(source.x1, title.x1)
                for line in sorted(lines, key=lambda l: l.y0):
                    if (
                        line.y0 > source.y0
                        and line.y0 - bottom < 6
                        and line.x0 >= min(title.x0, source.x0) - 5
                        and line.size <= source.size + 0.5
                    ):
                        bottom, right = line.y1, max(right, line.x1)
            else:
                later = [t.y0 for t in titles if t.y0 > title.y0 + 5 and abs(t.x0 - title.x0) < 30]
                bottom = min(later) - 2 if later else page.height
                if kind == "table":
                    right = max(l.x1 for l in lines)  # untitled-width tables span the text block
                else:
                    right = page.width - 30 if title.x0 > page.width * 0.4 else max(title.x1, page.width * 0.5)
            left = min(title.x0, sources[0].x0 if sources else title.x0)
            # Short "Source: World Bank." lines understate the width; table and chart text is
            # set smaller than body text, so widen over it without swallowing the other column
            right = max(
                [right] + [l.x1 for l in lines if title.y0 <= l.y0 <= bottom and l.x0 >= left - 5 and l.size < self.body_size * 0.95]
            )
            rect = pymupdf.Rect(left - 3, title.y0 - 2, right + 3, bottom + 1)
            regions.append(_Region(rect, kind, title, self._title_label(title.text)))
        return regions

    @staticmethod
    def _title_label(text: str) -> str:
        match = TITLE_RE.match(text)
        rest = re.sub(r"(\s\d{1,2})+$", "", match.group(3)).strip()  # trailing footnote markers
        return f"{match.group(1).title()} {match.group(2)} {rest}".strip()

    def _untitled_table(self, page: _Page, lines: list[Line], words: list[Word]) -> _Region | None:
        """Statistical-appendix style tables with no "TABLE x.y" title."""
        rows = _rows(words)
        for i, row in enumerate(rows):
            if sum(1 for w in row if PERIOD_RE.match(w.text)) < 4:
                continue
            numeric_rows = sum(1 for r in rows[i + 1:] if sum(1 for w in r if VALUE_RE.match(w.text)) >= 3)
            if numeric_rows < 5:
                continue
            header_top = min(w.y0 for w in row)
            heading = [
                l for l in lines
                if l.size >= self.body_size * 1.2 and header_top - 60 < l.y0 < header_top
            ]
            top = min(l.y0 for l in heading) if heading else header_top - 25
            rect = pymupdf.Rect(0, top - 2, page.width, page.height)
            title = heading[0] if heading else None
            label = CONTINUED_RE.sub("", title.text).strip() if title else f"Table on page {page.number}"
            return _Region(rect, "table", title, label)
        return None

    def _table_segments(self, page: _Page, region: _Region, lines: list[Line], words: list[Word], section: str, kind_hint: str) -> None:
        self.table_count += 1
        table_id = f"t{self.table_count}"
        inside = [l for l in lines if _in_rect(region.rect, l.cx, l.cy)]
        title = region.title
        subtitle = [
            l for l in inside
            if title and l is not title and 0 <= l.y0 - title.y1 < 14 and l.text.startswith("(") and abs(l.x0 - title.x0) < 30
        ]
        notes_top = min((l.y0 for l in inside if NOTES_START_RE.match(l.text) and (not title or l.y0 > title.y1)), default=None)
        notes = " ".join(l.text for l in sorted(inside, key=lambda l: (l.y0, l.x0)) if notes_top is not None and l.y0 >= notes_top - 0.5)
        excluded = ([title] if title else []) + subtitle

        def keep(w: Word) -> bool:
            if not _in_rect(region.rect, w.cx, w.cy):
                return False
            if notes_top is not None and w.y0 >= notes_top - 0.5:
                return False
            return not any(_in_rect(pymupdf.Rect(l.x0 - 1, l.y0 - 1, l.x1 + 1, l.y1 + 1), w.cx, w.cy) for l in excluded)

        name = region.label
        if subtitle:
            name += " " + " ".join(l.text for l in subtitle)
        rows, columns = _parse_table([w for w in words if keep(w)], notes)
        common = dict(page_start=page.number, page_end=page.number, section=section, chapter=self.chapter, table_id=table_id)

        overview = name
        if columns:
            overview += ". Columns: " + ", ".join(columns)
        if rows:
            labels = [r[0] for r in rows if r[0]]
            if labels:
                overview += ". Rows: " + ", ".join(dict.fromkeys(labels))
        if notes:
            overview += ". " + notes
        self.segments.append(Segment(overview, "table", standalone=True, **common))
        if not columns:
            # No period columns (country lists, regression output): keep rows together in blocks
            block: list[str] = []
            for _, body in rows + [("", "")]:
                if block and (not body or sum(len(b) for b in block) + len(body) > 1200):
                    self.segments.append(Segment(name + "\n" + "\n".join(block), "table", standalone=True, **common))
                    block = []
                if body:
                    block.append(body)
            return
        for label, body in rows:
            self.segments.append(Segment(f"{name} — {body}", "table", standalone=True, **common))

    def _figure_segment(self, page: _Page, region: _Region, lines: list[Line], section: str) -> None:
        inside = [l for l in lines if _in_rect(region.rect, l.cx, l.cy) and l is not region.title]
        notes_top = min((l.y0 for l in inside if NOTES_START_RE.match(l.text)), default=None)
        text_blocks: dict[int, list[Line]] = defaultdict(list)
        chart_labels: list[str] = []
        notes: list[Line] = []
        for line in inside:
            if notes_top is not None and line.y0 >= notes_top - 0.5:
                notes.append(line)
            elif line.size >= self.body_size * 0.6:
                text_blocks[line.block].append(line)
            elif re.search(r"[A-Za-z]{3,}", line.text) and not AXIS_TICK_RE.match(line.text):
                chart_labels.append(line.text)

        blocks = sorted(text_blocks.values(), key=lambda b: (round(b[0].y0 / 6), b[0].x0))
        parts = [region.label]
        parts += [self._join([l.text for l in block]) for block in blocks]
        if chart_labels:
            parts.append("Chart labels: " + "; ".join(dict.fromkeys(chart_labels)))
        if notes:
            parts.append(self._join([l.text for l in sorted(notes, key=lambda l: (l.y0, l.x0))]))
        self.segments.append(
            Segment("\n".join(parts), "figure", page.number, page.number, section, self.chapter, standalone=True)
        )

    # ----- prose streams ------------------------------------------------------

    def _add_prose(self, stream: _Stream, lines: list[Line], page: int, page_height: float, x0: float, x1: float) -> None:
        for para in self._paragraphs(self._reading_order(lines, x0, x1)):
            text = self._join([l.text for l in para])
            if not re.search(r"\w", text):
                continue  # stray bullets and rules
            level = self._heading_level(para)
            heading = CONTINUED_RE.sub("", text).strip()
            if level is not None and heading:
                heading = heading.title() if heading.isupper() and len(heading) > 3 else heading
                while stream.headings and stream.headings[-1][0] <= level + 0.3:
                    stream.headings.pop()
                stream.headings.append((level, heading))
                if level >= self.body_size * 1.2:
                    stream.in_references = bool(REFERENCES_RE.match(heading))
                stream.last_was_body = False
                continue

            size = max(l.size for l in para)
            if size <= self.body_size * 0.8 and para[0].y0 > page_height * 0.5:
                stream.footnotes.append(Segment(text, "footnote", page, page, stream.section(), self.chapter))
                continue

            kind = "references" if stream.in_references else stream.kind
            prev = stream.paras[-1] if stream.paras else None
            continues = (
                prev is not None
                and stream.last_was_body
                and prev.kind == kind
                and not prev.text.endswith(TERMINAL_PUNCT)
                and (text[:1].islower() or text[:1].isdigit() or prev.text.endswith(("-", ",", "—", "–")))
            )
            if continues:
                prev.text = self._join([prev.text, text])
                prev.page_end = page
            else:
                stream.paras.append(Segment(text, kind, page, page, stream.section(), self.chapter))
            stream.last_was_body = True

    # ----- driver -------------------------------------------------------------

    def run(self) -> list[Segment]:
        for page in self.pages:
            if not page.lines or self._is_toc(page):
                continue
            self._parse_page(page)

        for stream in [self.main, *self.boxes.values()]:
            self.segments.extend(stream.paras)
            self.segments.extend(stream.footnotes)
        return [s for s in self.segments if s.text.strip()]

    def _parse_page(self, page: _Page) -> None:
        lines = []
        for line in page.lines:
            if self._in_margin(line, page) and self._margin_key(line) in self.margin_keys:
                if match := re.search(r"\b(CHAPTER\s+\d+|STATISTICAL APPENDIX|APPENDIX\s*[A-Z0-9]*|ANNEX\s*[A-Z0-9.]*)", line.text, re.I):
                    chapter = match.group(1).title()
                    if chapter != self.chapter:
                        self.chapter = chapter
                continue
            lines.append(line)
        if not lines:
            return

        # Chapter opener pages ("GLOBAL OUTLOOK") start a fresh heading path
        big = [l for l in lines if self.body_size * 2.2 <= l.size <= self.body_size * 3]
        if big and len(lines) <= 8:
            title = " ".join(l.text for l in big)
            self.main.headings = [(max(l.size for l in big), title.title() if title.isupper() else title)]
            self.main.in_references = False
            return

        boxes = self._find_boxes(page, lines)
        regions = self._find_regions(page, lines)
        free_words = [w for w in page.words if not any(_in_rect(r.rect, w.cx, w.cy) for r in regions)]
        free_lines = [l for l in lines if not any(_in_rect(r.rect, l.cx, l.cy) for r in regions)]
        if not any(r.kind == "table" for r in regions):
            untitled = self._untitled_table(page, free_lines, free_words)
            if untitled:
                regions.append(untitled)

        for region in regions:
            box = next((b for b in boxes if _in_rect(b[0], region.rect.x0 + 5, region.rect.y0 + 5)), None)
            section = self.boxes[box[1]].section() if box else self.main.section()
            if region.kind == "table":
                self._table_segments(page, region, lines, page.words, section, "table")
            else:
                self._figure_segment(page, region, lines, section)

        remaining = [l for l in lines if not any(_in_rect(r.rect, l.cx, l.cy) for r in regions)]
        for rect, box_id, title in boxes:
            inside = [l for l in remaining if _in_rect(rect, l.cx, l.cy) and l is not title]
            remaining = [l for l in remaining if not _in_rect(rect, l.cx, l.cy)]
            self._add_prose(self.boxes[box_id], inside, page.number, page.height, rect.x0, rect.x1)

        if remaining:
            x0 = min(l.x0 for l in remaining)
            x1 = max(l.x1 for l in remaining)
            self._add_prose(self.main, remaining, page.number, page.height, x0, x1)

    def _find_boxes(self, page: _Page, lines: list[Line]) -> list[tuple[pymupdf.Rect, str, Line | None]]:
        boxes = []
        for rect, fill in page.fills:
            title = next(
                (l for l in lines if _in_rect(rect, l.cx, l.cy) and (m := TITLE_RE.match(l.text)) and m.group(1) == "BOX"),
                None,
            )
            if title:
                match = TITLE_RE.match(title.text)
                box_id = f"Box {match.group(2)}"
                if box_id not in self.boxes:
                    name = CONTINUED_RE.sub("", self._title_label(title.text)).strip()
                    self.boxes[box_id] = _Stream("box", [s for s in [self.main.section(), name] if s])
                self.open_box = (box_id, fill)
            elif self.open_box and self.open_box[1] == fill:
                box_id = self.open_box[0]  # continuation page without a repeated title
            else:
                continue
            boxes.append((rect, box_id, title))
        return boxes


def _rows(words: list[Word], tolerance: float = 3.0) -> list[list[Word]]:
    rows: list[list[Word]] = []
    for word in sorted(words, key=lambda w: w.cy):
        if rows and word.cy - rows[-1][0].cy <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda w: w.x0) for row in rows]


def _phrases(row: list[Word], gap: float = 10.0) -> list[tuple[float, float, str]]:
    phrases: list[list[Word]] = []
    for word in row:
        if phrases and word.x0 - phrases[-1][-1].x1 <= gap:
            phrases[-1].append(word)
        else:
            phrases.append([word])
    return [(p[0].x0, p[-1].x1, " ".join(w.text for w in p)) for p in phrases]


def _strip_markers(text: str) -> tuple[str, list[str]]:
    tokens = text.split()
    markers = []
    while len(tokens) > 1 and re.fullmatch(r"\d{1,2}", tokens[-1]):
        markers.insert(0, tokens.pop())
    return " ".join(tokens), markers


def _numbered_notes(notes: str) -> dict[str, str]:
    found = {}
    for match in re.finditer(r"(?:^|\s)(\d{1,2})\.\s+(.+?)(?=\s\d{1,2}\.\s+[A-Z“\"(]|$)", notes):
        found.setdefault(match.group(1), match.group(2).strip()[:300])
    return found


def _parse_table(words: list[Word], notes: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Rebuild rows from word positions so every value keeps its column header.

    Returns ([(row label, row text)], column labels). Tables without year-like column
    headers fall back to one text line per row.
    """
    rows = _rows(words)
    header_index = next(
        (i for i, row in enumerate(rows) if sum(1 for w in row if PERIOD_RE.match(w.text)) >= 3), None
    )
    if header_index is None:
        out = []
        for row in rows:
            text = " | ".join(p[2] for p in _phrases(row))
            out.append(("", text))
        return out, []

    note_lookup = _numbered_notes(notes)
    out: list[tuple[str, str]] = []

    def set_columns(header: list[Word]) -> tuple[list[Word], float, float]:
        cols = [w for w in header if PERIOD_RE.match(w.text)]
        centers = [c.cx for c in cols]
        spacing = min((b - a for a, b in zip(centers, centers[1:])), default=40) or 40
        boundary = centers[0] - spacing * 0.6
        return cols, spacing, boundary

    cols, spacing, boundary = set_columns(rows[header_index])
    groups: dict[int, list[str]] = defaultdict(list)

    def apply_groups(phrase_rows: list[list[Word]]) -> None:
        for row in phrase_rows:
            for x0, x1, text in _phrases(row):
                if x0 < boundary - 5:
                    continue
                text = re.sub(r"(\s\d{1,2})+$", "", text)
                for i, col in enumerate(cols):
                    if x0 - 15 <= col.cx <= x1 + 15:
                        groups[i].append(text)

    apply_groups(rows[:header_index])

    def labels() -> list[tuple[str, str]]:
        # Group headers only matter where a period repeats, e.g. a second "2026f" column
        # holding revisions against the previous forecast
        names = [" ".join(groups.get(i, [])) for i in range(len(cols))]
        repeated = {t for t, n in Counter(c.text for c in cols).items() if n > 1}
        return [(col.text, names[i] if col.text in repeated else "") for i, col in enumerate(cols)]

    def render_cells(values: dict[int, str]) -> str:
        parts, current = [], None
        for i in sorted(values):
            period, group = column_labels[i]
            cell = f"{period}: {values[i]}"
            if group != current and group:
                cell = f"| {group}: {cell}"
            current = group
            parts.append(cell)
        return "; ".join(parts).replace("; |", " |")

    column_labels = labels()
    pending_group: list[list[Word]] = []
    parents: list[tuple[float, str]] = []  # indentation hierarchy of row labels
    carried_label = ""

    for row in rows[header_index + 1:]:
        if sum(1 for w in row if PERIOD_RE.match(w.text)) >= 3 and not any(w.cx < boundary for w in row):
            cols, spacing, boundary = set_columns(row)
            groups.clear()
            column_labels = labels()
            continue
        label_words = [w for w in row if w.cx < boundary]
        value_words = [w for w in row if w.cx >= boundary]
        values: dict[int, str] = {}
        stray = []
        for word in value_words:
            if not VALUE_RE.match(word.text):
                stray.append(word)
                continue
            nearest = min(range(len(cols)), key=lambda i: abs(cols[i].cx - word.cx))
            if abs(cols[nearest].cx - word.cx) <= max(10, spacing * 0.55) and nearest not in values:
                values[nearest] = word.text.replace("−", "-").replace("–", "-")
            else:
                stray.append(word)

        label, markers = _strip_markers(" ".join(w.text for w in label_words))
        indent = label_words[0].x0 if label_words else boundary
        if len(values) <= 1 and sum(1 for w in stray if re.search(r"[A-Za-z]{2,}", w.text)) >= 3:
            continue  # an explanatory line inside the table, not a data row
        if not values:
            if label_words and out and re.search(r"\b(and|the|of|in|for|&)$|[,&-]$", out[-1]["label"]):
                # Second line of a wrapped label: "Latin America and the" + "Caribbean"
                out[-1]["label"] += " " + label
                out[-1]["markers"] += markers
                if parents and parents[-1][1] == out[-1]["label"][: -len(label) - 1]:
                    parents[-1] = (parents[-1][0], out[-1]["label"])
            elif label_words:
                carried_label = label  # wrapped label, or a group heading for the rows below
                while parents and parents[-1][0] >= indent - 2:
                    parents.pop()
                parents.append((indent, label.rstrip(":")))
            elif stray:
                pending_group.append(stray)  # a column-group header in the middle of the table
            continue
        if pending_group:
            groups.clear()
            apply_groups(pending_group)
            column_labels = labels()
            pending_group = []
        if not label:
            label = carried_label
            indent = parents[-1][0] if parents else indent
            if parents and parents[-1][1] == carried_label.rstrip(":"):
                parents.pop()
        carried_label = ""

        while parents and parents[-1][0] >= indent - 2:
            parents.pop()
        cells = render_cells(values)
        out.append({"parents": [p[1] for p in parents], "label": label, "cells": cells, "markers": markers})
        parents.append((indent, label))

    rendered = []
    for row in out:
        text = " > ".join(row["parents"] + [row["label"]]) + ": " + row["cells"]
        for marker in row["markers"]:
            if marker in note_lookup:
                text += f" (Note {marker}: {note_lookup[marker]})"
        rendered.append((row["label"], text))
    return rendered, [f"{period} ({group})" if group else period for period, group in column_labels]


def parse_pdf(path: Path) -> list[Segment]:
    with pymupdf.open(str(path)) as doc:
        pages = [_read_page(page, number) for number, page in enumerate(doc, start=1)]
    return _DocParser(pages).run()
