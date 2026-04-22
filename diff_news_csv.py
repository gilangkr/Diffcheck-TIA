#!/usr/bin/env python3
"""Build an HTML report that highlights edits in headlines and content."""

from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "TIA news tracker - news by AI.csv"
DEFAULT_OUTPUT = REPO_ROOT / "news_diff_report.html"

# Run this from the project folder with:
# python3 scripts/diff_news_csv.py
# That reads the default CSV in this repository and writes the HTML report to
# news_diff_report.html in the project folder.
#
# To use different files, run:
# python3 scripts/diff_news_csv.py --input "/full/path/to/input.csv" --output "/full/path/to/output.html"
#
# To see the available command-line options, run:
# python3 scripts/diff_news_csv.py --help

# This pattern keeps tags, spaces, words, and punctuation as separate pieces
# so the report can show exactly what changed without flattening the sentence.
TOKEN_PATTERN = re.compile(r"(<[^>]+?>|\s+|\w+|[^\w\s])", re.UNICODE)
P_OPEN_PATTERN = re.compile(r"<p\b[^>]*>", re.IGNORECASE)
P_CLOSE_PATTERN = re.compile(r"</p\s*>", re.IGNORECASE)
IFRAME_BLOCK_PATTERN = re.compile(r"<iframe\b[^>]*>.*?</iframe\s*>", re.IGNORECASE | re.DOTALL)
TIMELINE_SECTION_PATTERN = re.compile(
    r"<h2\b[^>]*>\s*Recent .*? developments\s*</h2>\s*(?:<p\b[^>]*>\s*)?"
    r"<iframe\b[^>]*>.*?</iframe\s*>(?:\s*</p>)?",
    re.IGNORECASE | re.DOTALL,
)
TRAILING_TIMELINE_HEADING_PATTERN = re.compile(
    r"<h2\b[^>]*>\s*Recent .*? developments\s*</h2>\s*(?:<p\b[^>]*>\s*</p>\s*)?$",
    re.IGNORECASE | re.DOTALL,
)
TIMELINE_HEADING_PATTERN = re.compile(
    r"<h2\b[^>]*>\s*Recent .*? developments\s*</h2>",
    re.IGNORECASE | re.DOTALL,
)
SOURCE_STYLE_PATTERN = re.compile(
    r"<style\b[^>]*>\s*\.source-ref\s*\{.*?\}\s*</style\s*>",
    re.IGNORECASE | re.DOTALL,
)
SOURCE_EM_PATTERN = re.compile(
    r"<em\b[^>]*class=(['\"])[^'\"]*\bsource-ref\b[^'\"]*\1[^>]*>.*?</em\s*>",
    re.IGNORECASE | re.DOTALL,
)
SOURCE_PARAGRAPH_PATTERN = re.compile(
    r"<p\b[^>]*>\s*(?:<em\b[^>]*class=(['\"])[^'\"]*\bsource-ref\b[^'\"]*\1[^>]*>.*?</em\s*>)\s*</p\s*>",
    re.IGNORECASE | re.DOTALL,
)
SPAN_OPEN_PATTERN = re.compile(r"<span\b[^>]*>", re.IGNORECASE)
SPAN_CLOSE_PATTERN = re.compile(r"</span\s*>", re.IGNORECASE)
TARGET_BLANK_PATTERN = re.compile(r'\s+target=(["\'])_blank\1', re.IGNORECASE)
REL_NOOPENER_PATTERN = re.compile(
    r'\s+rel=(["\'])noopener(?:\s+noreferrer)?\1', re.IGNORECASE
)
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
FOOD_FOR_THOUGHT_WRAPPER_PATTERN = re.compile(
    r'<div\b[^>]*id\s*=\s*["\']{1,2}food-for-thought-wrapper["\']{1,2}[^>]*>',
    re.IGNORECASE,
)


@dataclass
class RowResult:
    timestamp: str
    url: str
    original_content_html: str
    final_content_html: str
    content_diff_html: str
    changed: bool


def sanitize_href(href: str) -> str | None:
    """Allow only regular web links in rendered source cells."""
    parsed = urlparse(href)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return href
    return None


def normalize_display_text(text: str) -> str:
    # Timeline iframes are removed everywhere so the report only shows article text.
    # Paragraph tags are also flattened so each column uses the same spacing.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = TIMELINE_SECTION_PATTERN.sub("", normalized)
    normalized = IFRAME_BLOCK_PATTERN.sub("", normalized)
    normalized = TRAILING_TIMELINE_HEADING_PATTERN.sub("", normalized)
    normalized = TIMELINE_HEADING_PATTERN.sub("", normalized)
    normalized = TARGET_BLANK_PATTERN.sub("", normalized)
    normalized = REL_NOOPENER_PATTERN.sub("", normalized)
    normalized = SPAN_OPEN_PATTERN.sub("", normalized)
    normalized = SPAN_CLOSE_PATTERN.sub("", normalized)
    without_open = P_OPEN_PATTERN.sub("", normalized)
    with_breaks = P_CLOSE_PATTERN.sub("\n", without_open)
    tightened = re.sub(r"[ \t]+\n", "\n", with_breaks)
    tightened = re.sub(r"\n[ \t]+", "\n", tightened)
    return EXCESS_NEWLINES_PATTERN.sub("\n\n", tightened).strip()


def normalize_diff_text(text: str) -> str:
    # Source reference boilerplate is ignored in the diff so the comparison stays
    # focused on the article changes instead of the citation footer.
    normalized = normalize_display_text(text)
    normalized = SOURCE_STYLE_PATTERN.sub("", normalized)
    normalized = SOURCE_PARAGRAPH_PATTERN.sub("", normalized)
    normalized = SOURCE_EM_PATTERN.sub("", normalized)
    return EXCESS_NEWLINES_PATTERN.sub("\n\n", normalized).strip()


class SafeSnippetRenderer(HTMLParser):
    """Render snippets safely while keeping normal web links clickable."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.anchor_stack: list[bool] = []

    # This turns article snippets into safe HTML. Plain text stays plain text,
    # and only normal web links are kept as clickable links.
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            self.parts.append(html.escape(self.get_starttag_text() or f"<{tag}>"))
            return

        href = ""
        for name, value in attrs:
            if name.lower() == "href" and value:
                href = value
                break

        safe_href = sanitize_href(href)
        if safe_href:
            escaped_href = html.escape(safe_href, quote=True)
            self.parts.append(
                f'<a href="{escaped_href}" target="_blank" rel="noopener noreferrer">'
            )
            self.anchor_stack.append(True)
            return

        self.parts.append(html.escape(self.get_starttag_text() or "<a>"))
        self.anchor_stack.append(False)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.anchor_stack:
            if self.anchor_stack.pop():
                self.parts.append("</a>")
            else:
                self.parts.append("&lt;/a&gt;")
            return

        self.parts.append(html.escape(f"</{tag}>"))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(html.escape(self.get_starttag_text() or f"<{tag} />"))

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.parts.append(html.escape(f"<!--{data}-->"))

    def handle_decl(self, decl: str) -> None:
        self.parts.append(html.escape(f"<!{decl}>"))

    def close(self) -> str:  # type: ignore[override]
        super().close()
        while self.anchor_stack:
            if self.anchor_stack.pop():
                self.parts.append("</a>")
        return "".join(self.parts)


class VisibleTextExtractor(HTMLParser):
    """Collect only human-visible text from HTML for word counting."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    # Block-level tags need separators so adjacent paragraphs do not merge into one word.
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t]+\n", "\n", joined)
        joined = re.sub(r"\n[ \t]+", "\n", joined)
        return EXCESS_NEWLINES_PATTERN.sub("\n\n", joined).strip()


def render_text(text: str) -> str:
    text = normalize_display_text(text)
    renderer = SafeSnippetRenderer()
    renderer.feed(text)
    return renderer.close()


def tokenize(text: str) -> list[str]:
    text = normalize_diff_text(text)
    return TOKEN_PATTERN.findall(text)


def canonicalize_diff_token(token: str) -> str:
    # Curly apostrophes should not count as content edits in the diff view.
    return token.replace("’", "'")


def canonicalize_tag_token(token: str) -> str:
    # Tag attributes often change without affecting what readers see, so the
    # diff only compares the tag shape itself.
    match = re.match(r"<\s*(/?)\s*([a-zA-Z0-9:-]+)", token)
    if not match:
        return token

    slash, tag_name = match.groups()
    normalized_name = tag_name.lower()
    return f"</{normalized_name}>" if slash else f"<{normalized_name}>"


def normalize_display_diff_markup(text: str) -> str:
    # Entity-only text changes like R&D versus R&amp;D should compare the same
    # while still letting the diff column render HTML tags as code.
    return html.unescape(normalize_diff_text(text))


def build_display_diff_tokens(text: str) -> list[tuple[str, str]]:
    # The diff view still shows HTML code, but comparison ignores tag-attribute
    # noise and apostrophe style changes.
    tokens = TOKEN_PATTERN.findall(normalize_display_diff_markup(text))
    display_tokens: list[tuple[str, str]] = []
    for token in tokens:
        if token.startswith("<") and token.endswith(">"):
            compare_key = canonicalize_tag_token(token)
        else:
            compare_key = canonicalize_diff_token(token)
        display_tokens.append((compare_key, token))
    return display_tokens


def extract_visible_diff_text(text: str) -> str:
    # Diff decisions should follow what a reader can actually see, not HTML tag
    # formatting or attribute changes that leave the visible copy untouched.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = TIMELINE_SECTION_PATTERN.sub("", normalized)
    normalized = IFRAME_BLOCK_PATTERN.sub("", normalized)
    normalized = TRAILING_TIMELINE_HEADING_PATTERN.sub("", normalized)
    normalized = TIMELINE_HEADING_PATTERN.sub("", normalized)
    normalized = SOURCE_STYLE_PATTERN.sub("", normalized)
    normalized = SOURCE_PARAGRAPH_PATTERN.sub("", normalized)
    normalized = SOURCE_EM_PATTERN.sub("", normalized)

    extractor = VisibleTextExtractor()
    extractor.feed(normalized)
    extractor.close()
    return extractor.text()


def tokenize_visible_diff_text(text: str) -> list[str]:
    return [
        canonicalize_diff_token(token)
        for token in TOKEN_PATTERN.findall(extract_visible_diff_text(text))
    ]


def extract_news_body_text(body: str) -> str:
    # Word counts only include visible news-body text before the food-for-thought section.
    match = FOOD_FOR_THOUGHT_WRAPPER_PATTERN.search(body)
    if match:
        body = body[: match.start()]
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = TIMELINE_SECTION_PATTERN.sub("", body)
    body = IFRAME_BLOCK_PATTERN.sub("", body)
    body = TRAILING_TIMELINE_HEADING_PATTERN.sub("", body)
    body = TIMELINE_HEADING_PATTERN.sub("", body)
    body = SOURCE_STYLE_PATTERN.sub("", body)
    body = SOURCE_PARAGRAPH_PATTERN.sub("", body)
    body = SOURCE_EM_PATTERN.sub("", body)

    extractor = VisibleTextExtractor()
    extractor.feed(body)
    extractor.close()
    return extractor.text()


def count_news_words(_headline: str, body: str) -> int:
    # News word counts only cover the article body before the food-for-thought
    # section starts. Headlines are intentionally excluded.
    news_body_text = extract_news_body_text(body)
    return len(re.findall(r"\w+", news_body_text, re.UNICODE))


def render_article_cell(headline: str, body: str, news_word_count: int) -> str:
    headline_text = normalize_display_text(headline)
    body_text = normalize_display_text(body)
    parts: list[str] = []

    # This badge makes it clear that the count only covers the news section.
    parts.append(
        '<div class="article-stats">'
        f'<span class="word-count">News words: {news_word_count}</span>'
        "</div>"
    )

    # The headline is shown separately so it can be bold without changing the
    # spacing used for the body copy underneath it.
    if headline_text:
        parts.append(
            f'<div class="article-headline">{render_diff_tokens(tokenize(headline_text))}</div>'
        )
    if body_text:
        parts.append(f'<div class="article-body">{render_diff_tokens(tokenize(body_text))}</div>')

    return "".join(parts)


def render_diff_article_cell(headline: str, body: str) -> str:
    parts: list[str] = []

    # The diff column uses an invisible spacer so the headline and first body
    # line start at the same vertical position as the source columns.
    parts.append(
        '<div class="article-stats article-stats-spacer" aria-hidden="true">'
        '<span class="word-count">News words: 000</span>'
        "</div>"
    )

    # The diff headline uses the same structure as the source columns so the
    # first line stands out as the headline instead of blending into the body.
    if headline:
        parts.append(f'<div class="article-headline">{headline}</div>')
    if body:
        parts.append(f'<div class="article-body">{body}</div>')

    return "".join(parts)


def render_diff_tokens(tokens: list[str]) -> str:
    rendered: list[str] = []
    for token in tokens:
        if token.startswith("<") and token.endswith(">"):
            rendered.append(f'<span class="html-tag">{html.escape(token)}</span>')
        else:
            rendered.append(html.escape(token))
    return "".join(rendered)


def build_diff_html(original: str, final: str) -> tuple[str, bool]:
    original_tokens = tokenize_visible_diff_text(original)
    final_tokens = tokenize_visible_diff_text(final)

    # If the reader-visible text is the same, markup-only drift should not be
    # shown as an edit in the report.
    if original_tokens == final_tokens:
        final_display_tokens = build_display_diff_tokens(final)
        return render_diff_tokens([token for _, token in final_display_tokens]), False

    original_display_tokens = build_display_diff_tokens(original)
    final_display_tokens = build_display_diff_tokens(final)
    matcher = SequenceMatcher(
        a=[compare_key for compare_key, _ in original_display_tokens],
        b=[compare_key for compare_key, _ in final_display_tokens],
        autojunk=False,
    )

    parts: list[str] = []
    changed = False

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            parts.append(
                render_diff_tokens([token for _, token in final_display_tokens[j1:j2]])
            )
            continue

        changed = True
        if opcode in {"delete", "replace"} and i1 != i2:
            parts.append(
                "<del>"
                f"{render_diff_tokens([token for _, token in original_display_tokens[i1:i2]])}"
                "</del>"
            )
        if opcode in {"insert", "replace"} and j1 != j2:
            parts.append(
                "<ins>"
                f"{render_diff_tokens([token for _, token in final_display_tokens[j1:j2]])}"
                "</ins>"
            )

    return "".join(parts), changed


def build_rows(csv_path: Path) -> tuple[list[RowResult], dict[str, int]]:
    rows: list[RowResult] = []
    total_rows = 0
    processed_rows = 0
    changed_rows = 0
    total_original_news_words = 0
    total_final_news_words = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"headline", "content", "final_headline", "final_content"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing = ", ".join(sorted(required_columns - set(reader.fieldnames or [])))
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            total_rows += 1
            original_headline = row.get("headline", "") or ""
            final_headline = row.get("final_headline", "") or ""
            original_content = row.get("content", "") or ""
            final_content = row.get("final_content", "") or ""

            # This report only includes rows that have a final version to compare against.
            if not normalize_display_text(final_content):
                continue

            processed_rows += 1
            original_news_words = count_news_words(original_headline, original_content)
            final_news_words = count_news_words(final_headline, final_content)
            total_original_news_words += original_news_words
            total_final_news_words += final_news_words
            headline_diff_html, headline_changed = build_diff_html(
                original_headline, final_headline
            )
            body_diff_html, body_changed = build_diff_html(
                original_content, final_content
            )
            content_diff_html = render_diff_article_cell(
                headline_diff_html, body_diff_html
            )
            changed = headline_changed or body_changed
            if changed:
                changed_rows += 1

            rows.append(
                RowResult(
                    timestamp=(row.get("datetime", "") or "").strip(),
                    url=(row.get("URL", "") or "").strip(),
                    original_content_html=render_article_cell(
                        original_headline, original_content, original_news_words
                    ),
                    final_content_html=render_article_cell(
                        final_headline, final_content, final_news_words
                    ),
                    content_diff_html=content_diff_html,
                    changed=changed,
                )
            )

    average_original_news_words = (
        total_original_news_words / processed_rows if processed_rows else 0.0
    )
    average_final_news_words = total_final_news_words / processed_rows if processed_rows else 0.0

    summary = {
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "changed_rows": changed_rows,
        "unchanged_rows": processed_rows - changed_rows,
        "average_original_news_words": round(average_original_news_words, 1),
        "average_final_news_words": round(average_final_news_words, 1),
    }
    return rows, summary


def render_report(rows: list[RowResult], summary: dict[str, int], source_path: Path) -> str:
    row_html: list[str] = []
    for row in rows:
        timestamp = html.escape(row.timestamp or "")
        url_html = ""
        if row.url:
            escaped_url = html.escape(row.url, quote=True)
            visible_url = html.escape(row.url)
            url_html = (
                f'<div class="meta-link"><a href="{escaped_url}" '
                f'target="_blank" rel="noopener noreferrer">{visible_url}</a></div>'
            )

        row_html.append(
            f"""
            <tr class="{'changed' if row.changed else 'unchanged'}">
              <td class="meta" data-label="Meta">
                <div class="timestamp">{timestamp}</div>
                {url_html}
              </td>
              <td class="content-cell" data-label="Original">{row.original_content_html}</td>
              <td class="content-cell" data-label="Final">{row.final_content_html}</td>
              <td class="content-cell diff-cell" data-label="Diff">{row.content_diff_html}</td>
            </tr>
            """
        )

    source_name = html.escape(str(source_path.name))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>News Content Diff Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: rgba(255, 255, 255, 0.88);
      --panel-strong: #ffffff;
      --ink: #17202a;
      --muted: #64748b;
      --line: rgba(148, 163, 184, 0.24);
      --line-strong: rgba(148, 163, 184, 0.4);
      --changed: rgba(241, 245, 249, 0.78);
      --unchanged: rgba(255, 255, 255, 0.7);
      --remove: rgba(239, 68, 68, 0.14);
      --add: rgba(34, 197, 94, 0.16);
      --tag: #9a3412;
      --shadow: 0 22px 54px rgba(15, 23, 42, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(148, 163, 184, 0.18), transparent 34%),
        linear-gradient(180deg, #fbfcfd 0%, var(--bg) 100%);
    }}

    main {{
      width: min(95vw, 1680px);
      margin: 0 auto;
      padding: 32px 0 44px;
    }}

    .hero {{
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: clamp(1.8rem, 2.8vw, 2.6rem);
      letter-spacing: -0.03em;
    }}

    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 0.98rem;
      max-width: 78ch;
      line-height: 1.65;
    }}

    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 20px;
    }}

    .summary-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255, 255, 255, 0.7);
    }}

    .summary-card strong {{
      display: block;
      font-size: 1.55rem;
      margin-bottom: 4px;
      letter-spacing: -0.03em;
    }}

    .table-wrap {{
      overflow: visible;
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}

    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
    }}

    col.meta {{ width: 15%; }}
    col.content {{ width: 27%; }}
    col.content-diff {{ width: 31%; }}

    thead th {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(248, 250, 252, 0.92);
      backdrop-filter: blur(16px);
      text-align: left;
      vertical-align: top;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    thead th:first-child {{
      border-top-left-radius: 24px;
    }}

    thead th:last-child {{
      border-top-right-radius: 24px;
    }}

    .header-title {{
      display: block;
    }}

    .header-metric {{
      display: block;
      margin-top: 0.45rem;
      font-size: 0.76rem;
      font-weight: 600;
      letter-spacing: normal;
      text-transform: none;
      color: #475569;
    }}

    tbody tr.changed {{
      background: var(--changed);
    }}

    tbody tr.unchanged {{
      background: var(--unchanged);
    }}

    td {{
      vertical-align: top;
      padding: 18px;
      border-top: 1px solid var(--line);
    }}

    .timestamp,
    .meta-link {{
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }}

    .timestamp {{
      font-weight: 600;
      color: #334155;
      margin-bottom: 6px;
    }}

    .meta-link a,
    .content-cell a {{
      color: #0f5b84;
      text-decoration: none;
    }}

    .meta-link a:hover,
    .content-cell a:hover {{
      text-decoration: underline;
    }}

    .content-cell {{
      white-space: pre-wrap;
      line-height: 1.65;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    .article-stats {{
      margin: 0 0 0.8rem;
    }}

    .article-stats-spacer {{
      visibility: hidden;
    }}

    .word-count {{
      display: inline-flex;
      align-items: center;
      min-height: 1.9rem;
      padding: 0.18rem 0.72rem;
      border-radius: 999px;
      background: rgba(226, 232, 240, 0.65);
      color: #334155;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}

    .article-headline {{
      font-weight: 700;
      font-size: 1rem;
      line-height: 1.45;
      margin: 0 0 calc(0.45rem + var(--headline-extra-gap, 0px));
      white-space: normal;
    }}

    .article-body {{
      margin: 0;
      color: #334155;
    }}

    .diff-cell del {{
      text-decoration: line-through;
      background: var(--remove);
      color: #7f1d1d;
      padding: 0 1px;
    }}

    .diff-cell ins {{
      text-decoration: none;
      background: var(--add);
      color: #14532d;
      padding: 0 1px;
    }}

    .html-tag {{
      color: var(--tag);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.9em;
    }}

    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
      background: rgba(226, 232, 240, 0.7);
      padding: 0.15rem 0.35rem;
      border-radius: 6px;
    }}

    @media (max-width: 1100px) {{
      .table-wrap {{
        padding: 8px;
      }}

      table,
      thead,
      tbody,
      tr,
      td {{
        display: block;
        width: 100%;
      }}

      colgroup {{
        display: none;
      }}

      thead {{
        position: sticky;
        top: 0;
        z-index: 6;
      }}

      thead tr {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
      }}

      thead th {{
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
      }}

      tbody {{
        margin-top: 10px;
      }}

      tbody tr {{
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 12px;
        background: rgba(255, 255, 255, 0.76);
      }}

      td {{
        border-top: 1px solid var(--line);
        padding: 14px 16px;
      }}

      td:first-child {{
        border-top: 0;
      }}

      td::before {{
        content: attr(data-label);
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}
    }}

    @media (max-width: 900px) {{
      main {{
        width: min(98vw, 1680px);
        padding-top: 18px;
      }}

      .hero,
      .table-wrap {{
        border-radius: 18px;
      }}

      .hero {{
        padding: 22px 20px;
      }}

      thead tr {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 680px) {{
      thead tr {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>News Content Diff Report</h1>
      <p class="subtitle">
        Comparing the combined original headline and content against the combined
        final headline and content from <code>{source_name}</code>.
        <code>&lt;p&gt;</code> tags are ignored, and news word counts stop before
        <code>food-for-thought-wrapper</code>.
      </p>
      <div class="summary">
        <div class="summary-card"><strong>{summary['total_rows']}</strong>Total CSV rows</div>
        <div class="summary-card"><strong>{summary['processed_rows']}</strong>Rows with final content</div>
        <div class="summary-card"><strong>{summary['changed_rows']}</strong>Rows with edits</div>
        <div class="summary-card"><strong>{summary['unchanged_rows']}</strong>Rows without edits</div>
      </div>
    </section>

    <section class="table-wrap">
      <table id="diff-table">
        <colgroup>
          <col class="meta">
          <col class="content">
          <col class="content">
          <col class="content-diff">
        </colgroup>
        <thead>
          <tr>
            <th>
              <span class="header-title">Meta</span>
            </th>
            <th>
              <span class="header-title">Original</span>
              <span class="header-metric">
                Avg news words: {summary['average_original_news_words']}
              </span>
            </th>
            <th>
              <span class="header-title">Final</span>
              <span class="header-metric">
                Avg news words: {summary['average_final_news_words']}
              </span>
            </th>
            <th>
              <span class="header-title">Diff</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </section>
  </main>
  <script>
    (() => {{
      const desktopLayout = window.matchMedia('(min-width: 1101px)');

      const alignSourceBodyCopy = () => {{
        const rows = document.querySelectorAll('tbody tr');

        rows.forEach((row) => {{
          const sourceHeadlines = row.querySelectorAll('td.content-cell:not(.diff-cell) .article-headline');
          sourceHeadlines.forEach((headline) => {{
            headline.style.setProperty('--headline-extra-gap', '0px');
          }});
        }});

        if (!desktopLayout.matches) {{
          return;
        }}

        rows.forEach((row) => {{
          const diffBody = row.querySelector('.diff-cell .article-body');
          if (!diffBody) {{
            return;
          }}

          const diffTop = diffBody.getBoundingClientRect().top;
          const sourceCells = row.querySelectorAll('td.content-cell:not(.diff-cell)');

          sourceCells.forEach((cell) => {{
            const headline = cell.querySelector('.article-headline');
            const body = cell.querySelector('.article-body');
            if (!headline || !body) {{
              return;
            }}

            const sourceTop = body.getBoundingClientRect().top;
            const delta = Math.max(diffTop - sourceTop, 0);
            headline.style.setProperty('--headline-extra-gap', `${{delta}}px`);
          }});
        }});
      }};

      window.addEventListener('load', alignSourceBodyCopy);
      window.addEventListener('resize', alignSourceBodyCopy);
      desktopLayout.addEventListener('change', alignSourceBodyCopy);
    }})();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML diff report for combined headline and content text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV file to read (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML file to write (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(args.input)
    report_html = render_report(rows, summary, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_html, encoding="utf-8")

    print(f"Wrote report to: {args.output}")
    print(f"Total CSV rows: {summary['total_rows']}")
    print(f"Rows with final content: {summary['processed_rows']}")
    print(f"Rows with edits: {summary['changed_rows']}")
    print(f"Rows without edits: {summary['unchanged_rows']}")


if __name__ == "__main__":
    main()
