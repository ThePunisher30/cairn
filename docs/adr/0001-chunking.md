# ADR 0001: How the corpus is chunked

Status: accepted, 2026-10-20. Evidence: `docs/results.md`, section "Phase 2".

## Context

Retrieval returns chunks, so how the FastAPI corpus is cut decides what can be found. A cut through an
answer means no retriever can return it whole. I compared nine strategies on the 40 dev questions with one
metric: gold integrity, the share of gold spans that fit inside a single chunk. Chunk size matters too:
the embedding model planned for Phase 4 reads at most 512 tokens.

## Decision

- **Corpus scope:** `docs/en/docs/**/*.md` without `release-notes.md`, plus `fastapi/**/*.py` and
  `docs_src/**/*.py`: 671 documents. Translations and the repository's own `tests/` are excluded.
- **Docs (prose):** `MarkdownChunker`, budget 300 tokens. One chunk per heading section, with the heading
  path stored (and added to the search text); a section over budget is split by paragraph.
- **Code:** `PythonChunker`, based on the AST. One chunk per function or method, a header chunk per class,
  runs of top-level statements grouped, and `FixedChunker(200)` as the fallback for a file that does not parse.
- **Measured:** 3,267 chunks, mean 133 tokens, 0.88 of dev gold spans intact (fixed-400: 0.78, with chunks
  twice as large).
- Chunk locations are file path plus line range, so one answer key works for every chunker.

## Consequences

- 76 code chunks are over 512 tokens (largest: `FastAPI.__init__`, 7,124 tokens). They are fine for BM25
  but would be truncated by an embedding model. Revisit in Phase 4: split big functions, or embed a
  shortened form.
- Code intact is 0.70 at every size: questions about a whole class or a whole example file are cut by the
  AST rule. Revisit with parent-child retrieval (find a small chunk, return its parent).
- Gold span q005 ends on a blank line and can never be intact; the answer key is unchanged and the flaw is
  recorded in `docs/results.md`.
- The 0.88 is measured on dev questions. The test split has not been used.
- Line-range locations will not fit PDFs or database rows; supporting more formats means generalising the
  chunk location type.

## Alternatives considered

- **Fixed windows (100-400 tokens):** simple and uniform, but they ignore structure. Best was 0.78 at 400
  tokens, with chunks more than twice the size, and only 0.50 on semantic questions.
- **Recursive splitting (paragraphs, then lines), with or without overlap:** better than fixed at equal size
  (0.80 at 400) but still blind to headings and functions. Overlap added 0.06 intact for 18% more text.
- **Larger structure budget (600):** 0.92 intact, but the gain comes only from two over-budget docs
  sections, and it adds 29 more chunks over 512 tokens (105 against 76).
- **Smaller structure budget (150):** 0.66; sections get cut too often.
