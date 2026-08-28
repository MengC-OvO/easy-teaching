import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.schemas import IngestionResult, KnowledgeChunk, KnowledgeDocument, KnowledgeSourceType


class KnowledgeSourceSpec(BaseModel):
    source_id: str = Field(min_length=1)
    source_type: KnowledgeSourceType
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    uri: Optional[str] = None
    path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    notes: Optional[str] = None

    def to_document(self) -> KnowledgeDocument:
        return KnowledgeDocument(
            source_id=self.source_id,
            source_type=self.source_type,
            title=self.title,
            version=self.version,
            uri=self.uri,
        )


class ParsedTextBlock(BaseModel):
    content: str
    section: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class KnowledgeIngestionService:
    """Layout-aware parsing followed by heading and paragraph-aware chunking."""

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        target_tokens: int = 350,
        max_tokens: int = 500,
        min_tokens: int = 80,
        overlap_tokens: int = 40,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.project_root = project_root or Path.cwd()
        # Keep old character-based constructor arguments working during migration.
        if chunk_size is not None:
            max_tokens = max(1, chunk_size // 4)
            target_tokens = min(target_tokens, max_tokens)
        if chunk_overlap is not None:
            overlap_tokens = max(0, chunk_overlap // 4)
        if not 0 <= overlap_tokens < target_tokens <= max_tokens:
            raise ValueError("chunk settings must satisfy 0 <= overlap < target <= max")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min(min_tokens, target_tokens)
        self.overlap_tokens = overlap_tokens

    def load_sources(self, manifest_path: Path) -> List[KnowledgeSourceSpec]:
        raw_sources = json.loads(self._resolve_path(manifest_path).read_text(encoding="utf-8"))
        return [KnowledgeSourceSpec.model_validate(source) for source in raw_sources]

    def ingest_all(
        self,
        manifest_path: Path,
        *,
        max_pages_per_pdf: Optional[int] = None,
    ) -> List[IngestionResult]:
        return [
            self.ingest_source(source, max_pages_per_pdf=max_pages_per_pdf)
            for source in self.load_sources(manifest_path)
        ]

    def write_chunks_jsonl(
        self,
        results: Iterable[IngestionResult],
        output_path: Path,
    ) -> int:
        resolved_output = self._resolve_path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        seen_chunk_ids: set[str] = set()
        with resolved_output.open("w", encoding="utf-8") as file:
            for result in results:
                for chunk in result.chunks:
                    if chunk.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.chunk_id)
                    file.write(chunk.model_dump_json() + "\n")
                    written += 1
        return written

    def read_chunks_jsonl(self, input_path: Path) -> List[KnowledgeChunk]:
        resolved_input = self._resolve_path(input_path)
        chunks: List[KnowledgeChunk] = []
        seen_chunk_ids: set[str] = set()
        with resolved_input.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    chunk = KnowledgeChunk.model_validate_json(line)
                    if chunk.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.chunk_id)
                    chunks.append(chunk)
        return chunks

    def ingest_source(
        self,
        source: KnowledgeSourceSpec,
        *,
        max_pages_per_pdf: Optional[int] = None,
    ) -> IngestionResult:
        document = source.to_document()
        blocks = self._parse_source(
            source,
            self._resolve_path(Path(source.path)),
            max_pages_per_pdf=max_pages_per_pdf,
        )
        chunks = self._blocks_to_chunks(document, blocks)
        return IngestionResult(
            source_id=source.source_id,
            chunk_count=len(chunks),
            chunks=chunks,
        )

    def ingest_blocks(
        self,
        document: KnowledgeDocument,
        blocks: Iterable[ParsedTextBlock],
    ) -> IngestionResult:
        """Chunk already-parsed content without exposing private parser internals."""
        chunks = self._blocks_to_chunks(document, blocks)
        return IngestionResult(
            source_id=document.source_id,
            chunk_count=len(chunks),
            chunks=chunks,
        )

    def _parse_source(
        self,
        source: KnowledgeSourceSpec,
        path: Path,
        *,
        max_pages_per_pdf: Optional[int],
    ) -> List[ParsedTextBlock]:
        if source.format == "markdown":
            return self._parse_markdown(path)
        if source.format == "pdf":
            return self._parse_pdf(path, max_pages=max_pages_per_pdf)
        raise ValueError(f"Unsupported knowledge source format: {source.format}")

    def _parse_markdown(self, path: Path) -> List[ParsedTextBlock]:
        blocks, _ = self._markdown_blocks(
            path.read_text(encoding="utf-8"),
            page=None,
            initial_section="Document introduction",
            parser="markdown",
        )
        return blocks

    def _parse_pdf(self, path: Path, *, max_pages: Optional[int]) -> List[ParsedTextBlock]:
        try:
            import pymupdf
            import pymupdf4llm
        except ImportError as error:
            raise RuntimeError(
                "PDF ingestion requires pymupdf4llm. Install project dependencies first."
            ) from error

        with pymupdf.open(path) as document:
            page_count = document.page_count
        limit = min(page_count, max_pages) if max_pages is not None else page_count
        page_chunks = pymupdf4llm.to_markdown(
            str(path),
            pages=list(range(limit)),
            page_chunks=True,
            use_ocr=False,
            force_ocr=False,
            force_text=True,
            header=False,
            footer=False,
            write_images=False,
            embed_images=False,
            show_progress=False,
        )

        blocks: List[ParsedTextBlock] = []
        current_section = "Document introduction"
        for index, page_chunk in enumerate(page_chunks, start=1):
            metadata = page_chunk.get("metadata") or {}
            page_number = int(metadata.get("page_number") or index)
            page_blocks, current_section = self._markdown_blocks(
                page_chunk.get("text") or "",
                page=page_number,
                initial_section=current_section,
                parser="pymupdf4llm",
            )
            blocks.extend(page_blocks)
        return blocks

    def _markdown_blocks(
        self,
        text: str,
        *,
        page: Optional[int],
        initial_section: str,
        parser: str,
    ) -> Tuple[List[ParsedTextBlock], str]:
        blocks: List[ParsedTextBlock] = []
        current_section = initial_section
        buffer: List[str] = []

        def flush() -> None:
            content = self._normalize_text("\n".join(buffer))
            buffer.clear()
            if content:
                blocks.append(
                    ParsedTextBlock(
                        content=content,
                        section=current_section,
                        page=page,
                        metadata={"parser": parser},
                    )
                )

        for line in text.splitlines():
            heading = self._markdown_heading(line)
            if heading:
                flush()
                current_section = heading
            else:
                buffer.append(line)
        flush()
        return blocks, current_section

    def _blocks_to_chunks(
        self,
        document: KnowledgeDocument,
        blocks: Iterable[ParsedTextBlock],
    ) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        seen_chunk_ids: set[str] = set()
        for block in blocks:
            for index, content in enumerate(self._split_text(block.content), start=1):
                chunk = KnowledgeChunk.from_document(
                    document=document,
                    content=content,
                    section=block.section,
                    page=block.page,
                    metadata={
                        **block.metadata,
                        "chunk_in_block": str(index),
                        "approx_tokens": str(self._word_count(content)),
                    },
                )
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                chunks.append(chunk)
        return chunks

    def _split_text(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", normalized) if item.strip()]
        units: List[str] = []
        for paragraph in paragraphs:
            units.extend(self._split_oversized_unit(paragraph))

        chunks: List[str] = []
        current: List[str] = []
        current_count = 0
        for unit in units:
            unit_count = self._word_count(unit)
            proposed = current_count + unit_count
            if current and (
                proposed > self.max_tokens
                or (proposed > self.target_tokens and current_count >= self.min_tokens)
            ):
                completed = "\n\n".join(current).strip()
                chunks.append(completed)
                overlap = self._tail_words(completed, self.overlap_tokens)
                if self._word_count(overlap) + unit_count > self.max_tokens:
                    overlap = ""
                current = [overlap] if overlap else []
                current_count = self._word_count(overlap)
            current.append(unit)
            current_count += unit_count

        if current:
            final = "\n\n".join(current).strip()
            if (
                chunks
                and self._word_count(final) < self.min_tokens
                and self._word_count(chunks[-1]) + self._word_count(final) <= self.max_tokens
            ):
                chunks[-1] = f"{chunks[-1]}\n\n{final}"
            else:
                chunks.append(final)
        return chunks

    def _split_oversized_unit(self, text: str) -> List[str]:
        if self._word_count(text) <= self.max_tokens:
            return [text]
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            if item.strip()
        ]
        if len(sentences) == 1:
            words = re.findall(r"[\w’'-]+", text, flags=re.UNICODE)
            return [
                " ".join(words[start : start + self.max_tokens])
                for start in range(0, len(words), self.max_tokens)
            ]
        units: List[str] = []
        current: List[str] = []
        count = 0
        for sentence in sentences:
            sentence_count = self._word_count(sentence)
            if current and count + sentence_count > self.max_tokens:
                units.append(" ".join(current))
                current = []
                count = 0
            if sentence_count > self.max_tokens:
                units.extend(self._split_oversized_unit(sentence))
            else:
                current.append(sentence)
                count += sentence_count
        if current:
            units.append(" ".join(current))
        return units

    def _markdown_heading(self, line: str) -> Optional[str]:
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if not match:
            return None
        return re.sub(r"[*_`]+", "", match.group(1)).strip()

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\u00ad", "").replace("\r\n", "\n")
        text = re.sub(r"!\[[^\]]*]\([^)]*\)", "", text)
        paragraphs: List[str] = []
        for raw in re.split(r"\n\s*\n", text):
            lines = [" ".join(line.split()) for line in raw.splitlines() if line.strip()]
            if not lines:
                continue
            paragraphs.append(
                "\n".join(lines)
                if any(re.match(r"^[-*+]\s+", line) for line in lines)
                else " ".join(lines)
            )
        return "\n\n".join(paragraphs).strip()

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))

    def _tail_words(self, text: str, count: int) -> str:
        return " ".join(text.split()[-count:]) if count > 0 else ""

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path
