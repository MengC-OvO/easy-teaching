import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.schemas import (
    IngestionResult,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSourceType,
)


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
    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> None:
        self.project_root = project_root or Path.cwd()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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

        with resolved_output.open("w", encoding="utf-8") as file:
            for result in results:
                for chunk in result.chunks:
                    file.write(chunk.model_dump_json() + "\n")
                    written += 1

        return written

    def read_chunks_jsonl(self, input_path: Path) -> List[KnowledgeChunk]:
        resolved_input = self._resolve_path(input_path)
        chunks: List[KnowledgeChunk] = []

        with resolved_input.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                chunks.append(KnowledgeChunk.model_validate_json(stripped))

        return chunks

    def ingest_source(
        self,
        source: KnowledgeSourceSpec,
        *,
        max_pages_per_pdf: Optional[int] = None,
    ) -> IngestionResult:
        document = source.to_document()
        path = self._resolve_path(Path(source.path))
        blocks = self._parse_source(source, path, max_pages_per_pdf=max_pages_per_pdf)
        chunks = self._blocks_to_chunks(document, blocks)
        return IngestionResult(
            source_id=source.source_id,
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
        blocks: List[ParsedTextBlock] = []
        current_section: Optional[str] = None
        buffer: List[str] = []

        for line in path.read_text(encoding="utf-8").splitlines():
            heading = self._markdown_heading(line)
            if heading:
                self._flush_markdown_block(blocks, buffer, current_section)
                current_section = heading
                continue
            buffer.append(line)

        self._flush_markdown_block(blocks, buffer, current_section)
        return blocks

    def _flush_markdown_block(
        self,
        blocks: List[ParsedTextBlock],
        buffer: List[str],
        section: Optional[str],
    ) -> None:
        content = self._normalize_text("\n".join(buffer))
        buffer.clear()
        if not content:
            return
        blocks.append(
            ParsedTextBlock(
                content=content,
                section=section or "Document introduction",
            )
        )

    def _parse_pdf(self, path: Path, *, max_pages: Optional[int]) -> List[ParsedTextBlock]:
        reader = PdfReader(str(path))
        blocks: List[ParsedTextBlock] = []
        page_count = len(reader.pages)
        limit = min(page_count, max_pages) if max_pages is not None else page_count

        for index in range(limit):
            page_number = index + 1
            text = self._normalize_text(reader.pages[index].extract_text() or "")
            if not text:
                continue
            blocks.append(
                ParsedTextBlock(
                    content=text,
                    section=self._infer_pdf_section(text, page_number),
                    page=page_number,
                )
            )
        return blocks

    def _blocks_to_chunks(
        self,
        document: KnowledgeDocument,
        blocks: Iterable[ParsedTextBlock],
    ) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for block in blocks:
            for content in self._split_text(block.content):
                chunks.append(
                    KnowledgeChunk.from_document(
                        document=document,
                        content=content,
                        section=block.section,
                        page=block.page,
                        metadata=block.metadata,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if len(normalized) <= self.chunk_size:
            return [normalized] if normalized else []

        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            if end < len(normalized):
                sentence_end = normalized.rfind(". ", start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _markdown_heading(self, line: str) -> Optional[str]:
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        return match.group(1).strip() if match else None

    def _infer_pdf_section(self, text: str, page_number: int) -> str:
        for line in text.splitlines():
            candidate = line.strip()
            if 5 <= len(candidate) <= 120 and re.search(r"[A-Za-z]", candidate):
                return candidate
        return f"Page {page_number}"

    def _normalize_text(self, text: str) -> str:
        normalized_lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in normalized_lines if line).strip()

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path
