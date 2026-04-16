"""Shared test fixtures used across the unit and integration suites."""

from __future__ import annotations

import pytest

from taxonomy_rag.ingestion.models import Chunk, ParsedDocument
from taxonomy_rag.readers.base import AttachmentInfo


# ---------------------------------------------------------------------------
# Minimal Tool implementation for testing ToolKit without real tools
# ---------------------------------------------------------------------------

class MockTool:
    """Minimal Tool protocol implementation for ToolKit tests."""

    name = "mock_tool"
    description = "A mock tool for testing."
    input_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "Any string value."}
        },
        "required": ["value"],
    }

    def __init__(self, return_value: str = "mock_result") -> None:
        self._return_value = return_value
        self.calls: list[dict] = []

    def run(self, value: str) -> str:
        self.calls.append({"value": value})
        return f"{self._return_value}:{value}"


@pytest.fixture
def mock_tool() -> MockTool:
    return MockTool()


@pytest.fixture
def mock_tool_factory():
    """Callable that creates MockTool with configurable return value."""
    return MockTool


# ---------------------------------------------------------------------------
# Domain object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        source_path="/tmp/test.pdf",
        document_id="test_001",
        document_type="regulation",
        title="Test Regulation",
        pages=["Page one content about sustainable activities.", "Page two with more details."],
        metadata={"year": 2021},
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        content="This regulation concerns sustainable activities.",
        metadata={
            "source": "test.pdf",
            "document_id": "test_001",
            "chunk_strategy": "naive",
        },
    )


@pytest.fixture
def sample_attachment() -> AttachmentInfo:
    return AttachmentInfo(
        name="TEST-DOC-01",
        file_type="pdf",
        size_bytes=102400,
        path="/tmp/test.pdf",
    )
