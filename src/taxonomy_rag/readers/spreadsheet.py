"""Spreadsheet reader — stub awaiting implementation.

Intended implementation: use openpyxl (already in pyproject.toml) to iterate
over sheets and rows, formatting each row as tab-separated values with a
sheet header. Return all sheets concatenated.
"""

from __future__ import annotations


class SpreadsheetReader:
    """Extracts plain text from Excel (.xlsx) and CSV files."""

    def supports(self, source: str) -> bool:
        return source.lower().endswith((".xlsx", ".csv"))

    def read(self, source: str) -> str:
        raise NotImplementedError(
            "SpreadsheetReader is not yet implemented. "
            "Use openpyxl to load the workbook, iterate sheets and rows, "
            "and return formatted text."
        )
