"""Spreadsheet parser using openpyxl.

STUB — implement this file to enable XLSX ingestion.

Suggested approach:
    import openpyxl
    wb = openpyxl.load_workbook(file_path)
    for sheet in wb.worksheets:
        rows = [[str(cell.value or "") for cell in row] for row in sheet.iter_rows()]
        # convert rows to text representation suitable for chunking
"""

from taxonomy_rag.ingestion.models import ParsedDocument


class SpreadsheetParser:
    """Parses XLSX spreadsheets into a ParsedDocument.

    Implement parse() using openpyxl. Each sheet can be treated as a page.
    """

    def parse(self, file_path: str) -> ParsedDocument:
        raise NotImplementedError(
            "SpreadsheetParser.parse: implement XLSX extraction here using openpyxl. "
            "See module docstring for a starting point."
        )

    def supports(self, file_path: str) -> bool:
        raise NotImplementedError(
            "SpreadsheetParser.supports: return True if file_path ends with '.xlsx' (case-insensitive)."
        )
