"""Read the supplied CSVs and join facts by their explicit IDs."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from app.schemas import ProcurementModel, RFQ, Supplier, SupplierPerformance, SupplierQuote


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
Record = TypeVar("Record", bound=ProcurementModel)


class DataValidationError(ValueError):
    """A malformed source file or row."""


@dataclass(frozen=True)
class QuoteFacts:
    quote: SupplierQuote
    supplier: Supplier | None
    performance: SupplierPerformance | None


@dataclass(frozen=True)
class RFQFacts:
    rfq_id: str
    rfq: RFQ | None
    quotes: tuple[QuoteFacts, ...]
    missing_records: tuple[str, ...]


@dataclass(frozen=True)
class ProcurementData:
    rfqs: dict[str, RFQ]
    suppliers: dict[str, Supplier]
    performance: dict[str, SupplierPerformance]
    quotes: dict[str, SupplierQuote]

    def for_rfq(self, rfq_id: str) -> RFQFacts:
        rfq = self.rfqs.get(rfq_id)
        missing = {}
        if rfq is None:
            missing[f"rfqs.csv:{rfq_id}"] = None
        joined_quotes = []
        for quote in self.quotes.values():
            if quote.rfq_id != rfq_id:
                continue
            supplier_id = quote.supplier_id
            supplier = self.suppliers.get(supplier_id)
            performance = self.performance.get(supplier_id)
            if supplier is None:
                missing[f"suppliers.csv:{supplier_id}"] = None
            if performance is None:
                missing[f"supplier_performance.csv:{supplier_id}"] = None
            joined_quotes.append(QuoteFacts(quote, supplier, performance))
        return RFQFacts(rfq_id, rfq, tuple(joined_quotes), tuple(missing))


def _read_records(path: Path, model: type[Record], id_field: str) -> dict[str, Record]:
    records: dict[str, Record] = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            headers = reader.fieldnames or []
            if len(headers) != len(model.model_fields) or set(headers) != set(model.model_fields):
                raise DataValidationError(
                    f"{path}: expected columns {', '.join(model.model_fields)}"
                )
            for row in reader:
                location = f"{path}:{reader.line_num}"
                if None in row or None in row.values():
                    raise DataValidationError(f"{location}: wrong number of columns")
                try:
                    record = model.model_validate(row)
                except ValidationError as exc:
                    raise DataValidationError(f"{location}: {exc}") from exc
                record_id = getattr(record, id_field)
                if record_id in records:
                    raise DataValidationError(f"{location}: duplicate {id_field} {record_id}")
                records[record_id] = record
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataValidationError(f"{path}: {exc}") from exc
    return records


def load_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> ProcurementData:
    """Validate all four inputs. Missing join targets remain visible in RFQFacts."""
    directory = Path(data_dir)
    return ProcurementData(
        rfqs=_read_records(directory / "rfqs.csv", RFQ, "rfq_id"),
        suppliers=_read_records(directory / "suppliers.csv", Supplier, "supplier_id"),
        performance=_read_records(
            directory / "supplier_performance.csv", SupplierPerformance, "supplier_id"
        ),
        quotes=_read_records(directory / "supplier_quotes.csv", SupplierQuote, "quote_id"),
    )
