from datetime import date
from decimal import Decimal
import shutil

import pytest
from pydantic import ValidationError

from app.data_loader import DEFAULT_DATA_DIR, DataValidationError, load_data
from app.schemas import RFQRequest


@pytest.fixture
def data_dir(tmp_path):
    for source in DEFAULT_DATA_DIR.glob("*.csv"):
        shutil.copyfile(source, tmp_path / source.name)
    return tmp_path


def test_source_data_loads_and_joins_all_six_rfqs():
    data = load_data()
    assert list(data.rfqs) == [f"RFQ-{number}" for number in range(101, 107)]
    assert len(data.suppliers) == len(data.performance) == 7
    requests = [data.for_rfq(rfq_id) for rfq_id in data.rfqs]
    assert [len(request.quotes) for request in requests] == [3, 2, 2, 2, 3, 2]
    assert all(not request.missing_records for request in requests)
    assert data.quotes["Q-101"].unit_price_inr == Decimal("51000")
    assert data.quotes["Q-101"].promised_delivery == date(2026, 12, 10)
    assert requests[0].quotes[0].supplier.supplier_id == "S-01"
    assert requests[0].quotes[0].performance.defect_rate_pct == Decimal("1.5")
    assert requests[4].quotes[-1].quote.quote_id == "Q-114"


@pytest.mark.parametrize("tax", ["", "not-a-number", "-1", "NaN", "Infinity"])
def test_invalid_tax_reports_source_and_row(data_dir, tax):
    path = data_dir / "supplier_quotes.csv"
    lines = path.read_text().splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0] + "," + tax
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DataValidationError, match=r"supplier_quotes\.csv:2:") as error:
        load_data(data_dir)
    assert "tax_percent" in str(error.value)


def test_duplicate_primary_key_reports_source_row(data_dir):
    path = data_dir / "supplier_quotes.csv"
    with path.open("a") as output:
        output.write(path.read_text().splitlines()[1] + "\n")
    with pytest.raises(DataValidationError, match="duplicate quote_id Q-101"):
        load_data(data_dir)


def test_bad_headers_are_rejected(data_dir):
    path = data_dir / "rfqs.csv"
    path.write_text(path.read_text().replace("required_by", "unknown_column", 1))
    with pytest.raises(DataValidationError, match="expected columns"):
        load_data(data_dir)


def test_unknown_rfq_and_missing_performance_are_exposed(data_dir):
    path = data_dir / "supplier_performance.csv"
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.startswith("S-01,")))
    data = load_data(data_dir)

    unknown = data.for_rfq("RFQ-999")
    assert unknown.rfq is None
    assert unknown.missing_records == ("rfqs.csv:RFQ-999",)
    assert unknown.quotes == ()

    laptops = data.for_rfq("RFQ-101")
    assert laptops.quotes[0].performance is None
    assert laptops.missing_records == ("supplier_performance.csv:S-01",)


def test_request_validates_id_and_preserves_note():
    assert RFQRequest(rfq_id=" RFQ-101 ", note="Check delivery").model_dump() == {
        "rfq_id": "RFQ-101",
        "note": "Check delivery",
    }
    with pytest.raises(ValidationError):
        RFQRequest(rfq_id=" ")


def test_non_iso_date_reports_source_row(data_dir):
    path = data_dir / "rfqs.csv"
    path.write_text(path.read_text().replace("2026-12-15", "0", 1))
    with pytest.raises(DataValidationError, match=r"rfqs\.csv:2:") as error:
        load_data(data_dir)
    assert "YYYY-MM-DD" in str(error.value)
