
import pytest

from api import database


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Create an isolated SQLite database for each test."""
    db_path = tmp_path / "test_fraudguard.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.initialize_database()
    return db_path


def sample_prediction(predicted_illicit=True):
    return {
        "tx_id": "test_tx_001",
        "time_step": 10,
        "illicit_probability": 0.98 if predicted_illicit else 0.02,
        "predicted_illicit": predicted_illicit,
        "risk_level": "HIGH" if predicted_illicit else "LOW",
    }


def test_database_initialization(test_db):
    assert test_db.exists()

    with database.get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert "elliptic_predictions" in tables
    assert "investigation_cases" in tables


def test_save_prediction_creates_investigation(test_db):
    prediction_id = database.save_elliptic_prediction(
        sample_prediction(predicted_illicit=True)
    )

    predictions = database.get_elliptic_predictions()
    cases = database.get_investigation_cases()

    assert len(predictions) == 1
    assert predictions[0]["id"] == prediction_id
    assert len(cases) == 1
    assert cases[0]["prediction_id"] == prediction_id
    assert cases[0]["status"] == "Open"


def test_licit_prediction_does_not_create_investigation(test_db):
    database.save_elliptic_prediction(
        sample_prediction(predicted_illicit=False)
    )

    assert len(database.get_elliptic_predictions()) == 1
    assert database.get_investigation_cases() == []


def test_update_investigation_case(test_db):
    prediction_id = database.save_elliptic_prediction(
        sample_prediction()
    )
    case = database.get_investigation_cases()[0]

    updated = database.update_investigation_case(
        case_id=case["case_id"],
        status="Under Review",
        analyst_verdict="Unconfirmed",
        notes="Test investigation",
    )

    assert updated is True

    updated_case = database.get_investigation_cases()[0]
    assert updated_case["status"] == "Under Review"
    assert updated_case["notes"] == "Test investigation"
    assert updated_case["prediction_id"] == prediction_id


def test_update_nonexistent_case_returns_false(test_db):
    result = database.update_investigation_case(
        case_id=999,
        status="Open",
        analyst_verdict="Unconfirmed",
        notes="No such case",
    )

    assert result is False


def test_invalid_status_is_rejected(test_db):
    with pytest.raises(ValueError, match="Invalid investigation status"):
        database.update_investigation_case(
            case_id=1,
            status="Invalid Status",
            analyst_verdict="Unconfirmed",
            notes="",
        )


def test_invalid_verdict_is_rejected(test_db):
    with pytest.raises(ValueError, match="Invalid analyst verdict"):
        database.update_investigation_case(
            case_id=1,
            status="Open",
            analyst_verdict="Invalid Verdict",
            notes="",
        )


def test_investigation_statistics(test_db):
    database.save_elliptic_prediction(sample_prediction(True))
    database.save_elliptic_prediction(sample_prediction(False))

    case = database.get_investigation_cases()[0]

    database.update_investigation_case(
        case_id=case["case_id"],
        status="Resolved",
        analyst_verdict="True Positive",
        notes="Confirmed for test",
    )

    stats = database.get_investigation_statistics()

    assert stats == {
        "total_cases": 1,
        "open_cases": 0,
        "under_review": 0,
        "resolved_cases": 1,
    }


def test_prediction_limit(test_db):
    for index in range(5):
        prediction = sample_prediction(False)
        prediction["tx_id"] = f"test_tx_{index}"
        database.save_elliptic_prediction(prediction)

    results = database.get_elliptic_predictions(limit=2)

    assert len(results) == 2
