
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "fraudguard.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database():
    """Create prediction history and analyst investigation tables."""

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS elliptic_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT NOT NULL,
                time_step INTEGER NOT NULL,
                illicit_probability REAL NOT NULL,
                predicted_illicit INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigation_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                prediction_id INTEGER NOT NULL UNIQUE,

                status TEXT NOT NULL DEFAULT 'Open'
                    CHECK(status IN (
                        'Open',
                        'Under Review',
                        'Resolved'
                    )),

                analyst_verdict TEXT DEFAULT 'Unconfirmed'
                    CHECK(analyst_verdict IN (
                        'Unconfirmed',
                        'True Positive',
                        'False Positive'
                    )),

                notes TEXT DEFAULT '',

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(prediction_id)
                    REFERENCES elliptic_predictions(id)
                    ON DELETE CASCADE
            )
        """)


def save_elliptic_prediction(data):
    """Save a prediction and automatically create a case if flagged illicit."""

    with get_connection() as conn:

        cursor = conn.execute("""
            INSERT INTO elliptic_predictions (
                tx_id,
                time_step,
                illicit_probability,
                predicted_illicit,
                risk_level
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(data["tx_id"]),
            int(data["time_step"]),
            float(data["illicit_probability"]),
            int(data["predicted_illicit"]),
            data["risk_level"],
        ))

        prediction_id = cursor.lastrowid

        # Automatically create an investigation case for flagged transactions.
        if data["predicted_illicit"]:
            conn.execute("""
                INSERT INTO investigation_cases (prediction_id)
                VALUES (?)
            """, (prediction_id,))

        return prediction_id


def get_elliptic_predictions(limit=100):
    """Retrieve the latest prediction history."""

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM elliptic_predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


def get_investigation_cases(limit=100):
    """Retrieve investigation cases with their prediction details."""

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                c.id AS case_id,
                c.prediction_id,
                c.status,
                c.analyst_verdict,
                c.notes,
                c.created_at AS case_created_at,
                c.updated_at,

                p.tx_id,
                p.time_step,
                p.illicit_probability,
                p.predicted_illicit,
                p.risk_level,
                p.created_at AS prediction_created_at

            FROM investigation_cases c

            JOIN elliptic_predictions p
                ON c.prediction_id = p.id

            ORDER BY
                CASE c.status
                    WHEN 'Open' THEN 1
                    WHEN 'Under Review' THEN 2
                    ELSE 3
                END,
                c.updated_at DESC

            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


def update_investigation_case(
    case_id,
    status,
    analyst_verdict,
    notes,
):
    """Update an analyst's investigation status, verdict and notes."""

    valid_statuses = {
        "Open",
        "Under Review",
        "Resolved",
    }

    valid_verdicts = {
        "Unconfirmed",
        "True Positive",
        "False Positive",
    }

    if status not in valid_statuses:
        raise ValueError("Invalid investigation status")

    if analyst_verdict not in valid_verdicts:
        raise ValueError("Invalid analyst verdict")

    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE investigation_cases

            SET
                status = ?,
                analyst_verdict = ?,
                notes = ?,
                updated_at = CURRENT_TIMESTAMP

            WHERE id = ?
        """, (
            status,
            analyst_verdict,
            notes,
            case_id,
        ))

        return cursor.rowcount > 0


def get_investigation_statistics():
    """Return investigation case counts."""

    with get_connection() as conn:
        total = conn.execute("""
            SELECT COUNT(*) FROM investigation_cases
        """).fetchone()[0]

        open_cases = conn.execute("""
            SELECT COUNT(*) FROM investigation_cases
            WHERE status = 'Open'
        """).fetchone()[0]

        under_review = conn.execute("""
            SELECT COUNT(*) FROM investigation_cases
            WHERE status = 'Under Review'
        """).fetchone()[0]

        resolved = conn.execute("""
            SELECT COUNT(*) FROM investigation_cases
            WHERE status = 'Resolved'
        """).fetchone()[0]

        return {
            "total_cases": total,
            "open_cases": open_cases,
            "under_review": under_review,
            "resolved_cases": resolved,
        }
