import os
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
from dss.common.logging_setup import setup_logger

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")
load_dotenv()

logger = setup_logger("dss.database")

MNO_NAME_MAP = {
    0: "Airtel 6G",
    1: "Jio 6G",
    2: "Vi 6G",
    3: "BSNL 6G",
}


def get_db_url() -> str:
    """Return the database connection URL from the environment or default to local PostgreSQL / SQLite."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return "postgresql://postgres:Faizan9517@localhost:5432/Multi-MNO"
    return url


def get_connection(db_url: Optional[str] = None):
    """Open and return a live connection to PostgreSQL or SQLite."""
    url = db_url or get_db_url()
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(path)
    elif url.startswith(("postgresql://", "postgres://")):
        import psycopg2
        return psycopg2.connect(url)
    return sqlite3.connect(url)


@contextmanager
def db_session(db_url: Optional[str] = None):
    """Context manager for safe transactional database sessions."""
    conn = get_connection(db_url)
    try:
        yield conn
        conn.commit()
    except Exception as err:
        conn.rollback()
        logger.error("Transaction rolled back due to error: %s", err)
        raise
    finally:
        conn.close()


def init_db_schema(db_url: Optional[str] = None) -> None:
    """Ensure core relational tables exist for channel observations and MNO demands."""
    with db_session(db_url) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channel_obs (
                step_idx INTEGER NOT NULL,
                ts TIMESTAMP NOT NULL,
                channel_id INTEGER NOT NULL,
                pu_state INTEGER NOT NULL,
                noise_true_dbm REAL NOT NULL,
                noise_est_dbm REAL NOT NULL,
                interference_dbm REAL NOT NULL,
                rx_power_dbm REAL NOT NULL,
                snr_est_db REAL NOT NULL,
                PRIMARY KEY (step_idx, channel_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mno_demand (
                step_idx INTEGER NOT NULL,
                ts TIMESTAMP NOT NULL,
                mno_id INTEGER NOT NULL,
                mno_name VARCHAR(50) NOT NULL DEFAULT 'Airtel 6G',
                active_users INTEGER NOT NULL,
                traffic_demand_mbps REAL NOT NULL,
                requested_bw_mhz REAL NOT NULL,
                PRIMARY KEY (step_idx, mno_id)
            );
        """)


def get_latest_step(db_url: Optional[str] = None) -> int:
    """
    Fetch the latest common step index present in both channel and MNO demand records.
    Returns 4032 if the tables are empty or uninitialized.
    """
    try:
        with db_session(db_url) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(c.step_idx)
                FROM channel_obs c
                JOIN mno_demand m ON c.step_idx = m.step_idx;
            """)
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
    except Exception as err:
        logger.warning("Could not read latest step from database, falling back to 4032: %s", err)
    return 4032


def save_dataset_to_postgresql(
    channel_df: pd.DataFrame,
    mno_df: pd.DataFrame,
    db_url: Optional[str] = None
) -> None:
    """
    Persist channel observations and MNO demand records into the database.
    Performs conflict-safe bulk inserts.
    """
    url = db_url or get_db_url()
    init_db_schema(url)

    # Format MNO names cleanly if missing
    prepared_mno = mno_df.copy()
    if "mno_name" not in prepared_mno.columns:
        prepared_mno.insert(3, "mno_name", prepared_mno["mno_id"].map(MNO_NAME_MAP).fillna("Unknown 6G"))

    if url.startswith(("postgresql://", "postgres://")):
        import psycopg2
        from psycopg2.extras import execute_values

        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cursor:
                c_tuples = [tuple(row) for row in channel_df.itertuples(index=False)]
                execute_values(
                    cursor,
                    "INSERT INTO channel_obs VALUES %s ON CONFLICT (step_idx, channel_id) DO NOTHING;",
                    c_tuples
                )

                m_tuples = [tuple(row) for row in prepared_mno.itertuples(index=False)]
                execute_values(
                    cursor,
                    """
                    INSERT INTO mno_demand
                        (step_idx, ts, mno_id, mno_name, active_users, traffic_demand_mbps, requested_bw_mhz)
                    VALUES %s
                    ON CONFLICT (step_idx, mno_id) DO NOTHING;
                    """,
                    m_tuples
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        with db_session(url) as conn:
            channel_df.to_sql("channel_obs", conn, if_exists="append", index=False)
            prepared_mno.to_sql("mno_demand", conn, if_exists="append", index=False)


# Modern alias
save_dataset_to_db = save_dataset_to_postgresql


def sync_all_raw_datasets_to_db(raw_dir_path: Optional[Path] = None, db_url: Optional[str] = None) -> int:
    """
    Scan data/raw directory and commit all existing generated Parquet datasets to PostgreSQL.
    Returns the total number of synced datasets.
    """
    root_raw = raw_dir_path or (Path(__file__).resolve().parents[3] / "data" / "raw")
    if not root_raw.exists():
        return 0

    count = 0
    for folder in root_raw.iterdir():
        if not folder.is_dir():
            continue
        c_path = folder / "channel_obs.parquet"
        m_path = folder / "mno_demand.parquet"
        if c_path.exists() and m_path.exists():
            try:
                c_df = pd.read_parquet(c_path)
                m_df = pd.read_parquet(m_path)
                save_dataset_to_postgresql(c_df, m_df, db_url=db_url)
                count += 1
                logger.info("Synced raw dataset %s to database", folder.name)
            except Exception as err:
                logger.error("Failed to sync dataset %s: %s", folder.name, err)
    return count


