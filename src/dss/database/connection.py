import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load .env once
load_dotenv()


def get_db_url() -> str:
    """Returns DATABASE_URL from environment, defaulting to PostgreSQL or local SQLite database."""
    url = os.getenv("DATABASE_URL")
    if not url:
        db_path = Path(__file__).resolve().parents[3] / "data" / "dss_database.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    return url


def get_connection(db_url: str | None = None):
    """Opens and returns a raw database connection (SQLite or PostgreSQL based on URL)."""
    url = db_url or get_db_url()
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(path)
    elif url.startswith("postgresql://") or url.startswith("postgres://"):
        import psycopg2
        return psycopg2.connect(url)
    else:
        return sqlite3.connect(url)


@contextmanager
def db_session(db_url: str | None = None):
    """Context manager: opens a connection, commits on success, rolls back on error."""
    conn = get_connection(db_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db_schema(db_url: str | None = None):
    """Initializes channel_obs and mno_demand table schemas in PostgreSQL or SQLite."""
    conn = get_connection(db_url)
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
    
    # Check if mno_name column exists for existing tables
    try:
        cursor.execute("ALTER TABLE mno_demand ADD COLUMN IF NOT EXISTS mno_name VARCHAR(50) DEFAULT 'Airtel 6G';")
    except Exception:
        pass

    conn.commit()
    conn.close()


def save_dataset_to_postgresql(channel_df: pd.DataFrame, mno_df: pd.DataFrame, db_url: str | None = None):
    """Saves channel_obs and mno_demand DataFrames into PostgreSQL database using bulk execute_values."""
    url = db_url or get_db_url()
    init_db_schema(url)

    if url.startswith("postgresql://") or url.startswith("postgres://"):
        import psycopg2
        from psycopg2.extras import execute_values

        conn = psycopg2.connect(url)
        cursor = conn.cursor()

        channel_tuples = [tuple(x) for x in channel_df.to_numpy()]
        execute_values(cursor, "INSERT INTO channel_obs VALUES %s ON CONFLICT (step_idx, channel_id) DO NOTHING;", channel_tuples)

        # Ensure mno_name is in mno_df
        mno_names_map = {0: "Airtel 6G", 1: "Jio 6G", 2: "Vi 6G", 3: "BSNL 6G"}
        if "mno_name" not in mno_df.columns:
            mno_df = mno_df.copy()
            mno_df.insert(3, "mno_name", mno_df["mno_id"].map(mno_names_map))

        mno_tuples = [tuple(x) for x in mno_df.to_numpy()]
        execute_values(cursor, "INSERT INTO mno_demand (step_idx, ts, mno_id, mno_name, active_users, traffic_demand_mbps, requested_bw_mhz) VALUES %s ON CONFLICT (step_idx, mno_id) DO NOTHING;", mno_tuples)

        conn.commit()
        conn.close()
    else:
        conn = sqlite3.connect(url.replace("sqlite:///", ""))
        channel_df.to_sql("channel_obs", conn, if_exists="replace", index=False)
        mno_df.to_sql("mno_demand", conn, if_exists="replace", index=False)
        conn.close()

