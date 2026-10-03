import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

# Load .env once — all other modules import from here, never call load_dotenv() themselves
load_dotenv()


def get_db_url() -> str:
    """Returns DATABASE_URL from environment. Set it in .env — no hardcoded fallbacks."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL not set. Add it to your .env file.\n"
            "Example: DATABASE_URL=postgresql://user:pass@localhost:5432/dbname"
        )
    return url


def get_connection(db_url: str | None = None) -> psycopg2.extensions.connection:
    """Opens and returns a raw psycopg2 connection."""
    return psycopg2.connect(db_url or get_db_url())


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
