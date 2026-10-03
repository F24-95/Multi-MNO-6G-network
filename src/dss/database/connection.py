import os
from contextlib import contextmanager
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dss_db")

def get_connection(db_url: str | None = None):
    url = db_url or get_db_url()
    return psycopg2.connect(url)

@contextmanager
def db_session(db_url: str | None = None):
    conn = get_connection(db_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
