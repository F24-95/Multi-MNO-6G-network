import pytest
import psycopg2
from dss.database.connection import get_connection, db_session

def test_bad_db_url_raises():
    bad_url = "postgresql://invalid_user:invalid_pass@localhost:5432/non_existent_db"
    with pytest.raises(psycopg2.OperationalError):
        get_connection(bad_url)

# Note: Live DB connection test is tested when DB is initialized in Step 22
