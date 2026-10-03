import os
import sys
from dotenv import load_dotenv

load_dotenv()

print(f"Python: {sys.version}")

packages = ["numpy", "pandas", "sklearn", "torch", "flask"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[{pkg.upper()} OK] Version: {ver}")
    except Exception as e:
        print(f"[{pkg.upper()} FAIL] Error: {e}")

# Database Check
db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dss_db")
try:
    import psycopg2
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    res = cur.fetchone()
    cur.close()
    conn.close()
    if res and res[0] == 1:
        print("[DATABASE OK] PostgreSQL connection and SELECT 1 succeeded.")
    else:
        print(f"[DATABASE FAIL] Unexpected query result: {res}")
except Exception as e:
    print(f"[DATABASE FAIL] Connection failed: {e}")
