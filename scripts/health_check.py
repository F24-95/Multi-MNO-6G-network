import sys
from dss.database.connection import get_connection, get_db_url

print(f"Python: {sys.version}")

packages = ["numpy", "pandas", "sklearn", "torch", "flask"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[{pkg.upper()} OK] Version: {ver}")
    except Exception as e:
        print(f"[{pkg.upper()} FAIL] Error: {e}")

# Database check — uses connection.py as single source of truth
try:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT 1;")
    res = cur.fetchone()
    cur.close()
    conn.close()
    if res and res[0] == 1:
        print(f"[DATABASE OK] Connected to: {get_db_url()}")
    else:
        print(f"[DATABASE FAIL] Unexpected result: {res}")
except Exception as e:
    print(f"[DATABASE FAIL] {e}")
