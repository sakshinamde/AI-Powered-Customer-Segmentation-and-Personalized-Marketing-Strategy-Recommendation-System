"""
Database Viewer Utility for SmartSeg-AI
Run this script from your VS Code terminal to inspect your SQLite database.

Usage:
  python backend/view_db.py                  # Show all tables and row counts
  python backend/view_db.py <table_name>     # Show first 10 rows from <table_name>
  python backend/view_db.py <table_name> 20  # Show first 20 rows from <table_name>
  python backend/view_db.py --sql "SELECT * FROM users" # Run custom SQL query
"""

import sys
import os
import sqlite3

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Locate database
DB_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "smartseg.db"),
    os.path.join(os.path.dirname(__file__), "smartseg.db"),
    os.path.join(os.path.dirname(__file__), "smartseg_test.db"),
]

db_path = None
for p in DB_PATHS:
    if os.path.exists(p):
        db_path = p
        break

if not db_path:
    print("[!] No SQLite database file found (.db). Make sure the database has been initialized.")
    sys.exit(1)

print(f"[+] Connected to: {os.path.abspath(db_path)}\n")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

def print_tables():
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cur.fetchall()]
    print("=" * 60)
    print(f"{'Table Name':<30} | {'Row Count':<15}")
    print("-" * 60)
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM `{table}`;")
            count = cur.fetchone()[0]
            print(f"{table:<30} | {count:<15}")
        except Exception as e:
            print(f"{table:<30} | Error: {e}")
    print("=" * 60)
    print("\n[Tip] To view data in a table, run:")
    print("   python backend/view_db.py <table_name>")
    print("   e.g.: python backend/view_db.py datasets")
    print("   e.g.: python backend/view_db.py customers 5\n")

def print_table_data(table_name, limit=10):
    try:
        cur.execute(f"PRAGMA table_info(`{table_name}`);")
        cols_info = cur.fetchall()
        if not cols_info:
            print(f"[-] Table '{table_name}' does not exist.")
            return
        
        col_names = [col[1] for col in cols_info]
        cur.execute(f"SELECT * FROM `{table_name}` LIMIT {limit};")
        rows = cur.fetchall()

        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=col_names)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(f"[*] Table: {table_name} (Showing top {len(rows)} rows):")
            print("-" * 80)
            print(df.to_string(index=False))
            print("-" * 80)
        except ImportError:
            print(f"[*] Table: {table_name} (Showing top {len(rows)} rows):")
            print(" | ".join(col_names))
            print("-" * 80)
            for r in rows:
                print(" | ".join(str(x) for x in r))
            print("-" * 80)
    except Exception as e:
        print(f"[-] Error reading table '{table_name}': {e}")

def run_sql(query):
    try:
        cur.execute(query)
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        if not rows:
            print("[i] Query returned 0 rows or executed successfully.")
            return
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=col_names)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df.to_string(index=False))
        except ImportError:
            print(" | ".join(col_names))
            print("-" * 80)
            for r in rows:
                print(" | ".join(str(x) for x in r))
    except Exception as e:
        print(f"[-] Query execution error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_tables()
    elif sys.argv[1] == "--sql" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        run_sql(query)
    else:
        table_name = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
        print_table_data(table_name, limit)

conn.close()

