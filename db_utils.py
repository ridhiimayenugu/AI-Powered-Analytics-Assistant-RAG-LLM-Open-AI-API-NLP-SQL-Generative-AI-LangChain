"""
SQL Server-only utilities for:
- Creating SQLAlchemy engine via pyodbc
- Listing user databases
- Schema introspection (tables + columns)
- Running SELECT queries to pandas DataFrames

This module is intentionally SQL Server-specific to keep the project stable
before adding other database types.
"""
import os
import urllib.parse
from typing import List, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required environment variable: {name}")
    return val


def get_engine(database: Optional[str] = None) -> Engine:
    """
    Create a SQLAlchemy engine for SQL Server using a safe ODBC connection string.

    Required .env vars:
      SQL_SERVER_HOST=DESKTOP-IQOJMA1\\SQLSERVER2025   (or HOST,PORT)
      SQL_SERVER_USER=sa
      SQL_SERVER_PASSWORD=...

    Optional:
      SQL_SERVER_DB=AdventureWorksDW2025
      SQL_SERVER_DRIVER=ODBC Driver 18 for SQL Server
      SQL_SERVER_ENCRYPT=yes
      SQL_SERVER_TRUST_SERVER_CERT=yes
      SQL_SERVER_TIMEOUT=5
    """
    server = _require_env("SQL_SERVER_HOST")
    username = os.getenv("SQL_SERVER_USER", "sa")
    password = _require_env("SQL_SERVER_PASSWORD")
    db = database or os.getenv("SQL_SERVER_DB", "master")

    driver = os.getenv("SQL_SERVER_DRIVER", "ODBC Driver 18 for SQL Server")
    encrypt = os.getenv("SQL_SERVER_ENCRYPT", "yes")
    trust_cert = os.getenv("SQL_SERVER_TRUST_SERVER_CERT", "yes")
    timeout = os.getenv("SQL_SERVER_TIMEOUT", "5")

    # Curly braces around password help with special characters; quote_plus protects the full string.
    odbc = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={username};"
        f"PWD={{{password}}};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
        f"Connection Timeout={timeout};"
    )

    params = urllib.parse.quote_plus(odbc)
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True,
        future=True,
    )
    return engine


def list_user_databases(engine_master: Engine) -> List[str]:
    """
    List online user databases (excludes system DBs).
    """
    sql = text(
        """
        SELECT name
        FROM sys.databases
        WHERE database_id > 4
          AND state_desc = 'ONLINE'
        ORDER BY name
        """
    )
    with engine_master.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [r[0] for r in rows]


def fetch_schema_summary(
    engine: Engine,
    max_tables: int = 250,
    include_views: bool = False
) -> Tuple[str, List[Dict[str, object]]]:
    """
    Returns:
      - schema_summary (text): compact listing: Table + Columns
      - tables_struct (list): [{schema,name,type,columns:[{name,type},...]}, ...]

    Filters out sys/INFORMATION_SCHEMA and common noisy tables.
    """
    schema_filter = "t.TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')"
    name_filter = (
        "t.TABLE_NAME NOT LIKE 'spt_%' "
        "AND t.TABLE_NAME NOT LIKE 'MSreplication_%' "
        "AND t.TABLE_NAME NOT LIKE 'MSrepl_%' "
    )

    table_type_clause = "t.TABLE_TYPE = 'BASE TABLE'"
    if include_views:
        table_type_clause = "t.TABLE_TYPE IN ('BASE TABLE','VIEW')"

    tables_sql = text(
        f"""
        SELECT TOP (:max_tables)
            t.TABLE_SCHEMA,
            t.TABLE_NAME,
            t.TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE {schema_filter}
          AND {name_filter}
          AND {table_type_clause}
        ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
        """
    )

    cols_sql = text(
        """
        SELECT
            c.TABLE_SCHEMA,
            c.TABLE_NAME,
            c.COLUMN_NAME,
            c.DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA = :schema
          AND c.TABLE_NAME = :table
        ORDER BY c.ORDINAL_POSITION
        """
    )

    tables: List[Dict[str, object]] = []
    lines: List[str] = []

    with engine.connect() as conn:
        table_rows = conn.execute(tables_sql, {"max_tables": max_tables}).fetchall()

        for schema, table, table_type in table_rows:
            col_rows = conn.execute(cols_sql, {"schema": schema, "table": table}).fetchall()
            cols = [{"name": r[2], "type": r[3]} for r in col_rows]

            tables.append(
                {
                    "schema": schema,
                    "name": table,
                    "type": table_type,
                    "columns": cols,
                }
            )

            col_names = ", ".join([c["name"] for c in cols])
            lines.append(f"Table: {schema}.{table} ({table_type})")
            lines.append(f"Columns: {col_names}")
            lines.append("")

    summary = "\n".join(lines).strip()
    return summary, tables


def run_sql(engine: Engine, sql: str, max_rows: int = 2000) -> pd.DataFrame:
    """
    Run a SELECT query and return a DataFrame.
    """
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchmany(max_rows)
        df = pd.DataFrame(rows, columns=result.keys())
    return df
