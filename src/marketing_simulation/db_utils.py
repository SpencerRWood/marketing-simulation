# db_utils.py
from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv
from datetime import datetime, timezone
import os, io, csv, time, logging
from sqlalchemy import __version__ as SA_VER
from packaging.version import Version
from .models import visitors, interactions
from sqlalchemy import text
from .models import metadata, SCHEMA

# ----------------------------
# Logging setup
# ----------------------------
log = logging.getLogger("db")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

# ----------------------------
# Env / engine
# ----------------------------
load_dotenv()

PG_USER = os.getenv("DB_USER", "postgres")
PG_PASS = os.getenv("DB_PASSWORD", "postgres")
PG_HOST = os.getenv("DB_HOST", "localhost")
PG_PORT = os.getenv("DB_PORT", "5432")
PG_DB   = os.getenv("DB_DATABASE", "postgres")

DATABASE_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# Simple, SA 2.x-safe engine. COPY handles bulk speed.
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,  # optional but nice with SA 2.x APIs
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def ensure_schema_and_tables():
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    metadata.create_all(engine)
    # set search_path so your raw SQL without schema prefixes works
    with engine.begin() as conn:
        conn.execute(text(f"SET search_path TO {SCHEMA}, public"))
    logging.getLogger("db").info(
        "metadata_created",
        extra={"schema": SCHEMA, "tables": list(metadata.tables.keys())}
    )


def reset_tables(mode: str = "truncate", extra_tables: list[str] | None = None):
    t0 = time.time()
    mode = mode.lower()
    if mode not in {"truncate","recreate"}:
        raise ValueError("mode must be 'truncate' or 'recreate'")

    if mode == "truncate":
        # use bare table names so search_path resolves the schema
        table_names = [t.name for t in metadata.sorted_tables]
        if extra_tables:
            table_names += list(extra_tables)

        if not table_names:
            log.info("db_reset_done", extra={"event":"db_reset_done","mode":mode,"tables":0,"seconds":0.0})
            return

        sql = f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE"
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA}, public"))
            conn.execute(text(sql))

        dt = time.time() - t0
        log.info("db_reset_done", extra={"event":"db_reset_done","mode":mode,"tables":len(table_names),"seconds":round(dt,3)})
        return

    # recreate mode
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    metadata.drop_all(engine)
    metadata.create_all(engine)
    dt = time.time() - t0
    log.info("db_reset_done", extra={"event":"db_reset_done","mode":mode,"tables":len(metadata.tables),"seconds":round(dt,3)})

# ----------------------------
# Generic CSV COPY loader
# ----------------------------
def copy_csv(table: str, rows: list[dict], columns: list[str]) -> int:
    """High-speed bulk load using COPY (expects columns to match table DDL)."""
    if not rows:
        return 0
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in rows:
        writer.writerow([r.get(col) for col in columns])
    buf.seek(0)

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(
                f'COPY {table} ({", ".join(columns)}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)',
                buf
            )
        raw.commit()
    finally:
        raw.close()
    return len(rows)

# ----------------------------
# Phase 1a: Insert skeleton visitors
# ----------------------------
def ensure_skeleton_visitors(visitor_ids: set[str], created_at: datetime | None = None) -> int:
    """
    Ensure every visitor_id exists in `visitors`.
    Inserts minimal skeleton rows with visitor_id + created_at.
    """
    if not visitor_ids:
        return 0
    ts = created_at
    sql = text("""
        INSERT INTO visitors (visitor_id, created_at)
        VALUES (:visitor_id, :created_at)
        ON CONFLICT (visitor_id) DO NOTHING
    """)
    params = [{"visitor_id": vid, "created_at": ts} for vid in visitor_ids]

    t0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(sql, params)
    dt = time.perf_counter() - t0
    log.info(
        "visitor_skeleton_write_done",
        extra={
            "table": "visitors",
            "rows": len(params),
            "seconds": round(dt, 3),
            "rps": int(len(params)/dt) if dt else 0,
        },
    )
    return len(params)

# ----------------------------
# Phase 1b: Bulk insert interactions
# ----------------------------
def write_interactions_copy(rows: list[dict]) -> int:
    """Insert clickstream rows via COPY into `interactions`."""
    if not rows:
        return 0
    cols = ["visitor_id","session_id","channel","page","interaction","element","timestamp"]
    t0 = time.perf_counter()
    n = copy_csv("interactions", rows, cols)
    dt = time.perf_counter() - t0
    log.info(
    "interactions_inserted",
    extra={
        "table": "interactions",
        "rows": n,
        "seconds": round(dt, 3),
        "rps": int(n/dt) if dt else 0,
        },
    )
    return n

# ----------------------------
# Phase 2: Enrich visitors on signup/conversion
# ----------------------------
def upsert_visitor_enrichment(rows: list[dict]) -> int:
    if not rows:
        return 0

    # normalize (fix 'gener' -> 'gender' if present)
    # fixed = []
    # for r in rows:
    #     d = dict(r)
    #     if "gener" in d and "gender" not in d:
    #         d["gender"] = d.pop("gener")
    #     fixed.append(d)
    # rows = fixed

    all_cols = set().union(*(r.keys() for r in rows))
    if "visitor_id" not in all_cols:
        raise ValueError("upsert_visitor_enrichment: 'visitor_id' required")

    insert_cols = ["visitor_id"] + [c for c in all_cols if c != "visitor_id"]
    update_cols = [c for c in insert_cols if c not in ("visitor_id", "created_at")]

    target = f"{SCHEMA}.visitors" if SCHEMA else "visitors"

    def _set_expr(c: str) -> str:
        if c == "is_identified":
            return f'{c} = ({target}.{c} OR COALESCE(EXCLUDED.{c}, false))'
        if c == "identified_at":
            return f'{c} = COALESCE(EXCLUDED.{c}, {target}.{c})'
        if c in ("name", "gender", "email"):
            # do not overwrite with NULL or ""
            return f'{c} = COALESCE(NULLIF(EXCLUDED.{c}, \'\'), {target}.{c})'
        if c == "age":
            # keep existing if incoming is NULL; allow legitimate 0+ values
            return f'{c} = COALESCE(EXCLUDED.{c}, {target}.{c})'
        return f"{c} = EXCLUDED.{c}"

    insert_cols_sql = ", ".join(insert_cols)
    insert_vals_sql = ", ".join(f":{c}" for c in insert_cols)

    if update_cols:
        set_sql = ", ".join(_set_expr(c) for c in update_cols)
        sql = text(f"""
            INSERT INTO {target} ({insert_cols_sql})
            VALUES ({insert_vals_sql})
            ON CONFLICT (visitor_id) DO UPDATE SET {set_sql}
        """)
    else:
        sql = text(f"""
            INSERT INTO {target} ({insert_cols_sql})
            VALUES ({insert_vals_sql})
            ON CONFLICT (visitor_id) DO NOTHING
        """)

    normalized = [{c: r.get(c) for c in insert_cols} for r in rows]

    t0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(sql, normalized)
    dt = time.perf_counter() - t0
    logging.getLogger("db").info("visitors_upsert_done",
        extra={"event":"visitors_upsert_done"
               ,"table":"visitors"
               ,"rows":len(normalized)
               , "sample": {k: normalized[0].get(k) for k in ("visitor_id","is_identified","identified_at","name","gender","age","email") if k in insert_cols}
               ,"seconds":round(dt,3)})


    return len(normalized)