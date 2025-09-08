
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from typing import Iterable, Union, Dict, Any, List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text

import os
import csv
from io import StringIO

from .models import interactions, SCHEMA

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
try:
    from marketing_simulation.logging_utils import get_logger  # type: ignore
    log = get_logger("db")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    log = logging.getLogger("db")


# -----------------------------------------------------------------------------
# Engine & Schema
# -----------------------------------------------------------------------------
# load_dotenv()
# PG_USER = os.getenv("DB_USER", "postgres")
# PG_PASS = os.getenv("DB_PASSWORD", "postgres")
# PG_HOST = os.getenv("DB_HOST", "127.0.0.1")  # prefer IPv4 TCP over 'localhost'
# PG_PORT = os.getenv("DB_PORT", "5432")
# PG_DB   = os.getenv("DB_DATABASE", "postgres")

# DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN") or                f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# engine: Engine = create_engine(
#     DATABASE_URL,
#     pool_pre_ping=True,
#     future=True,
#     connect_args={"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5"))},
# )

def get_engine() -> Engine:
    load_dotenv()
    PG_USER = os.getenv("DB_USER", "postgres")
    PG_PASS = os.getenv("DB_PASSWORD", "postgres")
    PG_HOST = os.getenv("DB_HOST", "127.0.0.1")
    PG_PORT = os.getenv("DB_PORT", "5432")
    PG_DB   = os.getenv("DB_DATABASE", "postgres")
    DATABASE_URL = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_DSN")
        or f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
        connect_args={"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5"))},
    )

SCHEMA: Optional[str] = (os.getenv("DB_SCHEMA") or os.getenv("SCHEMA") or "").strip() or None

def _qual(table: str) -> str:
    return f"{SCHEMA}.{table}" if SCHEMA else table


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _coerce_utc(dt: datetime | str | None) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sticky_true_sql(col: str, target: str) -> str:
    return f"{col} = CASE WHEN COALESCE(EXCLUDED.{col}, FALSE) THEN TRUE ELSE {target}.{col} END"


def _exists(conn, table_name: str) -> bool:
    if SCHEMA:
        q = text("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :name
            LIMIT 1
        """)
        r = conn.execute(q, {"schema": SCHEMA, "name": table_name}).first()
    else:
        q = text("""
            SELECT 1
            FROM information_schema.tables
            WHERE (table_schema = 'public' OR table_schema = current_schema())
              AND table_name = :name
            LIMIT 1
        """)
        r = conn.execute(q, {"name": table_name}).first()
    return r is not None


# -----------------------------------------------------------------------------
# Visitor writes
# -----------------------------------------------------------------------------
def insert_visitor_created(
    visitor_id: str,
    *,
    created_at: datetime | str | None = None,
    marketing_funnel_stage: str = "Awareness",
    is_identified: bool = False,
) -> None:
    ts = _coerce_utc(created_at)
    target = _qual("visitors")
    sql = text(f"""
        INSERT INTO {target}
            (visitor_id, created_at, marketing_funnel_stage, stage_last_updated_date, is_identified)
        VALUES
            (:visitor_id, COALESCE(:created_at, NOW()), :marketing_funnel_stage, COALESCE(:created_at, NOW()), :is_identified)
        ON CONFLICT (visitor_id) DO NOTHING
    """)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, {
            "visitor_id": str(visitor_id),
            "created_at": ts,
            "marketing_funnel_stage": marketing_funnel_stage,
            "is_identified": is_identified,
        })
    log.info("visitor_created_write", extra={"visitor_id": str(visitor_id)})


def upsert_visitor_on_signup(
    rows: Union[Dict[str, Any], Iterable[Dict[str, Any]]]
) -> int:
    if isinstance(rows, dict):
        rows = [rows]

    norm: List[Dict[str, Any]] = []
    for r in rows:
        vid = r.get("visitor_id")
        if not vid:
            continue
        created_at = _coerce_utc(r.get("created_at"))
        sign_up_timestamp = _coerce_utc(r.get("sign_up_timestamp"))
        stage_ts = _coerce_utc(r.get("stage_last_updated_date")) or sign_up_timestamp or created_at

        norm.append({
            "visitor_id": str(vid),
            "created_at": created_at,
            "name": r.get("name"),
            "gender": r.get("gender"),
            "age": r.get("age"),
            "email": r.get("email"),
            "is_identified": r.get("is_identified", True),
            "signed_up": r.get("signed_up", True),
            "sign_up_timestamp": sign_up_timestamp,
            "marketing_funnel_stage": r.get("marketing_funnel_stage", "Signup"),
            "stage_last_updated_date": stage_ts,
        })

    if not norm:
        return 0

    target = _qual("visitors")
    sql = text(f"""
        INSERT INTO {target} (
            visitor_id, created_at,
            name, gender, age, email,
            is_identified, signed_up, sign_up_timestamp,
            marketing_funnel_stage, stage_last_updated_date
        )
        VALUES (
            :visitor_id, COALESCE(:created_at, NOW()),
            :name, :gender, :age, :email,
            :is_identified, :signed_up, :sign_up_timestamp,
            :marketing_funnel_stage, COALESCE(:stage_last_updated_date, NOW())
        )
        ON CONFLICT (visitor_id) DO UPDATE SET
            { _sticky_true_sql("is_identified", target) },
            { _sticky_true_sql("signed_up",  target) },

            name    = COALESCE(EXCLUDED.name,    {target}.name),
            gender  = COALESCE(EXCLUDED.gender,  {target}.gender),
            age     = COALESCE(EXCLUDED.age,     {target}.age),
            email   = COALESCE(EXCLUDED.email,   {target}.email),

            sign_up_timestamp           = COALESCE(EXCLUDED.sign_up_timestamp,           {target}.sign_up_timestamp),
            marketing_funnel_stage                  = COALESCE(EXCLUDED.marketing_funnel_stage,                  {target}.marketing_funnel_stage),
            stage_last_updated_date  = COALESCE(EXCLUDED.stage_last_updated_date,  {target}.stage_last_updated_date)
        ;
    """)
    t0 = time.perf_counter()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, norm)
    dt = time.perf_counter() - t0
    log.info("visitor_signup_upsert_done", extra={"rows": len(norm), "seconds": round(dt, 3)})
    return len(norm)


def ensure_skeleton_visitors(
    v_ids: Iterable[Union[str, Dict[str, Any]]],
    *,
    created_at: datetime | str | None = None,
) -> int:
    rows: List[Dict[str, Any]] = []
    default_ts = _coerce_utc(created_at)

    for x in v_ids or []:
        if isinstance(x, dict):
            vid = x.get("visitor_id") or x.get("id") or x.get("visitorId")
            ts = _coerce_utc(x.get("created_at")) if x.get("created_at") is not None else default_ts
        else:
            vid = x
            ts = default_ts
        if not vid:
            continue
        rows.append({"visitor_id": str(vid), "created_at": ts})

    if not rows:
        return 0

    target = _qual("visitors")
    sql = text(f"""
        INSERT INTO {target} (visitor_id, created_at)
        VALUES (:visitor_id, COALESCE(:created_at, NOW()))
        ON CONFLICT (visitor_id) DO NOTHING
    """)

    t0 = time.perf_counter()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)
    dt = time.perf_counter() - t0

    log.info("visitor_skeleton_write_done", extra={"rows": len(rows), "seconds": round(dt, 3)})
    return len(rows)


# -----------------------------------------------------------------------------
# Optional: generic enrichment upsert
# -----------------------------------------------------------------------------
IDENTITY_COLS = ("name", "gender", "age", "email")
STICKY_BOOL_COLS = {"is_identified", "signed_up"}


def upsert_visitor_enrichment(
    rows: List[Dict[str, Any]],
    *,
    skip_identity_cols: bool = True,
) -> int:
    if not rows:
        return 0

    norm: List[Dict[str, Any]] = []
    for r in rows:
        if "visitor_id" not in r:
            raise ValueError("upsert_visitor_enrichment: 'visitor_id' required")
        d = dict(r)
        if skip_identity_cols:
            for c in IDENTITY_COLS:
                d.pop(c, None)
        d.pop("created_at", None)
        for k, v in list(d.items()):
            if k.endswith("_at"):
                d[k] = _coerce_utc(v)
        norm.append(d)

    all_cols = set().union(*(r.keys() for r in norm))
    insert_cols = ["visitor_id"] + [c for c in all_cols if c != "visitor_id"]
    target = _qual("visitors")

    set_fragments = []
    for c in insert_cols:
        if c in ("visitor_id",):
            continue
        if c in STICKY_BOOL_COLS:
            set_fragments.append(_sticky_true_sql(c, target))
        else:
            set_fragments.append(f"{c} = COALESCE(EXCLUDED.{c}, {target}.{c})")
    set_sql = ",\n            ".join(set_fragments) or "visitor_id = {target}.visitor_id"

    cols_sql = ", ".join(insert_cols)
    vals_sql = ", ".join(f":{c}" for c in insert_cols)
    sql = text(f"""
        INSERT INTO {target} ({cols_sql})
        VALUES ({vals_sql})
        ON CONFLICT (visitor_id) DO UPDATE SET
            {set_sql}
    """)

    t0 = time.perf_counter()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, norm)
    dt = time.perf_counter() - t0

    log.info("visitors_upsert_done", extra={"rows": len(norm), "seconds": round(dt, 3)})
    return len(norm)


# -----------------------------------------------------------------------------
# Ensure schema (and optionally create minimal tables for dev)
# -----------------------------------------------------------------------------
def ensure_schema_and_tables(*, create_minimal: bool = False) -> None:
    visitors_sql = f"""
    CREATE TABLE IF NOT EXISTS {_qual("visitors")} (
        visitor_id UUID PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        -- session/attribution
        channel TEXT,

        -- identity & profile
        is_identified BOOLEAN NOT NULL DEFAULT FALSE,
        identified_at TIMESTAMPTZ,
        name TEXT,
        gender TEXT,
        age INT,
        email TEXT,

        -- signup
        signed_up BOOLEAN NOT NULL DEFAULT FALSE,
        sign_up_timestamp TIMESTAMPTZ,

        -- behavior / lifecycle
        return_visitor BOOLEAN NOT NULL DEFAULT FALSE,
        marketing_funnel_stage TEXT,
        stage_last_updated_date TIMESTAMPTZ,

        -- conversion
        converted BOOLEAN NOT NULL DEFAULT FALSE,
        converted_timestamp TIMESTAMPTZ
    );
    """

    interactions_sql = f"""
    CREATE TABLE IF NOT EXISTS {_qual("interactions")} (
        interaction_id UUID PRIMARY KEY,
        visitor_id UUID NOT NULL REFERENCES {_qual("visitors")} (visitor_id) ON DELETE CASCADE,
        session_id  UUID REFERENCES {_qual("sessions")} (session_id) ON DELETE SET NULL,
        event_ts    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        event_type  TEXT NOT NULL,
        page        TEXT,
        meta        JSONB
    );
    CREATE INDEX IF NOT EXISTS ix_interactions_visitor ON {_qual("interactions")} (visitor_id);
    CREATE INDEX IF NOT EXISTS ix_interactions_session ON {_qual("interactions")} (session_id);
    CREATE INDEX IF NOT EXISTS ix_interactions_event_ts ON {_qual("interactions")} (event_ts);
    """
    engine = get_engine()
    with engine.begin() as conn:
        if SCHEMA:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))

        missing = []
        for t in ["visitors", "sessions", "interactions"]:
            if not _exists(conn, t):
                missing.append(t)

        if missing and create_minimal:
            # Create minimal tables to unblock local/dev runs
            conn.execute(text(visitors_sql))
            conn.execute(text(interactions_sql))
            log.info("created_minimal_tables", extra={"tables": missing})
        else:
            log.info("schema_ready", extra={
                "schema": SCHEMA or "default",
                "missing": missing or [],
                "hint": "Pass create_minimal=True to create dev tables" if missing else None,
            })

# -----------------------------------------------------------------------------
# Reset tables (robust to partial schemas)
# -----------------------------------------------------------------------------
def reset_tables(mode: str = "truncate") -> None:
    """Dangerous: dev-only helper. TRUNCATE/DELETE only tables that exist."""
    ordered = ["interactions", "sessions", "visitors"]  # dependency order
    engine = get_engine()
    with engine.begin() as conn:
        existing = [t for t in ordered if _exists(conn, t)]

        if not existing:
            log.warning("reset_tables_noop", extra={"reason": "no known tables exist"})
            return

        if mode == "truncate":
            qualified = ", ".join(_qual(t) for t in existing)
            conn.execute(text(f"TRUNCATE TABLE {qualified} RESTART IDENTITY CASCADE"))
        elif mode == "delete":
            for t in existing:
                conn.execute(text(f"DELETE FROM {_qual(t)}"))
        else:
            raise ValueError("reset_tables: mode must be 'truncate' or 'delete'")

    log.info("db_reset_done", extra={"mode": mode, "tables": existing})


__all__ = [
    "engine",
    "SCHEMA",
    "insert_visitor_created",
    "upsert_visitor_on_signup",
    "ensure_skeleton_visitors",
    "upsert_visitor_enrichment",
    "ensure_schema_and_tables",
    "reset_tables",
]

def write_interactions_copy(rows: list[dict], engine: Engine | None = None) -> int:
    """
    High-speed bulk INSERT into interactions using PostgreSQL COPY.
    Falls back to executemany INSERT if COPY isn't available.
    Returns number of input rows.
    """
    if not rows:
        return 0
    
    engine = get_engine()

    # Determine columns to write: intersection of table columns and row keys.
    table_cols = [c.name for c in interactions.c]
    # Avoid autoincrement/PK if named 'id'
    cols = [c for c in table_cols if c != "id" and any(c in r for r in rows)]
    if not cols:
        return 0

    # Build CSV in-memory
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c) for c in cols})
    buf.seek(0)

    # Fully-qualify table name with schema
    schema = interactions.schema or os.getenv("DB_SCHEMA") or (SCHEMA if 'SCHEMA' in globals() else None)
    fq = f'"{schema}".{interactions.name}' if schema else interactions.name

    # Try COPY first; if it fails, fall back to executemany insert
    try:
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            sql = f'COPY {fq} ({", ".join(cols)}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)'
            try:
                # psycopg2 path
                cur.copy_expert(sql, buf)
            except AttributeError:
                # psycopg3 path
                cp = cur.copy(sql)
                cp.write(buf.getvalue().encode("utf-8"))
                cp.close()
            raw.commit()
        finally:
            raw.close()

        logging.getLogger("db").info(
            "interactions_insert_done",
            extra={"event": "interactions_insert_done", "table": fq, "rows_in_batch": len(rows), "path": "copy"},
        )
        return len(rows)

    except Exception:
        logging.getLogger("db").exception("interactions_copy_failed_falling_back_to_executemany")
        # Fallback: executemany INSERT
        normalized = [{c: r.get(c) for c in cols} for r in rows]
        with engine.begin() as conn:
            conn.execute(interactions.insert(), normalized)
        logging.getLogger("db").info(
            "interactions_insert_done",
            extra={"event": "interactions_insert_done", "table": fq, "rows_in_batch": len(rows), "path": "executemany"},
        )
        return len(rows)