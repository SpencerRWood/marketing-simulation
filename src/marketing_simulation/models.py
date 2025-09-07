# models.py
import os
from sqlalchemy import (
    Table, Column, MetaData, Integer, String, DateTime, Boolean, JSON,
    ForeignKey, Index, func, text
)

SCHEMA = os.getenv("DB_SCHEMA", "public")
metadata = MetaData(schema=SCHEMA)

visitors = Table(
    "visitors", metadata,
    Column("visitor_id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    # optional enrichment fields commonly upserted
    Column("channel", String),                      # <-- add this to store latest/first-touch
    Column("is_identified", Boolean, nullable=False, server_default=text("false")),
    Column("identified_at", DateTime(timezone=True)),
    Column("name", String),
    Column("gender", String),
    Column("age", Integer),
    Column("email", String),
    Column("signed_up", Boolean, nullable=False, server_default=text("false")),
    Column("sign_up_timestamp", DateTime(timezone=True)),
    Column("return_visitor", Boolean, nullable=False, server_default=text("false")),
    Column("marketing_funnel_stage", String, nullable=False, server_default=text("'Awareness'")),
    Column("stage_last_updated_date", DateTime(timezone=True)),
    Column("converted", Boolean, nullable=False, server_default=text("false")),
    Column("converted_timestamp", DateTime(timezone=True)),
)

interactions = Table(
    "interactions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("visitor_id", String, ForeignKey("visitors.visitor_id"), nullable=False),
    Column("session_id", String, nullable=False),
    Column("channel", String),
    Column("page", String, nullable=False),
    Column("interaction", String, nullable=False),
    Column("element", String),
    Column("timestamp", DateTime(timezone=True), nullable=False),
)
Index("ix_interactions_visitor_ts", interactions.c.visitor_id, interactions.c.timestamp)
Index("ix_interactions_session_ts", interactions.c.session_id, interactions.c.timestamp)

sim_runs = Table(
    "sim_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, unique=True, nullable=False),
    Column("started_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("ended_at", DateTime(timezone=True)),
    Column("seed", Integer),
    Column("notes", String),
    Column("schema_version", String, nullable=False, server_default=text("'v1'")),
    Column("success", Boolean, nullable=False, server_default=text("false")),
)
Index("ix_sim_runs_run_id", sim_runs.c.run_id, unique=True)

sim_params = Table(
    "sim_params", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, ForeignKey("sim_runs.run_id"), nullable=False),
    Column("key", String, nullable=False),
    Column("value", JSON, nullable=False),
)
Index("ix_sim_params_run_key", sim_params.c.run_id, sim_params.c.key)
