# src/main.py
import logging
from pathlib import Path
from datetime import datetime, timezone
import simpy
import random

from marketing_simulation.website import Page, WebsiteGraph
from marketing_simulation.simulation import run_simulation
from marketing_simulation import db_utils
from marketing_simulation.config_utils import load_yaml, website_factory_from_yaml, build_channel_plan
from marketing_simulation.logging_utils import init_logging, get_logger


init_logging(reset=True)
log=get_logger("app")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def main():
    log.info("starting_app")
    sim_cfg   = load_yaml(CONFIG_DIR / "simulation.default.yaml")
    ch_cfg    = load_yaml(CONFIG_DIR / "channels.yaml")
    web_cfg   = load_yaml(CONFIG_DIR / "website.graph.yaml")

    seed  = int(sim_cfg["seed"])
    grace = int(sim_cfg["grace_period_s"])

    # default NHPP config from simulation.default.yaml
    default_nhpp_cfg = sim_cfg.get("nhpp_cfg")
    log.info("params.arrival_profile", extra={"nhpp_default": default_nhpp_cfg})

    start_dt = _parse_utc(sim_cfg["start_dt_utc"])
    end_dt   = _parse_utc(sim_cfg["end_dt_utc"])
    if not (end_dt > start_dt):
        raise ValueError("end_date_utc must be strictly greater than start_date_utc")

    env = simpy.Environment()
    pages = website_factory_from_yaml(web_cfg) or {
        "landing": Page(name="landing", dropoff_prob=1.0, transitions=[], clickable_elements=[])
    }
    site = WebsiteGraph(env, pages, start_dt=start_dt)

    channel_plan = build_channel_plan(ch_cfg)

    db_utils.ensure_schema_and_tables()
    if sim_cfg.get("reset_tables", True):
        db_utils.reset_tables(mode="truncate")
    
    seed = random.randint(1, 100)

    summary = run_simulation(
        site=site,
        channel_plan=channel_plan,
        seed=seed,
        start_dt=start_dt,
        end_dt=end_dt,
        grace_period_s=grace,
        default_nhpp_cfg=default_nhpp_cfg,
    )
    log.info("run_summary", extra=summary)
    print(summary)
    log.info("app_completed")


if __name__ == "__main__":
    main()
