# src/marketing_simulation/simulation.py
from __future__ import annotations

import uuid, random, logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import simpy

from marketing_simulation.website import WebsiteGraph, Session as WebSession
from marketing_simulation.visitor import VisitorAgent
from marketing_simulation import db_utils

try:
    from marketing_simulation.logging_utils import get_logger  # type: ignore
    log = get_logger("sim")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    log = logging.getLogger("sim")


# ---------------------------------------------------------------------------
# Arrival process
# ---------------------------------------------------------------------------
def _generate_arrivals(
    env,
    website: WebsiteGraph,
    channel_name: str,
    rng: random.Random,
    logger: logging.Logger,
    sessions_out: List[WebSession],
    *,
    until_s: float | None = None,
    n_visitors: int | None = None,
    rate_per_s: float | None = None,
):
    """Generate arrivals either by exact count or Poisson process."""
    spawned = 0

    def next_gap():
        return rng.expovariate(rate_per_s) if rate_per_s and rate_per_s > 0 else 0.0

    def _spawn_one():
        nonlocal spawned
        v = VisitorAgent(
            unique_id=str(uuid.uuid4()),
            model=None,
            channel=channel_name,
            rng=rng,
        )
        sess = WebSession(env, website, v, channel=channel_name, logger=logger)
        sessions_out.append(sess)
        env.process(sess.simulate_site_interactions(start_page="landing"))
        spawned += 1
        logger.debug("agent_spawn", extra={"channel": channel_name, "env_now": float(env.now)})

    if n_visitors is not None:
        for _ in range(n_visitors):
            gap = next_gap()
            if gap > 0:
                yield env.timeout(gap)
            _spawn_one()
        logger.info("arrivals_done", extra={"channel": channel_name, "mode": "count", "planned": n_visitors, "actual": spawned})
        return

    if until_s is not None and rate_per_s:
        while env.now < until_s:
            gap = next_gap()
            if env.now + gap >= until_s:
                break
            yield env.timeout(gap)
            _spawn_one()

    logger.info("arrivals_done", extra={"channel": channel_name, "mode": "rate", "window_s": until_s, "actual": spawned})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rows_for_db_from_session_buffer(buf: List[dict]) -> List[dict]:
    return [
        {
            "visitor_id": r.get("visitor_id"),
            "session_id": r.get("session_id"),
            "channel":    r.get("channel"),
            "page":       r.get("page"),
            "interaction": r.get("interaction"),
            "element":     r.get("element"),
            "timestamp":   r.get("timestamp"),
        }
        for r in buf
    ]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def run_simulation(
    *,
    site: WebsiteGraph,
    channel_plan: Dict[str, Dict],
    seed: int = 42,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    grace_period_s: int = 600,
) -> dict:
    rng = random.Random(seed)

    start_dt = (start_dt or getattr(site, "start_dt", None)) or datetime.now(timezone.utc).replace(microsecond=0)
    if end_dt is None or not (end_dt > start_dt):
        raise ValueError("end_dt is required and must be strictly greater than start_dt")

    env = site.env
    window_s = int((end_dt - start_dt).total_seconds())

    sessions: List[WebSession] = []

    def _rate_from_cfg(cfg: Dict) -> Optional[float]:
        if cfg.get("visitors_per_day") is not None:
            return float(cfg["visitors_per_day"]) / 86400.0
        if cfg.get("visitors") is not None and window_s > 0:
            return float(cfg["visitors"]) / float(window_s)
        return None

    for ch, cfg in channel_plan.items():
        rate_per_s = _rate_from_cfg(cfg)
        env.process(_generate_arrivals(
            env=env,
            website=site,
            channel_name=ch,
            rng=rng,
            logger=log,
            sessions_out=sessions,
            until_s=window_s,
            n_visitors=None if rate_per_s else (int(cfg.get("visitors", 0)) if cfg.get("visitors") is not None else None),
            rate_per_s=rate_per_s,
        ))

    log.info("sim_start", extra={
        "run_id": f"run_{seed}_{int(start_dt.timestamp())}",
        "seed": seed,
        "start_dt": start_dt.isoformat(),
        "end_dt": end_dt.isoformat(),
        "sim_window_seconds": window_s,
        "sim_window_hours": round(window_s/3600, 3),
        "grace_period_s": grace_period_s,
        "channels": {
            k: {
                "visitors_per_day": v.get("visitors_per_day"),
                "visitors_total": v.get("visitors"),
            } for k, v in channel_plan.items()
        },
    })

    env.run(until=window_s + int(grace_period_s))

    # -----------------------------------------------------------------------
    # persistence: visitors, interactions, enrichment
    # -----------------------------------------------------------------------
    visitor_ids = {s.visitor.visitor_id for s in sessions}
    db_utils.ensure_skeleton_visitors(visitor_ids, created_at=start_dt)

    all_events = []
    for s in sessions:
        all_events.extend(_rows_for_db_from_session_buffer(s.data))
    db_utils.write_interactions_copy(all_events)

    enriched = []
    for s in sessions:
        v = s.visitor
        if getattr(v, "signed_up", False) or getattr(v, "converted", False):
            enriched.append(v.to_dict())
    db_utils.upsert_visitor_enrichment(enriched)

    conversions = sum(1 for s in sessions if getattr(s.visitor, "converted", False))
    signups = sum(1 for s in sessions if getattr(s.visitor, "signed_up", False))
    log.info("sim_complete", extra={
        "run_id": f"run_{seed}_{int(start_dt.timestamp())}",
        "seed": seed,
        "visitors": len(visitor_ids),
        "events": len(all_events),
        "signups": signups,
        "conversions": conversions,
        "ended_at": site.get_current_time().isoformat(),
    })

    return {
        "run_id": f"run_{seed}_{int(start_dt.timestamp())}",
        "seed": seed,
        "visitors": len(visitor_ids),
        "events": len(all_events),
        "signups": signups,
        "conversions": conversions,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "ended_at": site.get_current_time(),
        "sim_window_seconds": window_s,
    }
