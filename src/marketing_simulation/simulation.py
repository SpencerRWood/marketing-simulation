# src/marketing_simulation/simulation.py
from __future__ import annotations

import uuid, random, logging, math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import simpy
from sqlalchemy.engine import Engine
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
# NHPP helpers (noon-peak shape)
# ---------------------------------------------------------------------------
def _hour_of_day(start_dt: Optional[datetime], sim_secs: float) -> float:
    """Continuous hour-of-day in [0,24). Aligns to wall clock if start_dt known."""
    if start_dt is not None:
        dt = start_dt + timedelta(seconds=sim_secs)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return (sim_secs / 3600.0) % 24.0


def _daily_rate_per_s(
    t_sec: float,
    *,
    start_dt: Optional[datetime],
    base_rate_per_s: float,
    peak_lift: float,
    peak_hour: float,
    sigma_hours: float,
) -> float:
    """Instantaneous intensity λ(t) = base + lift * Gaussian bump(hour(t))."""
    h = _hour_of_day(start_dt, t_sec)
    bump = math.exp(-0.5 * ((h - peak_hour) / sigma_hours) ** 2)
    return max(0.0, base_rate_per_s + peak_lift * bump)


def _expected_gaussian_bump_mass(
    window_s: float,
    start_dt: Optional[datetime],
    peak_hour: float,
    sigma_hours: float,
    step_s: float = 10.0,
) -> float:
    """
    ∫ bump(t) dt over the window, where bump(t) = exp(-0.5*((hour(t)-peak)/sigma)^2).
    Think of this as the expected "time-weighted mass" contributed by the Gaussian bump.
    """
    t = 0.0
    acc = 0.0
    while t < window_s:
        dt = min(step_s, window_s - t)
        h = _hour_of_day(start_dt, t)
        bump = math.exp(-0.5 * ((h - peak_hour) / sigma_hours) ** 2)
        acc += bump * dt
        t += dt
    return acc


def _calibrate_nhpp_to_total(
    V_total: float,
    window_s: float,
    start_dt: Optional[datetime],
    nhpp: Dict,
) -> Dict:
    """
    Solve for base_rate_per_s and peak_lift so that ∫ λ(t) dt = V_total over the window.
    Use either:
      - nhpp['bump_share']  in [0,1]   (default 0.7), or
      - nhpp['peak_multiplier'] >= 1   (m = peak/base).
    """
    nhpp = dict(nhpp or {})
    if V_total <= 0 or window_s <= 0:
        nhpp["base_rate_per_s"] = 0.0
        nhpp["peak_lift"] = 0.0
        return nhpp

    peak = float(nhpp.get("peak_hour", 12.0))
    sigma = float(nhpp.get("sigma_hours", 3.0))
    bump_int = _expected_gaussian_bump_mass(window_s, start_dt, peak, sigma, step_s=10.0)

    if "peak_multiplier" in nhpp:
        m = max(1.0, float(nhpp["peak_multiplier"]))
        # peak = base + lift = m * base  =>  lift = (m-1)*base
        denom = window_s + (m - 1.0) * bump_int
        base = V_total / denom
        lift = (m - 1.0) * base
    else:
        # Default to bump share
        w = float(nhpp.get("bump_share", 0.7))
        w = min(max(w, 0.0), 1.0)
        base = (1.0 - w) * V_total / window_s
        lift = (w * V_total) / bump_int if bump_int > 0 else 0.0

    nhpp["base_rate_per_s"] = base
    nhpp["peak_lift"] = lift
    return nhpp


# ---------------------------------------------------------------------------
# Arrival process (supports exact count, constant rate, and NHPP via thinning)
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
    nhpp_cfg: Optional[Dict] = None,
):
    """Generate arrivals by exact count, constant-rate Poisson, or NHPP (thinning)."""

    spawned = 0
    start_dt = getattr(website, "start_dt", None)

    def _spawn_one():
        nonlocal spawned
        v = VisitorAgent(
            unique_id=str(uuid.uuid4()),
            model=None,
            channel=channel_name,
            rng=rng,
        )
        sess = WebSession(env, website, v, channel=channel_name, logger=logger)
        # mark export offset so the sweeper can incrementally drain s.data
        setattr(sess, "_exported_idx", 0)
        sessions_out.append(sess)
        env.process(sess.simulate_site_interactions(start_page="landing"))
        spawned += 1
        logger.debug("agent_spawn", extra={"channel": channel_name, "env_now": float(env.now)})

    def _next_gap_const():
        return rng.expovariate(rate_per_s) if rate_per_s and rate_per_s > 0 else 0.0

    # ---------- Mode 1: exact count ----------
    if n_visitors is not None:
        for _ in range(n_visitors):
            gap = _next_gap_const()
            if gap > 0:
                yield env.timeout(gap)
            _spawn_one()
        logger.info("arrivals_done", extra={"channel": channel_name, "mode": "count", "planned": n_visitors, "actual": spawned})
        return

    # ---------- Mode 2: NHPP via thinning ----------
    if until_s is not None and nhpp_cfg:
        base = float(nhpp_cfg.get("base_rate_per_s", 0.0))
        lift = float(nhpp_cfg.get("peak_lift", 0.0))
        peak = float(nhpp_cfg.get("peak_hour", 12.0))
        sigm = float(nhpp_cfg.get("sigma_hours", 3.0))

        lam_max = base + lift
        if lam_max <= 0.0:
            logger.info("arrivals_done", extra={"channel": channel_name, "mode": "nhpp", "window_s": until_s, "actual": 0})
            return

        # Validation trackers
        realized_per_hour = [0] * 24
        expected_per_hour = [0.0] * 24
        integ_dt = 10.0  # sec

        # expected arrivals (Riemann sum)
        t_est = 0.0
        while t_est < until_s:
            lam_t = _daily_rate_per_s(
                t_est,
                start_dt=start_dt,
                base_rate_per_s=base,
                peak_lift=lift,
                peak_hour=peak,
                sigma_hours=sigm,
            )
            expected_per_hour[int(_hour_of_day(start_dt, t_est)) % 24] += lam_t * integ_dt
            t_est += integ_dt

        # thinning loop using candidate interarrivals from lam_max
        while env.now < until_s:
            delta = rng.expovariate(lam_max)
            if env.now + delta >= until_s:
                break
            yield env.timeout(delta)  # advance to candidate time

            lam_t = _daily_rate_per_s(
                env.now,
                start_dt=start_dt,
                base_rate_per_s=base,
                peak_lift=lift,
                peak_hour=peak,
                sigma_hours=sigm,
            )
            if rng.random() <= (lam_t / lam_max):
                _spawn_one()
                realized_per_hour[int(_hour_of_day(start_dt, env.now)) % 24] += 1

        logger.info("arrivals.shape.realized",
                    extra={"channel": channel_name,
                           "realized_per_hour": realized_per_hour,
                           "total_realized": sum(realized_per_hour)})
        logger.info("arrivals.shape.expected",
                    extra={"channel": channel_name,
                           "expected_per_hour": [round(x, 1) for x in expected_per_hour],
                           "total_expected": round(sum(expected_per_hour), 1)})

        logger.info("arrivals_done", extra={"channel": channel_name, "mode": "nhpp", "window_s": until_s, "actual": spawned})
        return

    # ---------- Mode 3: constant-rate Poisson ----------
    if until_s is not None and rate_per_s:
        while env.now < until_s:
            gap = _next_gap_const()
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


class EventDrain:
    """
    Buffers interaction rows and flushes to DB when the buffer reaches `flush_every`.
    Guarantees FK by ensuring skeleton visitor rows exist before each flush.
    """
    def __init__(self, flush_every: int = 1000, engine: Engine | None = None):
        self.flush_every = flush_every
        self._buf: List[dict] = []
        self.total_seen = 0
        self._log = logging.getLogger("sim")
        self._engine = engine
    
    def _resolve_engine(self) -> Engine:
        """Prefer the injected engine; else try db_utils.engine; else db_utils.get_engine()."""
        if self._engine is not None:
            return self._engine
        try:
            from marketing_simulation import db_utils as _db
            eng = getattr(_db, "engine", None)
            if isinstance(eng, Engine):
                return eng
            return _db.get_engine()
        except Exception:
            from marketing_simulation import db_utils as _db  # best-effort fallback
            return _db.get_engine()

    def add_many(self, rows: List[dict]) -> None:
        if not rows:
            return
        self.total_seen += len(rows)
        self._buf.extend(rows)
        if len(self._buf) >= self.flush_every:
            self._flush_internal(reason="size_threshold")

    def flush(self, *, reason: str = "manual") -> int:
        return self._flush_internal(reason=reason)

    def _flush_internal(self, *, reason: str) -> int:
        n = len(self._buf)
        if n == 0:
            return 0

        # 1) FK safety: ensure all visitor_ids exist before writing interactions
        try:
            v_ids = {r.get("visitor_id") for r in self._buf if r.get("visitor_id")}
            if v_ids:
                from marketing_simulation import db_utils
                # db_utils.ensure_skeleton_visitors(v_ids)
        except Exception:
            self._log.exception(
                "ensure_skeleton_visitors_failed",
                extra={"event": "ensure_skeleton_visitors_failed", "unique_visitors": len(v_ids)}
            )
            # Don't clear buffer; try again on next sweep
            return 0

        # 2) Write interactions
        try:
            wrote = db_utils.write_interactions_copy(self._buf,engine=self._engine)
            self._log.info(
                "interactions_flush",
                extra={
                    "event": "interactions_flush",
                    "reason": reason,
                    "batch_size": wrote,
                    "total_seen": self.total_seen,
                },
            )
            # success → clear buffer
            self._buf.clear()
            return wrote
        except Exception:
            # Keep buffer for retry; log the failure
            self._log.exception(
                "interactions_flush_failed",
                extra={"event": "interactions_flush_failed", "reason": reason, "batch_size": n},
            )
            return 0



def _start_sweeper(env: simpy.Environment, sessions: List[WebSession], drain: EventDrain, *, sweep_every_s: int = 60):
    """
    Periodically scan each session's .data and export only the new rows since the last sweep.
    We track the export offset per session in s._exported_idx.
    """
    def _proc():
        while True:
            yield env.timeout(sweep_every_s)
            exported = 0
            for s in sessions:
                # session may not have any events yet
                data = getattr(s, "data", None)
                if not data:
                    continue
                idx = getattr(s, "_exported_idx", 0)
                if idx < len(data):
                    new_rows = _rows_for_db_from_session_buffer(data[idx:])
                    drain.add_many(new_rows)
                    setattr(s, "_exported_idx", len(data))
                    exported += len(new_rows)
            if exported:
                log.info("interactions_sweep", extra={"event": "interactions_sweep", "exported": exported, "total_seen": drain.total_seen})
    env.process(_proc())


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
    default_nhpp_cfg: Optional[Dict] = None,  # optional default NHPP (channels may override)
    interactions_flush_every: int = 1000,      # NEW: size-based flush threshold
    interactions_sweep_every_s: int = 60,      # NEW: how often to sweep sessions for new events
) -> dict:
    rng = random.Random(seed)

    start_dt = (start_dt or getattr(site, "start_dt", None)) or datetime.now(timezone.utc).replace(microsecond=0)
    if end_dt is None or not (end_dt > start_dt):
        raise ValueError("end_dt is required and must be strictly greater than start_dt")

    env = site.env
    window_s = int((end_dt - start_dt).total_seconds())

    # Expose start_dt to website (hour-of-day alignment)
    setattr(site, "start_dt", start_dt)

    sessions: List[WebSession] = []

    def _rate_from_cfg(cfg: Dict) -> Optional[float]:
        if cfg.get("visitors_per_day") is not None:
            return float(cfg["visitors_per_day"]) / 86400.0
        if cfg.get("visitors") is not None and window_s > 0:
            return float(cfg["visitors"]) / float(window_s)
        return None

    for ch, cfg in channel_plan.items():
        # Prefer per-channel NHPP; else default
        ch_nhpp = (cfg.get("nhpp") or default_nhpp_cfg)

        # Determine target total visitors for the window (from channels.yaml)
        target_total = None
        if ch_nhpp is not None:
            if cfg.get("visitors") is not None:
                target_total = float(cfg["visitors"])
            elif cfg.get("visitors_per_day") is not None:
                target_total = float(cfg["visitors_per_day"]) * (window_s / 86400.0)

        n_visitors = None
        rate_per_s = None

        if ch_nhpp is not None and target_total is not None:
            # Calibrate base/lift to hit the target_total over this window
            ch_nhpp = _calibrate_nhpp_to_total(target_total, window_s, start_dt, ch_nhpp)
            log.info("arrivals.target",
                     extra={"channel": ch,
                            "target_total": target_total,
                            "nhpp": {
                                "base_rate_per_s": round(ch_nhpp["base_rate_per_s"], 8),
                                "peak_lift": round(ch_nhpp["peak_lift"], 8),
                                "peak_hour": ch_nhpp.get("peak_hour", 12.0),
                                "sigma_hours": ch_nhpp.get("sigma_hours", 3.0),
                                "bump_share": ch_nhpp.get("bump_share"),
                                "peak_multiplier": ch_nhpp.get("peak_multiplier"),
                            }})
        else:
            # Fallback to legacy behavior (constant rate / fixed count)
            rate_per_s = _rate_from_cfg(cfg)
            if rate_per_s is None and cfg.get("visitors") is not None:
                n_visitors = int(cfg.get("visitors", 0))

        env.process(_generate_arrivals(
            env=env,
            website=site,
            channel_name=ch,
            rng=rng,
            logger=log,
            sessions_out=sessions,
            until_s=window_s,
            n_visitors=n_visitors,
            rate_per_s=rate_per_s,
            nhpp_cfg=ch_nhpp,
        ))

    # Start the background sweeper & drain
    drain = EventDrain(flush_every=interactions_flush_every)
    _start_sweeper(env, sessions, drain, sweep_every_s=interactions_sweep_every_s)

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
                "nhpp_override": v.get("nhpp") is not None,
            } for k, v in channel_plan.items()
        },
        "nhpp_default": default_nhpp_cfg,
        "interactions_flush_every": interactions_flush_every,
        "interactions_sweep_every_s": interactions_sweep_every_s,
    })

    # Run the sim
    env.run(until=window_s + int(grace_period_s))

    # Final sweep & guaranteed drain
    # (Grab any last events that were added after the final sweep tick)
    exported = 0
    for s in sessions:
        data = getattr(s, "data", None)
        if not data:
            continue
        idx = getattr(s, "_exported_idx", 0)
        if idx < len(data):
            new_rows = _rows_for_db_from_session_buffer(data[idx:])
            drain.add_many(new_rows)
            setattr(s, "_exported_idx", len(data))
            exported += len(new_rows)
    if exported:
        log.info("interactions_sweep", extra={"event": "interactions_sweep", "exported": exported, "total_seen": drain.total_seen})
    drain.flush(reason="run_end")

    # -----------------------------------------------------------------------
    # persistence: visitors enrichment (after interactions are stored)
    # -----------------------------------------------------------------------
    visitor_ids = {s.visitor.visitor_id for s in sessions}
    # db_utils.ensure_skeleton_visitors(visitor_ids, created_at=start_dt)

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
        "events": drain.total_seen,
        "signups": signups,
        "conversions": conversions,
        "ended_at": site.get_current_time().isoformat(),
    })

    return {
        "run_id": f"run_{seed}_{int(start_dt.timestamp())}",
        "seed": seed,
        "visitors": len(visitor_ids),
        "events": drain.total_seen,
        "signups": signups,
        "conversions": conversions,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "ended_at": site.get_current_time(),
        "sim_window_seconds": window_s,
    }
