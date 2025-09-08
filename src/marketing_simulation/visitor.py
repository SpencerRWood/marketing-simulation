
from __future__ import annotations

import uuid
import random
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from faker import Faker
from mesa import Agent
from marketing_simulation import db_utils
try:
    from marketing_simulation.logging_utils import get_logger  # type: ignore
    log = get_logger("visitor")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    log = logging.getLogger("visitor")

# DB helpers (new eager-write pattern)

fake = Faker()

def _utc_now() -> datetime:
    """Return timezone-aware UTC now()"""
    return datetime.now(timezone.utc)


class VisitorAgent(Agent):
    def __init__(
        self,
        unique_id: str,
        model,
        *,
        channel: str = "Direct",
        return_visitor: bool = False,
        visitor_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        rng: Optional[random.Random] = None,
        faker: Optional[Faker] = None,
    ):
        # IMPORTANT: no super().__init__
        self.unique_id = unique_id
        self.model = model

        # stable anonymous ID
        self.visitor_id = visitor_id or str(uuid.uuid4())
        self.created_at = created_at or datetime.now(timezone.utc)

        # identity / profile
        self.name: Optional[str] = fake.name()
        self.gender: Optional[str] = fake.random_element(["Male","Female","Other"])
        self.age: Optional[int] = fake.random_int(18, 75)
        self.email: Optional[str] = fake.unique.email()

        # funnel state
        self.is_identified = False
        self.identified_at: Optional[datetime] = None
        self.signed_up = False
        self.sign_up_timestamp: Optional[datetime] = None
        self.return_visitor = return_visitor
        self.marketing_funnel_stage = "Awareness"
        self.stage_last_updated_date: Optional[datetime] = None
        self.converted = False
        self.converted_timestamp: Optional[datetime] = None

        # sim-only (use private backing fields to avoid property collisions)
        self.channel = channel
        self.session_id = str(uuid.uuid4())
        self._rng = rng or random.Random()
        self._faker = faker or Faker()
        # Eager DB write of the skeleton visitor row
        try:
            db_utils.insert_visitor_created(
                self.visitor_id,
                created_at=self.created_at,
                marketing_funnel_stage=self.marketing_funnel_stage,
                is_identified=self.is_identified,
            )
        except Exception:
            # Keep the simulation alive; record failure
            log.exception("visitor_initial_write_failed", extra={"visitor_id": self.visitor_id})

    # ---------------------------------------------------------------------
    # Public actions
    # ---------------------------------------------------------------------
    def complete_signup(self, ts: datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self.signed_up = True
        self.sign_up_timestamp = ts
        self.marketing_funnel_stage = "Consideration"
        self.stage_last_updated_date = ts

        # if not getattr(self, "is_identified", False):
        #     self.identify_visitor(ts)  # sets name/gender/age/email in-memory only

        log.info("visitor_signup", extra={"visitor_id": self.visitor_id, "ts": ts.isoformat()})

    def identify(
        self,
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
        gender: Optional[str] = None,
        age: Optional[int] = None,
        ts: Optional[datetime] = None,
        marketing_funnel_stage: Optional[str] = "Identified",
    ) -> None:
        """Record that we captured identity (does NOT mark signup)."""
        if name:   self.name = name
        if email:  self.email = email
        if gender: self.gender = gender
        if age is not None: self.age = int(age)

        self.is_identified = True
        if marketing_funnel_stage:
            self.marketing_funnel_stage = marketing_funnel_stage
        self.stage_last_updated_date = ts or _utc_now()

        payload = {
            "visitor_id": self.visitor_id,
            "created_at": self.created_at,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "email": self.email,
            "is_identified": True,
            # Explicitly keep signup FALSE so sticky-true logic won't flip it on
            "signed_up": False,
            "marketing_funnel_stage": self.marketing_funnel_stage,
            "stage_last_updated_date": self.stage_last_updated_date,
        }

        try:
            # Use signup upsert with is_signed_up=False so it only enriches + sets identified
            db_utils.upsert_visitor_on_signup(payload)
        except Exception:
            log.exception("visitor_identify_upsert_failed", extra={"visitor_id": self.visitor_id})

    def complete_conversion(self, ts: Optional[datetime] = None) -> None:
        """Mark this visitor as signed up (conversion)."""
        ts = ts or _utc_now()
        self.signed_up = True
        self.is_identified = True  # signup implies we know who they are
        self.sign_up_timestamp = ts
        self.marketing_funnel_stage = "Signup"
        self.stage_last_updated_date = ts

        payload = {
            "visitor_id": self.visitor_id,
            "created_at": self.created_at,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "email": self.email,
            "is_identified": True,
            "signed_up": True,
            "sign_up_timestamp": self.sign_up_timestamp,
            "marketing_funnel_stage": self.marketing_funnel_stage,
            "stage_last_updated_date": self.stage_last_updated_date,
        }

        try:
            db_utils.upsert_visitor_on_signup(payload)
        except Exception:
            log.exception("visitor_signup_upsert_failed", extra={"visitor_id": self.visitor_id})

    # ---------------------------------------------------------------------
    # Utils
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "visitor_id": self.visitor_id,
            "created_at": self.created_at,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "email": self.email,
            "is_identified": self.is_identified,
            "signed_up": self.signed_up,
            "sign_up_timestamp": self.sign_up_timestamp,
            "marketing_funnel_stage": self.marketing_funnel_stage,
            "stage_last_updated_date": self.stage_last_updated_date,
        }

    def __repr__(self) -> str:
        return (
            f"VisitorAgent(visitor_id={self.visitor_id}, identified={self.is_identified}, "
            f"signed_up={self.signed_up}, marketing_funnel_stage={self.marketing_funnel_stage})"
        )
