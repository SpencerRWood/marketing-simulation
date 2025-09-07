# marketing_simulation/visitor.py
import uuid, random
from typing import Optional
from datetime import datetime, timezone
from mesa import Agent
from faker import Faker
import logging

fake = Faker()
log = logging.getLogger("visitor")

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
        self.name: Optional[str] = None
        self.gender: Optional[str] = None
        self.age: Optional[int] = None
        self.email: Optional[str] = None

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
        self.session_id = f"s_{self.visitor_id}"
        self._rng = rng or random.Random()
        self._faker = faker or Faker()

    # --- safe accessors (read-only) ---
    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def faker(self) -> Faker:
        return self._faker

    # --- state hooks ---
    def identify_visitor(self, ts: datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self.is_identified  = True
        self.identified_at  = self.identified_at or ts
        self.name   = getattr(self, "name", None)   or fake.name()
        self.gender = getattr(self, "gender", None) or fake.random_element(["Male","Female","Other"])
        self.age    = getattr(self, "age", None)    or fake.random_int(18, 75)
        self.email  = getattr(self, "email", None)  or fake.unique.email()

        # DO NOT write here; EOD batch will upsert
        # mark dirty if you like, but not required since your EOD collects signed_up/converted
        # self._enrichment_dirty = True

        log.info("identify_visitor",
                extra={"visitor_id": self.visitor_id, "email": self.email, "identified_at": ts.isoformat()})

    def complete_signup(self, ts: datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self.signed_up = True
        self.sign_up_timestamp = ts
        self.marketing_funnel_stage = "Consideration"
        self.stage_last_updated_date = ts

        if not getattr(self, "is_identified", False):
            self.identify_visitor(ts)  # sets name/gender/age/email in-memory only

        log.info("visitor_signup", extra={"visitor_id": self.visitor_id, "ts": ts.isoformat()})

    def complete_conversion(self, ts: datetime):
        self.converted = True
        self.converted_timestamp = ts
        self.marketing_funnel_stage = "Purchase"
        self.stage_last_updated_date = ts
        log.info("visitor_conversion", extra={"visitor_id": self.visitor_id, "ts": ts.isoformat()})

    def to_dict(self) -> dict:
        return {
            "visitor_id": self.visitor_id,
            "channel": self.channel,
            "return_visitor": self.return_visitor,
            "is_identified": self.is_identified,
            "identified_at": self.identified_at,
            "name": getattr(self, "name", None),
            "gender": getattr(self, "gender", None),
            "age": getattr(self, "age", None),
            "email": getattr(self, "email", None),
            "signed_up": self.signed_up,
            "sign_up_timestamp": self.sign_up_timestamp,
            "converted": self.converted,
            "converted_timestamp": self.converted_timestamp,
            "marketing_funnel_stage": self.marketing_funnel_stage,
            "stage_last_updated_date": self.stage_last_updated_date,
            "created_at": self.created_at,
        }

