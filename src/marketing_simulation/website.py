# website.py
import random, uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator
from datetime import datetime, timezone, timedelta

@dataclass
class Page:
    """Page config loaded from YAML."""
    name: str
    dropoff_prob: float = 0.0
    clickable_elements: Optional[List[str]] = None
    account_creation_element: Optional[str] = None   # e.g., "account_creation"
    conversion_element: Optional[str] = None         # e.g., "submit_order"
    transitions: Optional[List[Tuple[str, float]]] = None  # [(next_page, prob), ...]

class WebsiteGraph:
    """
    Holds the site structure and provides a simulation clock.
    env.now is seconds since sim start.
    """
    def __init__(self, env, pages: Dict[str, Page], start_dt: datetime):
        self.env = env
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = start_dt.astimezone(timezone.utc)
        self.start_dt = start_dt
        self.pages = pages or {}

    # ----- Authoritative timestamps (UTC) -----
    def get_current_time(self) -> datetime:
        return self.start_dt + timedelta(seconds=float(self.env.now))

    # ----- Page helpers -----
    def get_page(self, name: str) -> Optional[Page]:
        return self.pages.get(name)

    def next_page(self, current_name: str) -> Optional[str]:
        page = self.get_page(current_name)
        if not page or not page.transitions:
            return None
        names = [t[0] for t in page.transitions]
        probs = [t[1] for t in page.transitions]
        s = sum(probs)
        if s <= 0:
            return None
        probs = [p / s for p in probs]
        return random.choices(names, weights=probs, k=1)[0]

class Session:
    """
    Simulates a single session through the site.
    Requires visitor obj with:
      - attributes: signed_up (bool), converted (bool), visitor_id (str)
      - methods: complete_signup(ts: datetime), complete_conversion(ts: datetime)
    """
    def __init__(self, env, website: WebsiteGraph, visitor, channel="Direct", logger=None):
        self.env = env
        self.website = website
        self.visitor = visitor
        self.channel = channel
        self.log = logger
        self.session_id = str(uuid.uuid4())
        self.data: List[dict] = []

    # ----- Time & logging -----
    def _now(self) -> datetime:
        return self.website.get_current_time()

    def _log(self, payload: dict):
        # Centralized structured logging + capture in-memory for DB writes.
        row = {
            "timestamp": self._now(),
            "visitor_id": self.visitor.visitor_id,
            "session_id": self.session_id,
            "channel": self.channel,
            **payload,
        }
        self.data.append(row)
        if self.log:
            self.log.info(payload.get("interaction", "event"), extra=row)
        return row

    # ----- Core step: visit one page -----
    def visit_page(self, page_name: str) -> Optional[str]:
        page = self.website.get_page(page_name)
        if not page:
            self._log({"interaction": "error", "page": page_name, "reason": "missing_page"})
            return None

        # Pageview
        self._log({"interaction": "pageview", "page": page_name})

        # Dwell time on page
        yield self.env.timeout(1.0)

        # Generate clicks (0..len(clickable_elements)), random subset
        elements = list(page.clickable_elements or [])
        if elements:
            for element in random.sample(elements, random.randint(0, len(elements))):
                # Small delay between clicks
                yield self.env.timeout(0.2)
                self._log({"interaction": "click", "page": page_name, "element": element})

                # Account creation ONLY on its explicit element (deterministic)
                if (page.account_creation_element
                        and element == page.account_creation_element
                        and not self.visitor.signed_up):
                    self.visitor.complete_signup(self._now())
                    self._log({"interaction": "signup", "page": page_name, "element": element})

                # Conversion ONLY on its explicit element AND after signup
                if page.conversion_element and element == page.conversion_element:
                    if not self.visitor.signed_up:
                        self._log({
                            "interaction": "conversion_blocked",
                            "page": page_name,
                            "element": element,
                            "reason": "requires_account"
                        })
                    elif not self.visitor.converted:
                        self.visitor.complete_conversion(self._now())
                        self._log({"interaction": "conversion", "page": page_name, "element": element})

        # Drop-off after interacting with the page
        if random.random() < max(0.0, min(1.0, page.dropoff_prob)):
            self._log({"interaction": "dropoff", "page": page_name})
            return None

        # Navigate to next page (and log the nav click for traceability)
        next_name = self.website.next_page(page_name)
        if not next_name:
            # If transitions exist but probs invalid, optionally log a passive end
            if page.transitions:
                self._log({"interaction": "end_of_path", "page": page_name})
            return None

        yield self.env.timeout(0.2)
        self._log({"interaction": "click", "page": page_name, "element": next_name, "purpose": "navigate"})
        return next_name

    # ----- Full session traversal -----
    def simulate_site_interactions(self, start_page: str = "landing") -> List[dict]:
        current = start_page
        while current:
            nxt = (yield from self.visit_page(current))
            if not nxt:
                break
            current = nxt
        return self.data
